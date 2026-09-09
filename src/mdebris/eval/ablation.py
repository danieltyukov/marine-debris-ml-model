"""Sensor band ablation for the per-pixel spectral classifier.

Why this exists
---------------
Every optical satellite carries a different band set. Sentinel-2 has a red edge and
two shortwave infrared bands; PlanetScope has neither, and its 4-band Doves are what
the NASA IMPACT debris labels were drawn on. A score that depends on a band the next
sensor lacks is a score that does not transfer, and the honest way to find out which
bands carry the signal is to train the same model without them and measure.

The mechanism is deliberately dumb. :func:`mask_features` sets the hidden columns of
the feature matrix to NaN, which :class:`~mdebris.models.spectral.SpectralClassifier`
already treats as "missing" because histogram gradient boosting handles NaN natively.
An index that needs a hidden band is hidden with it: "no SWIR" means no FDI, no FAI
and no MNDWI, because that is what a sensor without SWIR actually gives you.

Thresholds
----------
Every threshold here is chosen on one set and scored on another. Picking the best-F1
threshold on the test split and then reporting the F1 at that threshold, as the
earlier sargassum report did, is an optimistic number. :func:`select_threshold` takes
the validation split and :func:`binary_metrics` scores the test split at whatever it
returned, so the two never see the same pixels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdebris.indices.spectral import BAND_ALIASES, INDEX_REGISTRY
from mdebris.models.spectral import FEATURE_BANDS, feature_names

__all__ = [
    "SENSOR_SCENARIOS",
    "binary_metrics",
    "cluster_bootstrap_f1",
    "index_threshold_baseline",
    "mask_features",
    "select_features",
    "select_threshold",
    "surviving_features",
]

#: Band subsets, each standing for a sensor. The values are Sentinel-2 band ids, and
#: a scenario that stands for another sensor keeps the Sentinel-2 band nearest each of
#: that sensor's bands rather than pretending the wavelengths match exactly.
SENSOR_SCENARIOS: dict[str, tuple[str, ...]] = {
    # The baseline: everything MARIDA ships.
    "sentinel2": FEATURE_BANDS,
    # A sensor with no shortwave infrared. FDI, FAI and MNDWI all need B11.
    "no_swir": tuple(b for b in FEATURE_BANDS if b not in ("B11", "B12")),
    # Landsat-like: visible, one NIR, two SWIR, no red edge.
    "no_red_edge": tuple(b for b in FEATURE_BANDS if b not in ("B05", "B06", "B07", "B8A")),
    # PlanetScope SuperDove, 8 bands: coastal blue, blue, green I, green, yellow, red,
    # red edge, NIR. Green I and yellow have no Sentinel-2 counterpart and are dropped.
    # Its NIR (865 nm) is nearer B8A by wavelength, but B08 is the band every NIR
    # index here is defined on, and a sensor with one NIR band would compute its
    # indices from it; B8A would leave the scenario with no index at all.
    "superdove": ("B01", "B02", "B03", "B04", "B05", "B08"),
    # PlanetScope Dove, 4 bands: the imagery the NASA IMPACT labels were drawn on.
    "dove4": ("B02", "B03", "B04", "B08"),
    # What a photographic prior sees.
    "rgb": ("B02", "B03", "B04"),
    # No raw band at all: the seven physically motivated indices alone.
    "indices_only": (),
}


def _canonical(band: str) -> str:
    """Canonical name of a Sentinel-2 band id, ``"B08"`` to ``"nir"``."""
    try:
        return BAND_ALIASES[band.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown band {band!r}") from exc


def _index_features() -> list[str]:
    """The index columns of the feature matrix, in column order."""
    return [f for f in feature_names() if f not in FEATURE_BANDS]


def surviving_features(
    keep_bands: Iterable[str], *, indices_from: Iterable[str] | None = None
) -> list[str]:
    """Feature columns that remain when only ``keep_bands`` are available.

    Args:
        keep_bands: Sentinel-2 band ids the sensor provides.
        indices_from: Bands the indices may be computed from, when that differs from
            ``keep_bands``. The ``indices_only`` scenario keeps no raw band as a
            feature but computes every index, and this is how it says so.

    Returns:
        Surviving feature names in :func:`feature_names` order.
    """
    kept = {b.strip().upper() for b in keep_bands}
    unknown = kept - set(FEATURE_BANDS)
    if unknown:
        raise ValueError(f"not model bands: {sorted(unknown)}")
    source = kept if indices_from is None else {b.strip().upper() for b in indices_from}
    canonical = {_canonical(b) for b in source}

    survivors: set[str] = set(kept)
    for name in _index_features():
        spec = INDEX_REGISTRY[name]
        if all(b in canonical for b in spec.bands):
            survivors.add(name)
    return [f for f in feature_names() if f in survivors]


def mask_features(
    features: np.ndarray,
    keep_bands: Iterable[str],
    *,
    indices_from: Iterable[str] | None = None,
) -> np.ndarray:
    """Copy of ``features`` with every column outside the scenario set to NaN.

    Args:
        features: ``(n_pixels, n_features)`` matrix in :func:`feature_names` order.
        keep_bands: Bands the scenario keeps; see :func:`surviving_features`.
        indices_from: See :func:`surviving_features`.

    Returns:
        A new float32 array. The input is not modified.

    Raises:
        ValueError: If the matrix does not have one column per model feature.
    """
    names = feature_names()
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(
            f"expected {len(names)} feature columns, got shape {x.shape}; "
            "build the matrix with mdebris.models.spectral.build_features"
        )
    keep = set(surviving_features(keep_bands, indices_from=indices_from))
    out = x.copy()
    for j, name in enumerate(names):
        if name not in keep:
            out[:, j] = np.nan
    return out


def select_features(
    features: np.ndarray,
    keep_bands: Iterable[str],
    *,
    indices_from: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """The scenario's surviving columns only, with their names.

    This is what the ablation trains on. :func:`mask_features` keeps the full width
    and NaN-fills, which is right for running a full-width model on a reduced scene,
    but scikit-learn cannot fit on a column that is NaN from top to bottom, so a
    model trained for a scenario sees only the columns that scenario has.

    Returns:
        ``(matrix, names)`` where ``matrix`` is ``(n_pixels, len(names))``.
    """
    names = feature_names()
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(
            f"expected {len(names)} feature columns, got shape {x.shape}; "
            "build the matrix with mdebris.models.spectral.build_features"
        )
    kept = surviving_features(keep_bands, indices_from=indices_from)
    cols = [names.index(n) for n in kept]
    return x[:, cols], kept


def _clean_scores(scores: ArrayLike) -> np.ndarray:
    """Scores as float64 with NaN pushed to -inf, so a NaN is never a positive."""
    s = np.asarray(scores, dtype=np.float64)
    return np.where(np.isnan(s), -np.inf, s)


def select_threshold(truth: ArrayLike, scores: ArrayLike) -> float:
    """The score cut that maximises F1 on this set, positives being ``score >= cut``.

    Ties go to the higher threshold, which is the more conservative operating point.
    Meant to be called on a validation set and applied elsewhere; see the module note.

    Raises:
        ValueError: If ``truth`` has no positive, since F1 is then undefined everywhere.
    """
    y = np.asarray(truth, dtype=bool).ravel()
    s = _clean_scores(scores).ravel()
    if y.shape != s.shape:
        raise ValueError(f"truth {y.shape} and scores {s.shape} disagree")
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("no positive in truth; F1 is undefined at every threshold")

    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    tp = np.cumsum(y[order])
    n_predicted = np.arange(1, len(s) + 1)
    fp = n_predicted - tp
    fn = positives - tp
    f1 = 2 * tp / (2 * tp + fp + fn)

    # Only the last row of a run of equal scores is a real cut: a threshold cannot
    # split two pixels with the same score.
    last_of_run = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    f1 = np.where(last_of_run, f1, -1.0)
    best = int(np.argmax(f1))  # argmax returns the first maximum, i.e. the higher cut
    return float(sorted_scores[best])


def binary_metrics(truth: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, Any]:
    """Counts and rates for ``score >= threshold`` against ``truth``.

    Zero denominators give 0.0 rather than NaN, so a scenario that never fires
    reports precision 0 and recall 0 instead of poisoning a table.
    """
    y = np.asarray(truth, dtype=bool).ravel()
    predicted = _clean_scores(scores).ravel() >= threshold
    if y.shape != predicted.shape:
        raise ValueError(f"truth {y.shape} and scores {predicted.shape} disagree")
    tp = int((y & predicted).sum())
    fp = int((~y & predicted).sum())
    fn = int((y & ~predicted).sum())
    tn = int((~y & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_positive": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def index_threshold_baseline(
    val_truth: ArrayLike,
    val_index: ArrayLike,
    test_truth: ArrayLike,
    test_index: ArrayLike,
) -> dict[str, Any]:
    """A single spectral index thresholded with no learning at all.

    The threshold is chosen on the validation set and the metrics are computed on
    the test set, the same protocol the learned scenarios follow, so the two kinds
    of row in the ablation table are comparable.
    """
    threshold = select_threshold(val_truth, val_index)
    return binary_metrics(test_truth, test_index, threshold)


def cluster_bootstrap_f1(
    truth: ArrayLike,
    scores: ArrayLike,
    threshold: float,
    groups: ArrayLike,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile interval for F1 from a bootstrap over groups, not pixels.

    Pixels inside one MARIDA patch share a scene, a date and an annotator, so a
    pixel bootstrap understates the uncertainty badly. Resampling whole patches
    with replacement is the cluster bootstrap, and it is cheap here because F1 is
    a function of per-patch counts: each resample is a weighted sum of those.

    Args:
        truth: Boolean outcome per pixel.
        scores: Score per pixel; positive when ``score >= threshold``.
        threshold: The cut to evaluate.
        groups: Patch id per pixel, any hashable dtype.
        n_boot: Resamples.
        seed: RNG seed.

    Returns:
        ``f1`` (the point estimate), ``low`` and ``high`` (2.5th and 97.5th
        percentiles), ``n_groups`` and ``n_boot``.
    """
    y = np.asarray(truth, dtype=bool).ravel()
    predicted = _clean_scores(scores).ravel() >= threshold
    g = np.asarray(groups).ravel()
    if y.shape != predicted.shape or g.shape != y.shape:
        raise ValueError(
            f"truth {y.shape}, scores {predicted.shape} and groups {g.shape} must match"
        )
    _ids, index = np.unique(g, return_inverse=True)
    n_groups = int(index.max()) + 1 if index.size else 0
    tp = np.bincount(index, weights=(y & predicted), minlength=n_groups)
    fp = np.bincount(index, weights=(~y & predicted), minlength=n_groups)
    fn = np.bincount(index, weights=(y & ~predicted), minlength=n_groups)

    def _f1(w: np.ndarray) -> np.ndarray:
        t, p, n = w @ tp, w @ fp, w @ fn
        denominator = 2 * t + p + n
        return np.divide(
            2 * t, denominator, out=np.zeros_like(t, dtype=float), where=denominator > 0
        )

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot)
    samples = _f1(weights.astype(float))
    point = float(_f1(np.ones((1, n_groups)))[0])
    return {
        "f1": point,
        "low": float(np.percentile(samples, 2.5)),
        "high": float(np.percentile(samples, 97.5)),
        "n_groups": n_groups,
        "n_boot": int(n_boot),
    }


def scenario_names() -> Sequence[str]:
    """Scenario names in table order."""
    return tuple(SENSOR_SCENARIOS)
