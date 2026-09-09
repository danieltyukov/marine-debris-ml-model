"""Is a probability an uncertainty? Reliability, calibration and abstention.

Why this exists
---------------
A classifier that says "0.9" is only telling a decision maker something if nine of
ten pixels it scores 0.9 are actually the class. Gradient boosting trained with
class rebalancing does not promise that, and a government that asks for an
uncertainty alongside a map is asking for exactly that promise. The tools here
measure how far the sargassum probability is from keeping it, fix what can be fixed
with a monotone remap fitted on held-out data, and define the interval in which the
honest output is "uncertain" rather than a yes or a no.

Everything is numpy over 1-D arrays. Only :class:`IsotonicCalibrator.fit` touches
scikit-learn, and the fitted map is stored as breakpoints, so applying and saving it
needs neither scikit-learn nor pickle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "IsotonicCalibrator",
    "abstention_band",
    "brier_score",
    "expected_calibration_error",
    "reliability_table",
]


def _as_unit_scores(scores: ArrayLike) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size and (np.nanmin(s) < 0.0 or np.nanmax(s) > 1.0):
        raise ValueError("scores must be probabilities in [0, 1]")
    return s


def _as_truth(truth: ArrayLike, n: int) -> np.ndarray:
    y = np.asarray(truth, dtype=bool).ravel()
    if y.size != n:
        raise ValueError(f"truth has {y.size} entries but scores has {n}")
    return y


def reliability_table(truth: ArrayLike, scores: ArrayLike, *, n_bins: int = 10) -> list[dict]:
    """Equal-width score bins with the observed positive fraction in each.

    Args:
        truth: Boolean outcome per pixel.
        scores: Predicted probability per pixel, in ``[0, 1]``.
        n_bins: Number of bins over ``[0, 1]``. A score of exactly 1.0 falls in the
            last bin rather than off the end.

    Returns:
        One dict per bin: ``lower``, ``upper``, ``count``, ``mean_score`` and
        ``fraction_positive``. The two rates are NaN for an empty bin, never 0, so
        an empty bin cannot be mistaken for a confidently wrong one.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    s = _as_unit_scores(scores)
    y = _as_truth(truth, s.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.minimum((s * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        inside = which == b
        count = int(inside.sum())
        rows.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "mean_score": float(s[inside].mean()) if count else float("nan"),
                "fraction_positive": float(y[inside].mean()) if count else float("nan"),
            }
        )
    return rows


def expected_calibration_error(truth: ArrayLike, scores: ArrayLike, *, n_bins: int = 10) -> float:
    """Count-weighted mean gap between predicted and observed rate across bins.

    Zero means every bin's positive fraction equals its mean score. This is the
    standard ECE with equal-width bins; it is what a reliability diagram shows,
    reduced to one number.
    """
    rows = reliability_table(truth, scores, n_bins=n_bins)
    n = sum(r["count"] for r in rows)
    if n == 0:
        return float("nan")
    return float(
        sum(
            r["count"] / n * abs(r["fraction_positive"] - r["mean_score"])
            for r in rows
            if r["count"]
        )
    )


def brier_score(truth: ArrayLike, scores: ArrayLike) -> float:
    """Mean squared error between probability and outcome; 0 is perfect, 1 is worst."""
    s = _as_unit_scores(scores)
    y = _as_truth(truth, s.size)
    if s.size == 0:
        return float("nan")
    return float(np.mean((s - y.astype(np.float64)) ** 2))


@dataclass(slots=True)
class IsotonicCalibrator:
    """A monotone remap from raw score to calibrated probability.

    Fitted with isotonic regression on a held-out set, then stored as the
    breakpoints of the resulting step-and-ramp function so applying it is a
    ``numpy.interp`` and saving it is a JSON file. Monotone by construction, so the
    ranking of pixels, and therefore any threshold-based operating point chosen on
    the raw scores, is preserved up to ties.
    """

    x: np.ndarray | None = None
    y: np.ndarray | None = None
    n_fit: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def fit(self, scores: ArrayLike, truth: ArrayLike) -> IsotonicCalibrator:
        """Fit on validation scores and outcomes. Needs scikit-learn."""
        from sklearn.isotonic import IsotonicRegression

        s = _as_unit_scores(scores)
        t = _as_truth(truth, s.size)
        if s.size < 2:
            raise ValueError("need at least two points to fit a calibrator")
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(s, t.astype(np.float64))
        self.x = np.asarray(model.X_thresholds_, dtype=np.float64)
        self.y = np.asarray(model.y_thresholds_, dtype=np.float64)
        self.n_fit = int(s.size)
        return self

    def apply(self, scores: ArrayLike) -> np.ndarray:
        """Calibrated probability for each raw score, clipped to ``[0, 1]``."""
        if self.x is None or self.y is None:
            raise RuntimeError("calibrator is not fitted; call fit() or load()")
        s = np.asarray(scores, dtype=np.float64)
        out = np.interp(s, self.x, self.y, left=float(self.y[0]), right=float(self.y[-1]))
        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        if self.x is None or self.y is None:
            raise RuntimeError("calibrator is not fitted; nothing to serialise")
        return {
            "kind": "isotonic",
            "version": 1,
            "n_fit": self.n_fit,
            "x": [float(v) for v in self.x],
            "y": [float(v) for v in self.y],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> IsotonicCalibrator:
        if blob.get("kind") != "isotonic":
            raise ValueError(f"not an isotonic calibrator: kind={blob.get('kind')!r}")
        return cls(
            x=np.asarray(blob["x"], dtype=np.float64),
            y=np.asarray(blob["y"], dtype=np.float64),
            n_fit=int(blob.get("n_fit", 0)),
            meta=dict(blob.get("meta", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def abstention_band(
    truth: ArrayLike,
    scores: ArrayLike,
    *,
    high_precision: float = 0.90,
    miss_rate: float = 0.02,
) -> dict[str, Any]:
    """The score interval in which the model should say "uncertain".

    Two edges, each tied to a promise the operating point makes:

    * ``high`` is the lowest cut at which precision still meets ``high_precision``,
      so everything at or above it can be called positive with that guarantee. It
      is ``None`` if no cut reaches the target.
    * ``low`` is the highest cut that still keeps a fraction ``1 - miss_rate`` of the
      true positives at or above it, so everything below it can be called negative
      while missing at most that share of the class.

    Scores in ``[low, high)`` are the abstention band. If the two edges cross, the
    band is empty and ``low`` is set to ``high``.

    Returns:
        ``low``, ``high``, ``n_uncertain``, ``fraction_uncertain``, and how many of
        the uncertain pixels were actually positive and negative.
    """
    if not 0.0 < high_precision <= 1.0:
        raise ValueError("high_precision must be in (0, 1]")
    if not 0.0 <= miss_rate < 1.0:
        raise ValueError("miss_rate must be in [0, 1)")
    s = _as_unit_scores(scores)
    y = _as_truth(truth, s.size)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("no positive in truth; the band is undefined")

    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    tp = np.cumsum(y[order])
    precision = tp / np.arange(1, s.size + 1)
    recall = tp / positives
    last_of_run = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    reachable = last_of_run & (precision >= high_precision)
    high: float | None
    if reachable.any():
        # Highest recall among reachable cuts; ties go to the higher cut.
        pick = int(np.argmax(np.where(reachable, recall, -1.0)))
        high = float(sorted_scores[pick])
    else:
        high = None

    keep = math.ceil((1.0 - miss_rate) * positives)
    keep = min(max(keep, 1), positives)
    positive_scores = np.sort(s[y])[::-1]
    low = float(positive_scores[keep - 1])
    if high is not None and low > high:
        low = high

    uncertain = (s >= low) & (s < high) if high is not None else s >= low
    n_uncertain = int(uncertain.sum())
    return {
        "low": low,
        "high": high,
        "high_precision": float(high_precision),
        "miss_rate": float(miss_rate),
        "n_uncertain": n_uncertain,
        "fraction_uncertain": n_uncertain / s.size if s.size else 0.0,
        "uncertain_positive": int((uncertain & y).sum()),
        "uncertain_negative": int((uncertain & ~y).sum()),
    }
