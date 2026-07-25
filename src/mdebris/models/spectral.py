"""A supervised per-pixel spectral classifier, trained on MARIDA.

Why this exists
---------------
Measurement drove this module. An open-vocabulary detector (OWLv2) was benchmarked
on the same imagery and does not transfer to 10 m ground sample distance: it returns
0.13 to 0.21 confidence with boxes covering most of the chip, against 0.67 to 0.79
with tight boxes on natural photographs. A 30 m debris patch is three pixels across
at Sentinel-2 resolution, and the texture and shape cues a photographic prior depends
on are not present.

What *is* present is the spectrum. Each pixel carries 11 reflectance bands, and the
difference between plastic, Sargassum, foam and sediment lives in the relationships
between them. That is a tabular classification problem, not a vision problem, and it
is what the MARIDA authors themselves benchmark with a Random Forest baseline
(Kikaki et al. 2022, PLoS ONE 17(1): e0262247).

So this trains on labelled pixels rather than prompting a pretrained detector. It
runs on CPU in minutes, which matters because the rest of the project assumes no GPU.

Relationship to the zero-shot path
----------------------------------
These are complementary rather than competing. The spectral classifier decides what a
pixel *is*; the open-vocabulary detector is still useful for rejecting whole-object
confusers such as vessels, and for higher-resolution imagery where objects span
enough pixels to look like objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mdebris.types import SurfaceClass

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

log = logging.getLogger(__name__)

__all__ = [
    "FEATURE_BANDS",
    "SpectralClassifier",
    "TrainingReport",
    "build_features",
    "feature_names",
]

# MARIDA ships these 11 Sentinel-2 bands. B09 and B10 (water vapour, cirrus) are
# excluded upstream because they carry no surface signal.
FEATURE_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)

# Index features computed from the bands. Adding these is not redundant with giving
# the model raw bands: a tree splits on one feature at a time, so it cannot express
# a ratio or a three-band baseline subtraction on its own without many splits.
_INDEX_FEATURES: tuple[str, ...] = ("FDI", "FAI", "NDVI", "NDWI", "PI", "KNDVI", "MNDWI")


def feature_names() -> list[str]:
    """Ordered feature names matching the columns produced by :func:`build_features`."""
    return [*FEATURE_BANDS, *_INDEX_FEATURES]


def build_features(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Stack reflectance bands and spectral indices into a per-pixel feature matrix.

    Args:
        bands: Reflectance arrays keyed by Sentinel-2 band id, all the same shape.
            Missing bands are filled with NaN rather than raising, so a scene with a
            reduced band set still runs and the model sees an explicit "unknown".

    Returns:
        Float32 array of shape (n_pixels, n_features). Row order is C-order raveling
        of the input grid, so a prediction can be reshaped straight back.

    Raises:
        ValueError: If no usable band is present or shapes disagree.
    """
    from mdebris.indices.spectral import compute_indices

    present = {k: np.asarray(v, dtype=np.float32) for k, v in bands.items() if k in FEATURE_BANDS}
    if not present:
        raise ValueError(f"none of the expected bands {FEATURE_BANDS} are present")
    shapes = {v.shape for v in present.values()}
    if len(shapes) != 1:
        raise ValueError(f"all bands must share a shape, got {shapes}")
    shape = shapes.pop()
    n = int(np.prod(shape))

    columns: list[np.ndarray] = []
    for name in FEATURE_BANDS:
        arr = present.get(name)
        columns.append(
            np.full(n, np.nan, dtype=np.float32)
            if arr is None
            else arr.reshape(-1).astype(np.float32)
        )

    indices = compute_indices(present)
    for name in _INDEX_FEATURES:
        arr = indices.get(name)
        columns.append(
            np.full(n, np.nan, dtype=np.float32)
            if arr is None
            else np.asarray(arr, dtype=np.float32).reshape(-1)
        )

    return np.column_stack(columns).astype(np.float32)


@dataclass(slots=True)
class TrainingReport:
    """Everything needed to judge whether a trained model is any good."""

    n_train: int
    n_test: int
    classes: list[str]
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0

    def debris_metrics(self) -> dict[str, float]:
        """Metrics for the class the project actually exists to find."""
        return self.per_class.get("Marine Debris", {})

    def to_markdown(self) -> str:
        lines = [
            f"Trained on {self.n_train:,} pixels, tested on {self.n_test:,} held-out pixels.",
            "",
            f"- overall accuracy: **{self.accuracy:.3f}**",
            f"- macro F1 across {len(self.classes)} classes: **{self.macro_f1:.3f}**",
        ]
        debris = self.debris_metrics()
        if debris:
            lines.append(
                f"- Marine Debris: precision **{debris.get('precision', 0):.3f}**, "
                f"recall **{debris.get('recall', 0):.3f}**, "
                f"F1 **{debris.get('f1', 0):.3f}** on {int(debris.get('support', 0)):,} pixels"
            )
        lines += ["", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
        for name in sorted(self.per_class, key=lambda k: -self.per_class[k].get("support", 0)):
            m = self.per_class[name]
            lines.append(
                f"| {name} | {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | "
                f"{m.get('f1', 0):.3f} | {int(m.get('support', 0)):,} |"
            )
        if self.feature_importance:
            top = sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:8]
            lines += ["", "Most informative features:", ""]
            lines += [f"- `{k}` {v:.4f}" for k, v in top]
        return "\n".join(lines)


class SpectralClassifier:
    """Per-pixel classifier over Sentinel-2 reflectance and spectral indices.

    Defaults to a histogram gradient boosting model. It handles NaN natively, which
    matters because no-data and missing bands are routine in satellite imagery, and it
    trains far faster than a forest of comparable accuracy on tabular data of this size.
    """

    name = "spectral"

    def __init__(
        self,
        *,
        model: Any | None = None,
        class_names: Sequence[str] | None = None,
        max_iter: int = 300,
        learning_rate: float = 0.1,
        random_state: int = 0,
        class_weight: str | dict | None = "balanced",
    ) -> None:
        self._model = model
        self.class_names: list[str] = list(class_names) if class_names else []
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._class_weight = class_weight

    # ---- training -----------------------------------------------------------

    def _new_model(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=self._max_iter,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            # Marine Debris is a small fraction of labelled pixels. Without
            # rebalancing, a model maximising accuracy simply predicts water.
            class_weight=self._class_weight,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        class_names: Sequence[str] | None = None,
    ) -> SpectralClassifier:
        """Fit on a feature matrix and integer or string labels."""
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if len(features) != len(labels):
            raise ValueError(f"features ({len(features)}) and labels ({len(labels)}) disagree")
        if class_names:
            self.class_names = list(class_names)
        self._model = self._new_model()
        self._model.fit(features, labels)
        return self

    # ---- inference ----------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("classifier is not trained; call fit() or load()")
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("classifier is not trained; call fit() or load()")
        return self._model.predict_proba(features)

    def predict_image(
        self, bands: dict[str, np.ndarray], *, return_proba: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Classify every pixel of a band stack, returning a label map of the input shape."""
        shape = next(iter(bands.values())).shape
        features = build_features(bands)
        labels = self.predict(features).reshape(shape)
        if not return_proba:
            return labels
        proba = self.predict_proba(features)
        return labels, proba.reshape(*shape, proba.shape[1])

    def debris_probability(self, bands: dict[str, np.ndarray]) -> np.ndarray:
        """Probability of the Marine Debris class per pixel, as a float map.

        More useful than a hard label for downstream thresholding, and it is what the
        cascade should screen on once a trained model is available.
        """
        if self._model is None:
            raise RuntimeError("classifier is not trained; call fit() or load()")
        shape = next(iter(bands.values())).shape
        proba = self.predict_proba(build_features(bands))
        classes = list(self._model.classes_)
        target = "Marine Debris" if "Marine Debris" in classes else None
        if target is None:
            raise ValueError(f"model has no Marine Debris class, has {classes}")
        return proba[:, classes.index(target)].reshape(shape)

    # ---- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if self._model is None:
            raise RuntimeError("nothing to save; the classifier is not trained")
        import joblib

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "class_names": self.class_names,
                "features": feature_names(),
                "version": 1,
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> SpectralClassifier:
        """Load a model saved by :meth:`save`.

        SECURITY: joblib is pickle-based, so loading a model file executes code inside
        it. Only load files you produced yourself with ``mdebris train`` or obtained
        from a source you trust as much as you trust running its code. This project
        never downloads a model file from the network, and does not ship one; the
        format exists so a user can persist their own training run. There is no safe
        way to load an untrusted scikit-learn estimator, which is why the trained
        artifact is deliberately not distributed.
        """
        import joblib

        blob = joblib.load(Path(path))
        expected = feature_names()
        stored = blob.get("features", expected)
        if stored != expected:
            raise ValueError(
                "the saved model expects different features than this version builds:\n"
                f"  saved:   {stored}\n  current: {expected}\n"
                "Retrain with `mdebris train`."
            )
        return cls(model=blob["model"], class_names=blob.get("class_names", []))

    # ---- evaluation ---------------------------------------------------------

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> TrainingReport:
        """Score on held-out data, reporting per-class metrics rather than accuracy alone.

        Accuracy is close to meaningless here: labelled MARIDA pixels are dominated by
        water, so a model that never predicts debris still scores well. The per-class
        recall on Marine Debris is the number that matters.
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )

        predicted = self.predict(features)
        classes = sorted(set(np.concatenate([np.unique(labels), np.unique(predicted)])).__iter__())
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predicted, labels=classes, zero_division=0
        )
        per_class = {
            str(c): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        }
        return TrainingReport(
            n_train=0,
            n_test=len(labels),
            classes=[str(c) for c in classes],
            accuracy=float(accuracy_score(labels, predicted)),
            macro_f1=float(f1_score(labels, predicted, average="macro", zero_division=0)),
            per_class=per_class,
            confusion=confusion_matrix(labels, predicted, labels=classes).tolist(),
        )


def to_surface_class(marida_label: str) -> SurfaceClass:
    """Map a MARIDA class name onto the project's SurfaceClass taxonomy."""
    from mdebris.data import MARIDA_TO_SURFACE

    return MARIDA_TO_SURFACE.get(marida_label, SurfaceClass.UNKNOWN)
