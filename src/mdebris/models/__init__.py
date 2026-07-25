"""Detectors, prompt sets and the box post-processing they share.

Import cost matters here. ``nms``, ``merge_tile_detections`` and the prompt sets are
pure Python and are imported eagerly, so the pipeline, the geo module and the tests
can use them without torch being installed at all. The detectors themselves are
exposed through :func:`__getattr__`, so ``from mdebris.models import nms`` does not
drag in transformers, and constructing a detector still does not download weights
(that happens on the first ``detect()`` call).

``Sam2Segmenter`` follows the same rule and is additionally optional at runtime:
mask refinement is a second model and a second forward pass, and the pipeline is
fully useful without it.

Typical use:

    >>> from mdebris.models import OWLv2Detector, DEFAULT_PROMPTS
    >>> det = OWLv2Detector(prompts=DEFAULT_PROMPTS)
    >>> dets = det.detect(chip, threshold=0.1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mdebris.models.base import (
    BaseDetector,
    ModelLoadError,
    as_uint8_rgb,
    clip_detections,
    merge_tile_detections,
    nms,
    offset_detections,
)
from mdebris.models.prompts import (
    CONFUSER_PROMPTS,
    DEFAULT_PROMPTS,
    MINIMAL_PROMPTS,
    PROMPT_SETS,
    TARGET_PROMPTS,
    PromptSet,
    get_prompt_set,
)

if TYPE_CHECKING:  # pragma: no cover - lets type checkers see the lazy names
    from mdebris.models.segment import Sam2Segmenter
    from mdebris.models.supervised import RTDetrDetector
    from mdebris.models.zeroshot import GroundingDinoDetector, OWLv2Detector

__all__ = [
    "CONFUSER_PROMPTS",
    "DEFAULT_PROMPTS",
    "DETECTORS",
    "MINIMAL_PROMPTS",
    "PROMPT_SETS",
    "TARGET_PROMPTS",
    "BaseDetector",
    "GroundingDinoDetector",
    "ModelLoadError",
    "OWLv2Detector",
    "PromptSet",
    "RTDetrDetector",
    "Sam2Segmenter",
    "as_uint8_rgb",
    "clip_detections",
    "get_detector",
    "get_prompt_set",
    "merge_tile_detections",
    "nms",
    "offset_detections",
]

# Attribute name -> module it lives in. Resolved on first access so that importing
# this package never imports torch.
_LAZY: dict[str, str] = {
    "OWLv2Detector": "mdebris.models.zeroshot",
    "GroundingDinoDetector": "mdebris.models.zeroshot",
    "RTDetrDetector": "mdebris.models.supervised",
    "Sam2Segmenter": "mdebris.models.segment",
}

#: Detector names usable from the CLI and config.
DETECTORS: dict[str, str] = {
    "owlv2": "OWLv2Detector",
    "grounding-dino": "GroundingDinoDetector",
    "rtdetr": "RTDetrDetector",
}


def __getattr__(name: str) -> Any:
    if (module := _LAZY.get(name)) is not None:
        import importlib

        return getattr(importlib.import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


def get_detector(name: str = "owlv2", **kwargs: Any) -> BaseDetector:
    """Construct a detector by name without importing the others.

    Args:
        name: One of :data:`DETECTORS`.
        **kwargs: Passed to the detector's constructor.

    Raises:
        KeyError: If the name is not a known detector.
    """
    try:
        cls_name = DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(DETECTORS)}") from None
    return __getattr__(cls_name)(**kwargs)
