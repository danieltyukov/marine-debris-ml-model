"""Detector base class and the box-level post-processing every detector shares.

Two things live here that are easy to get wrong and expensive to get wrong twice:

1. Lazy loading. A detector is constructed by the CLI, the API and the test suite
   alike, and only some of those code paths ever run a forward pass. Downloading
   500 MB of weights inside ``__init__`` would make ``mdebris --help`` hit the
   network, so weights load on the first ``detect()`` call instead.
2. Tile merging. The scene tiler overlaps tiles on purpose, which means one debris
   patch straddling a seam is detected twice, once per tile, in two different local
   coordinate systems. Merging is not optional bookkeeping: without it every seam
   in the scene produces duplicate detections, and duplicate detections inflate
   every downstream count and area estimate.

``nms`` and ``merge_tile_detections`` are deliberately pure Python plus the
geometry already on :class:`~mdebris.types.BBox`. They carry no torch or
torchvision import, so the correctness tests for them run in milliseconds and do
not need the ``models`` extra installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from mdebris.config import settings
from mdebris.types import BBox, Detection

__all__ = [
    "BaseDetector",
    "ModelLoadError",
    "as_uint8_rgb",
    "clip_detections",
    "merge_tile_detections",
    "nms",
    "offset_detections",
]

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when weights cannot be fetched or the optional extra is missing.

    A distinct type so callers can offer a targeted remedy (install the extra, go
    online, set HF_HOME) instead of surfacing a raw ImportError or HTTP error.
    """


# --------------------------------------------------------------------------------------
# image coercion
# --------------------------------------------------------------------------------------


def as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce an array to the HxWx3 uint8 RGB layout every detector expects.

    Satellite chips arrive in whatever dtype the reader produced: float reflectance
    in [0, 1], a single band, or an RGBA composite. Normalising once here means each
    detector's ``_detect`` can assume one layout, and a wrong-dtype array fails with
    a readable message rather than as a shape error deep inside a processor.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:  # single band, replicate to RGB so pretrained stems see 3 channels
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array, got shape {arr.shape}")
    if arr.shape[2] == 4:  # drop alpha, no detector uses it
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {arr.shape[2]}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        # Float imagery is reflectance in [0, 1] by convention. Anything noticeably
        # above 1 is already in 0..255, so scaling it again would clip to white.
        finite = arr[np.isfinite(arr)]
        scale = 255.0 if (finite.size == 0 or float(finite.max()) <= 1.5) else 1.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0 / scale, neginf=0.0) * scale
        return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Integer imagery wider than 8 bit (Sentinel-2 is 12 bit stored in uint16).
    info = np.iinfo(arr.dtype)
    scaled = arr.astype(np.float64) * (255.0 / float(info.max)) if info.max > 255 else arr
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------------------
# box post-processing
# --------------------------------------------------------------------------------------


def nms(
    dets: Sequence[Detection],
    iou_threshold: float = 0.5,
    *,
    class_agnostic: bool = False,
) -> list[Detection]:
    """Greedy non-maximum suppression, class-aware by default.

    Class-aware means a debris box and a sargassum box may overlap freely: they are
    competing hypotheses about the same water, and the pipeline wants to see both so
    it can reason about the disagreement. Only two boxes with the *same* label
    suppress each other. Pass ``class_agnostic=True`` to force a single winner per
    location, which is what you want just before writing final GeoJSON.

    Ties in score are broken by input order, so the result is deterministic.

    Args:
        dets: Detections in one shared coordinate system.
        iou_threshold: Boxes overlapping more than this are treated as duplicates.
        class_agnostic: Suppress across labels rather than within a label.

    Returns:
        The kept detections, highest score first. Input objects are returned as-is,
        not copied.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold {iou_threshold} outside [0, 1]")
    # Sort by descending score; the index tiebreaker keeps equal scores in input order.
    order = sorted(range(len(dets)), key=lambda i: (-dets[i].score, i))
    kept: list[int] = []
    for i in order:
        cand = dets[i]
        suppressed = False
        for j in kept:
            other = dets[j]
            if not class_agnostic and cand.label is not other.label:
                continue
            if cand.bbox.iou(other.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(i)
    return [dets[i] for i in kept]


def offset_detections(
    dets: Iterable[Detection], dx: float, dy: float
) -> list[Detection]:
    """Shift detections from tile-local pixels into scene pixels.

    Returns new :class:`Detection` objects. The originals are left alone because the
    caller may still hold the per-tile lists for debugging, and because ``indices``
    is a mutable dict that must not be shared between the two views of one detection.

    ``mask`` is carried over unchanged and therefore stays tile-shaped. Run mask
    refinement after merging, not before, if you need scene-aligned masks.
    """
    out: list[Detection] = []
    for d in dets:
        b = d.bbox
        out.append(
            Detection(
                bbox=BBox(b.xmin + dx, b.ymin + dy, b.xmax + dx, b.ymax + dy),
                score=d.score,
                label=d.label,
                geometry=d.geometry,
                tile=d.tile,
                scene=d.scene,
                mask=d.mask,
                indices=dict(d.indices),
                source_model=d.source_model,
            )
        )
    return out


def clip_detections(
    dets: Iterable[Detection], width: int, height: int, *, min_area: float = 1.0
) -> list[Detection]:
    """Clip boxes to a ``width`` x ``height`` frame, dropping anything that vanishes.

    Needed because a detector that pads its input (OWLv2 pads to a square) can place
    a box partly or wholly outside the real image. A box that survives clipping with
    less than ``min_area`` pixels was a padding artefact, not a detection.
    """
    out: list[Detection] = []
    for d in dets:
        b = d.bbox
        xmin, ymin = max(0.0, b.xmin), max(0.0, b.ymin)
        xmax, ymax = min(float(width), b.xmax), min(float(height), b.ymax)
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            continue
        clipped = BBox(xmin, ymin, xmax, ymax)
        if clipped.area < min_area:
            continue
        d.bbox = clipped
        out.append(d)
    return out


def merge_tile_detections(
    tile_detections: Iterable[tuple[tuple[int, int], Sequence[Detection]]],
    *,
    iou_threshold: float | None = None,
    scene_size: tuple[int, int] | None = None,
    class_agnostic: bool = False,
) -> list[Detection]:
    """Offset per-tile detections into scene coordinates and dedupe across seams.

    The scene tiler overlaps neighbouring tiles by ``settings.tile_overlap`` pixels so
    that a debris patch sitting on a seam is fully visible in at least one tile.
    The cost of that is duplication: the patch is detected once per tile it appears
    in, at different local coordinates. This function is where those duplicates are
    reconciled, and it is the only place that knows about tile origins.

    Args:
        tile_detections: Pairs of ``((x_offset, y_offset), detections)``. The offset
            is the tile's top-left corner in **scene pixel** coordinates, x first,
            matching :class:`BBox` ordering rather than array indexing order.
        iou_threshold: IoU above which two boxes are the same object. Defaults to
            ``settings.nms_iou_threshold``.
        scene_size: ``(width, height)`` of the full scene. When given, merged boxes
            are clipped to it, which removes boxes that a padding detector pushed
            past the scene edge.
        class_agnostic: Passed through to :func:`nms`.

    Returns:
        Scene-coordinate detections, highest score first.
    """
    thr = settings.nms_iou_threshold if iou_threshold is None else iou_threshold
    merged: list[Detection] = []
    for (dx, dy), dets in tile_detections:
        merged.extend(offset_detections(dets, float(dx), float(dy)))
    if scene_size is not None:
        merged = clip_detections(merged, scene_size[0], scene_size[1])
    return nms(merged, thr, class_agnostic=class_agnostic)


# --------------------------------------------------------------------------------------
# detector base
# --------------------------------------------------------------------------------------


class BaseDetector(ABC):
    """Shared plumbing for every detector: lazy weights, device, thread count.

    Subclasses implement :meth:`_load` and :meth:`_detect`. They can assume that by
    the time ``_detect`` runs the weights are on ``self.device`` and the image is a
    contiguous HxWx3 uint8 array.

    Satisfies the :class:`~mdebris.types.Detector` protocol.
    """

    #: Human-readable identifier written into ``Detection.source_model``.
    name: str = "base"

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        torch_threads: int | None = None,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._torch_threads = torch_threads
        self.model: Any | None = None
        self.processor: Any | None = None

    # ---- device and threads ----------------------------------------------------

    @property
    def device(self) -> str:
        """Resolved torch device string. Resolution is deferred so that constructing
        a detector does not import torch."""
        if self._device is None:
            self._device = settings.resolve_device()
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _configure_threads(self) -> None:
        import torch

        threads = (
            self._torch_threads if self._torch_threads is not None else settings.torch_threads
        )
        if threads is not None:
            # Only set it when asked. Overriding the torch default unconditionally
            # would fight with an outer parallel harness such as pytest-xdist.
            torch.set_num_threads(int(threads))

    # ---- lazy loading ----------------------------------------------------------

    def load(self) -> None:
        """Fetch weights and move them to the device. Idempotent, safe to call often."""
        if self.is_loaded:
            return
        self._configure_threads()
        try:
            self._load()
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised with an actionable message
            raise ModelLoadError(
                f"could not load {self.model_id!r} for {self.name}: {exc}. "
                "Install the model extra with `pip install 'mdebris[models]'`, check "
                "network access to huggingface.co, or point HF_HOME at a local cache."
            ) from exc
        log.info("loaded %s (%s) on %s", self.name, self.model_id, self.device)

    def unload(self) -> None:
        """Drop the weights. Useful when a long-running API process switches models."""
        self.model = None
        self.processor = None

    @abstractmethod
    def _load(self) -> None:
        """Populate ``self.model`` and ``self.processor``."""

    @abstractmethod
    def _detect(self, image: np.ndarray, threshold: float) -> list[Detection]:
        """Run one image. ``image`` is HxWx3 uint8 and the model is loaded."""

    # ---- public API ------------------------------------------------------------

    def detect(self, image: np.ndarray, *, threshold: float | None = None) -> list[Detection]:
        """Detect objects in an HxWx3 uint8 RGB array.

        Args:
            image: The chip or tile to run on.
            threshold: Score floor. Defaults to ``settings.score_threshold``.

        Returns:
            Detections in the image's own pixel coordinates, highest score first.
        """
        thr = settings.score_threshold if threshold is None else threshold
        if not 0.0 <= thr <= 1.0:
            raise ValueError(f"threshold {thr} outside [0, 1]")
        arr = as_uint8_rgb(image)
        self.load()
        dets = self._detect(arr, thr)
        dets.sort(key=lambda d: -d.score)
        return dets

    def detect_batch(
        self, images: Sequence[np.ndarray], *, threshold: float | None = None
    ) -> list[list[Detection]]:
        """Run several images. The base implementation is a loop.

        Subclasses override this only where a real batched forward pass is measurably
        faster. On CPU it usually is not: the matmuls already saturate every core, so
        batching trades memory for nothing.
        """
        return [self.detect(im, threshold=threshold) for im in images]

    def _to_device(self, inputs: Any) -> Any:
        """Move a processor's BatchFeature to the detector's device."""
        return inputs.to(self.device) if self.device != "cpu" else inputs

    def _autocast(self) -> Any:
        """No-op context manager placeholder kept explicit rather than implied.

        fp16 autocast is a CUDA-only win. On CPU it is slower than fp32 for these
        models, so nothing is enabled here and callers get plain fp32 everywhere.
        """
        import contextlib

        return contextlib.nullcontext()

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "lazy"
        return f"{type(self).__name__}(model_id={self.model_id!r}, device={self._device!r}, {state})"
