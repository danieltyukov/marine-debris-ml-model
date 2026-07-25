"""Scene-level orchestration: imagery in, georeferenced detections out.

This is the layer that composes geo, indices and models. It holds no domain logic
of its own beyond sequencing, which keeps the interesting parts (tile math, index
formulas, model wrappers) independently testable.

The detector is injected rather than constructed here so the pipeline can be
exercised with a stub in tests without touching model weights.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mdebris.config import settings
from mdebris.geo.raster import read_bands, to_rgb, window_transform
from mdebris.geo.tiles import windows_for_raster
from mdebris.models.base import clip_detections, merge_tile_detections
from mdebris.pipeline.cascade import ScreenResult, screen_tile, summarize_screening
from mdebris.types import Detection, DetectionSet, SceneRef

if TYPE_CHECKING:  # pragma: no cover
    import affine

    from mdebris.types import Detector

log = logging.getLogger(__name__)

__all__ = ["SceneResult", "detect_in_arrays", "detect_in_scene"]

# Bands the pipeline requests by default. B06 is included because the Biermann FDI
# formulation uses it as the red baseline, and SCL because cloud masking without it
# is guesswork.
DEFAULT_BANDS = ("B02", "B03", "B04", "B06", "B08", "B11", "SCL")


@dataclass(slots=True)
class SceneResult:
    """Detections for a scene plus the accounting needed to explain the run."""

    detections: DetectionSet
    screening: dict[str, float | int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    tiles: list[ScreenResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def to_geojson(self) -> dict[str, Any]:
        return self.detections.to_geojson()

    def summary(self) -> str:
        """One-line human-readable summary for CLI output."""
        s = self.screening
        secs = self.timings.get("total", 0.0)
        return (
            f"{len(self.detections)} detections | "
            f"{s.get('tiles_accepted', 0)}/{s.get('tiles_total', 0)} tiles screened in | "
            f"{100 * float(s.get('work_avoided', 0.0)):.0f}% detector calls avoided | "
            f"{secs:.1f}s"
        )


def detect_in_arrays(
    bands: Mapping[str, np.ndarray],
    detector: Detector,
    *,
    transform: affine.Affine | None = None,
    crs: Any | None = None,
    scene: SceneRef | None = None,
    threshold: float | None = None,
    use_cascade: bool | None = None,
    tile_size: int | None = None,
    overlap: int | None = None,
    segmenter: Any | None = None,
) -> SceneResult:
    """Run the pipeline over in-memory band arrays.

    Args:
        bands: Reflectance arrays keyed by band name, all on a common grid.
        detector: Anything satisfying the Detector protocol.
        transform: Affine transform of the array grid. Detections are only
            georeferenced when this is supplied.
        crs: CRS of the transform. Reprojected to EPSG:4326 when it is not already.
        scene: Provenance attached to every detection.
        threshold: Detector score threshold. Defaults to settings.
        use_cascade: Screen tiles before detecting. Defaults to settings.
        tile_size: Tile edge in pixels. Defaults to settings (960, matching OWLv2).
        overlap: Tile overlap in pixels, so debris on a seam is not cut in half.
        segmenter: Optional object with ``refine(image, detections)`` for SAM 2.

    Returns:
        A SceneResult with georeferenced detections and run statistics.

    Raises:
        ValueError: If bands are empty or not all the same shape.
    """
    if not bands:
        raise ValueError("detect_in_arrays requires at least one band array")

    shapes = {k: np.asarray(v).shape for k, v in bands.items()}
    distinct = set(shapes.values())
    if len(distinct) != 1:
        raise ValueError(
            f"all bands must share a grid, got {shapes}. "
            "Use geo.read_bands with target_shape to resample first."
        )

    height, width = next(iter(distinct))
    threshold = settings.score_threshold if threshold is None else threshold
    use_cascade = settings.use_cascade if use_cascade is None else use_cascade
    tile_size = settings.tile_size if tile_size is None else tile_size
    overlap = settings.tile_overlap if overlap is None else overlap

    timings: dict[str, float] = {}
    screen_results: list[ScreenResult] = []
    per_tile: list[tuple[tuple[int, int], list[Detection]]] = []

    t_start = time.perf_counter()
    t_screen = 0.0
    t_detect = 0.0

    for col_off, row_off, win_w, win_h in windows_for_raster(width, height, tile_size, overlap):
        rows = slice(row_off, row_off + win_h)
        cols = slice(col_off, col_off + win_w)
        tile_bands = {k: np.asarray(v)[rows, cols] for k, v in bands.items()}

        if use_cascade:
            t0 = time.perf_counter()
            screened = screen_tile(tile_bands)
            t_screen += time.perf_counter() - t0
            screen_results.append(screened)
            if not screened.accepted:
                continue
        else:
            screen_results.append(ScreenResult(accepted=True, reason="cascade disabled"))

        rgb = to_rgb(tile_bands)
        t0 = time.perf_counter()
        dets = detector.detect(rgb, threshold=threshold)
        t_detect += time.perf_counter() - t0

        if segmenter is not None and dets:
            try:
                dets = segmenter.refine(rgb, dets)
            except Exception as exc:
                # Segmentation is a refinement, not a requirement. Losing masks is
                # far better than losing the detections that were already found.
                log.warning("segmentation failed on tile (%d, %d): %s", col_off, row_off, exc)

        # Attach the index values at each detection centroid so a reviewer can see
        # the spectral evidence next to the model's confidence.
        if screen_results[-1].indices:
            _attach_index_values(dets, screen_results[-1].indices)

        per_tile.append(((col_off, row_off), dets))

    merged = merge_tile_detections(per_tile, scene_size=(width, height))
    merged = clip_detections(merged, width, height)

    for det in merged:
        det.scene = scene

    if transform is not None:
        from mdebris.geo.georef import georeference_detections

        merged = georeference_detections(merged, transform=transform, src_crs=crs)

    timings["screen"] = round(t_screen, 3)
    timings["detect"] = round(t_detect, 3)
    timings["total"] = round(time.perf_counter() - t_start, 3)

    stats = summarize_screening(screen_results)
    return SceneResult(
        detections=DetectionSet(
            detections=merged,
            scene=scene,
            meta={"width": width, "height": height, **stats},
        ),
        screening=stats,
        timings=timings,
        tiles=screen_results,
    )


def _attach_index_values(dets: Sequence[Detection], indices: Mapping[str, np.ndarray]) -> None:
    """Record each index value at the detection centroid, in place."""
    for det in dets:
        cx, cy = det.centroid if hasattr(det, "centroid") else det.bbox.centroid
        for name, arr in indices.items():
            a = np.asarray(arr)
            if a.ndim != 2:
                continue
            r = int(np.clip(round(cy), 0, a.shape[0] - 1))
            c = int(np.clip(round(cx), 0, a.shape[1] - 1))
            value = a[r, c]
            if np.isfinite(value):
                det.indices[name] = float(value)


def detect_in_scene(
    hrefs: Mapping[str, str | Path],
    detector: Detector,
    *,
    window: Any | None = None,
    scene: SceneRef | None = None,
    bands: Sequence[str] = DEFAULT_BANDS,
    output: str | Path | None = None,
    **kwargs: Any,
) -> SceneResult:
    """Read a scene (or a window of it) from COG hrefs and run the pipeline.

    Reads are windowed, so screening a coastline pulls the bytes for that window
    over HTTP range requests rather than downloading a multi-gigabyte scene.

    Args:
        hrefs: Band name to COG URL or path. Signed URLs expire, so pass fresh ones.
        detector: Anything satisfying the Detector protocol.
        window: Optional rasterio window limiting the read.
        scene: Provenance for the detections.
        bands: Band names to request. Missing ones are skipped rather than fatal.
        output: If given, write the GeoJSON here.
        **kwargs: Forwarded to detect_in_arrays.

    Returns:
        A SceneResult.

    Raises:
        ValueError: If none of the requested bands are available in hrefs.
    """
    available = {b: hrefs[b] for b in bands if b in hrefs}
    if not available:
        raise ValueError(
            f"none of the requested bands {list(bands)} are present in the supplied "
            f"hrefs {sorted(hrefs)}"
        )
    missing = [b for b in bands if b not in hrefs]
    if missing:
        log.info("bands not available for this scene, continuing without them: %s", missing)

    # The 10 m bands define the working grid. Resampling 20 m bands up to it is a
    # correctness requirement, not a nicety: FDI mixes B08 (10 m) with B11 (20 m),
    # and doing that arithmetic on mismatched grids silently misaligns the result.
    reference = next((b for b in ("B04", "B03", "B02", "B08") if b in available), None)

    arrays = read_bands(available, window, reference=reference)
    transform = None
    crs = None
    ref_href = available.get(reference) if reference else None
    if ref_href is not None:
        try:
            import rasterio

            with rasterio.open(ref_href) as src:
                crs = src.crs
                transform = (
                    window_transform(window, ref_href) if window is not None else src.transform
                )
        except Exception as exc:
            log.warning(
                "could not read geotransform, detections will not be georeferenced: %s", exc
            )

    result = detect_in_arrays(arrays, detector, transform=transform, crs=crs, scene=scene, **kwargs)

    if output is not None:
        from mdebris.geo.georef import write_geojson

        path = write_geojson(result.detections, output)
        result.detections.meta["output"] = str(path)

    return result
