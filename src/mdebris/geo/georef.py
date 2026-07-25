"""Turn pixel-space detections into geographic features.

This is the port of the legacy geo-registration step. The original built a
per-tile affine, shuffled TensorFlow's ``[ymin, xmin, ymax, xmax]`` boxes into
shapely's ``[xmin, ymin, xmax, ymax]`` order and pushed the result through
``shapely.affinity.affine_transform``. That core is preserved exactly; the
scaffolding around it is not, because it hid three real bugs:

* ``(bboxes * 256).astype(np.int)`` used an alias NumPy removed in 1.24, and the
  integer cast threw away sub-pixel precision for no benefit.
* ``np.squeeze`` collapsed a single detection to a 1-D array, so ``bbox[1]``
  indexed a float and raised ``TypeError``. A bare ``except TypeError: continue``
  wrapped the whole loop, so one detection on a tile silently discarded the
  entire tile. Tiles with exactly one piece of debris are the common case.
* The output path was hardcoded to ``./marine_litter/data_geo/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import affine
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform

from mdebris.config import settings
from mdebris.geo.tiles import tile_affine
from mdebris.types import BBox, Detection, DetectionSet, SurfaceClass, TileRef

if TYPE_CHECKING:  # pragma: no cover - import cost is only paid when actually used
    import geopandas as gpd

__all__ = [
    "WGS84",
    "boxes_to_detections",
    "default_output_path",
    "detections_to_geodataframe",
    "georeference_detections",
    "pixel_bbox_to_polygon",
    "read_geojson",
    "write_geojson",
]

WGS84 = "EPSG:4326"


def pixel_bbox_to_polygon(bbox: BBox, transform: affine.Affine) -> Polygon:
    """Map a pixel-space box through an affine transform into world coordinates.

    All four corners are transformed rather than just the opposite pair, so a
    rotated or sheared transform (legal in a GeoTIFF, though unusual) produces
    the correct rotated quadrilateral instead of a wrong axis-aligned envelope.

    Args:
        bbox: Box in pixel coordinates, ``xmin, ymin, xmax, ymax``.
        transform: Affine mapping ``(column, row)`` to world coordinates.

    Returns:
        The box as a polygon in the transform's target coordinate system.
    """
    # shapely's coefficient order is [a, b, d, e, xoff, yoff], which is not the
    # order the affine package stores them in. Getting this wrong transposes
    # every detection, so the mapping is written out once, here.
    coeffs = [
        transform.a,
        transform.b,
        transform.d,
        transform.e,
        transform.xoff,
        transform.yoff,
    ]
    return affine_transform(box(*bbox.as_xyxy()), coeffs)


def georeference_detections(
    dets: list[Detection],
    *,
    tile: TileRef | None = None,
    transform: affine.Affine | None = None,
    tile_size: int | None = None,
    src_crs: Any | None = None,
) -> list[Detection]:
    """Set ``.geometry`` on every detection, in EPSG:4326.

    Two addressing schemes are supported because the project reads both:

    * **Tile path** (``tile=``): rebuilds the legacy per-tile transform. Output
      is already in degrees, so no reprojection happens.
    * **Raster path** (``transform=``, optionally ``src_crs=``): uses a rasterio
      window transform. Sentinel-2 scenes are in UTM, so the polygons are
      reprojected to EPSG:4326 when ``src_crs`` is anything else.

    Detections are mutated in place and the same list is returned, so this can be
    chained. An empty list is returned unchanged; a single detection is handled
    exactly like a hundred (the legacy code was not).

    Args:
        dets: Detections carrying pixel-space boxes.
        tile: Tile the pixel coordinates belong to. Mutually exclusive with
            ``transform``.
        transform: Affine from pixel to ``src_crs`` coordinates. Mutually
            exclusive with ``tile``.
        tile_size: Tile image side in pixels, used only with ``tile``. Defaults
            to ``settings.tile_size``.
        src_crs: CRS of ``transform``'s output. Anything ``pyproj`` accepts:
            an EPSG code, a CRS string, a ``pyproj.CRS`` or a ``rasterio.crs.CRS``.
            ``None`` means the coordinates are already EPSG:4326.

    Returns:
        The same list, with ``geometry`` populated on each detection.

    Raises:
        ValueError: If neither or both of ``tile`` and ``transform`` are given.
    """
    if (tile is None) == (transform is None):
        raise ValueError("pass exactly one of tile= or transform=")

    if tile is not None:
        affine_tf = tile_affine(tile, tile_size)
        reproject = None
    else:
        affine_tf = transform
        reproject = _wgs84_transformer(src_crs)

    for det in dets:
        poly = pixel_bbox_to_polygon(det.bbox, affine_tf)
        if reproject is not None:
            poly = shapely_transform(reproject, poly)
        det.geometry = poly
        # Provenance the legacy code only wrote into the GeoJSON properties.
        # An existing tile is left alone so a caller can override it.
        if tile is not None and det.tile is None:
            det.tile = tile
    return dets


def _wgs84_transformer(src_crs: Any | None) -> Callable[..., Any] | None:
    """Build a coordinate transform to EPSG:4326, or None when already there."""
    if src_crs is None:
        return None
    from pyproj import CRS, Transformer

    crs = CRS.from_user_input(src_crs)
    if crs.equals(CRS.from_user_input(WGS84)):
        return None
    # always_xy keeps both sides in (x, y) = (lon, lat) order. Without it pyproj
    # honours the authority axis order and EPSG:4326 comes back as (lat, lon),
    # which silently produces mirrored geometry.
    return Transformer.from_crs(crs, CRS.from_user_input(WGS84), always_xy=True).transform


def boxes_to_detections(
    boxes: np.ndarray | Sequence[Sequence[float]],
    scores: np.ndarray | Sequence[float],
    *,
    width: int,
    height: int,
    classes: np.ndarray | Sequence[int] | None = None,
    normalized: bool = True,
    order: str = "yxyx",
    score_threshold: float = 0.0,
    label_map: Mapping[int, SurfaceClass] | None = None,
    source_model: str = "",
) -> list[Detection]:
    """Convert a detector's raw box array into ``Detection`` objects.

    This is the array-shape-handling half of the legacy bug. ``np.atleast_2d``
    is what makes one detection behave like N: the original relied on the array
    already being 2-D after ``np.squeeze``, which is false for a single row, and
    swallowed the resulting ``TypeError`` for the whole tile.

    Pixel coordinates stay floating point. The legacy ``.astype(np.int)`` both
    used a removed NumPy alias and truncated toward zero, shifting every box up
    to one pixel north-west (about 10 m of ground at Sentinel-2 resolution).

    Args:
        boxes: ``(N, 4)`` array of boxes, or a single ``(4,)`` box.
        scores: ``(N,)`` confidence scores, or a scalar.
        width: Image width in pixels, used to denormalize and to clip.
        height: Image height in pixels, used to denormalize and to clip.
        classes: Optional ``(N,)`` integer class ids.
        normalized: True when box coordinates are in ``[0, 1]``.
        order: ``"yxyx"`` (TensorFlow and most detection APIs) or ``"xyxy"``.
        score_threshold: Drop detections scoring below this.
        label_map: Class id to ``SurfaceClass``. Unmapped ids, and every id when
            this is None, become ``SurfaceClass.DEBRIS`` (the legacy model had
            exactly one class).
        source_model: Recorded on each detection for provenance.

    Returns:
        One ``Detection`` per surviving box, in input order.

    Raises:
        ValueError: If ``order`` is unknown or the array shapes disagree.
    """
    if order not in {"yxyx", "xyxy"}:
        raise ValueError(f"order must be 'yxyx' or 'xyxy', got {order!r}")

    box_arr = np.asarray(boxes, dtype=np.float64)
    if box_arr.size == 0:
        return []
    box_arr = np.atleast_2d(box_arr)
    if box_arr.ndim != 2 or box_arr.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {box_arr.shape}")

    score_arr = np.atleast_1d(np.asarray(scores, dtype=np.float64))
    if score_arr.shape[0] != box_arr.shape[0]:
        raise ValueError(
            f"got {box_arr.shape[0]} boxes but {score_arr.shape[0]} scores; they must match"
        )

    if classes is None:
        class_arr = np.zeros(box_arr.shape[0], dtype=np.int64)
    else:
        class_arr = np.atleast_1d(np.asarray(classes)).astype(np.int64)
        if class_arr.shape[0] != box_arr.shape[0]:
            raise ValueError(
                f"got {box_arr.shape[0]} boxes but {class_arr.shape[0]} classes; they must match"
            )

    if normalized:
        box_arr = box_arr * (
            [height, width, height, width] if order == "yxyx" else [width, height, width, height]
        )

    out: list[Detection] = []
    for row, score, cls in zip(box_arr, score_arr, class_arr, strict=True):
        if score < score_threshold:
            continue
        if order == "yxyx":
            ymin, xmin, ymax, xmax = row
        else:
            xmin, ymin, xmax, ymax = row
        # Detectors occasionally emit coordinates slightly outside the frame.
        # Clipping keeps BBox's non-degeneracy check satisfied without dropping
        # an otherwise good detection that grazes the edge.
        xmin, xmax = min(max(xmin, 0.0), width), min(max(xmax, 0.0), width)
        ymin, ymax = min(max(ymin, 0.0), height), min(max(ymax, 0.0), height)
        if xmax < xmin or ymax < ymin:
            continue
        label = SurfaceClass.DEBRIS
        if label_map is not None:
            label = label_map.get(int(cls), SurfaceClass.DEBRIS)
        out.append(
            Detection(
                bbox=BBox(xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax)),
                score=float(np.clip(score, 0.0, 1.0)),
                label=label,
                source_model=source_model,
            )
        )
    return out


def write_geojson(ds: DetectionSet, path: str | Path, *, indent: int | None = 2) -> Path:
    """Write a detection set as a GeoJSON FeatureCollection.

    Parent directories are created. The legacy writer targeted a fixed
    ``./marine_litter/data_geo/`` that had to exist beforehand, so a run on a
    fresh checkout died at the very end after doing all the inference work.

    Args:
        ds: Detections to serialize. Detections without geometry are skipped by
            ``DetectionSet.to_geojson``.
        path: Destination file. A directory is created for its parent.
        indent: JSON indentation. Pass None for a compact file.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(ds.to_geojson(), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def default_output_path(name: str) -> Path:
    """Resolve a GeoJSON output path under the configured output directory.

    Args:
        name: Scene id or other stem. A ``.geojson`` suffix is added if absent.

    Returns:
        Path under ``settings.output_dir``. Nothing is created.
    """
    stem = name if name.endswith(".geojson") else f"{name}.geojson"
    return settings.output_dir / stem


def detections_to_geodataframe(ds: DetectionSet) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from a detection set, for spatial joins and export.

    Args:
        ds: Detections to convert. Those without geometry are dropped.

    Returns:
        A GeoDataFrame in EPSG:4326 whose columns are the GeoJSON feature
        properties. An empty set yields an empty frame that still carries the
        CRS and geometry column, so downstream ``.to_file`` calls do not need a
        special case.
    """
    import geopandas as gpd
    import pandas as pd

    features = [d.to_feature() for d in ds.detections if d.geometry is not None]
    if not features:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=WGS84)
    frame = pd.DataFrame([f["properties"] for f in features])
    geoms = [d.geometry for d in ds.detections if d.geometry is not None]
    return gpd.GeoDataFrame(frame, geometry=geoms, crs=WGS84)


def read_geojson(path: str | Path) -> DetectionSet:
    """Read a DetectionSet back from a GeoJSON file written by :func:`write_geojson`.

    Round-tripping matters for evaluation: predictions and ground truth both arrive as
    GeoJSON on disk, and the scorer needs Detection objects with pixel boxes to compute
    IoU. Geographic polygons are converted back to a pixel-space bounding box in degrees
    so that IoU remains meaningful between two sets in the same CRS.

    Properties that the writer emitted (score, label, tile, scene_id, index values) are
    restored where present. Unknown labels fall back to ``SurfaceClass.UNKNOWN`` rather
    than raising, so a file produced by another tool still loads.

    Args:
        path: Path to a GeoJSON FeatureCollection.

    Returns:
        The detections, with ``geometry`` populated and ``bbox`` derived from it.

    Raises:
        ValueError: If the file is not a FeatureCollection.
    """
    import json as _json

    from shapely.geometry import shape

    from mdebris.types import BBox, Detection, DetectionSet, SceneRef, SurfaceClass

    data = _json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(
            f"{path} is not a GeoJSON FeatureCollection, got type={data.get('type')!r}"
        )

    detections: list[Detection] = []
    scene: SceneRef | None = None
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        geom = shape(geometry)
        props = dict(feature.get("properties") or {})
        minx, miny, maxx, maxy = geom.bounds

        try:
            label = SurfaceClass(props.get("label", "marine_debris"))
        except ValueError:
            label = SurfaceClass.UNKNOWN

        if scene is None and props.get("scene_id"):
            scene = SceneRef(
                scene_id=str(props["scene_id"]),
                collection=str(props.get("collection", "sentinel-2-l2a")),
                datetime=props.get("datetime"),
            )

        reserved = {"score", "label", "model", "tile", "scene_id", "collection", "datetime"}
        indices = {
            k: float(v)
            for k, v in props.items()
            if k not in reserved and isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        detections.append(
            Detection(
                bbox=BBox(xmin=minx, ymin=miny, xmax=maxx, ymax=maxy),
                score=float(props.get("score", 1.0)),
                label=label,
                geometry=geom,
                indices=indices,
                source_model=str(props.get("model", "")),
            )
        )

    return DetectionSet(detections=detections, scene=scene, meta=dict(data.get("properties") or {}))
