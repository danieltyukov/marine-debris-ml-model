"""Every Sentinel-2 pass over an island for a season, rolled onto named coast segments.

This is the engine behind ``scripts/run_island_season.py`` (and the Bonaire wrapper
``scripts/run_bonaire_season.py``). Per pass it reads the bands over HTTP with range
requests, removes land and mapped mangrove, classifies every non-cloud pixel on the
water side of the surf zones with the MARIDA-trained model, calibrates the sargassum
probability, and rolls the result onto the island's segments. One row per segment per
pass goes to a CSV; the season is then summarised into observability and gap
statistics per segment (:mod:`mdebris.coastal.season`) and written up by
:mod:`mdebris.coastal.report`.

Three states come out per segment per pass rather than two. A pixel is called
sargassum at or above the upper edge (calibrated 0.9), clear below the edge that keeps
98% of true sargassum on MARIDA's validation split, and uncertain in between.

There is no NDWI water gate. A raft has the near-infrared of vegetation, so on MARIDA's
test split no dense sargassum pixel and 3.6% of sparse ones have NDWI above zero: the
gate removes the target. Land is removed with the island's coastline polygons instead,
buffered 15 m seaward, and mapped mangrove and other vegetated wetland is removed with
it, buffered 20 m (:mod:`mdebris.coastal.masks` says why). Observability is the scene
classification cloud fraction over what is left of each zone.

A stationary flag is not floating material, so the runner accumulates per pixel how many
passes flagged it, how many could see it, and on how many it looked like leaf canopy.
The first two give the stationary count per segment, the third the persistent
vegetation diagnostic. Both are honesty checks on the detection columns: they name
flags that are probably not sargassum rather than silently dropping them.

The run is resumable: a scene already in the CSV is skipped. Persistence only counts the
passes read in one invocation, so delete the CSV for a fresh season.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mdebris.coastal.masks import (
    near,
    persistent_vegetation,
    polygon_mask,
    vegetated,
    zone_masks,
)
from mdebris.types import GeoBBox, SceneRef

__all__ = [
    "BANDS",
    "MIN_BLOB_PIXELS",
    "SARGASSUM_CLASSES",
    "STATIONARY_MIN_PASSES",
    "STATIONARY_SHARE",
    "OperatingPoints",
    "PassResult",
    "Persistence",
    "SeasonInputs",
    "SeasonOutputs",
    "load_operating_points",
    "process_pass",
    "run_season",
    "select_granules",
]

log = logging.getLogger("mdebris.season")

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
SARGASSUM_CLASSES = ("Dense Sargassum", "Sparse Sargassum")
MIN_BLOB_PIXELS = 4

# A pixel flagged on at least this share of the passes that could see it, with at
# least STATIONARY_MIN_PASSES such passes, is stationary: floating material is not.
STATIONARY_SHARE = 0.5
STATIONARY_MIN_PASSES = 5
# "Near" persistent vegetation: on it or within this many 10 m pixels of it.
VEGETATION_NEAR_PIXELS = 2


@dataclass(slots=True)
class OperatingPoints:
    """The two probability edges and where they came from."""

    low: float
    high: float
    calibrated: bool
    source: str


def load_operating_points(
    calibration_json: Path, *, calibrated: bool, high: float | None = None
) -> OperatingPoints:
    """The two probability edges.

    With calibrated probabilities the upper edge is a statement about the pixel
    itself: at or above 0.9 the model's own estimate is nine in ten, and the
    calibration report records the precision and recall that cut achieves on the
    held-out split. The lower edge is the one ``eval_calibration.py`` chose on
    validation to keep 98% of true sargassum above it. Without calibration both
    edges fall back to the raw abstention band from the same report.
    """
    blob = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    task = blob["tasks"]["sargassum"]
    if calibrated:
        low = float(task["band_calibrated"]["chosen_on_val"]["low"])
        edge = 0.9 if high is None else float(high)
        return OperatingPoints(
            low=low,
            high=edge,
            calibrated=True,
            source=f"{calibration_json}: calibrated >= {edge} is sargassum, "
            "low edge from band_calibrated",
        )
    chosen = task["band_raw"]["chosen_on_val"]
    if chosen["high"] is None:
        raise SystemExit("the calibration report found no 90%-precision operating point")
    return OperatingPoints(
        low=float(chosen["low"]),
        high=float(chosen["high"]) if high is None else float(high),
        calibrated=False,
        source=f"{calibration_json}:band_raw",
    )


# -- granules ------------------------------------------------------------------------


def select_granules(
    candidates: Sequence[tuple[SceneRef, Any]],
    *,
    tiles_ok: Any,
    zones: Sequence[Any],
    min_cover: float,
) -> tuple[list[SceneRef], dict[str, str]]:
    """Which granules count as passes, and why the others do not.

    Args:
        candidates: ``(scene, footprint)`` pairs from a STAC search; the footprint is
            the granule's data outline in EPSG:4326, or ``None`` if unknown.
        tiles_ok: Predicate on a scene id, true for the wanted tiles and orbits.
        zones: Surf-zone polygons (EPSG:4326) the granule has to cover.
        min_cover: Smallest acceptable share of any one zone inside the footprint.

    One datatake can be published as two granules of the same tile; only the one
    covering most of the least-covered zone is kept, ties going to the later
    processing. A granule that covers less than ``min_cover`` of some zone is left
    out: the rest of that zone would read as no-data, which counts as cloud, and turn
    a pass that never looked at the coast into a blind look.

    Returns:
        The kept scenes in time order, and a reason for every granule left out.
    """

    def cover(footprint: Any) -> float:
        if footprint is None:
            return 1.0
        return min(float(z.intersection(footprint).area / z.area) for z in zones) if zones else 1.0

    groups: dict[tuple[str, str], list[tuple[SceneRef, float]]] = defaultdict(list)
    for scene, footprint in candidates:
        if tiles_ok(scene.scene_id):
            groups[(scene.platform or "", str(scene.datetime))].append((scene, cover(footprint)))
    kept: list[SceneRef] = []
    dropped: dict[str, str] = {}
    for group in groups.values():
        best, best_cover = max(group, key=lambda sc: (sc[1], sc[0].scene_id))
        for scene, c in group:
            if scene is not best:
                dropped[scene.scene_id] = (
                    f"second granule of the same datatake as `{best.scene_id}`; its footprint "
                    f"covers {c:.0%} of the least-covered surf zone, the kept one {best_cover:.0%}"
                )
        if best_cover < min_cover:
            dropped[best.scene_id] = (
                f"its footprint covers only {best_cover:.1%} of one segment's surf zone, so the "
                "rest would read as no-data and count as cloud"
            )
            continue
        kept.append(best)
    kept.sort(key=lambda s: str(s.datetime))
    return kept, dropped


def find_passes(
    aoi: GeoBBox,
    start: str,
    end: str,
    *,
    tiles_ok: Any,
    zones: Sequence[Any],
    min_cover: float,
    limit: int = 1000,
) -> tuple[list[SceneRef], dict[str, str]]:
    """Search the archive and keep one full-cover granule per datatake (network)."""
    from shapely.geometry import shape

    from mdebris.data.stac import scene_ref, search_items

    # Tile cloud below 100%: a granule reported as fully clouded is not counted as a
    # pass, as in every season run so far.
    items = search_items(aoi, start, end, max_cloud=100.0, limit=limit)
    if len(items) >= limit:
        raise SystemExit(f"the scene search hit its limit ({limit}); passes would be missing")
    candidates = [
        (scene_ref(item), shape(item.geometry) if item.geometry else None) for item in items
    ]
    kept, dropped = select_granules(candidates, tiles_ok=tiles_ok, zones=zones, min_cover=min_cover)
    log.info(
        "%d granules over the area, %d kept as passes, %d left out",
        len(items),
        len(kept),
        len(dropped),
    )
    return kept, dropped


# -- one pass ------------------------------------------------------------------------


def _connected_boxes(mask: np.ndarray, *, min_pixels: int) -> list[tuple[int, int, int, int]]:
    from scipy import ndimage

    labelled, _count = ndimage.label(mask)
    boxes = []
    for rows, cols in ndimage.find_objects(labelled):
        if int((labelled[rows, cols] > 0).sum()) < min_pixels:
            continue
        boxes.append((cols.start, rows.start, cols.stop, rows.stop))
    return boxes


def _to_crs(bbox: GeoBBox, crs: Any) -> tuple[float, float, float, float]:
    from pyproj import CRS, Transformer

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform
    west, south = fwd(bbox.west, bbox.south)
    east, north = fwd(bbox.east, bbox.north)
    return west, south, east, north


@dataclass(slots=True)
class PassResult:
    scene: SceneRef
    rows: list[dict]
    zone_cloud_fraction: float
    hits: int
    hit_mask: np.ndarray
    usable_mask: np.ndarray
    vegetated_mask: np.ndarray
    masked: np.ndarray
    zones: dict[str, np.ndarray]
    transform: Any
    crs: Any
    # Kept only when a figure is wanted; None otherwise to bound memory.
    rgb: np.ndarray | None = None
    uncertain_mask: np.ndarray | None = None
    cloud_mask: np.ndarray | None = None
    # Kept only with keep_scores: the usable pixels (flat C-order indices), their
    # calibrated sargassum score and the FAI, FDI and NDVI feature columns.
    scores: dict[str, np.ndarray] | None = None


def process_pass(
    scene: SceneRef,
    *,
    aoi: GeoBBox,
    segments: Sequence[Any],
    zones: Mapping[str, Any],
    land: Sequence[Any],
    mangroves: Sequence[Any] | None,
    clf: Any,
    calibrator: Any | None,
    points: OperatingPoints,
    surf_zone_m: float,
    land_buffer_m: float = 15.0,
    mangrove_buffer_m: float = 20.0,
    keep_arrays: bool = False,
    keep_scores: bool = False,
) -> PassResult:
    """Read one pass over the island and roll it onto the segments.

    Args:
        zones: Surf zone of each segment, EPSG:4326, by segment id.
        land: Land polygons, EPSG:4326, removed with a ``land_buffer_m`` seaward strip.
        mangroves: Vegetated wetland polygons, removed with a ``mangrove_buffer_m``
            strip; ``None`` (or empty) to classify them as the 2.0 runner did.
        keep_arrays: Keep true colour and the class masks, for the pass figure.
        keep_scores: Keep every usable pixel's score and index values, for
            ``scripts/eval_index_vs_model.py``.
    """
    import rasterio

    from mdebris.coastal import aggregate_segments, segment_cloud_fractions
    from mdebris.data import get_scene_assets
    from mdebris.data.scaling import BOA_OFFSET, to_reflectance
    from mdebris.geo.georef import georeference_detections
    from mdebris.geo.raster import read_bands, window_transform
    from mdebris.indices.masks import cloud_mask_from_scl, water_mask
    from mdebris.models.spectral import build_features, feature_names
    from mdebris.types import BBox, Detection, DetectionSet, SurfaceClass

    hrefs = get_scene_assets(scene.scene_id, [*BANDS, "SCL"])
    with rasterio.open(hrefs["B04"]) as src:
        window = (
            rasterio.windows.from_bounds(*_to_crs(aoi, src.crs), transform=src.transform)
            .round_lengths()
            .round_offsets()
        )
        transform = window_transform(window, src.transform)
        crs = src.crs

    arrays = read_bands({b: hrefs[b] for b in BANDS}, window, reference="B04")
    scl = read_bands(
        {"B04": hrefs["B04"], "SCL": hrefs["SCL"]}, window, reference="B04", resampling="nearest"
    )["SCL"].astype(np.int16)
    reflectance = {b: to_reflectance(a, offset=BOA_OFFSET) for b, a in arrays.items()}
    shape = scl.shape

    cloudy = cloud_mask_from_scl(scl)
    water = water_mask(reflectance)
    land_mask = polygon_mask(land, transform, crs, shape, buffer_m=land_buffer_m)
    mangrove_mask = (
        polygon_mask(mangroves, transform, crs, shape, buffer_m=mangrove_buffer_m) & ~land_mask
        if mangroves
        else np.zeros(shape, dtype=bool)
    )
    zone_mask = zone_masks(zones, transform, crs, shape)
    union = np.zeros(shape, dtype=bool)
    for z in zone_mask.values():
        union |= z
    water_side = union & ~land_mask & ~mangrove_mask
    usable = water_side & ~cloudy
    # What the NDWI gate would have thrown away: surf, foam, glint, bright bottom,
    # and every dense raft. Recorded per segment, never used to exclude anything.
    not_water = usable & ~water
    leafy = usable & vegetated(reflectance)

    score_map = np.full(shape, np.nan, dtype=np.float32)
    kept: dict[str, np.ndarray] = {
        "flat": np.flatnonzero(usable).astype(np.int32),
        "score": np.zeros(0, np.float32),
        **{k: np.zeros(0, np.float32) for k in ("FAI", "FDI", "NDVI")},
    }
    if usable.any():
        features = build_features({b: reflectance[b][usable] for b in BANDS})
        if keep_scores:
            names = feature_names()
            kept.update({k: features[:, names.index(k)].copy() for k in ("FAI", "FDI", "NDVI")})
        proba = clf.predict_proba(features)
        classes = list(clf._model.classes_)
        idx = [classes.index(c) for c in SARGASSUM_CLASSES if c in classes]
        score = proba[:, idx].sum(axis=1)
        if calibrator is not None:
            score = calibrator.apply(score)
        score_map[usable] = score
        kept["score"] = np.asarray(score, dtype=np.float32)

    with np.errstate(invalid="ignore"):
        hits = score_map >= points.high
        uncertain = (score_map >= points.low) & (score_map < points.high)

    detections = [
        Detection(
            bbox=BBox(xmin=float(x0), ymin=float(y0), xmax=float(x1), ymax=float(y1)),
            score=float(np.nanmax(score_map[y0:y1, x0:x1])),
            label=SurfaceClass.SARGASSUM,
        )
        for x0, y0, x1, y1 in _connected_boxes(hits, min_pixels=MIN_BLOB_PIXELS)
    ]
    georeference_detections(detections, transform=transform, src_crs=crs)
    ds = DetectionSet(detections=detections, scene=scene)

    valid = ~land_mask & ~mangrove_mask
    scl_clouds = segment_cloud_fractions(
        segments, cloudy, transform, src_crs=crs, surf_zone_m=surf_zone_m, valid_mask=valid
    )
    observed_on = str(scene.datetime)[:10]
    report = aggregate_segments(
        ds, segments, surf_zone_m=surf_zone_m, cloud_fractions=scl_clouds, observed_on=observed_on
    )

    rows = []
    for obs in report:
        zone = zone_mask[obs.segment_id] & water_side
        seen = zone & usable
        row = obs.to_row()
        row.update(
            {
                "platform": scene.platform or "",
                "tile_cloud_pct": round(float(scene.cloud_cover or 0.0), 2),
                "water_side_pixels": int(zone.sum()),
                "scl_cloud_fraction": round(scl_clouds[obs.segment_id], 4),
                "not_water_fraction": (
                    round(float((zone & not_water).sum() / seen.sum()), 4) if seen.any() else ""
                ),
                "usable_pixels": int(seen.sum()),
                "sargassum_pixels": int((zone & hits).sum()),
                "uncertain_pixels": int((zone & uncertain).sum()),
                "max_probability": (
                    round(float(np.nanmax(score_map[seen])), 4) if seen.any() else ""
                ),
                "mangrove_masked_pixels": int((zone_mask[obs.segment_id] & mangrove_mask).sum()),
                "vegetated_pixels": int((zone & leafy).sum()),
                "vegetated_sargassum_pixels": int((zone & leafy & hits).sum()),
            }
        )
        rows.append(row)

    result = PassResult(
        scene=scene,
        rows=rows,
        zone_cloud_fraction=float(cloudy[water_side].mean()) if water_side.any() else 1.0,
        hits=int(hits.sum()),
        hit_mask=hits,
        usable_mask=usable,
        vegetated_mask=leafy,
        masked=mangrove_mask,
        zones={k: (v & water_side) for k, v in zone_mask.items()},
        transform=transform,
        crs=crs,
    )
    if keep_scores:
        result.scores = kept
    if keep_arrays:
        from mdebris.geo.raster import to_rgb

        result.rgb = to_rgb(reflectance)
        result.uncertain_mask = uncertain
        result.cloud_mask = cloudy
    return result


# -- across passes -------------------------------------------------------------------


@dataclass(slots=True)
class Persistence:
    """Per-pixel counts across a run: passes that flagged it, saw it, saw leaves on it."""

    hit_total: np.ndarray
    usable_total: np.ndarray
    zones: dict[str, np.ndarray]
    transform: Any
    crs: Any
    n_passes: int = 0
    vegetated_total: np.ndarray | None = None
    masked: np.ndarray | None = None

    @classmethod
    def start(cls, result: PassResult) -> Persistence:
        shape = result.hit_mask.shape
        return cls(
            hit_total=np.zeros(shape, dtype=np.int16),
            usable_total=np.zeros(shape, dtype=np.int16),
            vegetated_total=np.zeros(shape, dtype=np.int16),
            masked=result.masked.copy(),
            zones=result.zones,
            transform=result.transform,
            crs=result.crs,
        )

    def add(self, result: PassResult) -> None:
        if result.hit_mask.shape != self.hit_total.shape:
            raise ValueError("pass grid differs from the accumulated grid")
        self.hit_total += result.hit_mask
        self.usable_total += result.usable_mask
        if self.vegetated_total is None:
            self.vegetated_total = np.zeros(self.hit_total.shape, dtype=np.int16)
        self.vegetated_total += result.vegetated_mask
        self.n_passes += 1

    @property
    def stationary(self) -> np.ndarray:
        """Pixels flagged on at least STATIONARY_SHARE of the passes that saw them."""
        seen = self.usable_total >= STATIONARY_MIN_PASSES
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(seen, self.hit_total / np.maximum(self.usable_total, 1), 0.0)
        return seen & (share >= STATIONARY_SHARE)

    @property
    def persistent_vegetation(self) -> np.ndarray:
        """Pixels that looked like leaf canopy on at least 90% of 10 or more clear looks."""
        if self.vegetated_total is None:
            return np.zeros(self.hit_total.shape, dtype=bool)
        return persistent_vegetation(self.vegetated_total, self.usable_total)

    def per_segment(self) -> dict[str, dict[str, int | None]]:
        """Per segment: pixels ever flagged, stationary, and the vegetation check.

        A grid written before the vegetation counts existed (version 2.0) gives ``None``
        for the two vegetation numbers: not recorded, which is not the same as zero.
        """
        flagged = self.hit_total > 0
        stationary = self.stationary
        recorded = self.vegetated_total is not None
        leafy = self.persistent_vegetation
        near_leafy = near(leafy, VEGETATION_NEAR_PIXELS)
        return {
            seg: {
                "flagged_pixels": int((zone & flagged).sum()),
                "stationary_pixels": int((zone & stationary).sum()),
                "persistent_vegetation_pixels": int((zone & leafy).sum()) if recorded else None,
                "flagged_near_vegetation": (
                    int((zone & flagged & near_leafy).sum()) if recorded else None
                ),
            }
            for seg, zone in self.zones.items()
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.vegetated_total is not None:
            extra["vegetated_total"] = self.vegetated_total
        if self.masked is not None:
            extra["masked"] = self.masked
        np.savez_compressed(
            path,
            hit_total=self.hit_total,
            usable_total=self.usable_total,
            n_passes=self.n_passes,
            transform=np.asarray(list(self.transform)[:6], dtype=np.float64),
            crs=str(self.crs),
            zone_ids=np.asarray(list(self.zones), dtype=str),
            zones=np.stack(list(self.zones.values())),
            **extra,
        )

    @classmethod
    def load(cls, path: Path) -> Persistence:
        import affine
        from rasterio.crs import CRS

        blob = np.load(path, allow_pickle=False)
        zones = dict(zip(blob["zone_ids"].tolist(), blob["zones"], strict=True))
        return cls(
            hit_total=blob["hit_total"],
            usable_total=blob["usable_total"],
            zones=zones,
            transform=affine.Affine(*blob["transform"].tolist()),
            crs=CRS.from_string(str(blob["crs"])),
            n_passes=int(blob["n_passes"]),
            vegetated_total=blob.get("vegetated_total"),
            masked=blob.get("masked"),
        )


# -- CSV -----------------------------------------------------------------------------


def existing_scene_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return {row["scene_id"] for row in csv.DictReader(fh)}


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# -- the season ----------------------------------------------------------------------


@dataclass(slots=True)
class SeasonOutputs:
    """Where one run writes. Every path can be overridden from the command line."""

    csv: Path
    report: Path
    figure: Path
    pass_figure: Path
    persistence: Path
    persistence_figure: Path

    @classmethod
    def for_prefix(cls, prefix: str, *, docs: Path, assets: Path) -> SeasonOutputs:
        return cls(
            csv=docs / f"{prefix}_season.csv",
            report=docs / f"{prefix}_season.md",
            figure=assets / f"{prefix}_season.png",
            pass_figure=assets / f"{prefix}_pass.png",
            persistence=docs / f"{prefix}_persistence.npz",
            persistence_figure=assets / f"{prefix}_persistence.png",
        )


@dataclass(slots=True)
class SeasonInputs:
    """What one run reads, and how."""

    island: Any  # IslandConfig
    part: Any  # SeasonPart
    start: str
    end: str
    segments_path: Path
    island_path: Path
    mangroves_path: Path | None
    model: Path
    calibrator: Path | None
    calibration_json: Path
    high: float | None = None
    surf_zone_m: float = 500.0
    max_scenes: int | None = None
    command: str = ""
    extra_notes: list[str] = field(default_factory=list)


def _segments_for(inputs: SeasonInputs) -> list[Any]:
    from mdebris.coastal import load_segments

    segments = load_segments(inputs.segments_path)
    wanted = inputs.part.segment_ids
    if wanted:
        by_id = {s.segment_id: s for s in segments}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit(f"segments {missing} are not in {inputs.segments_path}")
        segments = [by_id[w] for w in wanted]
    return segments


def area_of_interest(inputs: SeasonInputs, zones: Mapping[str, Any], land: list[Any]) -> GeoBBox:
    """The part's pinned area, or one derived from the zones and the land they touch."""
    from mdebris.coastal.islands import derive_aoi

    if inputs.part.aoi is not None:
        return inputs.part.aoi
    return derive_aoi(
        list(zones.values()),
        land,
        margin_m=inputs.island.aoi_margin_m,
        utm_epsg=inputs.island.utm_epsg,
    )


def run_season(
    inputs: SeasonInputs,
    outputs: SeasonOutputs,
    *,
    summary_only: bool = False,
    pass_figure_scene: str | None = None,
) -> dict[str, Any]:
    """Run (or resume, or only summarise) one part of an island's season.

    Returns the JSON summary that is also written next to the report.
    """
    from mdebris.coastal import surf_zone
    from mdebris.coastal.masks import load_polygons
    from mdebris.coastal.report import season_markdown
    from mdebris.coastal.season import summarize_history
    from mdebris.eval.calibration import IsotonicCalibrator
    from mdebris.viz.season import pass_figure, persistence_figure, timeline_figure

    island, part = inputs.island, inputs.part
    segments = _segments_for(inputs)
    zones = {s.segment_id: surf_zone(s, inputs.surf_zone_m) for s in segments}
    land = load_polygons(inputs.island_path)
    mangroves = load_polygons(inputs.mangroves_path) if inputs.mangroves_path else None
    aoi = area_of_interest(inputs, zones, land)
    calibrated = inputs.calibrator is not None
    points = load_operating_points(inputs.calibration_json, calibrated=calibrated, high=inputs.high)
    calibrator = IsotonicCalibrator.load(inputs.calibrator) if calibrated else None
    log.info(
        "%s%s: %d segments, area %s; sargassum >= %.3f, uncertain in [%.3f, %.3f), "
        "%s probabilities; mangrove mask %s",
        island.name,
        f" ({part.label})" if part.label else "",
        len(segments),
        aoi,
        points.high,
        points.low,
        points.high,
        "calibrated" if calibrated else "raw",
        f"on ({len(mangroves)} polygons)" if mangroves else "off",
    )
    reader = {
        "aoi": aoi,
        "segments": segments,
        "zones": zones,
        "land": land,
        "mangroves": mangroves,
        "calibrator": calibrator,
        "points": points,
        "surf_zone_m": inputs.surf_zone_m,
        "land_buffer_m": island.land_buffer_m,
        "mangrove_buffer_m": island.mangrove_buffer_m,
    }

    def search() -> tuple[list[SceneRef], dict[str, str]]:
        return find_passes(
            aoi,
            inputs.start,
            inputs.end,
            tiles_ok=part.accepts,
            zones=list(zones.values()),
            min_cover=part.min_cover,
        )

    if pass_figure_scene:
        from mdebris.models.spectral import SpectralClassifier

        clf = SpectralClassifier.load(inputs.model)
        scenes, _dropped = search()
        scene = next(s for s in scenes if s.scene_id == pass_figure_scene)
        result = process_pass(scene, clf=clf, keep_arrays=True, **reader)
        pass_figure(result, segments, outputs.pass_figure, island=island, part=part)
        log.info("wrote %s", outputs.pass_figure)
        return {}

    previous: dict[str, Any] = {}
    json_path = outputs.report.with_suffix(".json")
    if json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
    failures: list[tuple[str, str]] = [tuple(f) for f in previous.get("failures", [])]
    dropped: dict[str, str] = dict(previous.get("dropped", {}))
    persistence: Persistence | None = None
    scenes: list[SceneRef] = []
    clf = None
    if not summary_only:
        from mdebris.models.spectral import SpectralClassifier

        clf = SpectralClassifier.load(inputs.model)
        scenes, dropped = search()
        if inputs.max_scenes:
            scenes = scenes[: inputs.max_scenes]
        done = existing_scene_ids(outputs.csv)
        failures = []
        log.info(
            "%d passes between %s and %s, %d already in %s",
            len(scenes),
            inputs.start,
            inputs.end,
            len(done),
            outputs.csv,
        )
        started_run = time.perf_counter()
        for i, scene in enumerate(scenes, 1):
            if scene.scene_id in done:
                continue
            started = time.perf_counter()
            try:
                result = process_pass(scene, clf=clf, **reader)
            except Exception as exc:
                log.warning("  %s failed: %s: %s", scene.scene_id, type(exc).__name__, exc)
                failures.append((scene.scene_id, f"{type(exc).__name__}: {exc}"))
                continue
            for row in result.rows:
                row["zone_cloud_fraction"] = round(result.zone_cloud_fraction, 4)
            append_rows(outputs.csv, result.rows)
            if persistence is None:
                persistence = Persistence.start(result)
            persistence.add(result)
            verdicts = Counter(r["observability"] for r in result.rows)
            log.info(
                "%3d/%d %s %s tile cloud %5.1f%%  zone cloud %5.1f%%  observed %d partial %d "
                "blind %d  sargassum px %d  (%.0fs)",
                i,
                len(scenes),
                str(scene.datetime)[:10],
                scene.platform or "",
                scene.cloud_cover or 0.0,
                100 * result.zone_cloud_fraction,
                verdicts.get("observed", 0),
                verdicts.get("partial", 0),
                verdicts.get("blind", 0),
                result.hits,
                time.perf_counter() - started,
            )
        log.info(
            "read %d passes in %.1f min", len(scenes), (time.perf_counter() - started_run) / 60
        )

    if persistence is not None:
        persistence.save(outputs.persistence)
        log.info("wrote %s", outputs.persistence)
    elif outputs.persistence.exists():
        persistence = Persistence.load(outputs.persistence)
    per_pixel = persistence.per_segment() if persistence is not None else {}

    rows = read_rows(outputs.csv)
    seasons = summarize_history(rows)
    passes = pass_table(rows)
    pixels = segment_pixels(rows)
    summary = {
        "island": island.key,
        "part": part.key,
        "window": [inputs.start, inputs.end],
        "area_of_interest": [aoi.west, aoi.south, aoi.east, aoi.north],
        "tiles": list(part.tiles),
        "orbits": list(part.orbits),
        "mangrove_mask": bool(mangroves),
        "operating_points": asdict(points),
        "segments": [s.to_row() for s in seasons],
        "stationary": per_pixel,
        "segment_pixels": pixels,
        "passes": passes,
        "dropped": dropped,
        "failures": failures,
    }
    outputs.report.parent.mkdir(parents=True, exist_ok=True)
    outputs.report.write_text(
        season_markdown(
            seasons,
            passes,
            points,
            inputs=inputs,
            segments=segments,
            per_pixel=per_pixel,
            segment_pixels=pixels,
            dropped=dropped,
            failures=failures,
            outputs=outputs,
            mangrove_mask=bool(mangroves),
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s and %s", outputs.report, json_path)

    timeline_figure(rows, segments, outputs.figure, island=island, part=part)
    log.info("wrote %s", outputs.figure)
    if persistence is not None:
        persistence_figure(
            persistence, segments, outputs.persistence_figure, island=island, part=part
        )
        log.info("wrote %s", outputs.persistence_figure)

    if not summary_only and clf is not None:
        best = max(passes.items(), key=lambda kv: kv[1]["sargassum_pixels"], default=None)
        if best and best[1]["sargassum_pixels"] > 0:
            scene = next(s for s in scenes if s.scene_id == best[0])
            log.info("re-reading %s for the pass figure", scene.scene_id)
            result = process_pass(scene, clf=clf, keep_arrays=True, **reader)
            pass_figure(result, segments, outputs.pass_figure, island=island, part=part)
            log.info("wrote %s", outputs.pass_figure)
    return summary


def pass_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """One record per pass: date, platform, cloud, verdict counts, flagged pixels."""
    passes: dict[str, dict[str, Any]] = {}
    for r in rows:
        meta = passes.setdefault(
            r["scene_id"],
            {
                "date": r["observed_on"],
                "platform": r.get("platform", ""),
                "tile_cloud_pct": float(r.get("tile_cloud_pct") or 0.0),
                "zone_cloud_fraction": float(r.get("zone_cloud_fraction") or 0.0),
                "scl_cloud_fractions": [],
                "not_water_fractions": [],
                "observed": 0,
                "partial": 0,
                "blind": 0,
                "sargassum_pixels": 0,
                "affected": [],
            },
        )
        meta[r["observability"]] += 1
        meta["scl_cloud_fractions"].append(float(r.get("scl_cloud_fraction") or 0.0))
        if r.get("not_water_fraction") not in ("", None):
            meta["not_water_fractions"].append(float(r["not_water_fraction"]))
        meta["sargassum_pixels"] += int(r.get("sargassum_pixels") or 0)
        if r["observability"] != "blind" and int(r.get("detection_count") or 0) > 0:
            meta["affected"].append(r["name"])
    return passes


def segment_pixels(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per segment: water-side pixels and pixels the mangrove mask removed.

    Both are fixed by the grid and the masks, so every pass gives the same numbers;
    the last row of each segment is kept.
    """
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out[str(r["segment_id"])] = {
            "water_side_pixels": int(r.get("water_side_pixels") or 0),
            "mangrove_masked_pixels": int(r.get("mangrove_masked_pixels") or 0),
        }
    return out
