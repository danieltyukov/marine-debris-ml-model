"""Every Sentinel-2 pass over Bonaire for a season, rolled onto named coast segments.

    python scripts/run_bonaire_season.py --start 2025-01-01 --end 2025-08-31

The question this answers is the one a monitoring proposal for a small island has
to answer before anything else: how often can the windward coast be seen at all
from an optical satellite, and how long does it go unseen. Each pass is read over
HTTP with range requests, the water pixels inside the surf zones are classified
with the MARIDA-trained model, the sargassum probability is calibrated with the
remap from ``scripts/eval_calibration.py``, and the result is rolled onto the
segments in ``assets/bonaire_segments.geojson`` with the same aggregation the
Cancun brief uses. One row per segment per pass goes to a CSV, and the season is
then summarised into observability and gap statistics per segment.

Three states come out per segment per pass rather than two. A pixel is called
sargassum above the 90%-precision edge, clear below the edge that keeps 98% of
true sargassum, and uncertain in between. Both edges were chosen on MARIDA's
validation split, and the counts of each are recorded so nobody has to take the
detection column on trust.

Land is removed with the island polygon in ``assets/bonaire_island.geojson``,
buffered 15 m seaward so the shoreline fringe of sand does not leak in, and every
remaining pixel that is not cloud by the scene classification layer is classified.
There is no NDWI water gate here, on purpose. The Cancun brief gated on water to
keep bright sand out of a model that has no land class, and that gate removes the
target: on MARIDA's test split no dense sargassum pixel and 3.6% of sparse ones
have NDWI above zero, because a raft has the near-infrared of vegetation. With a
real coastline the gate is not needed, and the Wageningen report on Bonaire (van
der Geest et al. 2024, C023/24) masked the island plus a 10 m seaward strip the
same way. Observability is the scene-classification cloud fraction over the water
side of each zone; the share of classified pixels the water gate would have
rejected (surf, foam, floating material) is kept as a column for the record.

A stationary "detection" is not sargassum. The classifier was trained on MARIDA,
which has no shallow lagoon in it, and the back-bay of Lac is flagged on clear
passes in January when nothing has arrived. So the runner also accumulates, per
pixel, how many passes flagged it and how many passes could see it, writes the
ratio as ``docs/bonaire_persistence.npz`` and a figure, and reports per segment how
many flagged pixels are stationary, meaning flagged on at least half of the passes
that saw them. Those are the lagoon bottom, the reef flat and the mangrove edge,
and they are subtracted from nothing: they are named, which is what a reader needs.

The run is resumable: a scene already in the CSV is skipped, so a network failure
halfway through costs one scene, not the season. Persistence counts only the
passes processed in one invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mdebris.coastal import aggregate_segments, load_segments, segment_cloud_fractions, surf_zone
from mdebris.coastal.season import SegmentSeason, summarize_history
from mdebris.data import get_scene_assets, search_scenes
from mdebris.data.scaling import BOA_OFFSET, to_reflectance
from mdebris.eval.calibration import IsotonicCalibrator
from mdebris.geo.georef import georeference_detections
from mdebris.geo.raster import read_bands, window_transform
from mdebris.indices.masks import cloud_mask_from_scl, water_mask
from mdebris.models.spectral import SpectralClassifier, build_features
from mdebris.types import BBox, Detection, DetectionSet, GeoBBox, SceneRef, SurfaceClass

log = logging.getLogger("bonaire")

# The island and its nearshore waters. Everything sits on Sentinel-2 tile 19PEP.
BONAIRE = GeoBBox(west=-68.43, south=12.01, east=-68.18, north=12.32)
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
SARGASSUM_CLASSES = ("Dense Sargassum", "Sparse Sargassum")
MIN_BLOB_PIXELS = 4

LAND_BUFFER_M = 15.0
# A pixel flagged on at least this share of the passes that could see it, with at
# least STATIONARY_MIN_PASSES such passes, is a stationary confuser, not a raft.
STATIONARY_SHARE = 0.5
STATIONARY_MIN_PASSES = 5

EXTRA_COLUMNS = (
    "platform",
    "tile_cloud_pct",
    "water_side_pixels",
    "scl_cloud_fraction",
    "not_water_fraction",
    "usable_pixels",
    "sargassum_pixels",
    "uncertain_pixels",
    "max_probability",
)


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
    calibration report records the precision and recall that cut actually achieves
    on the held-out split. The lower edge is the one ``eval_calibration.py`` chose on
    validation to keep 98% of true sargassum above it. Without calibration both
    edges fall back to the raw abstention band from the same report.
    """
    blob = json.loads(calibration_json.read_text(encoding="utf-8"))
    task = blob["tasks"]["sargassum"]
    if calibrated:
        low = float(task["band_calibrated"]["chosen_on_val"]["low"])
        return OperatingPoints(
            low=low,
            high=0.9 if high is None else float(high),
            calibrated=True,
            source=f"{calibration_json}: calibrated >= {0.9 if high is None else high} is sargassum, "
            f"low edge from band_calibrated",
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


def _connected_boxes(mask: np.ndarray, *, min_pixels: int) -> list[tuple[int, int, int, int]]:
    from scipy import ndimage

    labelled, _count = ndimage.label(mask)
    boxes = []
    for rows, cols in ndimage.find_objects(labelled):
        if int((labelled[rows, cols] > 0).sum()) < min_pixels:
            continue
        boxes.append((cols.start, rows.start, cols.stop, rows.stop))
    return boxes


def _to_crs(bbox: GeoBBox, crs) -> tuple[float, float, float, float]:
    from pyproj import CRS, Transformer

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform
    west, south = fwd(bbox.west, bbox.south)
    east, north = fwd(bbox.east, bbox.north)
    return west, south, east, north


def _zone_masks(segments, transform, crs, shape, surf_zone_m: float) -> dict[str, np.ndarray]:
    """Boolean raster of each segment's surf zone on the scene grid."""
    from pyproj import CRS, Transformer
    from rasterio.features import geometry_mask
    from shapely.ops import transform as shapely_transform

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform
    masks = {}
    for segment in segments:
        zone = shapely_transform(fwd, surf_zone(segment, surf_zone_m))
        masks[segment.segment_id] = geometry_mask(
            [zone], out_shape=shape, transform=transform, invert=True
        )
    return masks


def _land_mask(
    island_path: Path, transform, crs, shape, *, buffer_m: float = LAND_BUFFER_M
) -> np.ndarray:
    """True on land and within ``buffer_m`` of it, from the island polygon, on the scene grid."""
    from pyproj import CRS, Transformer
    from rasterio.features import geometry_mask
    from shapely.geometry import shape as to_shape
    from shapely.ops import transform as shapely_transform

    blob = json.loads(island_path.read_text(encoding="utf-8"))
    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform
    polygons = [
        shapely_transform(fwd, to_shape(f["geometry"])).buffer(buffer_m) for f in blob["features"]
    ]
    return geometry_mask(
        polygons, out_shape=shape, transform=transform, invert=True, all_touched=True
    )


@dataclass(slots=True)
class PassResult:
    scene: SceneRef
    rows: list[dict]
    zone_cloud_fraction: float
    hits: int
    hit_mask: np.ndarray
    usable_mask: np.ndarray
    zones: dict[str, np.ndarray]
    transform: object
    crs: object
    # Kept only when a figure is wanted; None otherwise to bound memory.
    rgb: np.ndarray | None = None
    uncertain_mask: np.ndarray | None = None
    cloud_mask: np.ndarray | None = None


def process_pass(
    scene: SceneRef,
    *,
    segments,
    island: Path,
    clf: SpectralClassifier,
    calibrator: IsotonicCalibrator | None,
    points: OperatingPoints,
    surf_zone_m: float,
    keep_arrays: bool = False,
) -> PassResult:
    """Read one pass over the island and roll it onto the segments."""
    import rasterio

    hrefs = get_scene_assets(scene.scene_id, [*BANDS, "SCL"])
    with rasterio.open(hrefs["B04"]) as src:
        window = (
            rasterio.windows.from_bounds(*_to_crs(BONAIRE, src.crs), transform=src.transform)
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
    land = _land_mask(island, transform, crs, shape)
    zones = _zone_masks(segments, transform, crs, shape, surf_zone_m)
    union = np.zeros(shape, dtype=bool)
    for z in zones.values():
        union |= z
    water_side = union & ~land
    usable = water_side & ~cloudy
    # What the NDWI gate would have thrown away: surf, foam, glint, bright bottom,
    # and every dense raft. Recorded per segment, never used to exclude anything.
    not_water = usable & ~water

    score_map = np.full(shape, np.nan, dtype=np.float32)
    if usable.any():
        features = build_features({b: reflectance[b][usable] for b in BANDS})
        proba = clf.predict_proba(features)
        classes = list(clf._model.classes_)
        idx = [classes.index(c) for c in SARGASSUM_CLASSES if c in classes]
        score = proba[:, idx].sum(axis=1)
        if calibrator is not None:
            score = calibrator.apply(score)
        score_map[usable] = score

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

    valid = ~land
    scl_clouds = segment_cloud_fractions(
        segments, cloudy, transform, src_crs=crs, surf_zone_m=surf_zone_m, valid_mask=valid
    )
    observed_on = str(scene.datetime)[:10]
    report = aggregate_segments(
        ds, segments, surf_zone_m=surf_zone_m, cloud_fractions=scl_clouds, observed_on=observed_on
    )

    rows = []
    for obs in report:
        zone = zones[obs.segment_id] & water_side
        row = obs.to_row()
        row.update(
            {
                "platform": scene.platform or "",
                "tile_cloud_pct": round(float(scene.cloud_cover or 0.0), 2),
                "water_side_pixels": int(zone.sum()),
                "scl_cloud_fraction": round(scl_clouds[obs.segment_id], 4),
                "not_water_fraction": (
                    round(float((zone & not_water).sum() / (zone & usable).sum()), 4)
                    if (zone & usable).any()
                    else ""
                ),
                "usable_pixels": int((zone & usable).sum()),
                "sargassum_pixels": int((zone & hits).sum()),
                "uncertain_pixels": int((zone & uncertain).sum()),
                "max_probability": (
                    round(float(np.nanmax(score_map[zone & usable])), 4)
                    if (zone & usable).any()
                    else ""
                ),
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
        zones={k: (v & water_side) for k, v in zones.items()},
        transform=transform,
        crs=crs,
    )
    if keep_arrays:
        from mdebris.geo.raster import to_rgb

        result.rgb = to_rgb(reflectance)
        result.uncertain_mask = uncertain
        result.cloud_mask = cloudy
    return result


@dataclass(slots=True)
class Persistence:
    """Per-pixel counts across a run: passes that flagged it, passes that saw it."""

    hit_total: np.ndarray
    usable_total: np.ndarray
    zones: dict[str, np.ndarray]
    transform: object
    crs: object
    n_passes: int = 0

    def add(self, result: PassResult) -> None:
        if result.hit_mask.shape != self.hit_total.shape:
            raise ValueError("pass grid differs from the accumulated grid")
        self.hit_total += result.hit_mask
        self.usable_total += result.usable_mask
        self.n_passes += 1

    @property
    def stationary(self) -> np.ndarray:
        """Pixels flagged on at least STATIONARY_SHARE of the passes that saw them."""
        seen = self.usable_total >= STATIONARY_MIN_PASSES
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(seen, self.hit_total / np.maximum(self.usable_total, 1), 0.0)
        return seen & (share >= STATIONARY_SHARE)

    def per_segment(self) -> dict[str, dict[str, int]]:
        flagged = self.hit_total > 0
        stationary = self.stationary
        return {
            seg: {
                "flagged_pixels": int((zone & flagged).sum()),
                "stationary_pixels": int((zone & stationary).sum()),
            }
            for seg, zone in self.zones.items()
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            hit_total=self.hit_total,
            usable_total=self.usable_total,
            n_passes=self.n_passes,
            transform=np.asarray(list(self.transform)[:6], dtype=np.float64),
            crs=str(self.crs),
            zone_ids=np.asarray(list(self.zones), dtype=str),
            zones=np.stack(list(self.zones.values())),
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
        )


def _existing_scene_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return {row["scene_id"] for row in csv.DictReader(fh)}


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--segments", type=Path, default=Path("assets/bonaire_segments.geojson"))
    parser.add_argument("--island", type=Path, default=Path("assets/bonaire_island.geojson"))
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument(
        "--calibrator", type=Path, default=Path("models/sargassum_calibration.json")
    )
    parser.add_argument("--calibration-json", type=Path, default=Path("docs/calibration.json"))
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--high", type=float, default=None, help="Override the sargassum edge.")
    parser.add_argument("--surf-zone-m", type=float, default=500.0)
    parser.add_argument("--csv", type=Path, default=Path("docs/bonaire_season.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/bonaire_season.md"))
    parser.add_argument("--figure", type=Path, default=Path("assets/bonaire_season.png"))
    parser.add_argument("--pass-figure", type=Path, default=Path("assets/bonaire_pass.png"))
    parser.add_argument("--persistence", type=Path, default=Path("docs/bonaire_persistence.npz"))
    parser.add_argument(
        "--persistence-figure", type=Path, default=Path("assets/bonaire_persistence.png")
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip downloads; rebuild the report from the CSV.",
    )
    parser.add_argument(
        "--pass-figure-scene",
        default=None,
        help="Only redraw the pass figure for this scene id; nothing else runs.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    segments = load_segments(args.segments)
    calibrated = not args.no_calibration
    points = load_operating_points(args.calibration_json, calibrated=calibrated, high=args.high)
    calibrator = IsotonicCalibrator.load(args.calibrator) if calibrated else None
    log.info(
        "%d segments; sargassum >= %.3f, uncertain in [%.3f, %.3f), %s probabilities (%s)",
        len(segments),
        points.high,
        points.low,
        points.high,
        "calibrated" if calibrated else "raw",
        points.source,
    )

    if args.pass_figure_scene:
        clf = SpectralClassifier.load(args.model)
        scenes = search_scenes(BONAIRE, args.start, args.end, max_cloud=100.0, limit=500)
        scene = next(s for s in scenes if s.scene_id == args.pass_figure_scene)
        result = process_pass(
            scene,
            segments=segments,
            island=args.island,
            clf=clf,
            calibrator=calibrator,
            points=points,
            surf_zone_m=args.surf_zone_m,
            keep_arrays=True,
        )
        _pass_figure(result, segments, args.pass_figure)
        log.info("wrote %s", args.pass_figure)
        return

    failures: list[tuple[str, str]] = []
    pass_meta: dict[str, dict] = {}
    persistence: Persistence | None = None
    if not args.summary_only:
        clf = SpectralClassifier.load(args.model)
        scenes = search_scenes(BONAIRE, args.start, args.end, max_cloud=100.0, limit=500)
        scenes.sort(key=lambda s: str(s.datetime))
        if args.max_scenes:
            scenes = scenes[: args.max_scenes]
        done = _existing_scene_ids(args.csv)
        log.info(
            "%d passes between %s and %s, %d already in %s",
            len(scenes),
            args.start,
            args.end,
            len(done),
            args.csv,
        )

        for i, scene in enumerate(scenes, 1):
            if scene.scene_id in done:
                continue
            started = time.perf_counter()
            try:
                result = process_pass(
                    scene,
                    segments=segments,
                    island=args.island,
                    clf=clf,
                    calibrator=calibrator,
                    points=points,
                    surf_zone_m=args.surf_zone_m,
                )
            except Exception as exc:
                log.warning("  %s failed: %s: %s", scene.scene_id, type(exc).__name__, exc)
                failures.append((scene.scene_id, f"{type(exc).__name__}: {exc}"))
                continue
            for row in result.rows:
                row["zone_cloud_fraction"] = round(result.zone_cloud_fraction, 4)
            _append_rows(args.csv, result.rows)
            if persistence is None:
                persistence = Persistence(
                    hit_total=np.zeros(result.hit_mask.shape, dtype=np.int16),
                    usable_total=np.zeros(result.hit_mask.shape, dtype=np.int16),
                    zones=result.zones,
                    transform=result.transform,
                    crs=result.crs,
                )
            persistence.add(result)
            verdicts = Counter(r["observability"] for r in result.rows)
            log.info(
                "%3d/%d %s %s tile cloud %5.1f%%  zone cloud %5.1f%%  observed %d partial %d blind %d  sargassum px %d  (%.0fs)",
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

    if persistence is not None:
        persistence.save(args.persistence)
        log.info("wrote %s", args.persistence)
    elif args.persistence.exists():
        persistence = Persistence.load(args.persistence)
    stationary = persistence.per_segment() if persistence is not None else {}

    rows = _read_rows(args.csv)
    seasons = summarize_history(rows)
    for r in rows:
        meta = pass_meta.setdefault(
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

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _markdown(seasons, pass_meta, points, args, failures, segments, stationary),
        encoding="utf-8",
    )
    args.report.with_suffix(".json").write_text(
        json.dumps(
            {
                "window": [args.start, args.end],
                "operating_points": asdict(points),
                "segments": [s.to_row() for s in seasons],
                "stationary": stationary,
                "passes": pass_meta,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", args.report)

    _timeline_figure(rows, segments, args.figure)
    log.info("wrote %s", args.figure)
    if persistence is not None:
        _persistence_figure(persistence, segments, args.persistence_figure)
        log.info("wrote %s", args.persistence_figure)

    if not args.summary_only:
        best = max(pass_meta.items(), key=lambda kv: kv[1]["sargassum_pixels"], default=None)
        if best and best[1]["sargassum_pixels"] > 0:
            scene = next(s for s in scenes if s.scene_id == best[0])
            log.info("re-reading %s for the pass figure", scene.scene_id)
            result = process_pass(
                scene,
                segments=segments,
                island=args.island,
                clf=clf,
                calibrator=calibrator,
                points=points,
                surf_zone_m=args.surf_zone_m,
                keep_arrays=True,
            )
            _pass_figure(result, segments, args.pass_figure)
            log.info("wrote %s", args.pass_figure)


def _markdown(
    seasons: list[SegmentSeason],
    passes: dict,
    points: OperatingPoints,
    args,
    failures,
    segments,
    stationary: dict[str, dict[str, int]] | None = None,
) -> str:
    n_pass = len(passes)
    platforms = Counter(p["platform"] for p in passes.values())
    by_name = {s.segment_id: s for s in segments}
    lines = [
        "# Bonaire, one season of Sentinel-2 passes",
        "",
        f"Produced by `python scripts/run_bonaire_season.py --start {args.start} --end {args.end}`.",
        f"{n_pass} passes over tile 19PEP"
        + (
            ", " + ", ".join(f"{n} from {p}" for p, n in sorted(platforms.items()))
            if platforms
            else ""
        )
        + ". Segments are cut from the OpenStreetMap coastline at named landmarks",
        "(`assets/bonaire_segments.geojson`); the surf zone is the water within",
        f"{args.surf_zone_m:.0f} m of each, land removed with the OpenStreetMap island polygon plus a",
        f"{LAND_BUFFER_M:.0f} m seaward strip. A pixel is called sargassum at calibrated probability",
        f">= {points.high:.2f}, which after calibration means the model's own estimate is at least",
        f"{points.high:.0%}; it is uncertain in [{points.low:.3f}, {points.high:.2f}) and clear below. The",
        "lower edge keeps 98% of true sargassum on MARIDA's validation split, and",
        "`docs/calibration.md` records what the upper edge achieves on the test split.",
        "",
        "## How often each stretch of coast could be seen",
        "",
        "| segment | exposure | passes | observed | partial | blind | usable | longest gap, days | median gap | passes with sargassum | most front affected, m | pixels ever flagged | of which stationary |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in seasons:
        exposure = (
            by_name[s.segment_id].properties.get("exposure", "") if s.segment_id in by_name else ""
        )
        lines.append(
            f"| {s.name} | {exposure} | {s.n_passes} | {s.n_observed} | {s.n_partial} | {s.n_blind} | "
            f"{100 * s.usable_fraction:.0f}% | {s.longest_gap_days if s.longest_gap_days is not None else '-'} | "
            f"{s.median_gap_days if s.median_gap_days is not None else '-'} | {s.n_detection_passes} | "
            f"{s.max_affected_front_m:.0f} | "
            + (
                f"{stationary[s.segment_id]['flagged_pixels']:,} | "
                f"{stationary[s.segment_id]['stationary_pixels']:,} |"
                if stationary and s.segment_id in stationary
                else "- | - |"
            )
        )
    lines += [
        "",
        "`usable` is observed plus partial: passes on which a detection means something.",
        "A segment is blind on a pass when more than 70% of the water side of its surf",
        "zone is cloud by the scene classification layer, partial above 20%. Every",
        "non-cloud pixel on the water side is classified, surf included: there is no NDWI",
        "water gate, because on MARIDA no dense sargassum pixel passes one. The longest",
        "gap is the most days between two consecutive usable looks, which is the number",
        "that decides whether a satellite product can give a beach any warning at all.",
        "",
        "The last two columns are the honesty check on the detection columns. A pixel",
        f"flagged on at least {STATIONARY_SHARE:.0%} of the passes that could see it (with at least",
        f"{STATIONARY_MIN_PASSES} such passes) is stationary, and floating material is not. Those pixels",
        "are the shallow back-bay of Lac, the reef flat at Cai and the mangrove edge: bottom",
        "types the MARIDA classifier was never trained on. Detection counts in segments",
        "where most flagged pixels are stationary are that confuser, not sargassum, and the",
        "figure `assets/bonaire_persistence.png` shows where it sits. The Wageningen report",
        "asked for separate models for the bays and the open sea for this reason.",
        "",
        "## Pass by pass",
        "",
        "| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _scene_id, p in sorted(passes.items(), key=lambda kv: kv[1]["date"]):
        not_water = (
            100 * sum(p["not_water_fractions"]) / len(p["not_water_fractions"])
            if p["not_water_fractions"]
            else 0.0
        )
        lines.append(
            f"| {p['date']} | {p['platform']} | {p['tile_cloud_pct']:.0f} | "
            f"{100 * p['zone_cloud_fraction']:.0f} | {not_water:.0f} | "
            f"{p['observed']} | {p['partial']} | {p['blind']} | "
            f"{p['sargassum_pixels']} | {', '.join(p['affected']) or '-'} |"
        )
    if failures:
        lines += ["", "## Passes that could not be read", ""]
        lines += [f"- `{sid}`: {why}" for sid, why in failures]
    lines += [
        "",
        "## What this does not say",
        "",
        "No pixel here has been checked against the beach. The classifier was trained",
        "and calibrated on MARIDA scenes from Central America, not Bonaire, and the",
        "reef flat, seagrass and the mangrove edge of Lac are surfaces it has never been",
        "graded on. The observability columns do not depend on the classifier at all",
        "and are the part of this table to trust first: they say how often an optical",
        "satellite could look at each stretch of coast, which is a property of the",
        "weather and the orbit, not of the model.",
        "",
        "The Kralendijk waterfront is a leeward control. Sargassum arrives on the",
        "windward side; if that row fills with detections, the model is finding",
        "something other than sargassum.",
        "",
    ]
    return "\n".join(lines)


OBSERVABILITY_STYLE = {
    "observed": ("#1f77b4", "o"),
    "partial": ("#d95f02", "s"),
    "blind": ("#7b3294", "x"),
}


def _timeline_figure(rows: list[dict], segments, path: Path) -> None:
    """One row per segment, one marker per pass, coloured and shaped by observability."""
    import matplotlib

    matplotlib.use("Agg")
    from datetime import date, timedelta

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from mdebris.viz.plots import save_figure

    order = [s.segment_id for s in segments]
    names = {s.segment_id: s.name for s in segments}
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    for row in rows:
        y = order.index(row["segment_id"])
        when = date.fromisoformat(row["observed_on"][:10])
        colour, marker = OBSERVABILITY_STYLE[row["observability"]]
        ax.plot(
            when,
            y,
            marker=marker,
            color=colour,
            markersize=6,
            linestyle="none",
            markeredgewidth=1.4,
        )
        if row["observability"] != "blind" and int(row.get("detection_count") or 0) > 0:
            ax.plot(
                when,
                y,
                marker="o",
                markersize=13,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.4,
                linestyle="none",
            )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[s] for s in order])
    ax.invert_yaxis()
    dates = sorted({date.fromisoformat(r["observed_on"][:10]) for r in rows})
    if dates:
        pad = timedelta(days=4)
        ax.set_xlim(dates[0] - pad, dates[-1] + pad)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color="0.9")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D(
            [], [], marker=m, color=c, linestyle="none", markersize=7, markeredgewidth=1.4, label=k
        )
        for k, (c, m) in OBSERVABILITY_STYLE.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="black",
            linestyle="none",
            label="sargassum detected",
        )
    )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False
    )
    ax.set_title("Bonaire coast segments, every Sentinel-2 pass: could the beach be seen?")
    save_figure(fig, path)


def _persistence_figure(persistence: Persistence, segments, path: Path) -> None:
    """How often each pixel was flagged, over the passes that saw it, around Lac Bay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    from mdebris.viz.plots import save_figure

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(persistence.crs), always_xy=True
    ).transform
    to_pixel = ~persistence.transform
    seen = persistence.usable_total >= STATIONARY_MIN_PASSES
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(
            seen, persistence.hit_total / np.maximum(persistence.usable_total, 1), np.nan
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    windows = {
        "whole island": ((-68.43, 12.32), (-68.18, 12.01)),
        "Lac Bay and Cai": ((-68.255, 12.130), (-68.195, 12.070)),
    }
    for ax, (title, ((lon0, lat0), (lon1, lat1))) in zip(axes, windows.items(), strict=True):
        x0, y0 = to_pixel * fwd(lon0, lat0)
        x1, y1 = to_pixel * fwd(lon1, lat1)
        r0, r1 = sorted((max(0, int(y0)), int(y1)))
        c0, c1 = sorted((max(0, int(x0)), int(x1)))
        crop = share[r0:r1, c0:c1]
        # Land and never-seen water in grey, so white means "seen and never flagged".
        unseen = np.where(seen[r0:r1, c0:c1], np.nan, 0.55)
        ax.imshow(unseen, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
        image = ax.imshow(crop, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        for segment in segments:
            projected = shapely_transform(fwd, segment.geometry)
            xs, ys = [], []
            for x, y in projected.coords:
                col, row = to_pixel * (x, y)
                xs.append(col - c0)
                ys.append(row - r0)
            ax.plot(xs, ys, lw=1.0, color="#444444", alpha=0.8)
        ax.set_xlim(0, c1 - c0)
        ax.set_ylim(r1 - r0, 0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        image,
        ax=axes[1],
        fraction=0.046,
        pad=0.02,
        label="share of the passes that saw the pixel on which it was called sargassum",
    )

    fig.suptitle(
        f"Persistence over {persistence.n_passes} passes: how often each pixel was called "
        "sargassum when it could be seen",
        fontsize=11,
    )
    save_figure(fig, path)


def _pass_figure(result: PassResult, segments, path: Path) -> None:
    """True colour, the three-state classification, and the segment verdicts, for one pass."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    from mdebris.viz.plots import save_figure

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(result.crs), always_xy=True
    ).transform
    to_pixel = ~result.transform
    verdict = {r["segment_id"]: r["observability"] for r in result.rows}

    def draw(ax, colour_of=None):
        for segment in segments:
            projected = shapely_transform(fwd, segment.geometry)
            xs, ys = [], []
            for x, y in projected.coords:
                col, row = to_pixel * (x, y)
                xs.append(col)
                ys.append(row)
            ax.plot(xs, ys, lw=2.0, color=colour_of(segment) if colour_of else "white")

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.5), constrained_layout=True)
    axes[0].imshow(result.rgb)
    axes[0].set_title(f"{str(result.scene.datetime)[:10]}, true colour")
    draw(axes[0])

    axes[1].imshow(result.rgb)
    overlay = np.zeros((*result.hit_mask.shape, 4))
    overlay[result.uncertain_mask] = (0.85, 0.37, 0.01, 0.9)
    overlay[result.hit_mask] = (0.12, 0.47, 0.71, 1.0)
    overlay[result.cloud_mask] = (0.6, 0.6, 0.6, 0.35)
    axes[1].imshow(overlay)
    axes[1].set_title("sargassum (blue), uncertain (orange), cloud (grey)")
    draw(axes[1])
    axes[1].legend(
        handles=[
            Patch(color="#1f77b4", label="sargassum"),
            Patch(color="#d95f02", label="uncertain"),
            Patch(color="0.6", label="cloud"),
        ],
        loc="lower left",
    )

    axes[2].imshow(result.rgb)
    draw(axes[2], colour_of=lambda s: OBSERVABILITY_STYLE[verdict[s.segment_id]][0])
    axes[2].set_title("segment verdict")
    axes[2].legend(
        handles=[Patch(color=c, label=k) for k, (c, _m) in OBSERVABILITY_STYLE.items()],
        loc="lower left",
    )
    # Lac Bay and the coast just north of it, where the material arrives, at a scale
    # where a 10 m detection is visible.
    x0, y0 = to_pixel * fwd(-68.255, 12.130)
    x1, y1 = to_pixel * fwd(-68.195, 12.070)
    r0, r1 = sorted((int(y0), int(y1)))
    c0, c1 = sorted((int(x0), int(x1)))
    axes[3].imshow(result.rgb[r0:r1, c0:c1])
    axes[3].imshow(overlay[r0:r1, c0:c1])
    for segment in segments:
        projected = shapely_transform(fwd, segment.geometry)
        xs, ys = [], []
        for x, y in projected.coords:
            col, row = to_pixel * (x, y)
            xs.append(col - c0)
            ys.append(row - r0)
        axes[3].plot(xs, ys, lw=1.2, color="white", alpha=0.8)
    axes[3].set_xlim(0, c1 - c0)
    axes[3].set_ylim(r1 - r0, 0)
    axes[3].set_title("Lac Bay and Cai, same pass")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(fig, path)


if __name__ == "__main__":
    main()
