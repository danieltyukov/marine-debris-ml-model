"""Run the whole chain over a real coastline and write a per-beach-segment brief.

    python scripts/make_beach_segments.py

This is the end-to-end demonstration of the operational product: free Sentinel-2
imagery in, a table naming which beach is affected out. It reads a live scene over
Cancun, classifies every water pixel with the MARIDA-trained model, measures the
cloud over each beach segment, and writes the segment report plus a figure.

Nothing here is synthetic. If the scene it picks is cloudy over a segment, the
segment comes out BLIND, and that is the correct answer rather than a failure of
the script.

The segment geometry in ``assets/qroo_segments.geojson`` is a hand-drawn
approximation of the Cancun hotel-zone beaches from the shoreline in the imagery.
It is good enough to demonstrate the aggregation and is not a survey product; a
real deployment would use the municipality's own ZOFEMAT boundaries, which is one
of the things worth asking a customer for.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from mdebris.coastal import aggregate_segments, load_segments, segment_cloud_fractions
from mdebris.data import get_scene_assets, search_scenes
from mdebris.geo.raster import read_bands, window_transform
from mdebris.indices.masks import CLOUD_SCL_CLASSES, water_mask
from mdebris.models.spectral import SpectralClassifier, build_features
from mdebris.types import BBox, Detection, DetectionSet, GeoBBox, SurfaceClass

log = logging.getLogger("beaches")

# The Cancun hotel zone, the stretch that carries most of the USD 150M a year
# Quintana Roo hotels spend clearing beaches.
CANCUN = GeoBBox(west=-86.80, south=21.02, east=-86.72, north=21.18)

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")

# Reflectance offset applied to Sentinel-2 processing baseline 04.00 and later.
# Passing raw DNs to a model trained on offset-corrected reflectance shifts every
# band by 0.1, which is larger than the FDI signal itself.
BOA_OFFSET = -1000.0
QUANTIFICATION = 10_000.0

# The dispatch operating point from docs/sargassum_report.md: 90% precision, the
# highest recall available under it. A false positive costs a crew shift.
SARGASSUM_THRESHOLD = 0.736
SARGASSUM_CLASSES = ("Dense Sargassum", "Sparse Sargassum")


def _connected_boxes(mask: np.ndarray, *, min_pixels: int = 4) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of connected True regions, as ``(xmin, ymin, xmax, ymax)``.

    Small blobs are dropped: at 10 m a three-pixel cluster is 300 m2, below the
    size where a per-pixel spectral call is trustworthy on its own.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    boxes = []
    for rows, cols in ndimage.find_objects(labelled):
        region = labelled[rows, cols]
        if int((region > 0).sum()) < min_pixels:
            continue
        boxes.append((cols.start, rows.start, cols.stop, rows.stop))
    log.info("  %d connected regions, %d above the size floor", count, len(boxes))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=Path("assets/qroo_segments.geojson"))
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-07-30")
    parser.add_argument("--max-cloud", type=float, default=40.0)
    parser.add_argument(
        "--scene",
        default=None,
        help="Use this scene id instead of the first match. Pick a cloudy one to see "
        "what the product does when it cannot see a beach.",
    )
    parser.add_argument("--surf-zone-m", type=float, default=500.0)
    parser.add_argument("--out", type=Path, default=Path("docs/beach_segments.md"))
    parser.add_argument("--geojson", type=Path, default=Path("docs/beach_segments.geojson"))
    parser.add_argument("--figure", type=Path, default=Path("assets/beach_segments.png"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    segments = load_segments(args.segments)
    log.info("%d beach segments from %s", len(segments), args.segments)

    scenes = search_scenes(CANCUN, args.start, args.end, max_cloud=args.max_cloud, limit=20)
    if not scenes:
        raise SystemExit(
            f"no Sentinel-2 scene over Cancun between {args.start} and {args.end} under "
            f"{args.max_cloud}% cloud. Widen the window; this is the revisit problem, "
            "and it is the honest reason a detection product is not a forecast."
        )
    if args.scene:
        matches = [s for s in scenes if s.scene_id == args.scene]
        if not matches:
            raise SystemExit(
                f"{args.scene} is not among the {len(scenes)} scenes found. "
                f"Available: {', '.join(s.scene_id for s in scenes[:5])}"
            )
        scene = matches[0]
    else:
        scene = scenes[0]
    log.info("scene %s, %s, %.1f%% cloud", scene.scene_id, scene.datetime, scene.cloud_cover or 0)

    hrefs = get_scene_assets(scene.scene_id, [*BANDS, "SCL"])
    import rasterio

    with rasterio.open(hrefs["B04"]) as src:
        window = (
            rasterio.windows.from_bounds(*_to_utm(CANCUN, src.crs), transform=src.transform)
            .round_lengths()
            .round_offsets()
        )
        transform = window_transform(window, src.transform)
        crs = src.crs
    log.info("window %dx%d px", int(window.width), int(window.height))

    arrays = read_bands(hrefs, window, reference="B04")
    scl = arrays.pop("SCL").astype(np.int16)
    reflectance = {
        name: (arr.astype(np.float32) + BOA_OFFSET) / QUANTIFICATION for name, arr in arrays.items()
    }

    log.info("classifying %s pixels", f"{scl.size:,}")
    clf = SpectralClassifier.load(args.model)
    classes = list(clf._model.classes_)
    proba = clf.predict_proba(build_features(reflectance))
    idx = [classes.index(c) for c in SARGASSUM_CLASSES if c in classes]
    sargassum = proba[:, idx].sum(axis=1).reshape(scl.shape)

    cloudy = np.isin(scl, list(CLOUD_SCL_CLASSES))
    # MARIDA has no land class. All 15 of its labels are sea-surface classes, so a
    # land pixel is forced into whichever marine category it most resembles, and
    # bright sand and coastal vegetation resemble floating biomass. Gating on water
    # is a correctness requirement here, not a speed optimisation.
    water = water_mask(reflectance)
    hits = (sargassum >= SARGASSUM_THRESHOLD) & water & ~cloudy
    log.info(
        "  %.0f%% water, %.1f%% cloud in window, %s pixels over threshold",
        100 * water.mean(),
        100 * cloudy.mean(),
        f"{int(hits.sum()):,}",
    )

    detections = [
        Detection(
            bbox=BBox(xmin=float(x0), ymin=float(y0), xmax=float(x1), ymax=float(y1)),
            score=float(sargassum[y0:y1, x0:x1].max()),
            label=SurfaceClass.SARGASSUM,
        )
        for x0, y0, x1, y1 in _connected_boxes(hits)
    ]
    from mdebris.geo.georef import georeference_detections

    georeference_detections(detections, transform=transform, src_crs=crs)
    ds = DetectionSet(detections=detections, scene=scene)

    clouds = segment_cloud_fractions(
        segments, cloudy, transform, src_crs=crs, surf_zone_m=args.surf_zone_m
    )
    report = aggregate_segments(ds, segments, surf_zone_m=args.surf_zone_m, cloud_fractions=clouds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_markdown(report, scene), encoding="utf-8")
    report.write_geojson(args.geojson)
    log.info("wrote %s and %s", args.out, args.geojson)

    _figure(reflectance, hits, cloudy, transform, crs, segments, report, args.figure)
    log.info("wrote %s", args.figure)


def _to_utm(bbox: GeoBBox, crs) -> tuple[float, float, float, float]:
    from pyproj import CRS, Transformer

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform
    west, south = fwd(bbox.west, bbox.south)
    east, north = fwd(bbox.east, bbox.north)
    return west, south, east, north


def _markdown(report, scene) -> str:
    lines = [
        "# Beach-segment brief, Cancun hotel zone",
        "",
        f"Scene `{scene.scene_id}`, {str(scene.datetime)[:10]}, "
        f"{scene.cloud_cover or 0:.1f}% cloud over the tile.",
        "",
        "Produced by `python scripts/make_beach_segments.py`. Sargassum probability is",
        f"thresholded at {SARGASSUM_THRESHOLD}, the 90%-precision operating point from",
        "`sargassum_report.md`, and cloud is measured per segment from the scene's own",
        "classification layer.",
        "",
        "| segment | observed | cloud % | cover % | affected front m | detections |",
        "|---|---|---|---|---|---|",
    ]
    for obs in report.ranked():
        blind = obs.observability.value == "blind"
        lines.append(
            f"| {obs.name} | {obs.observability.value} | {100 * obs.cloud_fraction:.0f} | "
            f"{'—' if blind else f'{100 * obs.coverage:.2f}'} | "
            f"{'—' if blind else f'{obs.affected_front_m:.0f}'} | "
            f"{'—' if blind else obs.detection_count} |"
        )
    lines.append("")
    if report.blind:
        lines += [
            f"{len(report.blind)} of {len(report)} segments were not observed. Those rows carry",
            "no coverage figure on purpose: a beach under cloud has not been measured, and",
            "reporting 0% for it is the specific mistake that sends nobody to a beach that",
            "needed clearing.",
        ]
    else:
        lines += [
            "Every segment was observed on this pass, so an absence of detections here does",
            "mean a clear beach. That is not the usual case. LANOT, who run the nearest",
            "comparable Sentinel-2 platform, report cloud above 90% often enough that a",
            "fully clear day over this coast is close to non-existent.",
        ]
    lines += [
        "",
        "## What this is not",
        "",
        "This is a detection on the day of the overpass, not a landfall forecast. Nothing",
        "here models drift between the overpass and the arrival, and no field observation",
        "has confirmed any of these detections. The numbers say where floating material",
        "was, at 10 m, at one instant.",
        "",
    ]
    return "\n".join(lines)


def _figure(reflectance, hits, cloudy, transform, crs, segments, report, path: Path) -> None:
    """Scene, detections and segment verdicts side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    from mdebris.geo.raster import to_rgb
    from mdebris.viz.plots import save_figure

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), constrained_layout=True)
    rgb = to_rgb(reflectance)

    to_pixel = ~transform
    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform

    def _draw_segments(ax, colour_of=None):
        for segment in segments:
            projected = shapely_transform(fwd, segment.geometry)
            xs, ys = [], []
            for x, y in projected.coords:
                col, row = to_pixel * (x, y)
                xs.append(col)
                ys.append(row)
            ax.plot(xs, ys, lw=2.5, color=colour_of(segment) if colour_of else "white")

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true colour")
    _draw_segments(axes[0])

    axes[1].imshow(rgb)
    overlay = np.zeros((*hits.shape, 4))
    overlay[hits] = (1.0, 0.45, 0.0, 0.9)
    overlay[cloudy] = (0.6, 0.6, 0.6, 0.35)
    axes[1].imshow(overlay)
    axes[1].set_title("sargassum (orange), cloud (grey)")
    _draw_segments(axes[1])

    verdict = {o.segment_id: o.observability.value for o in report}
    colours = {"observed": "#2e7d32", "partial": "#f9a825", "blind": "#c62828"}
    axes[2].imshow(rgb)
    _draw_segments(axes[2], colour_of=lambda s: colours[verdict[s.segment_id]])
    axes[2].set_title("segment verdict")
    axes[2].legend(handles=[Patch(color=c, label=k) for k, c in colours.items()], loc="lower right")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(fig, path)


if __name__ == "__main__":
    main()
