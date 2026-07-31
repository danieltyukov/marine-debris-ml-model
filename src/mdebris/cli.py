"""Command line interface.

Model imports are deliberately deferred into the command bodies. Importing torch
costs several seconds, and ``mdebris --help`` or ``mdebris samples`` should not pay
for a machine learning stack they never touch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mdebris import __version__
from mdebris.config import settings

app = typer.Typer(
    name="mdebris",
    help="Detect marine debris in satellite imagery with open-vocabulary models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mdebris {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise typer.BadParameter(
            f"expected 'west,south,east,north', got {text!r} with {len(parts)} values"
        )
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"bbox values must be numbers: {text!r}") from exc
    return west, south, east, north


@app.command()
def samples() -> None:
    """List the sample chips bundled with the package."""
    from mdebris.data import list_samples, load_sample

    table = Table(title="Bundled Sentinel-2 samples")
    for col in ("name", "scene", "date", "size", "cloud %", "where"):
        table.add_column(col, overflow="fold")
    for name in list_samples():
        _, meta = load_sample(name)
        table.add_row(
            name,
            meta.get("scene_id", "")[:32],
            str(meta.get("datetime", ""))[:10],
            f"{meta.get('width')}x{meta.get('height')}",
            f"{meta.get('cloud_cover', 0):.1f}",
            (meta.get("description", "") or "")[:60],
        )
    console.print(table)


@app.command()
def indices(
    sample: Annotated[str, typer.Option(help="Bundled sample name.")] = "accra",
    output: Annotated[Path | None, typer.Option(help="Write a PNG figure here.")] = None,
) -> None:
    """Compute spectral indices over a sample chip and report their statistics."""
    import numpy as np

    from mdebris.data import sample_reflectance
    from mdebris.indices.masks import water_mask
    from mdebris.indices.spectral import compute_indices

    bands, meta = sample_reflectance(sample)
    refl = {k: v for k, v in bands.items() if k != "SCL"}
    values = compute_indices(refl)
    water = water_mask(refl)

    table = Table(title=f"Spectral indices over {sample} ({meta.get('scene_id', '')[:24]})")
    for col in ("index", "mean", "p50", "p99", "p99.9", "over water p99"):
        table.add_column(col, justify="right")
    for name, arr in sorted(values.items()):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        wet = arr[water & np.isfinite(arr)]
        table.add_row(
            name,
            f"{finite.mean():+.5f}",
            f"{np.percentile(finite, 50):+.5f}",
            f"{np.percentile(finite, 99):+.5f}",
            f"{np.percentile(finite, 99.9):+.5f}",
            f"{np.percentile(wet, 99):+.5f}" if wet.size else "n/a",
        )
    console.print(table)
    console.print(f"water fraction: {water.mean() * 100:.1f}%")

    if output is not None:
        import matplotlib

        matplotlib.use("Agg")
        from mdebris.viz.plots import figure_grid, plot_index_heatmap, save_figure

        names = [n for n in ("FDI", "FAI", "NDVI", "NDWI", "PI", "KNDVI") if n in values]
        fig, axes = figure_grid(2, 3)
        for ax, name in zip(axes, names, strict=False):
            plot_index_heatmap(values[name], name=name, ax=ax)
        path = save_figure(fig, output)
        console.print(f"wrote {path}")


@app.command()
def search(
    bbox: Annotated[str, typer.Option(help="west,south,east,north in degrees.")],
    start: Annotated[str, typer.Option(help="Start date, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="End date, YYYY-MM-DD.")],
    max_cloud: Annotated[float, typer.Option(help="Maximum cloud cover percent.")] = 20.0,
    limit: Annotated[int, typer.Option(help="Maximum scenes to return.")] = 10,
) -> None:
    """Search free Sentinel-2 imagery for an area and date range."""
    from mdebris.data import search_scenes
    from mdebris.types import GeoBBox

    west, south, east, north = _parse_bbox(bbox)
    scenes = search_scenes(
        GeoBBox(west=west, south=south, east=east, north=north),
        start,
        end,
        max_cloud=max_cloud,
        limit=limit,
    )
    if not scenes:
        console.print("[yellow]No scenes matched. Try widening the dates or the cloud limit.")
        raise typer.Exit(code=1)

    table = Table(title=f"{len(scenes)} scenes")
    for col in ("scene id", "date", "cloud %", "platform"):
        table.add_column(col, overflow="fold")
    for s in scenes:
        table.add_row(
            s.scene_id,
            (s.datetime or "")[:10],
            f"{s.cloud_cover:.1f}" if s.cloud_cover is not None else "?",
            s.platform or "",
        )
    console.print(table)


@app.command()
def detect(
    sample: Annotated[str | None, typer.Option(help="Bundled sample to run on.")] = None,
    bbox: Annotated[
        str | None, typer.Option(help="west,south,east,north for a live search.")
    ] = None,
    start: Annotated[str | None, typer.Option(help="Start date for a live search.")] = None,
    end: Annotated[str | None, typer.Option(help="End date for a live search.")] = None,
    model: Annotated[
        str, typer.Option(help="Detector: owlv2, grounding-dino or rtdetr.")
    ] = "owlv2",
    threshold: Annotated[float, typer.Option(help="Score threshold.")] = 0.10,
    cascade: Annotated[bool, typer.Option(help="Screen tiles with spectral indices first.")] = True,
    segment: Annotated[bool, typer.Option(help="Refine boxes into masks with SAM 2.")] = False,
    targets_only: Annotated[
        bool, typer.Option(help="Drop confuser classes from the output.")
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="GeoJSON output path.")
    ] = None,
    figure: Annotated[Path | None, typer.Option(help="Also write a detection overlay PNG.")] = None,
) -> None:
    """Detect debris in a bundled sample or in live Sentinel-2 imagery.

    Either pass --sample, or pass --bbox with --start and --end to search for a scene.
    """
    import rasterio.transform as rt

    from mdebris.models import get_detector
    from mdebris.pipeline import detect_in_arrays
    from mdebris.types import SceneRef

    if sample is None and bbox is None:
        raise typer.BadParameter("pass either --sample or --bbox with --start and --end")
    if bbox is not None and (start is None or end is None):
        raise typer.BadParameter("--bbox requires both --start and --end")

    detector = get_detector(model)
    segmenter = None
    if segment:
        from mdebris.models import Sam2Segmenter

        segmenter = Sam2Segmenter()

    if sample is not None:
        from mdebris.data import sample_reflectance

        bands, meta = sample_reflectance(sample)
        transform = rt.Affine(*meta["transform"])
        scene = SceneRef(scene_id=meta.get("scene_id", sample), datetime=meta.get("datetime"))
        crs = meta.get("crs")
        console.print(f"[cyan]{sample}[/cyan]: {scene.scene_id}")
    else:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        from mdebris.data import StacClient, reflectance_params_for_item, scale_bands
        from mdebris.geo.raster import read_bands
        from mdebris.types import GeoBBox

        west, south, east, north = _parse_bbox(bbox)  # type: ignore[arg-type]
        client = StacClient()
        items = client.search_items(
            GeoBBox(west=west, south=south, east=east, north=north), start, end, limit=1
        )
        if not items:
            console.print("[red]No scene matched that area and date range.")
            raise typer.Exit(code=1)
        item = items[0]
        scene = SceneRef(
            scene_id=item.id,
            datetime=str(item.datetime),
            cloud_cover=item.properties.get("eo:cloud_cover"),
        )
        console.print(f"[cyan]scene[/cyan]: {scene.scene_id} ({scene.cloud_cover or 0:.0f}% cloud)")

        hrefs = client.asset_hrefs(item, ["B02", "B03", "B04", "B06", "B08", "B11", "SCL"])
        # The 10 m bands define the grid. Reading the window in the scene's own CRS
        # avoids a reprojection of the imagery itself.
        with rasterio.open(hrefs["B04"]) as src:
            crs = src.crs
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, west, south, east, north
            )
            window = from_bounds(left, bottom, right, top, src.transform)
            transform = src.window_transform(window)

        raw = read_bands(hrefs, window, reference="B04")
        # Baseline 04.00 and later carry BOA_ADD_OFFSET, so DN/10000 is wrong by 0.1.
        scale, offset = reflectance_params_for_item(item)
        bands = scale_bands(raw, scale=scale, offset=offset)

    with console.status(f"running {detector.name}, this is slow on CPU"):
        result = detect_in_arrays(
            bands,
            detector,
            transform=transform,
            crs=crs,
            scene=scene,
            threshold=threshold,
            use_cascade=cascade,
            segmenter=segmenter,
        )

    detections = result.detections.targets_only() if targets_only else result.detections
    console.print(f"[green]{result.summary()}")

    if len(detections):
        table = Table(title="Detections")
        for col in ("class", "score", "lon", "lat", "area m2"):
            table.add_column(col, justify="right")
        for d in sorted(detections.detections, key=lambda x: -x.score)[:20]:
            lon, lat = d.geometry.centroid.coords[0] if d.geometry else (float("nan"),) * 2
            area = d.area_m2
            table.add_row(
                str(d.label),
                f"{d.score:.3f}",
                f"{lon:.5f}",
                f"{lat:.5f}",
                f"{area:.0f}" if area else "n/a",
            )
        console.print(table)
    else:
        console.print("[yellow]No detections above the threshold.")

    if output is not None:
        from mdebris.geo.georef import write_geojson

        console.print(f"wrote {write_geojson(detections, output)}")

    if figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mdebris.geo.raster import to_rgb
        from mdebris.viz.plots import plot_detections, save_figure

        fig, ax = plt.subplots(figsize=(9, 9))
        plot_detections(to_rgb(bands), detections.detections, ax=ax, title=f"{scene.scene_id[:40]}")
        console.print(f"wrote {save_figure(fig, figure)}")


@app.command()
def evaluate(
    predictions: Annotated[Path, typer.Argument(help="GeoJSON of predictions.")],
    ground_truth: Annotated[Path, typer.Argument(help="GeoJSON of ground truth.")],
    iou: Annotated[float, typer.Option(help="IoU threshold for a match.")] = 0.5,
    output: Annotated[Path | None, typer.Option(help="Write a markdown report here.")] = None,
) -> None:
    """Score predictions against ground truth with mAP, precision, recall and F1."""
    from mdebris.eval import evaluate as run_eval
    from mdebris.eval.report import format_markdown
    from mdebris.geo import read_geojson

    result = run_eval(read_geojson(predictions), read_geojson(ground_truth), iou_threshold=iou)
    report = format_markdown(result)
    console.print(report)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        console.print(f"wrote {output}")


@app.command()
def beaches(
    detections: Annotated[Path, typer.Argument(help="GeoJSON produced by detect.")],
    segments: Annotated[
        Path, typer.Option("--segments", "-s", help="GeoJSON of named beach segments.")
    ],
    surf_zone_m: Annotated[
        float, typer.Option(help="How far offshore to look, in metres.")
    ] = 500.0,
    clouds: Annotated[
        Path | None,
        typer.Option(help="JSON mapping segment_id to cloud fraction, 0 to 1."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write per-segment GeoJSON here.")
    ] = None,
    csv_path: Annotated[
        Path | None, typer.Option("--csv", help="Write the per-segment table here.")
    ] = None,
    history: Annotated[
        Path | None, typer.Option(help="Append this run to a dated per-segment CSV record.")
    ] = None,
    all_labels: Annotated[
        bool, typer.Option(help="Count every detection class, not just sargassum and debris.")
    ] = False,
) -> None:
    """Roll detections up onto named beach segments.

    Without --clouds every segment is reported as fully observed, which is only
    true on a clear scene. Pass the cloud fractions whenever they are known; an
    unseen beach reported as clean is the failure mode this command exists to
    avoid.
    """
    from mdebris.coastal import aggregate_segments, append_history, load_segments
    from mdebris.geo import read_geojson

    beach_segments = load_segments(segments)
    detection_set = read_geojson(detections)
    cloud_fractions = json.loads(clouds.read_text(encoding="utf-8")) if clouds else None

    report = aggregate_segments(
        detection_set,
        beach_segments,
        surf_zone_m=surf_zone_m,
        cloud_fractions=cloud_fractions,
        labels=None if all_labels else _default_beach_labels(),
    )

    table = Table(title=f"Beach segments{f' — {report.observed_on}' if report.observed_on else ''}")
    table.add_column("segment")
    table.add_column("cover %", justify="right")
    table.add_column("front m", justify="right")
    table.add_column("dets", justify="right")
    table.add_column("cloud %", justify="right")
    table.add_column("seen")
    for obs in report.ranked():
        blind = obs.observability.value == "blind"
        table.add_row(
            obs.name,
            "[dim]—[/dim]" if blind else f"{100 * obs.coverage:.2f}",
            "[dim]—[/dim]" if blind else f"{obs.affected_front_m:.0f}",
            "[dim]—[/dim]" if blind else str(obs.detection_count),
            f"{100 * obs.cloud_fraction:.0f}",
            {"observed": "[green]OK", "partial": "[yellow]PARTIAL", "blind": "[red]BLIND"}[
                obs.observability.value
            ],
        )
    console.print(table)

    if report.blind:
        console.print(
            f"[yellow]{len(report.blind)} of {len(report)} segments were not observed; "
            "their absence of detections means nothing."
        )

    if output is not None:
        console.print(f"wrote {report.write_geojson(output)}")
    if csv_path is not None:
        console.print(f"wrote {report.write_csv(csv_path)}")
    if history is not None:
        console.print(f"appended {len(report)} rows to {append_history(report, history)}")


def _default_beach_labels():
    """Detection classes a beach-clearing customer cares about."""
    from mdebris.coastal.segments import DEFAULT_TARGET_LABELS

    return DEFAULT_TARGET_LABELS


@app.command()
def config() -> None:
    """Show the resolved configuration and the detected compute device."""
    table = Table(title="mdebris configuration")
    table.add_column("setting")
    table.add_column("value", overflow="fold")
    for key, value in settings.model_dump().items():
        if "key" in key and value:
            value = "<set>"  # never print a credential
        table.add_row(key, str(value))
    table.add_row("[bold]resolved device", f"[bold]{settings.resolve_device()}")
    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the HTTP API."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "the API extra is not installed: pip install 'mdebris[api]'"
        ) from exc
    uvicorn.run("mdebris.api.app:app", host=host, port=port, reload=reload)


@app.command(name="export-geojson")
def export_geojson(
    input_path: Annotated[Path, typer.Argument(help="GeoJSON produced by detect.")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    min_score: Annotated[float, typer.Option(help="Drop detections below this score.")] = 0.0,
) -> None:
    """Filter an existing GeoJSON by score."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    features = [
        f for f in data.get("features", []) if f.get("properties", {}).get("score", 0) >= min_score
    ]
    data["features"] = features
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"kept {len(features)} features, wrote {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
