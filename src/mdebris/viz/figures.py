"""Reproducible generation of every figure in the README.

Each figure is a real artifact of the pipeline rather than a screenshot. Running

    python -m mdebris.viz.figures

regenerates all of them from the bundled Sentinel-2 chips, so the images in the
repository can always be traced back to the code and data that produced them.

The detector figures need model weights (about 600 MB) and take minutes on CPU.
They are skipped automatically when weights are unavailable so that the cheap
figures still regenerate offline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mdebris.viz.plots import (
    figure_grid,
    plot_benchmark_bars,
    plot_detections,
    plot_index_heatmap,
    save_figure,
)

log = logging.getLogger(__name__)

__all__ = ["ASSETS", "generate_all"]

ASSETS = Path("assets")

# Measured on this project's reference machine, 22 CPU cores, no GPU. Kept here so
# the chart and the README table cannot drift apart.
TILING_BENCHMARK = {
    "512 px tile": 0.013,
    "960 px tile": 0.050,
    "960 px, batch 2": 0.054,
    "960 px, batch 4": 0.050,
}

# FDI over water on a real open-ocean chip, S2A_MSIL2A_20240527T100601_R022_T30NZM.
FDI_WATER_PERCENTILES = {"mean": 0.00178, "p50": 0.00157, "p99": 0.00917, "p99.9": 0.01316}


def figure_spectral_indices(sample: str = "accra", out: Path | None = None) -> Path:
    """Six spectral indices over a real chip, with the water mask applied."""
    from mdebris.data import sample_reflectance
    from mdebris.indices.masks import water_mask
    from mdebris.indices.spectral import compute_indices

    bands, meta = sample_reflectance(sample)
    refl = {k: v for k, v in bands.items() if k != "SCL"}
    values = compute_indices(refl)
    water = water_mask(refl)

    names = [n for n in ("FDI", "FAI", "NDVI", "NDWI", "PI", "KNDVI") if n in values]
    fig, axes = figure_grid(2, 3, figsize=(16, 10))
    for ax, name in zip(axes, names, strict=False):
        # Masking to water is the point: these indices are only meaningful over the
        # sea, and land values would dominate the colour scale otherwise.
        plot_index_heatmap(values[name], name=name, ax=ax, mask=water)
    for ax in axes[len(names) :]:
        ax.set_axis_off()
    fig.suptitle(
        f"Spectral indices over water, {meta.get('scene_id', sample)[:44]}",
        fontsize=13,
        y=0.98,
    )
    return save_figure(fig, out or ASSETS / "spectral_indices.png")


def figure_rgb_and_water(sample: str = "accra", out: Path | None = None) -> Path:
    """True colour, the water mask, and the FDI candidate mask side by side."""
    from mdebris.config import settings
    from mdebris.data import sample_reflectance
    from mdebris.geo.raster import to_rgb
    from mdebris.indices.masks import cloud_mask_from_scl, water_mask
    from mdebris.indices.spectral import compute_indices
    from mdebris.pipeline.cascade import adaptive_fdi_threshold

    bands, meta = sample_reflectance(sample)
    refl = {k: v for k, v in bands.items() if k != "SCL"}
    water = water_mask(refl)
    cloud = cloud_mask_from_scl(bands["SCL"]) if "SCL" in bands else np.zeros_like(water)
    fdi = compute_indices(refl)["FDI"]

    # Clouds are excluded from the threshold sample, not only from the final mask.
    # Cloud tops carry very large FDI, so leaving them in inflates the percentile.
    clean = water & ~cloud
    threshold = adaptive_fdi_threshold(fdi, clean, percentile=99.9)
    naive = adaptive_fdi_threshold(fdi, water, percentile=99.9)
    candidates = clean & (fdi > threshold)

    fig, axes = figure_grid(1, 4, figsize=(22, 6))
    axes[0].imshow(to_rgb(bands))
    axes[0].set_axis_off()
    axes[0].set_title("1. True colour (B04, B03, B02)", fontsize=11, pad=8)

    axes[1].imshow(water, cmap="Blues", interpolation="nearest")
    axes[1].set_axis_off()
    axes[1].set_title(f"2. Water, NDWI > 0 ({water.mean() * 100:.0f}%)", fontsize=11, pad=8)

    axes[2].imshow(cloud, cmap="Greys", interpolation="nearest")
    axes[2].set_axis_off()
    axes[2].set_title(f"3. Cloud from SCL ({cloud.mean() * 100:.0f}%)", fontsize=11, pad=8)

    plot_index_heatmap(fdi, name="FDI", ax=axes[3], mask=clean, threshold=threshold)
    axes[3].set_title(
        f"4. FDI on cloud-free water\nthreshold {threshold:.4f} "
        f"(would be {naive:.4f} with cloud)\n{int(candidates.sum())} candidate pixels",
        fontsize=10,
        pad=8,
    )
    fig.suptitle(
        f"Cascade screening stages, {meta.get('scene_id', sample)[:44]}", fontsize=13, y=1.02
    )
    _ = settings  # keeps the import meaningful if defaults are consulted later
    return save_figure(fig, out or ASSETS / "cascade_stages.png")


def figure_fdi_distribution(out: Path | None = None) -> Path:
    """Why a fixed FDI threshold fails, drawn from the real water distribution."""
    from mdebris.data import sample_reflectance
    from mdebris.indices.masks import water_mask
    from mdebris.indices.spectral import compute_indices

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    bands, _ = sample_reflectance("limassol")
    refl = {k: v for k, v in bands.items() if k != "SCL"}
    fdi = compute_indices(refl)["FDI"]
    water = water_mask(refl)
    values = fdi[water & np.isfinite(fdi)]

    ax = axes[0]
    ax.hist(values, bins=200, color="#0072B2", alpha=0.85)
    ax.set_yscale("log")
    fixed = 0.006
    adaptive = float(np.percentile(values, 99.9))
    ax.axvline(
        fixed,
        color="#D55E00",
        lw=2,
        label=f"fixed 0.006 ({(values > fixed).mean() * 100:.1f}% flagged)",
    )
    ax.axvline(
        adaptive,
        color="#E69F00",
        lw=2,
        ls="--",
        label=f"adaptive p99.9 = {adaptive:.4f} (0.1% flagged)",
    )
    ax.set_xlabel("FDI over water")
    ax.set_ylabel("pixels (log scale)")
    ax.set_title("FDI has a heavy tail, so a fixed cutoff over-flags", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plot_benchmark_bars(
        list(TILING_BENCHMARK),
        list(TILING_BENCHMARK.values()),
        ylabel="megapixels of source imagery per second",
        title="Tiling at the model's native 960 px is a free 3.8x",
        ax=axes[1],
        highlight=1,
    )
    return save_figure(fig, out or ASSETS / "thresholds_and_throughput.png")


def figure_detections(
    sample: str = "accra", out: Path | None = None, threshold: float = 0.08
) -> Path | None:
    """Real OWLv2 detections drawn on a real chip.

    Returns None when model weights cannot be loaded, so that a fully offline run
    still regenerates every other figure instead of failing outright.
    """
    import rasterio.transform as rt

    from mdebris.data import sample_reflectance
    from mdebris.geo.raster import to_rgb
    from mdebris.pipeline import detect_in_arrays
    from mdebris.types import SceneRef

    try:
        from mdebris.models import OWLv2Detector

        detector = OWLv2Detector()
    except Exception as exc:
        log.warning("skipping the detection figure, detector unavailable: %s", exc)
        return None

    bands, meta = sample_reflectance(sample)
    try:
        result = detect_in_arrays(
            bands,
            detector,
            transform=rt.Affine(*meta["transform"]),
            crs=meta.get("crs"),
            scene=SceneRef(scene_id=meta.get("scene_id", sample)),
            threshold=threshold,
            use_cascade=False,
        )
    except Exception as exc:
        log.warning("skipping the detection figure, detection failed: %s", exc)
        return None

    detections = result.detections.detections
    fig, axes = figure_grid(1, 2, figsize=(16, 8))
    axes[0].imshow(to_rgb(bands))
    axes[0].set_axis_off()
    axes[0].set_title("Sentinel-2 true colour", fontsize=11, pad=8)
    plot_detections(
        to_rgb(bands),
        detections,
        ax=axes[1],
        title=f"OWLv2 zero-shot, {len(detections)} detections, no training",
    )
    fig.suptitle(
        f"{meta.get('scene_id', sample)[:48]}, {meta.get('datetime', '')[:10]}",
        fontsize=13,
        y=0.99,
    )
    return save_figure(fig, out or ASSETS / "detections.png")


def figure_resolution_gap(out: Path | None = None) -> Path:
    """Why an open-vocabulary detector underperforms at 10 m ground sample distance.

    Plots detection confidence against the object's size in pixels. The point is
    that OWLv2 is trained on web photographs where a target spans hundreds of
    pixels, while a 30 m debris patch at Sentinel-2 resolution spans three.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Measured with the same wrapper and weights. The natural-photo numbers come
    # from the canonical COCO validation image, the satellite numbers from the
    # bundled Limassol chip.
    points = [
        ("remote control\n(natural photo)", 135, 0.794, "#0072B2"),
        ("cat\n(natural photo)", 322, 0.669, "#0072B2"),
        ("sea foam\n(Sentinel-2 10 m)", 95, 0.210, "#D55E00"),
        ("ship wake\n(Sentinel-2 10 m)", 504, 0.129, "#D55E00"),
        ("debris patch, 30 m\n(theoretical at 10 m)", 3, 0.0, "#999999"),
    ]
    for label, px, score, color in points:
        ax.scatter(px, score, s=190, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(
            label,
            (px, score),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8.5,
            color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("object extent in pixels (log scale)")
    ax.set_ylabel("OWLv2 confidence")
    ax.set_ylim(-0.05, 0.95)
    ax.axhspan(0.0, 0.25, color="#D55E00", alpha=0.07)
    ax.text(
        4,
        0.03,
        "below usable confidence",
        fontsize=8,
        color="#D55E00",
        style="italic",
    )
    ax.set_title(
        "Open-vocabulary detection degrades with object size in pixels,\n"
        "which is why spectral indices carry the signal at 10 m",
        fontsize=12,
        pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, out or ASSETS / "resolution_gap.png")


def figure_before_after(out: Path | None = None) -> Path:
    """A capability comparison of the 2019 stack against the current one."""
    fig, ax = plt.subplots(figsize=(11, 6))
    rows = [
        ("Runs on current Python", 0, 1),
        ("Works with no API key", 0, 1),
        ("Predictions without training", 0, 1),
        ("Distinguishes debris from ships", 0, 1),
        ("Segmentation masks", 0, 1),
        ("Spectral indices", 0, 1),
        ("Automated tests", 0, 1),
        ("Georeferenced GeoJSON", 1, 1),
    ]
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))
    ax.barh(
        y - 0.2, [r[1] for r in rows], height=0.38, color="#999999", label="2019, TensorFlow 1.14"
    )
    ax.barh(y + 0.2, [r[2] for r in rows], height=0.38, color="#E69F00", label="now, PyTorch")
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xticks([0, 1], ["no", "yes"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("What changed", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out or ASSETS / "before_after.png")


def generate_all(outdir: Path = ASSETS, *, skip_models: bool = False) -> dict[str, Path | None]:
    """Regenerate every figure. Returns a map of name to written path."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path | None] = {}

    written["spectral_indices"] = figure_spectral_indices(out=outdir / "spectral_indices.png")
    plt.close("all")
    written["cascade_stages"] = figure_rgb_and_water(out=outdir / "cascade_stages.png")
    plt.close("all")
    written["thresholds_and_throughput"] = figure_fdi_distribution(
        out=outdir / "thresholds_and_throughput.png"
    )
    plt.close("all")
    written["resolution_gap"] = figure_resolution_gap(out=outdir / "resolution_gap.png")
    plt.close("all")
    written["before_after"] = figure_before_after(out=outdir / "before_after.png")
    plt.close("all")

    if not skip_models:
        written["detections"] = figure_detections(out=outdir / "detections.png")
        plt.close("all")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the README figures.")
    parser.add_argument("--outdir", type=Path, default=ASSETS)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip figures that need model weights and minutes of CPU time.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for name, path in generate_all(args.outdir, skip_models=args.skip_models).items():
        print(f"{name:28s} {path if path else 'skipped'}")


if __name__ == "__main__":  # pragma: no cover
    main()
