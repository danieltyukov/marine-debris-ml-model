"""Plotting primitives for figures and reports.

These functions take plain arrays and detections rather than pipeline objects, so
they can be unit tested without network access, model weights or a STAC endpoint.
Composition into the actual README figures happens in ``mdebris.viz.figures``.

Matplotlib is imported lazily inside each function. Importing pyplot at module
scope costs about a second and pulls in a GUI backend probe, which is wasteful for
the many code paths that never draw anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mdebris.types import Detection, SurfaceClass

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "CLASS_COLORS",
    "plot_benchmark_bars",
    "plot_confusion_matrix",
    "plot_detections",
    "plot_index_heatmap",
    "plot_mask_overlay",
    "save_figure",
    "stretch_to_uint8",
]

# A colourblind-safe qualitative palette (Okabe-Ito). Debris is the vivid orange so
# it stands out against ocean blue, and the confuser classes get visually distinct
# hues, since the whole point of showing them is to tell them apart at a glance.
CLASS_COLORS: dict[SurfaceClass, str] = {
    SurfaceClass.DEBRIS: "#E69F00",
    SurfaceClass.SARGASSUM: "#009E73",
    SurfaceClass.SHIP: "#CC79A7",
    SurfaceClass.WAKE: "#56B4E9",
    SurfaceClass.FOAM: "#F0E442",
    SurfaceClass.CLOUD: "#FFFFFF",
    SurfaceClass.SEDIMENT: "#D55E00",
    SurfaceClass.WATER: "#0072B2",
    SurfaceClass.UNKNOWN: "#999999",
}


def stretch_to_uint8(
    array: np.ndarray,
    *,
    low: float = 2.0,
    high: float = 98.0,
    per_channel: bool = True,
) -> np.ndarray:
    """Percentile contrast stretch to display range.

    Open water occupies a narrow, dark slice of the reflectance range, so a linear
    map from [0, 1] to [0, 255] renders an ocean chip as almost pure black. Clipping
    to percentiles of the actual data is what makes the imagery legible.

    NaN is treated as no-data and rendered as zero rather than propagating through
    the percentile computation.

    Args:
        array: 2-D (H, W) or 3-D (H, W, C) float array.
        low: Lower percentile clipped to 0.
        high: Upper percentile clipped to 255.
        per_channel: Stretch each channel independently. Per-channel acts as a
            white balance, which usually looks better but shifts relative colour
            between bands. Pass False to preserve true colour ratios.

    Returns:
        uint8 array of the same shape.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {low}, {high}")

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2 or not per_channel:
        return _stretch_plane(data, low, high)

    out = np.empty_like(data, dtype=np.uint8)
    for c in range(data.shape[2]):
        out[..., c] = _stretch_plane(data[..., c], low, high)
    return out


def _stretch_plane(plane: np.ndarray, low: float, high: float) -> np.ndarray:
    finite = np.isfinite(plane)
    if not finite.any():
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(plane[finite], [low, high])
    if hi <= lo:
        # A constant plane has no contrast to stretch; mid-grey is more honest
        # than an arbitrary saturation to black or white.
        return np.full(plane.shape, 128, dtype=np.uint8)
    scaled = (plane - lo) / (hi - lo)
    return (np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0) * 255.0).astype(np.uint8)


def plot_detections(
    image: np.ndarray,
    detections: list[Detection],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_scores: bool = True,
    min_score: float = 0.0,
) -> Axes:
    """Draw detection boxes over an image chip, coloured by class.

    Args:
        image: HxWx3 uint8 RGB, or a float array which will be stretched.
        detections: Detections in pixel coordinates of this image.
        ax: Existing axes to draw on. A new figure is created when omitted.
        title: Optional axes title.
        show_scores: Annotate each box with its class and confidence.
        min_score: Skip detections below this score.

    Returns:
        The axes drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rgb = image if image.dtype == np.uint8 else stretch_to_uint8(image)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    for det in detections:
        if det.score < min_score:
            continue
        color = CLASS_COLORS.get(det.label, CLASS_COLORS[SurfaceClass.UNKNOWN])
        ax.add_patch(
            Rectangle(
                (det.bbox.xmin, det.bbox.ymin),
                det.bbox.width,
                det.bbox.height,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
            )
        )
        if show_scores:
            ax.text(
                det.bbox.xmin,
                max(det.bbox.ymin - 4.0, 0.0),
                f"{det.label} {det.score:.2f}",
                color="black",
                fontsize=7,
                bbox={"facecolor": color, "alpha": 0.85, "pad": 1.2, "edgecolor": "none"},
            )
    return ax


def plot_index_heatmap(
    index: np.ndarray,
    *,
    name: str,
    ax: Axes | None = None,
    cmap: str = "viridis",
    mask: np.ndarray | None = None,
    threshold: float | None = None,
) -> Axes:
    """Render a spectral index as a heatmap.

    Args:
        index: 2-D index array, may contain NaN for no-data.
        name: Index name used in the title and colourbar label.
        ax: Existing axes, or None to create a figure.
        cmap: Matplotlib colormap.
        mask: Optional boolean mask. False pixels are dimmed out, which is how a
            water-only view of an index is produced.
        threshold: If given, draw a contour at this value to show the decision
            boundary the cascade actually applies.

    Returns:
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    shown = np.array(index, dtype=np.float32, copy=True)
    if mask is not None:
        shown[~mask] = np.nan

    finite = np.isfinite(shown)
    if finite.any():
        vmin, vmax = np.percentile(shown[finite], [2, 98])
        if vmax <= vmin:
            vmin, vmax = float(np.nanmin(shown)), float(np.nanmax(shown)) or 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(name, fontsize=11, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=name)

    if threshold is not None and finite.any() and vmin < threshold < vmax:
        ax.contour(
            np.nan_to_num(shown, nan=vmin),
            levels=[threshold],
            colors="#E69F00",
            linewidths=1.2,
        )
    return ax


def plot_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    ax: Axes | None = None,
    color: str = "#E69F00",
    alpha: float = 0.45,
    title: str | None = None,
) -> Axes:
    """Overlay a boolean mask on an image, for SAM 2 segmentation output."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    rgb = image if image.dtype == np.uint8 else stretch_to_uint8(image)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = to_rgb(color)
    overlay[..., 3] = np.where(mask.astype(bool), alpha, 0.0)
    ax.imshow(overlay)
    return ax


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    *,
    ax: Axes | None = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> Axes:
    """Confusion matrix heatmap with rows as ground truth and columns as predictions.

    The orientation matches the legacy evaluation output so old and new tables can
    be read the same way.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(matrix, dtype=np.float64)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            row_sums = data.sum(axis=1, keepdims=True)
            data = np.where(row_sums > 0, data / row_sums, 0.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.1 * len(labels) + 2))

    im = ax.imshow(data, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=11, pad=10)

    # Annotate each cell, flipping text colour on dark cells so it stays readable.
    hi = data.max() if data.size and data.max() > 0 else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}" if normalize else f"{int(val)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val > 0.6 * hi else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_benchmark_bars(
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    ax: Axes | None = None,
    highlight: int | None = None,
    value_fmt: str = "{:.3f}",
) -> Axes:
    """Horizontal bar chart for benchmark comparisons.

    Args:
        labels: Bar labels.
        values: Bar values, same length as labels.
        ylabel: Axis label describing the quantity.
        title: Chart title.
        ax: Existing axes or None.
        highlight: Index of the bar to emphasise, for example the configuration
            the project actually ships.
        value_fmt: Format string for the value annotation.
    """
    import matplotlib.pyplot as plt

    if len(labels) != len(values):
        raise ValueError(f"labels ({len(labels)}) and values ({len(values)}) length mismatch")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 2))

    colors = ["#0072B2"] * len(values)
    if highlight is not None:
        colors[highlight] = "#E69F00"

    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=12, pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    span = max(values) if values else 1.0
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + span * 0.015,
            bar.get_y() + bar.get_height() / 2,
            value_fmt.format(val),
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, span * 1.18)
    return ax


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 150, tight: bool = True) -> Path:
    """Write a figure to disk, creating parent directories.

    Returns the resolved path so callers can log or assert on it.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    return out


def figure_grid(nrows: int, ncols: int, *, figsize: tuple[float, float] | None = None) -> Any:
    """Create a figure and a flat list of axes, a small convenience over subplots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5.0 * ncols, 4.6 * nrows))
    flat = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
    return fig, flat
