"""Figures for a season run: the timeline, the persistence map and one pass up close.

Titles and zoom windows come from the island's ``island.toml``, so the same three
functions draw Bonaire, Aruba, Curaçao or any island added later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["OBSERVABILITY_STYLE", "pass_figure", "persistence_figure", "timeline_figure"]

OBSERVABILITY_STYLE = {
    "observed": ("#1f77b4", "o"),
    "partial": ("#d95f02", "s"),
    "blind": ("#7b3294", "x"),
}
MANGROVE_RGBA = (0.20, 0.55, 0.25, 0.55)


def _title(island: Any, part: Any) -> str:
    return f"{island.name}, {part.label}" if part.label else island.name


def timeline_figure(
    rows: Sequence[Mapping[str, Any]],
    segments: Sequence[Any],
    path: Path,
    *,
    island: Any,
    part: Any,
) -> None:
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
    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.5 * len(order)), constrained_layout=True)
    for row in rows:
        if row["segment_id"] not in names:
            continue
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
    ax.set_ylim(len(order) - 0.5, -0.5)
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
            label="flagged as sargassum (unconfirmed)",
        )
    )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False
    )
    tiles = " and ".join(part.tiles)
    ax.set_title(
        f"{_title(island, part)} (tile {tiles}): every Sentinel-2 pass, could the coast be seen?"
    )
    save_figure(fig, path)


def _projector(crs: Any) -> Any:
    from pyproj import CRS, Transformer

    return Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform


def _segment_pixels(segment: Any, fwd: Any, to_pixel: Any, c0: int = 0, r0: int = 0):
    from shapely.ops import transform as shapely_transform

    xs, ys = [], []
    for x, y in shapely_transform(fwd, segment.geometry).coords:
        col, row = to_pixel * (x, y)
        xs.append(col - c0)
        ys.append(row - r0)
    return xs, ys


def _crop(
    window: Any, fwd: Any, to_pixel: Any, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    (lon0, lat0), (lon1, lat1) = window.corners
    x0, y0 = to_pixel * fwd(lon0, lat0)
    x1, y1 = to_pixel * fwd(lon1, lat1)
    r0, r1 = sorted((int(y0), int(y1)))
    c0, c1 = sorted((int(x0), int(x1)))
    return max(0, r0), min(shape[0], r1), max(0, c0), min(shape[1], c1)


def _full(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return 0, shape[0], 0, shape[1]


def persistence_figure(
    persistence: Any, segments: Sequence[Any], path: Path, *, island: Any, part: Any
) -> None:
    """How often each pixel was flagged, over the passes that saw it, with the mangrove mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mdebris.coastal.runner import STATIONARY_MIN_PASSES
    from mdebris.viz.plots import save_figure

    fwd = _projector(persistence.crs)
    to_pixel = ~persistence.transform
    shape = persistence.hit_total.shape
    seen = persistence.usable_total >= STATIONARY_MIN_PASSES
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(
            seen, persistence.hit_total / np.maximum(persistence.usable_total, 1), np.nan
        )
    masked = persistence.masked if persistence.masked is not None else np.zeros(shape, bool)

    panels = [
        (
            part.whole.title if part.whole else "whole area read",
            _crop(part.whole, fwd, to_pixel, shape) if part.whole else _full(shape),
        )
    ]
    if part.zoom is not None:
        panels.append((part.zoom.title, _crop(part.zoom, fwd, to_pixel, shape)))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7), squeeze=False)
    image = None
    for ax, (title, (r0, r1, c0, c1)) in zip(axes[0], panels, strict=True):
        # Land and never-seen water in grey, so white means "seen and never flagged".
        unseen = np.where(seen[r0:r1, c0:c1], np.nan, 0.55)
        ax.imshow(unseen, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
        overlay = np.zeros((r1 - r0, c1 - c0, 4))
        overlay[masked[r0:r1, c0:c1]] = MANGROVE_RGBA
        ax.imshow(overlay, interpolation="nearest")
        image = ax.imshow(
            share[r0:r1, c0:c1], cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        for segment in segments:
            xs, ys = _segment_pixels(segment, fwd, to_pixel, c0, r0)
            ax.plot(xs, ys, lw=1.0, color="#444444", alpha=0.8)
        ax.set_xlim(0, c1 - c0)
        ax.set_ylim(r1 - r0, 0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    if masked.any():
        axes[0][0].legend(
            handles=[Patch(color=MANGROVE_RGBA, label="removed by the mangrove mask")],
            loc="lower left",
        )
    fig.colorbar(
        image,
        ax=axes[0][-1],
        fraction=0.046,
        pad=0.02,
        label="share of the passes that saw the pixel on which it was called sargassum",
    )
    fig.suptitle(
        f"{_title(island, part)}, persistence over {persistence.n_passes} passes: how often each "
        "pixel was called sargassum when it could be seen",
        fontsize=11,
    )
    save_figure(fig, path)


def pass_figure(
    result: Any, segments: Sequence[Any], path: Path, *, island: Any, part: Any
) -> None:
    """True colour, the three-state classification, and the segment verdicts, for one pass."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mdebris.viz.plots import save_figure

    fwd = _projector(result.crs)
    to_pixel = ~result.transform
    shape = result.hit_mask.shape
    verdict = {r["segment_id"]: r["observability"] for r in result.rows}

    def draw(ax: Any, colour_of: Any = None) -> None:
        for segment in segments:
            xs, ys = _segment_pixels(segment, fwd, to_pixel)
            ax.plot(xs, ys, lw=2.0, color=colour_of(segment) if colour_of else "white")

    n_panels = 4 if part.zoom is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 7.5), constrained_layout=True)
    axes[0].imshow(result.rgb)
    axes[0].set_title(f"{island.name} {str(result.scene.datetime)[:10]}, true colour")
    draw(axes[0])

    axes[1].imshow(result.rgb)
    overlay = np.zeros((*shape, 4))
    overlay[result.masked] = MANGROVE_RGBA
    overlay[result.uncertain_mask] = (0.85, 0.37, 0.01, 0.9)
    overlay[result.hit_mask] = (0.12, 0.47, 0.71, 1.0)
    overlay[result.cloud_mask] = (0.6, 0.6, 0.6, 0.35)
    axes[1].imshow(overlay)
    axes[1].set_title("flagged (blue), uncertain (orange), cloud (grey)")
    draw(axes[1])
    legend = [
        Patch(color="#1f77b4", label="flagged as sargassum"),
        Patch(color="#d95f02", label="uncertain"),
        Patch(color="0.6", label="cloud"),
    ]
    if result.masked.any():
        legend.append(Patch(color=MANGROVE_RGBA, label="mangrove mask"))
    axes[1].legend(handles=legend, loc="lower left")

    axes[2].imshow(result.rgb)
    draw(axes[2], colour_of=lambda s: OBSERVABILITY_STYLE[verdict[s.segment_id]][0])
    axes[2].set_title("segment verdict")
    axes[2].legend(
        handles=[Patch(color=c, label=k) for k, (c, _m) in OBSERVABILITY_STYLE.items()],
        loc="lower left",
    )
    if part.zoom is not None:
        r0, r1, c0, c1 = _crop(part.zoom, fwd, to_pixel, shape)
        axes[3].imshow(result.rgb[r0:r1, c0:c1])
        axes[3].imshow(overlay[r0:r1, c0:c1])
        for segment in segments:
            xs, ys = _segment_pixels(segment, fwd, to_pixel, c0, r0)
            axes[3].plot(xs, ys, lw=1.2, color="white", alpha=0.8)
        axes[3].set_xlim(0, c1 - c0)
        axes[3].set_ylim(r1 - r0, 0)
        axes[3].set_title(f"{part.zoom.title}, same pass")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(fig, path)
