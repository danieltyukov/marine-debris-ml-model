"""How much of a season's water side and flags an island's mangrove mask covers.

    git show 7d7847d:docs/bonaire_persistence.npz > bonaire_persistence_v2_0.npz
    python scripts/eval_mangrove_mask.py --island bonaire --persistence bonaire_persistence_v2_0.npz

Reads a persistence grid written by the season runner (``docs/<island>_persistence.npz``)
and the island's ``mangroves.geojson``, rasterises the mask on the grid with the same
buffer the runner uses, and prints per segment the water-side pixels, how many of them the
mask covers, the pixels ever flagged and how many of those the mask covers, and the same
for flag events (one pixel flagged on one pass). Nothing is downloaded.

Run on a grid from a run made without the mask (version 2.0, or ``--no-mangrove-mask``) it
says what the mask would have removed; run on a grid from a masked run it should report no
covered flags. ``--figure`` also draws, for the island's close-up window, how often each
pixel was flagged with the mask over it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdebris.coastal.islands import load_island
from mdebris.coastal.masks import load_polygons, polygon_mask
from mdebris.coastal.runner import Persistence


def _mask(island, grid):
    return polygon_mask(
        load_polygons(island.mangroves_path),
        grid.transform,
        grid.crs,
        grid.hit_total.shape,
        buffer_m=island.mangrove_buffer_m,
    )


def coverage(island_key: str, persistence_path: Path) -> dict[str, dict[str, int]]:
    island = load_island(island_key)
    grid = Persistence.load(persistence_path)
    mask = _mask(island, grid)
    flagged = grid.hit_total > 0
    out = {}
    for seg, zone in grid.zones.items():
        out[seg] = {
            "water_side_pixels": int(zone.sum()),
            "covered_by_mask": int((zone & mask).sum()),
            "flagged_pixels": int((zone & flagged).sum()),
            "flagged_pixels_covered": int((zone & flagged & mask).sum()),
            "flag_events": int(grid.hit_total[zone].sum()),
            "flag_events_covered": int(grid.hit_total[zone & mask].sum()),
        }
    return out


def figure(island_key: str, persistence_path: Path, path: Path, title: str) -> None:
    """Flag counts in the close-up window, with the mangrove mask drawn over them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from pyproj import CRS, Transformer

    from mdebris.viz.plots import save_figure

    island = load_island(island_key)
    grid = Persistence.load(persistence_path)
    window = island.part().zoom
    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(grid.crs), always_xy=True
    ).transform
    to_pixel = ~grid.transform
    (lon0, lat0), (lon1, lat1) = window.corners
    c0, r0 = (int(v) for v in to_pixel * fwd(lon0, lat0))
    c1, r1 = (int(v) for v in to_pixel * fwd(lon1, lat1))
    zone = np.zeros(grid.hit_total.shape, bool)
    for z in grid.zones.values():
        zone |= z
    hits = np.where(grid.hit_total > 0, grid.hit_total, np.nan)[r0:r1, c0:c1]
    mask = _mask(island, grid)[r0:r1, c0:c1]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.imshow(np.where(zone[r0:r1, c0:c1], np.nan, 0.55), cmap="Greys", vmin=0, vmax=1)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = (0.20, 0.55, 0.25, 0.45)
    ax.imshow(overlay)
    image = ax.imshow(
        hits,
        cmap="magma_r",
        vmin=1,
        vmax=max(2, int(np.nanmax(hits)) if np.isfinite(hits).any() else 2),
    )
    fig.colorbar(
        image, ax=ax, fraction=0.046, pad=0.02, label="passes on which the pixel was flagged"
    )
    ax.legend(
        handles=[
            Patch(
                color=(0.20, 0.55, 0.25, 0.45), label="mangrove mask (OpenStreetMap, 20 m buffer)"
            ),
            Patch(color="0.55", label="land, or outside the surf zones"),
        ],
        loc="lower left",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--island", required=True)
    parser.add_argument("--persistence", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="Also write the numbers here.")
    parser.add_argument("--figure", type=Path, help="Also draw the close-up window here.")
    parser.add_argument("--title", default="Flagged pixels and the mangrove mask")
    args = parser.parse_args()

    result = coverage(args.island, args.persistence)
    print(
        "| segment | water-side pixels | covered by the mask | pixels ever flagged "
        "| of which covered | flag events | of which covered |"
    )
    print("|---|---|---|---|---|---|---|")
    for seg, c in result.items():
        print(
            f"| {seg} | {c['water_side_pixels']:,} | {c['covered_by_mask']:,} | "
            f"{c['flagged_pixels']:,} | {c['flagged_pixels_covered']:,} | "
            f"{c['flag_events']:,} | {c['flag_events_covered']:,} |"
        )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        figure(args.island, args.persistence, args.figure, args.title)


if __name__ == "__main__":
    main()
