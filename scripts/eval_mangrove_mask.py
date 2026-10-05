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
covered flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdebris.coastal.islands import load_island
from mdebris.coastal.masks import load_polygons, polygon_mask
from mdebris.coastal.runner import Persistence


def coverage(island_key: str, persistence_path: Path) -> dict[str, dict[str, int]]:
    island = load_island(island_key)
    grid = Persistence.load(persistence_path)
    mask = polygon_mask(
        load_polygons(island.mangroves_path),
        grid.transform,
        grid.crs,
        grid.hit_total.shape,
        buffer_m=island.mangrove_buffer_m,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--island", required=True)
    parser.add_argument("--persistence", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="Also write the numbers here.")
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


if __name__ == "__main__":
    main()
