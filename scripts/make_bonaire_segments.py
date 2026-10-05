"""Cut Bonaire's coast into named segments from the OpenStreetMap coastline.

    python scripts/make_bonaire_segments.py --fetch
    python scripts/make_bonaire_segments.py --osm-json bonaire_coastline.json

Kept so the documented command still runs. It is
``scripts/make_island_segments.py --island bonaire`` with the 2.0 flags: ``--out`` for the
segments and ``--island`` for the land polygon. The landmarks, the segments between
them and the leeward control (the Kralendijk waterfront) are listed in
``assets/islands/bonaire/island.toml``; ``mdebris.coastal.build`` has the method. With
``--fetch`` the mangrove mask is rebuilt as well.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from mdebris.coastal.build import (
    build_land,
    build_segments,
    build_wetlands,
    coastline_query,
    overpass,
    wetland_query,
)
from mdebris.coastal.islands import load_island

log = logging.getLogger("bonaire")


def main() -> None:
    island = load_island("bonaire")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm-json", type=Path, help="A saved Overpass response to use.")
    parser.add_argument("--fetch", action="store_true", help="Query Overpass live.")
    parser.add_argument("--out", type=Path, default=island.segments_path)
    parser.add_argument("--island", type=Path, default=island.island_path)
    parser.add_argument("--mangroves", type=Path, default=island.mangroves_path)
    parser.add_argument("--simplify-m", type=float, default=island.simplify_m)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    island = replace(island, simplify_m=args.simplify_m)

    if args.osm_json:
        osm = json.loads(args.osm_json.read_text(encoding="utf-8"))
    elif args.fetch:
        osm = overpass(coastline_query(island.osm_bbox))
    else:
        raise SystemExit("give --osm-json <file> or --fetch")

    for path, blob in (
        (args.out, build_segments(osm, island)),
        (args.island, build_land(osm, island)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info("wrote %s (%d features)", path, len(blob["features"]))
    if args.fetch:
        wet = build_wetlands(overpass(wetland_query(island.osm_bbox)), island)
        args.mangroves.write_text(
            json.dumps(wet, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log.info("wrote %s (%d features)", args.mangroves, len(wet["features"]))


if __name__ == "__main__":
    main()
