"""Build an island's segments, land polygons and mangrove mask from OpenStreetMap.

    python scripts/make_island_segments.py --island aruba --fetch
    python scripts/make_island_segments.py --island aruba --osm-json coast.json --wetland-json wet.json

Reads ``assets/islands/<island>/island.toml`` and writes, next to it:

- ``segments.geojson``: arcs of the merged OpenStreetMap coastline between the named
  landmarks the configuration lists, each snapped to the nearest coastline point;
- ``island.geojson``: the land polygons the season runner removes, buffered, before
  classifying (the main ring, or every closed ring, as ``osm.land`` says);
- ``mangroves.geojson``: mapped mangrove and other vegetated wetland, which the runner
  removes from the water side as well.

``--fetch`` queries Overpass (several public mirrors, tried in turn) and ``--save-osm``
keeps the raw responses so the build can be repeated offline. The raw responses are
not committed: the built GeoJSON files carry the OpenStreetMap attribution and the
retrieval date. ``mdebris.coastal.build`` has the method.
"""

from __future__ import annotations

import argparse
import json
import logging
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

log = logging.getLogger("segments")


def _write(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote %s (%d features)", path, len(blob["features"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--island", required=True, help="Island key, folder or island.toml.")
    parser.add_argument("--osm-json", type=Path, help="A saved Overpass coastline response.")
    parser.add_argument("--wetland-json", type=Path, help="A saved Overpass wetland response.")
    parser.add_argument("--fetch", action="store_true", help="Query Overpass live.")
    parser.add_argument("--save-osm", type=Path, help="Folder to keep the raw responses in.")
    parser.add_argument(
        "--segments-out", type=Path, help="Default: <island folder>/segments.geojson"
    )
    parser.add_argument("--land-out", type=Path, help="Default: <island folder>/island.geojson")
    parser.add_argument(
        "--mangroves-out", type=Path, help="Default: <island folder>/mangroves.geojson"
    )
    parser.add_argument("--simplify-m", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    island = load_island(args.island)
    if args.simplify_m is not None:
        from dataclasses import replace

        island = replace(island, simplify_m=args.simplify_m)
    if island.osm_bbox is None:
        raise SystemExit(f"{island.key}: island.toml has no osm.bbox to query")

    coast = wet = None
    if args.osm_json:
        coast = json.loads(args.osm_json.read_text(encoding="utf-8"))
    if args.wetland_json:
        wet = json.loads(args.wetland_json.read_text(encoding="utf-8"))
    if args.fetch:
        coast = coast or overpass(coastline_query(island.osm_bbox))
        wet = wet or overpass(wetland_query(island.osm_bbox))
    if coast is None and wet is None:
        raise SystemExit("give --osm-json and/or --wetland-json, or --fetch")
    if args.save_osm:
        args.save_osm.mkdir(parents=True, exist_ok=True)
        for name, blob in (("coastline", coast), ("wetland", wet)):
            if blob is not None:
                (args.save_osm / f"{island.key}_{name}.json").write_text(
                    json.dumps(blob), encoding="utf-8"
                )

    if coast is not None:
        _write(args.segments_out or island.segments_path, build_segments(coast, island))
        _write(args.land_out or island.island_path, build_land(coast, island))
    if wet is not None:
        _write(args.mangroves_out or island.mangroves_path, build_wetlands(wet, island))


if __name__ == "__main__":
    main()
