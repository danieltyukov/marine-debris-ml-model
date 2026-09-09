"""Cut Bonaire's coast into named segments from the OpenStreetMap coastline.

    python scripts/make_bonaire_segments.py --osm-json bonaire_coastline.json
    python scripts/make_bonaire_segments.py --fetch

The Cancun segments in ``assets/qroo_segments.geojson`` were hand-traced. For
Bonaire the coastline is taken from OpenStreetMap instead, so the geometry has a
source anyone can check and a licence (ODbL) that is recorded in the file. The
segments are the arcs of that coastline between named landmarks, south tip to
north coast along the windward side, plus one leeward stretch on the Kralendijk
waterfront that sargassum should not reach and which therefore acts as a control.

Landmark coordinates are OpenStreetMap nodes, looked up on 2026-09-09. Each is
snapped to the nearest point on the coastline, and the shorter of the two arcs
between consecutive landmarks is the segment, which for every pair here is the
obvious one.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import linemerge, substring
from shapely.ops import transform as shapely_transform

log = logging.getLogger("bonaire")

OVERPASS = "https://overpass-api.de/api/interpreter"
QUERY = '[out:json][timeout:120];(way["natural"="coastline"](12.0,-68.45,12.33,-68.17););out geom;'
UTM19N = "EPSG:32619"
_FWD = Transformer.from_crs("EPSG:4326", UTM19N, always_xy=True).transform
_BACK = Transformer.from_crs(UTM19N, "EPSG:4326", always_xy=True).transform

# OpenStreetMap nodes, (lon, lat). Snapped to the coastline at build time.
LANDMARKS: dict[str, tuple[str, float, float]] = {
    "willemstoren": ("Willemstoren lighthouse", -68.2373, 12.0283),
    "sorobon": ("Sorobon", -68.2358, 12.0921),
    "cai": ("Cai", -68.2225, 12.1032),
    "washikemba": ("Boka Washikemba", -68.2101, 12.1736),
    "spelonk": ("Spelonk lighthouse", -68.1966, 12.2123),
    "onima": ("Boka Onima", -68.3114, 12.2533),
    "chikitu": ("Playa Chikitu", -68.3492, 12.2797),
    "kokolishi": ("Boka Kokolishi", -68.3687, 12.3048),
    "te_amo": ("Te Amo Beach", -68.2798, 12.1346),
    "playa_lechi": ("Playa Lechi", -68.2833, 12.1615),
}

# id, name, from, to, exposure, note
SEGMENTS: list[tuple[str, str, str, str, str, str]] = [
    (
        "willemstoren_sorobon",
        "Willemstoren to Sorobon",
        "willemstoren",
        "sorobon",
        "windward",
        "South-east coast past the Pekelmeer salt flats to the south shore of Lac.",
    ),
    (
        "lac_bay",
        "Lac Bay",
        "sorobon",
        "cai",
        "windward",
        "The lagoon behind the reef, Sorobon to Cai, mangroves included.",
    ),
    (
        "cai_washikemba",
        "Cai to Boka Washikemba",
        "cai",
        "washikemba",
        "windward",
        "Open east coast north of Lac.",
    ),
    (
        "washikemba_spelonk",
        "Boka Washikemba to Spelonk",
        "washikemba",
        "spelonk",
        "windward",
        "Lagun and the coast up to the easternmost point.",
    ),
    (
        "spelonk_onima",
        "Spelonk to Boka Onima",
        "spelonk",
        "onima",
        "windward",
        "The long north-east coast.",
    ),
    (
        "onima_chikitu",
        "Boka Onima to Playa Chikitu",
        "onima",
        "chikitu",
        "windward",
        "North coast towards Washington Slagbaai.",
    ),
    (
        "chikitu_kokolishi",
        "Playa Chikitu to Boka Kokolishi",
        "chikitu",
        "kokolishi",
        "windward",
        "Inside Washington Slagbaai National Park.",
    ),
    (
        "kralendijk",
        "Kralendijk waterfront",
        "te_amo",
        "playa_lechi",
        "leeward",
        "Leeward control: Te Amo Beach to Playa Lechi, which sargassum should not reach.",
    ),
]


def fetch_osm() -> dict:
    data = urllib.parse.urlencode({"data": QUERY}).encode()
    with urllib.request.urlopen(OVERPASS, data=data, timeout=180) as resp:
        return json.load(resp)


def coastline_ring(osm: dict) -> LineString:
    """Merge the coastline ways into the island's outline, in UTM metres.

    The largest merged component is the main island; Klein Bonaire is a
    separate, much shorter ring and is dropped.
    """
    lines = []
    for element in osm.get("elements", []):
        if element.get("type") != "way":
            continue
        coords = [(n["lon"], n["lat"]) for n in element.get("geometry", [])]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    if not lines:
        raise SystemExit("no coastline ways in the Overpass response")
    merged = linemerge(lines)
    parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    projected = [shapely_transform(_FWD, p) for p in parts]
    ring = max(projected, key=lambda g: g.length)
    log.info(
        "%d coastline ways merged into %d components; keeping the %.1f km one",
        len(lines),
        len(parts),
        ring.length / 1000,
    )
    return ring


def _forward_arc(ring: LineString, start: float, end: float) -> LineString:
    """Arc from ``start`` to ``end`` metres along the ring in the ring's direction."""
    if start <= end:
        return substring(ring, start, end)
    first = substring(ring, start, ring.length)
    second = substring(ring, 0.0, end)
    merged = linemerge([first, second])
    if merged.geom_type != "LineString":
        # The two pieces did not share the wrap point exactly; join them by hand.
        merged = LineString([*first.coords, *second.coords[1:]])
    return merged


def cut_segment(ring: LineString, a: Point, b: Point) -> LineString:
    """Shorter coastline arc between two points snapped onto the ring."""
    da, db = ring.project(a), ring.project(b)
    forward = _forward_arc(ring, da, db)
    backward = _forward_arc(ring, db, da)
    return forward if forward.length <= backward.length else backward


def build(osm: dict, *, simplify_m: float) -> dict:
    ring = coastline_ring(osm)
    snapped = {key: Point(_FWD(lon, lat)) for key, (_name, lon, lat) in LANDMARKS.items()}
    features = []
    for seg_id, name, start, end, exposure, note in SEGMENTS:
        arc = cut_segment(ring, snapped[start], snapped[end])
        arc = arc.simplify(simplify_m, preserve_topology=False)
        length_m = arc.length
        geom = shapely_transform(_BACK, arc)
        log.info("%-22s %6.1f km  %s", seg_id, length_m / 1000, name)
        features.append(
            {
                "type": "Feature",
                "id": seg_id,
                "properties": {
                    "name": name,
                    "exposure": exposure,
                    "from": LANDMARKS[start][0],
                    "to": LANDMARKS[end][0],
                    "length_m": round(length_m),
                    "note": note,
                },
                "geometry": mapping(geom),
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "properties": {
            "title": "Bonaire coastal segments",
            "source": "Coastline and landmark positions from OpenStreetMap, natural=coastline "
            "ways within 12.0..12.33N, 68.45..68.17W, retrieved 2026-09-09 via Overpass.",
            "license": "Geometry (c) OpenStreetMap contributors, ODbL 1.0, "
            "https://www.openstreetmap.org/copyright",
            "method": "Arcs of the merged coastline between named landmarks, snapped to the "
            f"nearest coastline point, simplified to {simplify_m:g} m. Built by "
            "scripts/make_bonaire_segments.py.",
            "island": "Bonaire, Caribbean Netherlands",
        },
        "features": features,
    }


def island_polygon(osm: dict, *, simplify_m: float) -> dict:
    """The main island as one WGS84 polygon, for masking land out of the surf zones.

    A surf zone is a buffer on both sides of the shoreline. Without a land mask the
    landward half counts as "clear" in every observability figure and halves them.
    """
    ring = coastline_ring(osm)
    polygon = Polygon(ring.coords).buffer(0).simplify(simplify_m, preserve_topology=True)
    if polygon.geom_type != "Polygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    log.info("island polygon %.1f km2", polygon.area / 1e6)
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "properties": {
            "title": "Bonaire land polygon",
            "source": "OpenStreetMap natural=coastline ways, retrieved 2026-09-09 via Overpass, "
            "merged into the island outline.",
            "license": "Geometry (c) OpenStreetMap contributors, ODbL 1.0, "
            "https://www.openstreetmap.org/copyright",
            "method": f"Largest merged coastline ring, simplified to {simplify_m:g} m. Built by "
            "scripts/make_bonaire_segments.py.",
        },
        "features": [
            {
                "type": "Feature",
                "id": "bonaire",
                "properties": {"name": "Bonaire"},
                "geometry": mapping(shapely_transform(_BACK, polygon)),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm-json", type=Path, help="A saved Overpass response to use.")
    parser.add_argument("--fetch", action="store_true", help="Query Overpass live.")
    parser.add_argument("--out", type=Path, default=Path("assets/bonaire_segments.geojson"))
    parser.add_argument("--island", type=Path, default=Path("assets/bonaire_island.geojson"))
    parser.add_argument("--simplify-m", type=float, default=15.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.osm_json:
        osm = json.loads(args.osm_json.read_text(encoding="utf-8"))
    elif args.fetch:
        osm = fetch_osm()
    else:
        raise SystemExit("give --osm-json <file> or --fetch")

    collection = build(osm, simplify_m=args.simplify_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collection, indent=1), encoding="utf-8")
    log.info("wrote %s (%d segments)", args.out, len(collection["features"]))
    island = island_polygon(osm, simplify_m=args.simplify_m)
    args.island.write_text(json.dumps(island, indent=1), encoding="utf-8")
    log.info("wrote %s", args.island)


if __name__ == "__main__":
    main()
