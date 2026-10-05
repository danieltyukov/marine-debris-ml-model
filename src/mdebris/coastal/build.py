"""Build an island's segments, land polygons and mangrove mask from OpenStreetMap.

The coastline comes from OpenStreetMap so the geometry has a source anyone can check
and a licence (ODbL) that is recorded in every file. Segments are arcs of the merged
coastline between named landmarks listed in the island's ``island.toml``: each landmark
is snapped to the nearest point of the coastline, and the shorter of the two arcs
between a segment's two landmarks is the segment.

Three things vary between islands, and the configuration says which applies:

- ``utm_epsg``: the metric CRS the arcs are cut and simplified in (UTM 19N for the ABC
  islands, 20N for Sint Maarten).
- ``osm.segment_rings``: how many coastline rings segments may be cut on. Usually one,
  the main island. Saint Martin needs two, because Simpson Bay Lagoon is open to the
  sea at both ends and the Lowlands form a ring of their own.
- ``osm.land``: ``"main"`` masks the main island only; ``"all"`` masks every closed
  coastline ring in the query box, so an islet inside a surf zone is not classified
  as water.

The functions here are pure apart from :func:`overpass`, so the builder is tested on a
small synthetic island without the network.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from mdebris.coastal.masks import is_vegetated_wetland

__all__ = [
    "OVERPASS_MIRRORS",
    "build_land",
    "build_segments",
    "build_wetlands",
    "coastline_components",
    "coastline_query",
    "cut_segment",
    "overpass",
    "wetland_query",
]

log = logging.getLogger("mdebris.build")

OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
# Overpass answers 406 to a request without a User-Agent.
_HEADERS = {
    "User-Agent": "mdebris (+https://github.com/danieltyukov/marine-debris-ml-model)",
    "Accept": "application/json",
}
_LICENSE = (
    "Geometry (c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright"
)
_CRS84 = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}


def _bbox_text(bbox: Sequence[float]) -> str:
    south, west, north, east = bbox
    return f"{south},{west},{north},{east}"


def coastline_query(bbox: Sequence[float]) -> str:
    """Overpass query for every ``natural=coastline`` way in ``(south, west, north, east)``."""
    return f'[out:json][timeout:120];(way["natural"="coastline"]({_bbox_text(bbox)}););out geom;'


def wetland_query(bbox: Sequence[float]) -> str:
    """Overpass query for every ``natural=wetland`` way and relation in the box."""
    box = _bbox_text(bbox)
    return (
        f'[out:json][timeout:120];(way["natural"="wetland"]({box});'
        f'relation["natural"="wetland"]({box}););out geom;'
    )


def overpass(query: str, *, mirrors: Sequence[str] = OVERPASS_MIRRORS, attempts: int = 6) -> dict:
    """Run an Overpass query, moving to the next mirror on a busy or failed server."""
    last: Exception | None = None
    for attempt in range(attempts):
        url = mirrors[attempt % len(mirrors)]
        try:
            data = urllib.parse.urlencode({"data": query}).encode()
            req = urllib.request.Request(url, data=data, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # HTTP 429/504 are routine on the public servers
            last = exc
            log.warning("Overpass %s: %s; trying again", url, exc)
            time.sleep(10)
    raise RuntimeError(f"every Overpass attempt failed: {last}")


def _transformers(epsg: int) -> tuple[Any, Any]:
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    back = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    return fwd, back


def coastline_components(osm: Mapping[str, Any], *, epsg: int) -> list[Any]:
    """Coastline ways merged into lines, projected to ``epsg``, longest first."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge
    from shapely.ops import transform as shapely_transform

    lines = []
    for element in osm.get("elements", []):
        if element.get("type") != "way":
            continue
        coords = [(n["lon"], n["lat"]) for n in element.get("geometry", [])]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    if not lines:
        raise ValueError("no coastline ways in the Overpass response")
    merged = linemerge(lines)
    parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    fwd, _back = _transformers(epsg)
    projected = [shapely_transform(fwd, p) for p in parts]
    log.info(
        "%d coastline ways merged into %d components; the longest is %.1f km",
        len(lines),
        len(parts),
        max(p.length for p in projected) / 1000,
    )
    return sorted(projected, key=lambda g: -g.length)


def _rings(components: Sequence[Any]) -> list[Any]:
    """Closed coastline rings, longest first; the longest component if none is closed.

    A query box that cuts through a coastline leaves open fragments, and those must
    not outrank a small closed island.
    """
    closed = [c for c in components if c.is_ring]
    return closed or list(components[:1])


def _forward_arc(ring: Any, start: float, end: float) -> Any:
    """Arc from ``start`` to ``end`` metres along the ring in the ring's direction."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge, substring

    if start <= end:
        return substring(ring, start, end)
    first = substring(ring, start, ring.length)
    second = substring(ring, 0.0, end)
    # A landmark exactly on the ring's start point leaves one piece of zero length.
    if second.geom_type != "LineString" or second.length == 0:
        return first
    if first.geom_type != "LineString" or first.length == 0:
        return second
    merged = linemerge([first, second])
    if merged.geom_type != "LineString":
        # The two pieces did not share the wrap point exactly; join them by hand.
        merged = LineString([*first.coords, *second.coords[1:]])
    return merged


def cut_segment(ring: Any, a: Any, b: Any) -> Any:
    """Shorter coastline arc between two points snapped onto the ring."""
    da, db = ring.project(a), ring.project(b)
    forward = _forward_arc(ring, da, db)
    backward = _forward_arc(ring, db, da)
    return forward if forward.length <= backward.length else backward


def _retrieved(osm: Mapping[str, Any]) -> str:
    return str(osm.get("osm3s", {}).get("timestamp_osm_base", "unknown"))[:10]


def build_segments(osm: Mapping[str, Any], island: Any) -> dict[str, Any]:
    """The island's segments as a GeoJSON FeatureCollection (EPSG:4326)."""
    from shapely.geometry import Point, mapping
    from shapely.ops import transform as shapely_transform

    fwd, back = _transformers(island.utm_epsg)
    rings = _rings(coastline_components(osm, epsg=island.utm_epsg))[: max(1, island.segment_rings)]
    snapped: dict[str, Any] = {}
    ring_of: dict[str, int] = {}
    for key, mark in island.landmarks.items():
        point = Point(fwd(mark.lon, mark.lat))
        idx = min(range(len(rings)), key=lambda i: rings[i].distance(point))
        ring_of[key] = idx
        snapped[key] = point
        log.info("%-18s snapped %5.0f m to coastline ring %d", key, rings[idx].distance(point), idx)
    features = []
    for spec in island.segments:
        if ring_of[spec.start] != ring_of[spec.end]:
            raise ValueError(f"{spec.segment_id}: its two landmarks are on different rings")
        ring = rings[ring_of[spec.start]]
        arc = cut_segment(ring, snapped[spec.start], snapped[spec.end])
        arc = arc.simplify(island.simplify_m, preserve_topology=False)
        length_m = arc.length
        start, end = island.landmarks[spec.start], island.landmarks[spec.end]
        props: dict[str, Any] = {"name": spec.name, "exposure": spec.exposure, **spec.extra}
        props["from"] = start.label
        if start.osm or end.osm:
            props["from_osm"] = start.osm
        props["to"] = end.label
        if start.osm or end.osm:
            props["to_osm"] = end.osm
        props.update({"length_m": round(length_m), "note": spec.note})
        log.info("%-26s %6.1f km  %s", spec.segment_id, length_m / 1000, spec.name)
        features.append(
            {
                "type": "Feature",
                "id": spec.segment_id,
                "properties": props,
                "geometry": mapping(shapely_transform(back, arc)),
            }
        )
    south, west, north, east = island.osm_bbox or (0, 0, 0, 0)
    return {
        "type": "FeatureCollection",
        "crs": _CRS84,
        "properties": {
            "title": f"{island.name} coastal segments",
            "source": "Coastline from OpenStreetMap, natural=coastline ways within "
            f"{south}..{north}N, {west}..{east}E, retrieved {_retrieved(osm)} via Overpass; "
            "landmark positions from OpenStreetMap, listed in island.toml.",
            "license": _LICENSE,
            "method": "Arcs of the merged coastline between named landmarks, snapped to the "
            f"nearest coastline point, cut in EPSG:{island.utm_epsg} and simplified to "
            f"{island.simplify_m:g} m. Built by scripts/make_island_segments.py.",
            "island": island.name,
        },
        "features": features,
    }


def build_land(osm: Mapping[str, Any], island: Any) -> dict[str, Any]:
    """The land polygons removed before classification, as a FeatureCollection."""
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as shapely_transform

    _fwd, back = _transformers(island.utm_epsg)
    rings = _rings(coastline_components(osm, epsg=island.utm_epsg))
    if island.land == "main":
        rings = rings[:1]
    features = []
    for i, ring in enumerate(rings):
        polygon = Polygon(ring.coords).buffer(0)
        # The rings segments are cut on are simplified like the segments; islets are not.
        if i < max(1, island.segment_rings):
            polygon = polygon.simplify(island.simplify_m, preserve_topology=True)
        if polygon.geom_type != "Polygon":
            polygon = max(polygon.geoms, key=lambda g: g.area)
        features.append(
            {
                "type": "Feature",
                "id": island.key if i == 0 else f"ring_{i}",
                "properties": {
                    "name": island.name if i == 0 else "islet or second ring",
                    "area_m2": round(polygon.area),
                },
                "geometry": mapping(shapely_transform(back, polygon)),
            }
        )
    log.info(
        "land: %d polygon(s), main %.1f km2",
        len(features),
        features[0]["properties"]["area_m2"] / 1e6,
    )
    which = (
        "The longest closed coastline ring"
        if island.land == "main"
        else "Every closed merged coastline ring in the query box"
    )
    return {
        "type": "FeatureCollection",
        "crs": _CRS84,
        "properties": {
            "title": f"{island.name} land polygons",
            "source": f"OpenStreetMap natural=coastline ways, retrieved {_retrieved(osm)} via "
            "Overpass.",
            "license": _LICENSE,
            "method": f"{which}; the rings segments are cut on simplified to "
            f"{island.simplify_m:g} m, islets left as mapped. Built by "
            "scripts/make_island_segments.py.",
        },
        "features": features,
    }


def _element_polygons(element: Mapping[str, Any]) -> list[Any]:
    """Polygons of one Overpass way or multipolygon relation (``out geom``)."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import linemerge, polygonize, unary_union

    def coords(geometry: Sequence[Mapping[str, float]]) -> list[tuple[float, float]]:
        return [(p["lon"], p["lat"]) for p in geometry if p]

    if element.get("type") == "way":
        pts = coords(element.get("geometry", []))
        if len(pts) >= 4 and pts[0] == pts[-1]:
            poly = Polygon(pts).buffer(0)
            return [poly] if not poly.is_empty else []
        return []
    if element.get("type") != "relation":
        return []
    outer, inner = [], []
    for member in element.get("members", []):
        if member.get("type") != "way" or len(member.get("geometry", [])) < 2:
            continue
        line = LineString(coords(member["geometry"]))
        (inner if member.get("role") == "inner" else outer).append(line)
    if not outer:
        return []
    shell = unary_union(list(polygonize(linemerge(outer))))
    if inner:
        shell = shell.difference(unary_union(list(polygonize(linemerge(inner)))))
    shell = shell.buffer(0)
    if shell.is_empty:
        return []
    return list(shell.geoms) if shell.geom_type == "MultiPolygon" else [shell]


def build_wetlands(osm: Mapping[str, Any], island: Any) -> dict[str, Any]:
    """Mapped mangrove and other vegetated wetland, as a FeatureCollection.

    Only elements that :func:`~mdebris.coastal.masks.is_vegetated_wetland` accepts are
    kept, so seagrass beds and untagged salt ponds stay classifiable.
    """
    from shapely.geometry import mapping

    features = []
    for element in osm.get("elements", []):
        tags = element.get("tags", {})
        if not is_vegetated_wetland(tags):
            continue
        for j, poly in enumerate(_element_polygons(element)):
            features.append(
                {
                    "type": "Feature",
                    "id": f"{element['type']}/{element['id']}" + (f"#{j}" if j else ""),
                    "properties": {
                        "wetland": tags.get("wetland", ""),
                        "name": tags.get("name", ""),
                    },
                    "geometry": mapping(poly),
                }
            )
    log.info("wetland: %d vegetated polygons kept", len(features))
    return {
        "type": "FeatureCollection",
        "crs": _CRS84,
        "properties": {
            "title": f"{island.name} mangrove and vegetated wetland",
            "source": f"OpenStreetMap natural=wetland ways and relations, retrieved {_retrieved(osm)} "
            "via Overpass.",
            "license": _LICENSE,
            "method": "Kept: wetland=mangrove, swamp, marsh, saltmarsh, reedbed or wet_meadow. Left "
            "out: seagrass beds, tidal flats, salt pans and untagged wetland. The season runner "
            "buffers these polygons and removes them from the water side before classifying. "
            "Built by scripts/make_island_segments.py.",
        },
        "features": features,
    }
