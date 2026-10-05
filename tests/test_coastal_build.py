"""The segment builder, on a small synthetic island instead of a live Overpass query.

The island is a 4 km square whose coastline arrives as two ways (so they have to be
merged), with a 200 m islet off one corner. Landmarks sit near three corners; the
shorter arc between two of them is one side of the square.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from mdebris.coastal.build import (
    build_land,
    build_segments,
    build_wetlands,
    coastline_query,
    cut_segment,
    wetland_query,
)
from mdebris.coastal.islands import parse_island

UTM = "EPSG:32619"
_BACK = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform
_FWD = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform
X0, Y0 = 500_000.0, 1_340_000.0


def _node(x, y):
    lon, lat = _BACK(x, y)
    return {"lon": lon, "lat": lat}


def _way(wid, pts, **tags):
    return {"type": "way", "id": wid, "tags": tags, "geometry": [_node(x, y) for x, y in pts]}


@pytest.fixture
def coast():
    s = 4000.0
    # Counter-clockwise square in two pieces that share their end nodes.
    first = [(X0, Y0), (X0 + s, Y0), (X0 + s, Y0 + s)]
    second = [(X0 + s, Y0 + s), (X0, Y0 + s), (X0, Y0)]
    islet = [(X0 - 600, Y0 - 600), (X0 - 400, Y0 - 600), (X0 - 400, Y0 - 400), (X0 - 600, Y0 - 400), (X0 - 600, Y0 - 600)]  # fmt: skip
    return {
        "osm3s": {"timestamp_osm_base": "2026-10-04T00:00:00Z"},
        "elements": [
            _way(1, first, natural="coastline"),
            _way(2, second, natural="coastline"),
            _way(3, islet, natural="coastline"),
        ],
    }


def _island(land="main", rings=1):
    def mark(x, y, label):
        lon, lat = _BACK(x, y)
        return {"label": label, "lon": lon, "lat": lat}

    return parse_island(
        {
            "name": "Squareland",
            "utm_epsg": 32619,
            "osm": {"bbox": [12.0, -68.6, 12.2, -68.4], "land": land, "segment_rings": rings},
            "season": {"leeward_control": "north", "parts": [{"tiles": ["19PEP"]}]},
            # Landmarks 30 m off the coast, as a beach label or a lighthouse would be.
            "landmarks": {
                "sw": mark(X0 - 30, Y0 - 30, "South-west point"),
                "se": mark(X0 + 4030, Y0 - 30, "South-east point"),
                "ne": mark(X0 + 4030, Y0 + 4030, "North-east point"),
                "nw": mark(X0 - 30, Y0 + 4030, "North-west point"),
                "islet": mark(X0 - 500, Y0 - 650, "Islet"),
            },
            "segments": [
                {"id": "south", "name": "South coast", "from": "sw", "to": "se"},
                {"id": "east", "name": "East coast", "from": "se", "to": "ne", "tile": "19PEP"},
                {
                    "id": "north",
                    "name": "North coast",
                    "from": "ne",
                    "to": "nw",
                    "exposure": "leeward",
                },
            ],
        },
        directory=Path("squareland"),
    )


def test_queries_use_south_west_north_east():
    assert "(12.0,-68.6,12.2,-68.4)" in coastline_query((12.0, -68.6, 12.2, -68.4))
    assert 'relation["natural"="wetland"]' in wetland_query((12.0, -68.6, 12.2, -68.4))


def test_segments_are_the_shorter_arc_between_landmarks(coast):
    collection = build_segments(coast, _island())
    assert [f["id"] for f in collection["features"]] == ["south", "east", "north"]
    for feature in collection["features"]:
        line = shapely_transform(_FWD, shape(feature["geometry"]))
        assert line.length == pytest.approx(4000.0, abs=1.0)
        assert feature["properties"]["length_m"] == pytest.approx(4000, abs=1)
    east = collection["features"][1]["properties"]
    assert east["tile"] == "19PEP"
    assert east["from"] == "South-east point" and east["to"] == "North-east point"
    assert "OpenStreetMap" in collection["properties"]["source"]
    assert "ODbL" in collection["properties"]["license"]
    assert "2026-10-04" in collection["properties"]["source"]


def test_land_is_the_main_ring_or_every_ring(coast):
    main = build_land(coast, _island(land="main"))
    every = build_land(coast, _island(land="all"))
    assert len(main["features"]) == 1
    assert len(every["features"]) == 2
    assert main["features"][0]["properties"]["area_m2"] == pytest.approx(16e6, rel=1e-3)
    assert every["features"][1]["properties"]["area_m2"] == pytest.approx(4e4, rel=1e-2)


def test_a_segment_across_two_rings_is_refused(coast):
    island = _island(rings=2)
    bad = parse_island(
        {
            "name": "Squareland",
            "utm_epsg": 32619,
            "osm": {"segment_rings": 2},
            "season": {"parts": [{"tiles": ["19PEP"]}]},
            "landmarks": {
                k: {"label": v.label, "lon": v.lon, "lat": v.lat}
                for k, v in island.landmarks.items()
            },
            "segments": [{"id": "x", "name": "X", "from": "sw", "to": "islet"}],
        },
        directory=island.directory,
    )
    with pytest.raises(ValueError, match="different rings"):
        build_segments(coast, bad)


def test_cut_segment_wraps_round_the_ring_start():
    from shapely.geometry import LineString, Point

    ring = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    # From just above the origin on the west side to just right of it on the south
    # side: the short way crosses the ring's start point.
    arc = cut_segment(ring, Point(0, 2), Point(2, 0))
    assert arc.length == pytest.approx(4.0)


def test_wetlands_keep_mangrove_and_drop_seagrass():
    sq = [(X0, Y0), (X0 + 100, Y0), (X0 + 100, Y0 + 100), (X0, Y0 + 100), (X0, Y0)]
    hole = [(X0 + 40, Y0 + 40), (X0 + 60, Y0 + 40), (X0 + 60, Y0 + 60), (X0 + 40, Y0 + 60), (X0 + 40, Y0 + 40)]  # fmt: skip
    relation = {
        "type": "relation",
        "id": 9,
        "tags": {"natural": "wetland", "wetland": "mangrove", "type": "multipolygon"},
        "members": [
            # The outer ring arrives in two open ways that have to be joined.
            {"type": "way", "role": "outer", "geometry": [_node(*p) for p in sq[:3]]},
            {"type": "way", "role": "outer", "geometry": [_node(*p) for p in sq[2:]]},
            {"type": "way", "role": "inner", "geometry": [_node(*p) for p in hole]},
        ],
    }
    osm = {
        "elements": [
            _way(1, sq, natural="wetland", wetland="mangrove"),
            _way(2, sq, natural="wetland", wetland="seagrass"),
            _way(3, sq, natural="wetland"),
            _way(4, sq[:3], natural="wetland", wetland="mangrove"),  # not closed
            relation,
        ]
    }
    collection = build_wetlands(osm, _island())
    ids = [f["id"] for f in collection["features"]]
    assert ids == ["way/1", "relation/9"]
    areas = [shapely_transform(_FWD, shape(f["geometry"])).area for f in collection["features"]]
    assert areas[0] == pytest.approx(10_000, rel=1e-3)
    assert areas[1] == pytest.approx(10_000 - 400, rel=1e-3)
    assert "seagrass" in collection["properties"]["method"]
