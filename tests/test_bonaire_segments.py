"""The Bonaire segment file ships in ``assets/islands/bonaire/`` and has to be usable offline.

These tests pin what the season runner relies on: every segment loads, is named,
sits on Bonaire, is a plausible stretch of coast rather than a stray fragment, and
exactly one of them is the leeward control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyproj import Transformer
from shapely.ops import transform as shapely_transform

from mdebris.coastal import load_segments

ASSET = Path(__file__).resolve().parents[1] / "assets" / "islands" / "bonaire" / "segments.geojson"

# Bonaire and its waters, generously. Klein Bonaire sits inside this box too, which
# is fine: the point is to catch a coordinate typo, not to draw a border.
WEST, SOUTH, EAST, NORTH = -68.45, 12.00, -68.17, 12.33
_TO_UTM19N = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform


@pytest.fixture(scope="module")
def segments():
    return load_segments(ASSET)


def test_file_carries_its_osm_attribution():
    blob = json.loads(ASSET.read_text(encoding="utf-8"))
    assert "OpenStreetMap" in json.dumps(blob["properties"])
    assert "ODbL" in json.dumps(blob["properties"])


def test_every_segment_is_named_and_on_bonaire(segments):
    assert len(segments) >= 7
    for seg in segments:
        assert seg.name, seg.segment_id
        minx, miny, maxx, maxy = seg.geometry.bounds
        assert WEST <= minx <= maxx <= EAST, seg.segment_id
        assert SOUTH <= miny <= maxy <= NORTH, seg.segment_id


def test_segments_are_coast_length_not_fragments(segments):
    for seg in segments:
        metres = shapely_transform(_TO_UTM19N, seg.geometry).length
        assert 800.0 <= metres <= 40_000.0, (seg.segment_id, metres)


def test_exactly_one_leeward_control(segments):
    exposure = [seg.properties.get("exposure") for seg in segments]
    assert exposure.count("leeward") == 1
    assert all(e in {"windward", "leeward"} for e in exposure)


def test_lac_bay_is_a_segment(segments):
    assert any("Lac" in seg.name for seg in segments)


ISLAND = Path(__file__).resolve().parents[1] / "assets" / "islands" / "bonaire" / "island.geojson"


def test_island_polygon_holds_rincon_and_not_the_sea():
    from shapely.geometry import Point, shape

    blob = json.loads(ISLAND.read_text(encoding="utf-8"))
    island = shape(blob["features"][0]["geometry"])
    assert island.geom_type == "Polygon"
    assert island.contains(Point(-68.3200, 12.2400))  # Rincon, inland
    assert not island.contains(Point(-68.1500, 12.2000))  # open sea east of Spelonk
    assert "OpenStreetMap" in json.dumps(blob["properties"])
