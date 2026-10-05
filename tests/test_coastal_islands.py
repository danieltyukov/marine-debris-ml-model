"""Island configuration: the shipped islands load, and a bad island.toml fails loudly.

Everything the season runner used to hard-code for Bonaire now comes from
``assets/islands/<key>/island.toml``, so these tests pin what the runner relies on:
every configured segment exists in the segment file, the leeward control is one of
them, and each part reads a tile and orbit that its granules can be matched against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import LineString, box

from mdebris.coastal import load_segments
from mdebris.coastal.islands import (
    DEFAULT_ISLANDS_DIR,
    SeasonPart,
    derive_aoi,
    list_islands,
    load_island,
    parse_island,
)

SHIPPED = ("aruba", "bonaire", "curacao", "sint_maarten")


def test_the_islands_folder_is_found_from_the_package():
    assert DEFAULT_ISLANDS_DIR.name == "islands"
    assert set(SHIPPED) <= set(list_islands())


@pytest.mark.parametrize("key", SHIPPED)
def test_shipped_island_matches_its_segment_file(key):
    island = load_island(key)
    assert island.key == key
    segments = load_segments(island.segments_path)
    ids = [s.segment_id for s in segments]
    assert ids == [spec.segment_id for spec in island.segments]
    assert island.leeward_control in ids
    leeward = [s for s in segments if s.properties["exposure"] == "leeward"]
    assert [s.segment_id for s in leeward] == [island.leeward_control]
    for part in island.parts:
        assert set(part.segment_ids) <= set(ids)
        assert part.tiles and part.orbits
    assert island.island_path.is_file()
    assert island.mangroves_path.is_file()


@pytest.mark.parametrize("key", SHIPPED)
def test_shipped_files_carry_the_osm_attribution(key):
    island = load_island(key)
    for path in (island.segments_path, island.island_path, island.mangroves_path):
        props = json.dumps(json.loads(path.read_text(encoding="utf-8"))["properties"])
        assert "OpenStreetMap" in props and "ODbL" in props, path


def test_curacao_parts_split_the_coast_between_two_tiles():
    island = load_island("curacao")
    west, east = island.part("west"), island.part("east")
    assert (west.tiles, east.tiles) == (("19PDP",), ("19PEP",))
    assert west.orbits == east.orbits == ("R082",)
    assert not set(west.segment_ids) & set(east.segment_ids)
    assert set(west.segment_ids) | set(east.segment_ids) == {s.segment_id for s in island.segments}
    assert west.prefix(island.key) == "curacao_west"
    assert west.zoom is not None and west.zoom.title == "Watamula and Shete Boka"


def test_sint_maarten_second_orbit_is_not_comparable():
    island = load_island("sint_maarten")
    main, second = island.part(), island.part("r139")
    assert main.comparable and not second.comparable
    assert main.prefix(island.key) == "sint_maarten"
    assert second.prefix(island.key) == "sint_maarten_r139"
    assert island.leeward_control not in second.segment_ids
    with pytest.raises(KeyError, match="no part"):
        island.part("r999")


def test_part_accepts_only_its_tiles_and_orbits():
    part = SeasonPart(key="", label="", tiles=("19PEP",), orbits=("R082",))
    assert part.accepts("S2A_MSIL2A_20250116T150721_R082_T19PEP_20250116T180950")
    assert not part.accepts("S2A_MSIL2A_20250116T150721_R082_T19PDP_20250116T180950")
    assert not part.accepts("S2A_MSIL2A_20250116T150721_R125_T19PEP_20250116T180950")
    any_orbit = SeasonPart(key="", label="", tiles=("19PEP",))
    assert any_orbit.accepts("S2A_MSIL2A_20250116T150721_R125_T19PEP_20250116T180950")


def _minimal() -> dict:
    return {
        "name": "Testa",
        "utm_epsg": 32619,
        "season": {"leeward_control": "b_c", "parts": [{"tiles": ["T19PEP"], "orbits": [82]}]},
        "landmarks": {
            "a": {"label": "A", "lon": -68.30, "lat": 12.10},
            "b": {"label": "B", "lon": -68.20, "lat": 12.10},
            "c": {"label": "C", "lon": -68.20, "lat": 12.20},
        },
        "segments": [
            {"id": "a_b", "name": "A to B", "from": "a", "to": "b"},
            {"id": "b_c", "name": "B to C", "from": "b", "to": "c", "exposure": "leeward"},
        ],
    }


def test_parse_normalises_tiles_and_orbits(tmp_path):
    island = parse_island(_minimal(), tmp_path / "testa")
    part = island.part()
    assert part.tiles == ("19PEP",)
    assert part.orbits == ("R082",)
    assert island.key == "testa"
    assert island.segments[0].exposure == "windward"
    assert island.land == "main" and island.segment_rings == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b.pop("name"), "no `name`"),
        (lambda b: b["season"].update(parts=[]), r"no \[\[season.parts\]\]"),
        (lambda b: b["season"]["parts"][0].update(tiles=[]), "at least one Sentinel-2 tile"),
        (lambda b: b["season"]["parts"][0].update(min_cover=1.5), "min_cover"),
        (lambda b: b["segments"][0].update(to="nowhere"), "names no landmark"),
        (lambda b: b["segments"][0].update(exposure="sideways"), "exposure"),
        (lambda b: b["season"].update(leeward_control="x"), "leeward_control"),
        (lambda b: b.update(osm={"land": "some"}), "osm.land"),
        (lambda b: b["season"]["parts"][0].update(segments=["zzz"]), "unknown segments"),
        (lambda b: b["season"]["parts"][0].update(aoi=[1, 2, 0, 3]), "aoi"),
        (
            lambda b: b["season"]["parts"].append({"tiles": ["19PEP"]}),
            "share a key",
        ),
        (
            lambda b: b.update(figures={"zoom": {"title": "z", "box": [0, 0, 1, 1]}}),
            "window",
        ),
    ],
)
def test_parse_rejects_a_bad_configuration(tmp_path, mutate, message):
    blob = _minimal()
    mutate(blob)
    with pytest.raises(ValueError, match=message):
        parse_island(blob, tmp_path / "testa")


def test_load_island_by_folder_and_by_file(tmp_path):
    folder = tmp_path / "testa"
    folder.mkdir()
    (folder / "island.toml").write_text(
        'name = "Testa"\nutm_epsg = 32619\n[[season.parts]]\ntiles = ["19PEP"]\n',
        encoding="utf-8",
    )
    assert load_island(folder).name == "Testa"
    assert load_island(folder / "island.toml").name == "Testa"
    assert load_island("testa", root=tmp_path).name == "Testa"
    assert list_islands(tmp_path) == ["testa"]
    with pytest.raises(FileNotFoundError, match="islands found: testa"):
        load_island("nowhere", root=tmp_path)


def test_derived_area_holds_the_zones_and_the_land_they_touch():
    zone = LineString([(-68.30, 12.10), (-68.20, 12.10)]).buffer(0.005)
    touching = box(-68.32, 12.04, -68.18, 12.105)
    far = box(-69.5, 12.0, -69.4, 12.1)
    aoi = derive_aoi([zone], [touching, far], margin_m=1000.0, utm_epsg=32619)
    assert aoi.west < -68.32 and aoi.east > -68.18
    assert aoi.south < 12.04 and aoi.north > 12.105
    # The island that touches no zone does not drag the area west by a degree.
    assert aoi.west > -68.4
    with pytest.raises(ValueError, match="no surf zones"):
        derive_aoi([], [touching], margin_m=0.0, utm_epsg=32619)


def test_bonaire_reads_the_area_the_previous_runner_read():
    part = load_island("bonaire").part()
    assert part.aoi is not None
    assert (part.aoi.west, part.aoi.south, part.aoi.east, part.aoi.north) == (
        -68.43,
        12.01,
        -68.18,
        12.32,
    )
    assert Path(load_island("bonaire").directory).name == "bonaire"
