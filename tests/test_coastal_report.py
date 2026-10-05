"""The season report renders from rows alone and says what the masks did.

The report is the public face of a season run, so the test checks the words that carry
meaning: which mask was applied, the leeward control, and that a second-orbit run says
it is not comparable.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mdebris.coastal.islands import load_island
from mdebris.coastal.report import season_markdown
from mdebris.coastal.runner import OperatingPoints, SeasonOutputs, pass_table, segment_pixels
from mdebris.coastal.season import summarize_history


def _rows():
    base = {
        "scene_id": "S2A_X",
        "platform": "Sentinel-2A",
        "tile_cloud_pct": "10",
        "zone_cloud_fraction": "0.1",
        "scl_cloud_fraction": "0.1",
        "not_water_fraction": "0.05",
        "detection_count": "0",
        "affected_front_m": "0",
        "sargassum_pixels": "0",
        "water_side_pixels": "1000",
        "mangrove_masked_pixels": "0",
    }
    rows = []
    for seg, name in (("lac_bay", "Lac Bay"), ("kralendijk", "Kralendijk waterfront")):
        for day, obs in (
            ("2025-03-12", "observed"),
            ("2025-03-17", "partial"),
            ("2025-06-17", "observed"),
        ):
            rows.append(
                base
                | {
                    "segment_id": seg,
                    "name": name,
                    "observed_on": day,
                    "observability": obs,
                    "scene_id": f"S2A_{day}",
                    "mangrove_masked_pixels": "8752" if seg == "lac_bay" else "0",
                }
            )
    return rows


def _render(part_key=None, mangrove_mask=True, island_key="bonaire"):
    island = load_island(island_key)
    part = island.part(part_key)
    segments = [
        SimpleNamespace(segment_id=s.segment_id, name=s.name, properties={"exposure": s.exposure})
        for s in island.segments
    ]
    rows = _rows()
    inputs = SimpleNamespace(
        island=island,
        part=part,
        start="2025-01-01",
        end="2025-08-31",
        command="python scripts/run_island_season.py --island bonaire",
        segments_path=Path("assets/islands/bonaire/segments.geojson"),
        mangroves_path=Path("assets/islands/bonaire/mangroves.geojson"),
        surf_zone_m=500.0,
        extra_notes=["An extra note."],
    )
    return season_markdown(
        summarize_history(rows),
        pass_table(rows),
        OperatingPoints(low=0.056, high=0.9, calibrated=True, source="test"),
        inputs=inputs,
        segments=segments,
        per_pixel={
            "lac_bay": {
                "flagged_pixels": 3,
                "stationary_pixels": 0,
                "persistent_vegetation_pixels": 10,
                "flagged_near_vegetation": 2,
            }
        },
        segment_pixels=segment_pixels(rows),
        dropped={"S2B_DUP": "second granule of the same datatake"},
        failures=[],
        outputs=SeasonOutputs.for_prefix("bonaire", docs=Path("docs"), assets=Path("assets")),
        mangrove_mask=mangrove_mask,
    )


def test_report_names_the_mask_the_control_and_the_clear_gap():
    text = _render()
    assert text.startswith("# Bonaire: one season of Sentinel-2 passes")
    assert "mapped mangrove and other vegetated" in text
    assert "`assets/islands/bonaire/mangroves.geojson`" in text
    assert "Kralendijk waterfront is a leeward control" in text
    assert "97 days (12 Mar to 17 Jun)" in text
    assert "| Lac Bay | 1,000 | 8,752 | 10 | 3 | 2 |" in text
    assert "## Granules not counted as passes" in text
    assert "An extra note." in text
    assert chr(0x2014) not in text and chr(0x2013) not in text  # no em or en dashes


def test_report_says_when_the_mask_was_off():
    assert "The mangrove mask was off for this run" in _render(mangrove_mask=False)


def test_second_orbit_report_says_it_is_not_comparable():
    text = _render(part_key="r139", island_key="sint_maarten")
    assert text.startswith("# Sint Maarten, windward, second orbit R139:")
    assert "not mixed into tables that compare islands" in text
