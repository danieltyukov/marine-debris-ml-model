"""scripts/compare_islands.py and scripts/eval_mangrove_mask.py, offline.

The comparison is built from season outputs only, so a missing run must be named
rather than silently dropped, and the markers in docs/islands.md must be respected.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def compare():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("compare_islands")
    finally:
        sys.path.remove(str(SCRIPTS))


def _write_season(docs: Path, prefix: str, segments: list[str]) -> None:
    rows = []
    for day, obs in (
        ("2025-03-12", "observed"),
        ("2025-03-17", "blind"),
        ("2025-06-17", "observed"),
    ):
        for seg in segments:
            rows.append(
                {
                    "segment_id": seg,
                    "name": seg,
                    "observed_on": day,
                    "observability": obs,
                    "detection_count": "0",
                    "scene_id": f"S_{day}",
                }
            )
    with (docs / f"{prefix}_season.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_pixel = {
        s: {"flagged_pixels": 2, "stationary_pixels": 0, "flagged_near_vegetation": 1}
        for s in segments
    }
    (docs / f"{prefix}_season.json").write_text(
        json.dumps({"stationary": per_pixel}), encoding="utf-8"
    )


def test_missing_runs_are_named(compare, tmp_path):
    lines, records = compare.build(tmp_path)
    text = "\n".join(lines)
    assert "Not run yet:" in text
    assert "Bonaire (bonaire)" in text and "Curaçao (curacao_west)" in text
    assert records == {"comparable": [], "other": []}


def test_an_island_run_is_tabulated_and_written_between_markers(compare, tmp_path, monkeypatch):
    from mdebris.coastal.islands import load_island

    segments = [s.segment_id for s in load_island("bonaire").segments]
    _write_season(tmp_path, "bonaire", segments)
    lines, records = compare.build(tmp_path)
    text = "\n".join(lines)
    assert "| Bonaire | lac_bay | windward | 67% (2/3) | 97 | 97 days (12 Mar to 17 Jun) |" in text
    assert "- Bonaire: on 3 pass dates, every windward segment was usable on 2" in text
    assert len(records["comparable"]) == len(segments)

    page = tmp_path / "islands.md"
    page.write_text(f"intro\n{compare.START}\nold\n{compare.END}\noutro\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_islands.py", "--docs-dir", str(tmp_path), "--write"])
    compare.main()
    body = page.read_text(encoding="utf-8")
    assert body.startswith("intro\n") and body.endswith("outro\n")
    assert "old" not in body and "| Bonaire | lac_bay |" in body
    assert json.loads((tmp_path / "islands.json").read_text(encoding="utf-8"))["comparable"]


def test_mask_coverage_counts_flags_under_the_mask(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        module = importlib.import_module("eval_mangrove_mask")
    finally:
        sys.path.remove(str(SCRIPTS))
    # A 40 x 40 grid of 10 m pixels centred on a mapped Lac Bay mangrove polygon.
    from pyproj import Transformer

    from mdebris.coastal.islands import load_island
    from mdebris.coastal.masks import load_polygons
    from mdebris.coastal.runner import Persistence

    island = load_island("bonaire")
    poly = max(load_polygons(island.mangroves_path), key=lambda p: p.area)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform(
        poly.representative_point().x, poly.representative_point().y
    )
    from affine import Affine

    shape = (40, 40)
    hits = np.zeros(shape, np.int16)
    hits[20, 20] = 3  # under the polygon
    hits[0, 0] = 1  # 200 m away, probably not
    grid = Persistence(
        hit_total=hits,
        usable_total=np.full(shape, 10, np.int16),
        zones={"lac_bay": np.ones(shape, bool)},
        transform=Affine(10.0, 0.0, x - 205.0, 0.0, -10.0, y + 205.0),
        crs="EPSG:32619",
        n_passes=10,
    )
    path = tmp_path / "p.npz"
    grid.save(path)
    result = module.coverage("bonaire", path)["lac_bay"]
    assert result["flagged_pixels"] == 2
    assert result["flagged_pixels_covered"] >= 1
    assert result["flag_events_covered"] >= 3
    assert result["covered_by_mask"] > 0
