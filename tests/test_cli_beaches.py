"""Tests for the ``mdebris beaches`` command.

The command is the only place a customer-facing number is rendered, so these
check the wiring and the presentation rather than the arithmetic, which
``test_coastal_segments`` already pins down.
"""

from __future__ import annotations

import csv
import json

import pytest
from pyproj import Transformer
from shapely.geometry import LineString, box, mapping
from shapely.ops import transform as shapely_transform
from typer.testing import CliRunner

from mdebris.cli import app

runner = CliRunner()

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True).transform
_TO_WGS = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True).transform
ORIGIN_E, ORIGIN_N = _TO_UTM(-86.75, 21.10)


def _wgs(geom):
    shifted = shapely_transform(lambda x, y, z=None: (x + ORIGIN_E, y + ORIGIN_N), geom)
    return shapely_transform(_TO_WGS, shifted)


@pytest.fixture
def segments_file(tmp_path):
    features = [
        {
            "type": "Feature",
            "id": "delfines",
            "properties": {"name": "Playa Delfines"},
            "geometry": mapping(_wgs(LineString([(0, 0), (0, 2000)]))),
        },
        {
            "type": "Feature",
            "id": "nizuc",
            "properties": {"name": "Punta Nizuc"},
            "geometry": mapping(_wgs(LineString([(0, 5000), (0, 7000)]))),
        },
    ]
    path = tmp_path / "segments.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    return path


@pytest.fixture
def detections_file(tmp_path):
    """One sargassum patch off Playa Delfines and a ship off Punta Nizuc."""
    features = [
        {
            "type": "Feature",
            "geometry": mapping(_wgs(box(100, 400, 400, 1200))),
            "properties": {
                "score": 0.91,
                "label": "sargassum",
                "scene_id": "S2A_TEST",
                "datetime": "2026-07-24T16:20:31Z",
            },
        },
        {
            "type": "Feature",
            "geometry": mapping(_wgs(box(100, 5400, 200, 5600))),
            "properties": {"score": 0.88, "label": "ship", "scene_id": "S2A_TEST"},
        },
    ]
    path = tmp_path / "detections.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    return path


def test_beaches_names_the_affected_segment(segments_file, detections_file):
    result = runner.invoke(app, ["beaches", str(detections_file), "-s", str(segments_file)])
    assert result.exit_code == 0, result.output
    assert "Playa Delfines" in result.output
    assert "Punta Nizuc" in result.output
    assert "2026-07-24" in result.output


def test_beaches_ignores_a_ship_by_default(segments_file, detections_file, tmp_path):
    """A vessel offshore is not a clearing job, so it must not raise a beach's number."""
    csv_path = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        ["beaches", str(detections_file), "-s", str(segments_file), "--csv", str(csv_path)],
    )
    assert result.exit_code == 0, result.output
    rows = {r["segment_id"]: r for r in csv.DictReader(csv_path.read_text().splitlines())}
    assert int(rows["delfines"]["detection_count"]) == 1
    assert int(rows["nizuc"]["detection_count"]) == 0


def test_all_labels_counts_the_ship_too(segments_file, detections_file, tmp_path):
    csv_path = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        [
            "beaches",
            str(detections_file),
            "-s",
            str(segments_file),
            "--all-labels",
            "--csv",
            str(csv_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = {r["segment_id"]: r for r in csv.DictReader(csv_path.read_text().splitlines())}
    assert int(rows["nizuc"]["detection_count"]) == 1


def test_a_blind_segment_is_flagged_and_its_numbers_withheld(
    segments_file, detections_file, tmp_path
):
    """Printing 0.00% next to a 94%-cloud segment is the mistake this guards against."""
    clouds = tmp_path / "clouds.json"
    clouds.write_text(json.dumps({"nizuc": 0.94}), encoding="utf-8")

    result = runner.invoke(
        app,
        ["beaches", str(detections_file), "-s", str(segments_file), "--clouds", str(clouds)],
    )
    assert result.exit_code == 0, result.output
    assert "BLIND" in result.output
    assert "1 of 2 segments were not observed" in result.output


def test_beaches_writes_geojson_and_history(segments_file, detections_file, tmp_path):
    out = tmp_path / "nested" / "segments_out.geojson"
    history = tmp_path / "history.csv"
    for _ in range(2):
        result = runner.invoke(
            app,
            [
                "beaches",
                str(detections_file),
                "-s",
                str(segments_file),
                "-o",
                str(out),
                "--history",
                str(history),
            ],
        )
        assert result.exit_code == 0, result.output

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2
    assert data["properties"]["surf_zone_m"] == 500.0

    rows = list(csv.DictReader(history.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 4, "the dated record must accumulate across runs, not overwrite"


def test_surf_zone_option_changes_what_counts(segments_file, detections_file, tmp_path):
    """A 50 m zone cannot reach a patch that starts 100 m offshore."""
    csv_path = tmp_path / "narrow.csv"
    result = runner.invoke(
        app,
        [
            "beaches",
            str(detections_file),
            "-s",
            str(segments_file),
            "--surf-zone-m",
            "50",
            "--csv",
            str(csv_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = {r["segment_id"]: r for r in csv.DictReader(csv_path.read_text().splitlines())}
    assert float(rows["delfines"]["detected_area_m2"]) == 0.0


def test_beaches_rejects_a_missing_segments_file(detections_file, tmp_path):
    result = runner.invoke(
        app, ["beaches", str(detections_file), "-s", str(tmp_path / "nope.geojson")]
    )
    assert result.exit_code != 0
