"""The season runner, offline: granule choice, one pass end to end, persistence.

``process_pass`` is run on a synthetic scene written to local GeoTIFFs: a strip of
land, a block of mangrove canopy on the water side of the coastline (the Lac Bay
situation), a small bright raft in open water, and a stub classifier that calls
anything with a bright near-infrared sargassum. With the mangrove mask on, only the
raft is flagged and the canopy is out of the water side; with it off, the canopy is
flagged too, which is what the 2.0 runner did at Lac Bay.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, box, mapping

from mdebris.coastal import load_segments, surf_zone
from mdebris.coastal.runner import (
    BANDS,
    OperatingPoints,
    PassResult,
    Persistence,
    load_operating_points,
    pass_table,
    process_pass,
    segment_pixels,
    select_granules,
)
from mdebris.types import GeoBBox, SceneRef

pytest.importorskip("rasterio")
pytest.importorskip("scipy")

CRS = "EPSG:32619"
X0, Y0 = 500_000.0, 1_340_600.0  # top-left corner, UTM 19N
PIX = 10.0
N = 60  # 600 m square


def _to_lonlat(geom):
    from pyproj import Transformer
    from shapely.ops import transform

    return transform(Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform, geom)


def _rect(c0, r0, c1, r1):
    """Pixel rectangle [c0, c1) x [r0, r1) as a lon/lat polygon."""
    return _to_lonlat(box(X0 + c0 * PIX, Y0 - r1 * PIX, X0 + c1 * PIX, Y0 - r0 * PIX))


class StubClassifier:
    """Calls a pixel dense sargassum when its B08 reflectance is above 0.1."""

    class _Model:
        classes_ = np.array(["Marine Water", "Dense Sargassum", "Sparse Sargassum"])

    _model = _Model()

    def predict_proba(self, features):
        from mdebris.models.spectral import FEATURE_BANDS

        bright = (features[:, FEATURE_BANDS.index("B08")] > 0.1).astype(float)
        return np.column_stack([1.0 - bright, bright, np.zeros_like(bright)])


@pytest.fixture
def scene(tmp_path, monkeypatch):
    import rasterio

    water = dict.fromkeys(BANDS, 0.02) | {"B08": 0.01, "B8A": 0.01}
    canopy = dict.fromkeys(BANDS, 0.03) | {
        "B04": 0.02,
        "B05": 0.10,
        "B06": 0.30,
        "B07": 0.35,
        "B08": 0.35,
        "B8A": 0.36,
    }
    raft = water | {"B05": 0.10, "B06": 0.15, "B07": 0.15, "B08": 0.15, "B8A": 0.14}
    land = dict.fromkeys(BANDS, 0.3)
    refl = {b: np.full((N, N), water[b], dtype=np.float32) for b in BANDS}
    for b in BANDS:
        refl[b][:, :15] = land[b]  # land: columns 0 to 14
        refl[b][20:30, 20:30] = canopy[b]  # mangrove canopy on the water side
        refl[b][40:44, 40:44] = raft[b]  # a raft in open water
    transform = Affine(PIX, 0.0, X0, 0.0, -PIX, Y0)
    hrefs = {}
    for name in [*BANDS, "SCL"]:
        path = tmp_path / f"{name}.tif"
        data = (
            np.full((N, N), 6, dtype=np.uint16)  # SCL water everywhere
            if name == "SCL"
            else np.round((refl[name] + 0.1) * 10_000).astype(np.uint16)
        )
        with rasterio.open(
            path, "w", driver="GTiff", width=N, height=N, count=1, dtype="uint16",
            crs=CRS, transform=transform,
        ) as dst:  # fmt: skip
            dst.write(data, 1)
        hrefs[name] = str(path)
    monkeypatch.setattr("mdebris.data.get_scene_assets", lambda scene_id, bands: hrefs)

    coast = _to_lonlat(LineString([(X0 + 150, Y0 - 5), (X0 + 150, Y0 - N * PIX + 5)]))
    seg_path = tmp_path / "segments.geojson"
    seg_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "coast",
                        "properties": {"name": "Test coast", "exposure": "windward"},
                        "geometry": mapping(coast),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    segments = load_segments(seg_path)
    inset = _to_lonlat(box(X0 + 5, Y0 - N * PIX + 5, X0 + N * PIX - 5, Y0 - 5)).bounds
    return {
        "aoi": GeoBBox(west=inset[0], south=inset[1], east=inset[2], north=inset[3]),
        "segments": segments,
        "zones": {s.segment_id: surf_zone(s, 500.0) for s in segments},
        "land": [_rect(-5, -5, 15, N + 5)],
        "mangrove": [_rect(20, 20, 30, 30)],
    }


def _run(scene, mangroves):
    return process_pass(
        SceneRef(scene_id="S2A_MSIL2A_20250116T150721_R082_T19PEP_X", datetime="2025-01-16"),
        aoi=scene["aoi"],
        segments=scene["segments"],
        zones=scene["zones"],
        land=scene["land"],
        mangroves=mangroves,
        clf=StubClassifier(),
        calibrator=None,
        points=OperatingPoints(low=0.056, high=0.9, calibrated=False, source="test"),
        surf_zone_m=500.0,
        land_buffer_m=15.0,
        mangrove_buffer_m=20.0,
    )


def test_mangrove_mask_keeps_the_canopy_out_and_the_raft_in(scene):
    (row,) = _run(scene, scene["mangrove"]).rows
    assert row["observability"] == "observed"
    assert row["sargassum_pixels"] == 16  # the 4 x 4 raft, nothing else
    assert row["detection_count"] == 1
    assert row["mangrove_masked_pixels"] >= 100  # the canopy and its 20 m strip
    # A floating mat looks like leaves on the day it is there; only the persistence
    # check across passes tells it from a canopy.
    assert row["vegetated_pixels"] == 16
    assert row["vegetated_sargassum_pixels"] == 16


def test_without_the_mask_the_canopy_is_flagged_as_in_version_2_0(scene):
    masked = _run(scene, scene["mangrove"])
    (row,) = (unmasked := _run(scene, None)).rows
    assert row["mangrove_masked_pixels"] == 0
    assert row["sargassum_pixels"] == 16 + 100  # raft plus the whole canopy block
    assert row["vegetated_pixels"] == 100 + 16
    assert row["vegetated_sargassum_pixels"] == 100 + 16
    assert (
        row["water_side_pixels"] - masked.rows[0]["water_side_pixels"]
        == (masked.rows[0]["mangrove_masked_pixels"])
    )
    assert not masked.hit_mask[20:30, 20:30].any()
    assert unmasked.hit_mask[20:30, 20:30].all()
    assert masked.masked[20:30, 20:30].all() and not unmasked.masked.any()


def test_land_and_its_strip_are_never_classified(scene):
    result = _run(scene, None)
    assert not result.usable_mask[:, :16].any()
    assert result.usable_mask[:, 17:].any()


def _scene(sid, platform="Sentinel-2A", when="2025-01-16T15:07:21Z"):
    return SceneRef(scene_id=sid, datetime=when, platform=platform)


def test_select_granules_keeps_one_full_cover_granule_per_datatake():
    zone = box(0, 0, 1, 1)
    ok = lambda sid: "_T19PEP_" in sid and "_R082_" in sid  # noqa: E731
    candidates = [
        (_scene("A_R082_T19PEP_1"), box(-1, -1, 2, 2)),
        # same datatake, a second granule covering a sliver: dropped as duplicate
        (_scene("A_R082_T19PEP_0"), box(0.9, 0.9, 2, 2)),
        # another datatake whose only granule covers half the zone: dropped
        (_scene("B_R082_T19PEP_1", when="2025-01-21T15:07:21Z"), box(0.5, -1, 2, 2)),
        # wrong orbit and wrong tile never enter
        (_scene("C_R125_T19PEP_1", when="2025-01-26T15:07:21Z"), box(-1, -1, 2, 2)),
        (_scene("D_R082_T19PDP_1", when="2025-01-31T15:07:21Z"), box(-1, -1, 2, 2)),
        # earlier in time, listed later: comes back first
        (_scene("E_R082_T19PEP_1", when="2025-01-11T15:07:21Z"), box(-1, -1, 2, 2)),
    ]
    kept, dropped = select_granules(candidates, tiles_ok=ok, zones=[zone], min_cover=0.99)
    assert [s.scene_id for s in kept] == ["E_R082_T19PEP_1", "A_R082_T19PEP_1"]
    assert set(dropped) == {"A_R082_T19PEP_0", "B_R082_T19PEP_1"}
    assert "second granule" in dropped["A_R082_T19PEP_0"]
    assert "50.0%" in dropped["B_R082_T19PEP_1"]


def test_select_granules_breaks_a_tie_towards_the_later_processing():
    zone = box(0, 0, 1, 1)
    full = box(-1, -1, 2, 2)
    kept, dropped = select_granules(
        [(_scene("A_T19PEP_20250707"), full), (_scene("A_T19PEP_20250622"), full)],
        tiles_ok=lambda sid: True,
        zones=[zone],
        min_cover=0.99,
    )
    assert [s.scene_id for s in kept] == ["A_T19PEP_20250707"]
    assert list(dropped) == ["A_T19PEP_20250622"]


def _result(hits, usable, leafy, zone):
    return PassResult(
        scene=_scene("x"),
        rows=[],
        zone_cloud_fraction=0.0,
        hits=int(hits.sum()),
        hit_mask=hits,
        usable_mask=usable,
        vegetated_mask=leafy,
        masked=np.zeros_like(hits),
        zones={"z": zone},
        transform=Affine(10.0, 0.0, X0, 0.0, -10.0, Y0),
        crs=CRS,
    )


def test_persistence_counts_stationary_and_vegetation(tmp_path):
    shape = (6, 6)
    zone = np.ones(shape, dtype=bool)
    usable = np.ones(shape, dtype=bool)
    leafy = np.zeros(shape, dtype=bool)
    leafy[0, 0] = True  # a canopy pixel the map missed
    first = np.zeros(shape, dtype=bool)
    first[0, 1] = True  # next to the canopy, flagged on every pass
    persistence = None
    for i in range(10):
        hits = first.copy()
        if i == 3:
            hits[5, 5] = True  # one transient flag in open water
        result = _result(hits, usable, leafy, zone)
        persistence = persistence or Persistence.start(result)
        persistence.add(result)
    counts = persistence.per_segment()["z"]
    assert counts == {
        "flagged_pixels": 2,
        "stationary_pixels": 1,
        "persistent_vegetation_pixels": 1,
        "flagged_near_vegetation": 1,
    }
    path = tmp_path / "p.npz"
    persistence.save(path)
    again = Persistence.load(path)
    assert again.n_passes == 10
    assert again.per_segment() == persistence.per_segment()


def test_persistence_from_a_2_0_file_has_no_vegetation_counts(tmp_path):
    shape = (3, 3)
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        hit_total=np.zeros(shape, np.int16),
        usable_total=np.full(shape, 12, np.int16),
        n_passes=12,
        transform=np.asarray([10.0, 0.0, X0, 0.0, -10.0, Y0]),
        crs=CRS,
        zone_ids=np.asarray(["z"]),
        zones=np.ones((1, *shape), dtype=bool),
    )
    old = Persistence.load(path)
    assert old.vegetated_total is None and old.masked is None
    assert old.per_segment()["z"]["persistent_vegetation_pixels"] == 0


def test_pass_table_and_segment_pixels_roll_up_rows():
    rows = [
        {
            "scene_id": "s1",
            "observed_on": "2025-01-16",
            "observability": "observed",
            "name": "A",
            "segment_id": "a",
            "detection_count": "1",
            "sargassum_pixels": "5",
            "water_side_pixels": "100",
            "mangrove_masked_pixels": "7",
        },
        {
            "scene_id": "s1",
            "observed_on": "2025-01-16",
            "observability": "blind",
            "name": "B",
            "segment_id": "b",
            "detection_count": "2",
            "sargassum_pixels": "3",
            "water_side_pixels": "50",
        },
    ]
    table = pass_table(rows)
    assert table["s1"]["observed"] == 1 and table["s1"]["blind"] == 1
    assert table["s1"]["sargassum_pixels"] == 8
    assert table["s1"]["affected"] == ["A"]  # a detection under a blind verdict is not a look
    assert segment_pixels(rows) == {
        "a": {"water_side_pixels": 100, "mangrove_masked_pixels": 7},
        "b": {"water_side_pixels": 50, "mangrove_masked_pixels": 0},
    }


def test_operating_points_come_from_the_calibration_report():
    report = Path(__file__).resolve().parents[1] / "docs" / "calibration.json"
    points = load_operating_points(report, calibrated=True)
    assert points.high == pytest.approx(0.9)
    assert 0.0 < points.low < points.high
    raw = load_operating_points(report, calibrated=False)
    assert not raw.calibrated and raw.low < raw.high
