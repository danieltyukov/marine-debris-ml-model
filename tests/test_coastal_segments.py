"""Tests for rolling detections up onto named beach segments.

The geometry here is chosen so the expected answers can be computed by hand. All
fixtures sit off Cancun (about 21.1 N, 86.75 W, UTM zone 16N), because that is
the coastline the module is aimed at and because it is far from any zone edge, so
a metre is a metre.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box
from shapely.ops import transform as shapely_transform

from mdebris.coastal.segments import (
    BeachSegment,
    Observability,
    SegmentObservation,
    SegmentReport,
    _utm_epsg,
    aggregate_segments,
    append_history,
    load_segments,
    segment_cloud_fractions,
    surf_zone,
)
from mdebris.types import BBox, Detection, DetectionSet, SceneRef, SurfaceClass

# Fixtures are laid out in the projected CRS and converted back to WGS84, rather
# than approximated with metres-per-degree constants. A spherical approximation
# is off by roughly half a percent against the WGS84 ellipsoid at this latitude,
# which is large enough to swamp the tolerances these tests want to assert at.
UTM16N = "EPSG:32616"
_TO_UTM = Transformer.from_crs("EPSG:4326", UTM16N, always_xy=True).transform
_TO_WGS = Transformer.from_crs(UTM16N, "EPSG:4326", always_xy=True).transform
ORIGIN_E, ORIGIN_N = _TO_UTM(-86.75, 21.10)  # off Cancun, well inside zone 16 north


def _wgs(geom):
    """Move a geometry laid out in metres from the origin into WGS84."""
    shifted = shapely_transform(lambda x, y, z=None: (x + ORIGIN_E, y + ORIGIN_N), geom)
    return shapely_transform(_TO_WGS, shifted)


def _utm(geom):
    """Inverse of :func:`_wgs`, for measuring a result back in metres."""
    projected = shapely_transform(_TO_UTM, geom)
    return shapely_transform(lambda x, y, z=None: (x - ORIGIN_E, y - ORIGIN_N), projected)


def _segment(seg_id: str, length_m: float = 2000.0, name: str = "") -> BeachSegment:
    """A north-south shoreline of the given length, running up from the origin."""
    return BeachSegment(
        segment_id=seg_id,
        geometry=_wgs(LineString([(0.0, 0.0), (0.0, length_m)])),
        name=name,
    )


def _detection(
    west_m: float,
    south_m: float,
    east_m: float,
    north_m: float,
    *,
    label: SurfaceClass = SurfaceClass.SARGASSUM,
) -> Detection:
    """A detection whose footprint is a rectangle placed in metres from the origin."""
    return Detection(
        bbox=BBox(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0),
        score=0.9,
        label=label,
        geometry=_wgs(box(west_m, south_m, east_m, north_m)),
    )


def _set(*dets: Detection, scene: SceneRef | None = None) -> DetectionSet:
    return DetectionSet(detections=list(dets), scene=scene)


# ----------------------------------------------------------- UTM selection ----


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [
        (-86.75, 21.1, "EPSG:32616"),  # Cancun, zone 16 north
        (4.35, 52.0, "EPSG:32631"),  # Delft, zone 31 north
        (-70.6, -33.4, "EPSG:32719"),  # Santiago, zone 19 south
        (-180.0, 0.0, "EPSG:32601"),  # west edge, first zone
        (179.9, 0.0, "EPSG:32660"),  # east edge, last zone
    ],
)
def test_utm_epsg_picks_the_zone_containing_the_point(lon, lat, expected):
    assert _utm_epsg(lon, lat) == expected


def test_utm_epsg_clamps_at_the_antimeridian():
    """Longitude exactly 180 would compute zone 61, which does not exist."""
    assert _utm_epsg(180.0, 0.0) == "EPSG:32660"


# ------------------------------------------------------------- BeachSegment ----


def test_segment_rejects_an_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        BeachSegment(segment_id="", geometry=LineString([(0, 0), (1, 1)]))


def test_segment_rejects_empty_geometry():
    with pytest.raises(ValueError, match="empty geometry"):
        BeachSegment(segment_id="a", geometry=Polygon())


def test_segment_label_falls_back_to_the_id():
    assert _segment("s1").label == "s1"
    assert _segment("s1", name="Playa Delfines").label == "Playa Delfines"


# ---------------------------------------------------------------- loading ----


def _write(tmp_path, payload):
    path = tmp_path / "segments.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_segments_reads_ids_and_names(tmp_path):
    path = _write(
        tmp_path,
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "delfines",
                    "properties": {"name": "Playa Delfines", "zofemat": "Benito Juarez"},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-86.75, 21.1], [-86.75, 21.12]],
                    },
                }
            ],
        },
    )
    (segment,) = load_segments(path)
    assert segment.segment_id == "delfines"
    assert segment.name == "Playa Delfines"
    assert segment.properties["zofemat"] == "Benito Juarez"


def test_load_segments_falls_back_through_id_sources(tmp_path):
    """A GIS export with no id column still loads, positionally."""
    path = _write(
        tmp_path,
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"segment_id": "from_property"},
                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 1]]},
                },
                {
                    "type": "Feature",
                    "properties": {"name": "from_name"},
                    "geometry": {"type": "LineString", "coordinates": [[1, 0], [1, 1]]},
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "LineString", "coordinates": [[2, 0], [2, 1]]},
                },
            ],
        },
    )
    assert [s.segment_id for s in load_segments(path)] == [
        "from_property",
        "from_name",
        "segment_002",
    ]


def test_load_segments_skips_null_geometry_features(tmp_path):
    path = _write(
        tmp_path,
        {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": "a", "properties": {}, "geometry": None},
                {
                    "type": "Feature",
                    "id": "b",
                    "properties": {},
                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 1]]},
                },
            ],
        },
    )
    assert [s.segment_id for s in load_segments(path)] == ["b"]


def test_load_segments_rejects_a_bare_geometry(tmp_path):
    path = _write(tmp_path, {"type": "LineString", "coordinates": [[0, 0], [0, 1]]})
    with pytest.raises(ValueError, match="FeatureCollection"):
        load_segments(path)


def test_load_segments_rejects_a_collection_with_nothing_usable(tmp_path):
    path = _write(tmp_path, {"type": "FeatureCollection", "features": []})
    with pytest.raises(ValueError, match="no features"):
        load_segments(path)


# --------------------------------------------------------------- surf zone ----


def test_surf_zone_buffers_by_the_requested_metres():
    """A 2 km line buffered by 500 m is a 3 km by 1 km stadium: 2e6 + pi*500^2 m^2."""
    measured = _utm(surf_zone(_segment("s1", 2000.0), 500.0))
    west, south, east, north = measured.bounds
    assert north - south == pytest.approx(3000.0, rel=1e-3)
    assert east - west == pytest.approx(1000.0, rel=1e-3)
    # shapely approximates the semicircular caps with segments, so the area comes
    # in a hair under the analytic stadium.
    assert measured.area == pytest.approx(2000.0 * 1000.0 + math.pi * 500.0**2, rel=1e-3)


def test_surf_zone_rejects_a_non_positive_distance():
    with pytest.raises(ValueError, match="positive distance"):
        surf_zone(_segment("s1"), 0.0)


# ------------------------------------------------------------- aggregation ----


def test_a_detection_inside_the_surf_zone_is_measured_in_square_metres():
    """A 200 m by 100 m patch 100 m offshore contributes exactly its own area."""
    report = aggregate_segments(
        _set(_detection(100, 500, 300, 600)), [_segment("s1", 2000.0)], surf_zone_m=500.0
    )
    (obs,) = report.observations
    assert obs.detected_area_m2 == pytest.approx(200 * 100, rel=1e-3)
    assert obs.detection_count == 1


def test_material_beyond_the_surf_zone_is_clipped_away():
    """Only the part of a patch inside the zone counts, not the whole patch."""
    # Spans 300 m to 700 m offshore; a 500 m zone keeps the first 200 m of it.
    report = aggregate_segments(
        _set(_detection(300, 500, 700, 600)), [_segment("s1", 2000.0)], surf_zone_m=500.0
    )
    (obs,) = report.observations
    assert obs.detected_area_m2 == pytest.approx(200 * 100, rel=1e-2)


def test_material_entirely_outside_the_zone_is_not_counted():
    report = aggregate_segments(
        _set(_detection(2000, 500, 2200, 600)), [_segment("s1", 2000.0)], surf_zone_m=500.0
    )
    (obs,) = report.observations
    assert obs.detected_area_m2 == 0.0
    assert obs.detection_count == 0
    assert obs.affected_front_m == 0.0


def test_overlapping_detections_are_unioned_not_summed():
    """Two detections covering the same water must not double-count it.

    Cross-tile merging leaves overlapping footprints, so summing areas can push
    coverage above 1.0, which would make it meaningless as a fraction.
    """
    overlapping = _set(_detection(100, 500, 300, 700), _detection(200, 500, 400, 700))
    (obs,) = aggregate_segments(overlapping, [_segment("s1", 2000.0)]).observations
    # Union is 300 m by 200 m, not the 2 x (200 x 200) a sum would give.
    assert obs.detected_area_m2 == pytest.approx(300 * 200, rel=1e-2)


def test_coverage_is_a_fraction_of_the_surf_zone_area():
    report = aggregate_segments(
        _set(_detection(100, 500, 300, 600)), [_segment("s1", 2000.0)], surf_zone_m=500.0
    )
    (obs,) = report.observations
    zone_area = 2000.0 * 1000.0 + math.pi * 500.0**2
    assert obs.coverage == pytest.approx(20_000 / zone_area, rel=1e-2)
    assert 0.0 <= obs.coverage <= 1.0


def test_affected_front_measures_shoreline_not_water():
    """A patch 400 m long offshore affects the shoreline it reaches, not its own length.

    The patch spans 500 m to 900 m north and sits 100 m offshore, so with a 500 m
    reach it covers that 400 m of front plus 500 m of taper at each end.
    """
    report = aggregate_segments(
        _set(_detection(100, 500, 300, 900)), [_segment("s1", 3000.0)], surf_zone_m=500.0
    )
    (obs,) = report.observations
    assert obs.front_length_m == pytest.approx(3000.0, rel=1e-3)
    assert 400.0 < obs.affected_front_m < 1400.0
    assert obs.affected_front_fraction == pytest.approx(obs.affected_front_m / 3000.0)


def test_a_polygon_segment_measures_its_perimeter_as_front():
    """A beach held as a parcel rather than a line still gets a coverage figure.

    ``front_length_m`` becomes the parcel perimeter, which is not a beach front,
    so ``affected_front_m`` is not meaningful for polygons. Coverage is.
    """
    parcel = BeachSegment(segment_id="parcel", geometry=_wgs(box(0, 0, 200, 400)))
    (obs,) = aggregate_segments(_set(_detection(50, 50, 150, 350)), [parcel]).observations
    assert obs.front_length_m == pytest.approx(2 * (200 + 400), rel=1e-3)
    assert obs.coverage > 0.0


def test_labels_filter_what_counts():
    """A ship in the surf zone is not a clearing job."""
    mixed = _set(
        _detection(100, 500, 300, 600, label=SurfaceClass.SARGASSUM),
        _detection(100, 700, 300, 800, label=SurfaceClass.SHIP),
    )
    default = aggregate_segments(mixed, [_segment("s1", 2000.0)])
    assert default.observations[0].detection_count == 1

    everything = aggregate_segments(mixed, [_segment("s1", 2000.0)], labels=None)
    assert everything.observations[0].detection_count == 2


def test_segments_are_returned_in_input_order():
    segments = [_segment(f"s{i}") for i in range(4)]
    report = aggregate_segments(_set(), segments)
    assert [o.segment_id for o in report.observations] == ["s0", "s1", "s2", "s3"]


def test_the_observation_date_comes_from_the_scene():
    scene = SceneRef(scene_id="S2A_TEST", datetime="2026-07-24T16:20:31Z")
    report = aggregate_segments(_set(scene=scene), [_segment("s1")])
    assert report.observed_on == "2026-07-24"
    assert report.scene_id == "S2A_TEST"
    assert report.observations[0].observed_on == "2026-07-24"


def test_an_explicit_date_overrides_the_scene():
    scene = SceneRef(scene_id="S2A_TEST", datetime="2026-07-24T16:20:31Z")
    report = aggregate_segments(_set(scene=scene), [_segment("s1")], observed_on="2026-07-25")
    assert report.observed_on == "2026-07-25"


def test_detections_without_geometry_are_ignored():
    """A detection that was never georeferenced has no place on a map."""
    ungeoreferenced = Detection(bbox=BBox(0, 0, 10, 10), score=0.9, label=SurfaceClass.SARGASSUM)
    report = aggregate_segments(_set(ungeoreferenced), [_segment("s1")])
    assert report.observations[0].detection_count == 0


def test_aggregate_rejects_an_empty_segment_list():
    with pytest.raises(ValueError, match="no segments"):
        aggregate_segments(_set(), [])


def test_aggregate_rejects_a_non_positive_surf_zone():
    with pytest.raises(ValueError, match="must be positive"):
        aggregate_segments(_set(), [_segment("s1")], surf_zone_m=-1.0)


@pytest.mark.parametrize(
    ("partial", "blind"),
    [(0.8, 0.5), (-0.1, 0.5), (0.2, 1.5)],
)
def test_aggregate_rejects_unordered_thresholds(partial, blind):
    with pytest.raises(ValueError, match="partial_above"):
        aggregate_segments(_set(), [_segment("s1")], partial_above=partial, blind_above=blind)


def test_aggregate_rejects_a_cloud_fraction_outside_zero_to_one():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aggregate_segments(_set(), [_segment("s1")], cloud_fractions={"s1": 1.4})


# ----------------------------------------------------------- observability ----


@pytest.mark.parametrize(
    ("cloud", "expected"),
    [
        (0.00, Observability.OBSERVED),
        (0.20, Observability.OBSERVED),  # boundary is exclusive
        (0.21, Observability.PARTIAL),
        (0.70, Observability.PARTIAL),
        (0.71, Observability.BLIND),
        (1.00, Observability.BLIND),
    ],
)
def test_cloud_fraction_maps_to_an_observability_state(cloud, expected):
    report = aggregate_segments(_set(), [_segment("s1")], cloud_fractions={"s1": cloud})
    assert report.observations[0].observability is expected


def test_a_segment_with_no_cloud_information_is_treated_as_clear():
    """Callers with no cloud data get the old two-state behaviour, not a crash."""
    report = aggregate_segments(_set(), [_segment("s1")], cloud_fractions={"other": 0.9})
    assert report.observations[0].observability is Observability.OBSERVED


def test_only_observed_absences_are_actionable():
    assert Observability.OBSERVED.is_actionable
    assert not Observability.PARTIAL.is_actionable
    assert not Observability.BLIND.is_actionable


def test_a_blind_segment_still_reports_what_was_detected():
    """Cloud makes an absence unreliable. It does not make a positive detection false."""
    report = aggregate_segments(
        _set(_detection(100, 500, 300, 600)),
        [_segment("s1", 2000.0)],
        cloud_fractions={"s1": 0.95},
    )
    (obs,) = report.observations
    assert obs.observability is Observability.BLIND
    assert obs.detected_area_m2 > 0.0
    assert obs.detection_count == 1


def test_blind_segments_sort_last_not_first():
    """A segment nobody saw is unknown, not low risk, so it must not head the list."""
    segments = [_segment("clean"), _segment("blind"), _segment("dirty")]
    report = aggregate_segments(
        _set(_detection(100, 500, 400, 900)),
        segments,
        cloud_fractions={"blind": 0.99},
    )
    # 'clean' and 'dirty' share geometry here, so rank by state alone.
    assert report.ranked()[-1].segment_id == "blind"
    assert all(o.observability is Observability.OBSERVED for o in report.ranked()[:2])


def test_report_partitions_blind_and_affected():
    segments = [_segment("a"), _segment("b"), _segment("c")]
    report = aggregate_segments(
        _set(_detection(100, 500, 300, 600)),
        segments,
        cloud_fractions={"c": 0.99},
    )
    assert {o.segment_id for o in report.blind} == {"c"}
    # 'c' has detections but is blind, so it is excluded from 'affected'.
    assert {o.segment_id for o in report.affected} == {"a", "b"}


# ------------------------------------------------------- cloud from raster ----


def _raster_grid(cloud_rows: int, total_rows: int = 40, cols: int = 40):
    """A 40x40 grid of 100 m pixels with the northern rows marked cloudy.

    The origin is placed 1 km west and 4 km north of the fixture origin, so the
    4 km square grid contains the segment fixtures and their surf zones.
    """
    mask = np.zeros((total_rows, cols), dtype=bool)
    mask[:cloud_rows] = True
    return mask, from_origin(ORIGIN_E - 1000.0, ORIGIN_N + 4000.0, 100.0, 100.0)


def test_segment_cloud_fraction_is_all_clear_over_a_clear_raster():
    mask, transform = _raster_grid(cloud_rows=0)
    fractions = segment_cloud_fractions(
        [_segment("s1", 2000.0)], mask, transform, src_crs="EPSG:32616"
    )
    assert fractions["s1"] == pytest.approx(0.0)


def test_segment_cloud_fraction_is_all_cloud_over_a_full_raster():
    mask, transform = _raster_grid(cloud_rows=40)
    fractions = segment_cloud_fractions(
        [_segment("s1", 2000.0)], mask, transform, src_crs="EPSG:32616"
    )
    assert fractions["s1"] == pytest.approx(1.0)


def test_segment_cloud_fraction_is_partial_over_a_half_cloudy_raster():
    mask, transform = _raster_grid(cloud_rows=20)
    fractions = segment_cloud_fractions(
        [_segment("s1", 2000.0)], mask, transform, src_crs="EPSG:32616"
    )
    assert 0.0 < fractions["s1"] < 1.0


def test_a_segment_outside_the_raster_counts_as_unobserved_not_clear():
    """Absence of coverage is the same kind of ignorance as cloud, and must read that way."""
    mask, transform = _raster_grid(cloud_rows=0)
    far_away = BeachSegment(segment_id="far", geometry=LineString([(0.0, 0.0), (0.0, 0.01)]))
    fractions = segment_cloud_fractions([far_away], mask, transform, src_crs="EPSG:32616")
    assert fractions["far"] == 1.0


def test_segment_cloud_fractions_rejects_a_non_2d_mask():
    _, transform = _raster_grid(cloud_rows=0)
    with pytest.raises(ValueError, match="must be 2-D"):
        segment_cloud_fractions(
            [_segment("s1")], np.zeros((2, 4, 4), dtype=bool), transform, src_crs="EPSG:32616"
        )


def test_raster_cloud_fractions_feed_straight_into_aggregation():
    """The two halves compose: measure cloud, then classify observability with it."""
    mask, transform = _raster_grid(cloud_rows=40)
    segments = [_segment("s1", 2000.0)]
    fractions = segment_cloud_fractions(segments, mask, transform, src_crs="EPSG:32616")
    report = aggregate_segments(_set(), segments, cloud_fractions=fractions)
    assert report.observations[0].observability is Observability.BLIND


# ----------------------------------------------------------------- outputs ----


def test_report_geojson_carries_metrics_as_properties():
    report = aggregate_segments(
        _set(_detection(100, 500, 300, 600)),
        [_segment("s1", 2000.0, name="Playa Delfines")],
        cloud_fractions={"s1": 0.1},
    )
    data = report.to_geojson()
    assert data["type"] == "FeatureCollection"
    (feature,) = data["features"]
    assert feature["id"] == "s1"
    assert feature["properties"]["name"] == "Playa Delfines"
    assert feature["properties"]["observability"] == "observed"
    assert feature["properties"]["detected_area_m2"] > 0
    assert data["properties"]["surf_zone_m"] == 500.0
    assert data["properties"]["metric_crs"] == "EPSG:32616"


def test_report_geojson_round_trips_through_json():
    report = aggregate_segments(_set(_detection(100, 500, 300, 600)), [_segment("s1")])
    assert json.loads(json.dumps(report.to_geojson()))["features"]


def test_observation_without_geometry_refuses_to_become_a_feature():
    obs = SegmentObservation(
        segment_id="s1",
        name="s1",
        observability=Observability.OBSERVED,
        cloud_fraction=0.0,
        coverage=0.0,
        detected_area_m2=0.0,
        affected_front_m=0.0,
        front_length_m=100.0,
        detection_count=0,
    )
    with pytest.raises(ValueError, match="no surf-zone geometry"):
        obs.to_feature()


def test_write_csv_produces_one_row_per_segment(tmp_path):
    report = aggregate_segments(_set(), [_segment("a"), _segment("b")])
    path = report.write_csv(tmp_path / "out" / "segments.csv")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert {r["segment_id"] for r in rows} == {"a", "b"}


def test_write_csv_of_an_empty_report_writes_an_empty_file(tmp_path):
    path = SegmentReport().write_csv(tmp_path / "empty.csv")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_append_history_writes_a_header_once_then_appends(tmp_path):
    """The dated record is the product. A rerun must extend it, not replace it."""
    path = tmp_path / "history.csv"
    segments = [_segment("a"), _segment("b")]
    append_history(aggregate_segments(_set(), segments, observed_on="2026-07-24"), path)
    append_history(aggregate_segments(_set(), segments, observed_on="2026-07-29"), path)

    text = path.read_text(encoding="utf-8")
    assert text.count("segment_id,name") == 1
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 4
    assert {r["observed_on"] for r in rows} == {"2026-07-24", "2026-07-29"}


def test_append_history_of_an_empty_report_is_a_no_op(tmp_path):
    path = tmp_path / "history.csv"
    append_history(SegmentReport(), path)
    assert not path.exists() or path.read_text(encoding="utf-8") == ""
