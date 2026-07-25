"""Tests for georeferencing detections and writing them out.

Several tests here are regressions for bugs in the TF1 script this module
replaces; each names the bug it pins down.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import affine
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from mdebris.geo.georef import (
    boxes_to_detections,
    default_output_path,
    detections_to_geodataframe,
    georeference_detections,
    pixel_bbox_to_polygon,
    write_geojson,
)
from mdebris.geo.tiles import WEB_MERCATOR_MAX_LAT, tile_affine, tile_bounds
from mdebris.types import BBox, Detection, DetectionSet, SceneRef, SurfaceClass, TileRef

ROOT_TILE = TileRef(0, 0, 0)

# EPSG:32633 (UTM zone 33N) has its central meridian at 15 degrees east by
# definition, so easting 500000 is exactly 15 E. Northing 4000000 is about
# 36.14 N. Both are used below as hand-checkable reprojection targets.
UTM33N = "EPSG:32633"
UTM_ORIGIN_LON = 15.0
UTM_ORIGIN_LAT = 36.14471809881776


def _det(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> Detection:
    return Detection(bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), score=score)


# ------------------------------------------------------ pixel to polygon ----


def test_pixel_bbox_to_polygon_hand_computed_on_the_root_tile():
    """A box on the middle half of the 256 px root tile spans half the world.

    The root tile is 360 degrees wide over 256 columns (1.40625 deg/px) and
    2 * 85.0511 degrees tall over 256 rows. Columns 64 and 192 are therefore
    -90 and +90 longitude, and rows 64 and 192 are +/- half the Mercator limit.
    """
    poly = pixel_bbox_to_polygon(BBox(64, 64, 192, 192), tile_affine(ROOT_TILE, 256))
    assert isinstance(poly, Polygon)
    west, south, east, north = poly.bounds
    assert west == pytest.approx(-90.0)
    assert east == pytest.approx(90.0)
    assert north == pytest.approx(WEB_MERCATOR_MAX_LAT / 2.0)
    assert south == pytest.approx(-WEB_MERCATOR_MAX_LAT / 2.0)


def test_pixel_bbox_to_polygon_centre_pixel_of_a_real_tile():
    """The centre column of tile 12-34-6 is the midpoint of its longitude span."""
    tile = TileRef(12, 34, 6)
    b = tile_bounds(tile)
    poly = pixel_bbox_to_polygon(BBox(127, 127, 129, 129), tile_affine(tile, 256))
    lon, lat = poly.centroid.x, poly.centroid.y
    assert lon == pytest.approx((b.west + b.east) / 2.0)
    assert lat == pytest.approx((b.south + b.north) / 2.0)
    # Zoom 6 tiles are 5.625 degrees wide; the west edge lands on a clean value.
    assert b.west == pytest.approx(-112.5)


def test_pixel_bbox_to_polygon_is_north_up():
    """Row 0 is the northern edge. A sign error here mirrors every detection."""
    a = tile_affine(ROOT_TILE, 256)
    top = pixel_bbox_to_polygon(BBox(0, 0, 10, 10), a)
    bottom = pixel_bbox_to_polygon(BBox(0, 246, 10, 256), a)
    assert top.centroid.y > bottom.centroid.y


def test_pixel_bbox_to_polygon_honours_a_rotated_transform():
    """All four corners are transformed, so a rotation yields a rotated polygon."""
    rotated = affine.Affine.rotation(30.0) * affine.Affine.scale(1.0, 1.0)
    poly = pixel_bbox_to_polygon(BBox(0, 0, 10, 10), rotated)
    assert poly.area == pytest.approx(100.0)
    # An axis-aligned result would have an envelope of exactly the same area.
    assert poly.envelope.area > poly.area


# --------------------------------------------------- detection array port ----


def test_boxes_to_detections_handles_zero_one_and_many():
    """Bug: one detection per tile used to discard the whole tile.

    ``np.squeeze`` collapsed a (1, 4) array to (4,), ``bbox[1]`` then indexed a
    float, and a bare ``except TypeError: continue`` wrapped the entire loop.
    """
    assert boxes_to_detections(np.zeros((0, 4)), np.zeros(0), width=256, height=256) == []

    one = boxes_to_detections(
        np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([0.9]), width=256, height=256
    )
    assert len(one) == 1

    many = boxes_to_detections(
        np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.6, 0.6], [0.0, 0.0, 1.0, 1.0]]),
        np.array([0.9, 0.8, 0.7]),
        width=256,
        height=256,
    )
    assert len(many) == 3
    assert [round(d.score, 2) for d in many] == [0.9, 0.8, 0.7]


def test_single_detection_survives_the_legacy_squeeze():
    """Regression: the exact array shapes the legacy pipeline produced.

    ``boxes[indices]`` followed by ``np.squeeze`` yields a 1-D (4,) box array and
    a 0-D score for a tile holding exactly one detection.
    """
    boxes = np.zeros((100, 4), dtype=np.float32)
    scores = np.zeros(100, dtype=np.float32)
    boxes[0] = [0.10, 0.20, 0.30, 0.40]  # ymin, xmin, ymax, xmax
    scores[0] = 0.87

    indices = np.argwhere(scores >= 0.2)
    squeezed_boxes = np.squeeze(boxes[indices])
    squeezed_scores = np.squeeze(scores[indices])
    assert squeezed_boxes.shape == (4,), "precondition: this is what the legacy code saw"
    assert squeezed_scores.shape == ()

    # What the legacy loop did with that array, reproduced.
    with pytest.raises(TypeError):
        _ = [row[1] for row in squeezed_boxes.tolist()]

    dets = boxes_to_detections(squeezed_boxes, squeezed_scores, width=256, height=256)
    assert len(dets) == 1
    d = dets[0]
    assert d.bbox.xmin == pytest.approx(0.20 * 256, rel=1e-5)
    assert d.bbox.ymin == pytest.approx(0.10 * 256, rel=1e-5)
    assert d.bbox.xmax == pytest.approx(0.40 * 256, rel=1e-5)
    assert d.bbox.ymax == pytest.approx(0.30 * 256, rel=1e-5)
    assert d.score == pytest.approx(0.87, rel=1e-5)


def test_single_detection_still_produces_one_geojson_feature():
    """End to end version of the same bug: one detection must reach the output."""
    dets = boxes_to_detections(
        np.squeeze(np.array([[0.4, 0.4, 0.6, 0.6]])), np.float32(0.5), width=256, height=256
    )
    georeference_detections(dets, tile=TileRef(12, 34, 6), tile_size=256)
    gj = DetectionSet(detections=dets).to_geojson()
    assert len(gj["features"]) == 1


def test_boxes_to_detections_keeps_sub_pixel_precision():
    """Bug: ``(bboxes * 256).astype(np.int)`` used a removed alias and truncated.

    A box 0.256 px from the top must not snap to row 0.
    """
    dets = boxes_to_detections(
        np.array([[0.001, 0.002, 0.501, 0.502]]), np.array([0.5]), width=1000, height=1000
    )
    assert dets[0].bbox.ymin == pytest.approx(1.0)
    assert dets[0].bbox.xmin == pytest.approx(2.0)
    dets_256 = boxes_to_detections(
        np.array([[0.001, 0.002, 0.501, 0.502]]), np.array([0.5]), width=256, height=256
    )
    assert dets_256[0].bbox.ymin == pytest.approx(0.256)
    assert dets_256[0].bbox.ymin != 0.0


def test_geo_module_never_references_the_removed_numpy_aliases():
    """``np.int`` and friends were removed in NumPy 1.24 and must not reappear.

    The check walks the AST rather than grepping, so the docstrings that quote
    the legacy code do not trip it.
    """
    removed = {"int", "float", "bool", "object", "str", "complex", "long", "unicode"}
    geo_dir = Path(__file__).resolve().parents[1] / "src" / "mdebris" / "geo"
    offenders = []
    for path in sorted(geo_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"np", "numpy"}
                and node.attr in removed
            ):
                offenders.append(f"{path.name}:{node.lineno} np.{node.attr}")
    assert offenders == []


def test_boxes_to_detections_respects_order_and_threshold():
    xyxy = boxes_to_detections(
        np.array([[0.2, 0.1, 0.4, 0.3]]),
        np.array([0.9]),
        width=100,
        height=100,
        order="xyxy",
    )
    assert xyxy[0].bbox.as_xyxy() == pytest.approx((20.0, 10.0, 40.0, 30.0))

    kept = boxes_to_detections(
        np.array([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]]),
        np.array([0.05, 0.75]),
        width=100,
        height=100,
        score_threshold=0.5,
    )
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.75)


def test_boxes_to_detections_accepts_absolute_pixel_boxes():
    dets = boxes_to_detections(
        np.array([[10.0, 20.0, 30.0, 40.0]]),
        np.array([0.6]),
        width=256,
        height=256,
        normalized=False,
    )
    assert dets[0].bbox.as_xyxy() == pytest.approx((20.0, 10.0, 40.0, 30.0))


def test_boxes_to_detections_clips_boxes_that_overrun_the_frame():
    dets = boxes_to_detections(
        np.array([[-0.05, -0.10, 1.20, 1.05]]), np.array([0.6]), width=100, height=100
    )
    assert dets[0].bbox.as_xyxy() == pytest.approx((0.0, 0.0, 100.0, 100.0))


def test_boxes_to_detections_maps_class_ids():
    dets = boxes_to_detections(
        np.array([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]]),
        np.array([0.9, 0.9]),
        classes=np.array([1, 2]),
        width=100,
        height=100,
        label_map={1: SurfaceClass.DEBRIS, 2: SurfaceClass.SARGASSUM},
    )
    assert [d.label for d in dets] == [SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM]


def test_boxes_to_detections_defaults_every_class_to_debris():
    dets = boxes_to_detections(
        np.array([[0.0, 0.0, 0.1, 0.1]]),
        np.array([0.9]),
        classes=np.array([7]),
        width=10,
        height=10,
    )
    assert dets[0].label is SurfaceClass.DEBRIS


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order": "xywh"}, "order must be"),
        ({}, "they must match"),
    ],
)
def test_boxes_to_detections_rejects_bad_input(kwargs, match):
    boxes = np.array([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]])
    scores = np.array([0.9]) if not kwargs else np.array([0.9, 0.8])
    with pytest.raises(ValueError, match=match):
        boxes_to_detections(boxes, scores, width=10, height=10, **kwargs)


def test_boxes_to_detections_rejects_wrong_column_count():
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        boxes_to_detections(np.zeros((2, 5)), np.zeros(2), width=10, height=10)


# ------------------------------------------------------------ tile path ----


def test_georeference_detections_tile_path_matches_the_legacy_transform():
    """Parity check against the original inline computation."""
    import mercantile
    import shapely
    from shapely import geometry

    tile = TileRef(12, 34, 6)
    b = mercantile.bounds(tile.x, tile.y, tile.z)
    a = affine.Affine((b[2] - b[0]) / 256, 0.0, b[0], 0.0, (0 - (b[3] - b[1]) / 256), b[3])
    a_lst = [a.a, a.b, a.d, a.e, a.xoff, a.yoff]
    legacy_box = [51, 25, 102, 76]  # xyxy after the legacy index shuffle
    legacy = shapely.affinity.affine_transform(geometry.box(*legacy_box), a_lst)

    dets = [_det(51, 25, 102, 76)]
    georeference_detections(dets, tile=tile, tile_size=256)
    assert dets[0].geometry.equals_exact(legacy, tolerance=1e-12)


def test_georeference_detections_is_a_no_op_on_an_empty_list():
    assert georeference_detections([], tile=ROOT_TILE, tile_size=256) == []


def test_georeference_detections_handles_many_detections_independently():
    dets = [_det(0, 0, 64, 64), _det(96, 96, 160, 160), _det(192, 192, 256, 256)]
    georeference_detections(dets, tile=ROOT_TILE, tile_size=256)
    lons = [d.geometry.centroid.x for d in dets]
    lats = [d.geometry.centroid.y for d in dets]
    assert lons == sorted(lons), "west to east ordering must be preserved"
    assert lats == sorted(lats, reverse=True), "north to south ordering must be preserved"
    assert lons[1] == pytest.approx(0.0)
    assert lats[1] == pytest.approx(0.0)


def test_georeference_detections_records_provenance():
    tile = TileRef(12, 34, 6)
    dets = [_det(10, 10, 20, 20)]
    georeference_detections(dets, tile=tile, tile_size=256)
    assert dets[0].tile == tile
    assert dets[0].to_feature()["properties"]["tile"] == "12-34-6"


def test_georeference_detections_does_not_clobber_an_existing_tile():
    explicit = TileRef(1, 1, 3)
    dets = [_det(10, 10, 20, 20)]
    dets[0].tile = explicit
    georeference_detections(dets, tile=TileRef(12, 34, 6), tile_size=256)
    assert dets[0].tile == explicit


def test_georeference_detections_returns_the_same_objects():
    dets = [_det(0, 0, 8, 8)]
    assert georeference_detections(dets, tile=ROOT_TILE, tile_size=256) is dets


def test_georeference_detections_tile_size_changes_the_scale():
    small = [_det(0, 0, 128, 128)]
    large = [_det(0, 0, 480, 480)]
    georeference_detections(small, tile=ROOT_TILE, tile_size=256)
    georeference_detections(large, tile=ROOT_TILE, tile_size=960)
    # Half of each tile image covers half the tile either way.
    assert small[0].geometry.bounds == pytest.approx(large[0].geometry.bounds)


# ---------------------------------------------------------- raster path ----


def test_georeference_detections_transform_path_in_wgs84():
    transform = affine.Affine(0.001, 0.0, 10.0, 0.0, -0.001, 50.0)
    dets = [_det(0, 0, 100, 100)]
    georeference_detections(dets, transform=transform)
    assert dets[0].geometry.bounds == pytest.approx((10.0, 49.9, 10.1, 50.0))


def test_georeference_detections_reprojects_from_utm():
    """A UTM scene must come out as lon/lat, not as metres or a swapped pair."""
    transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
    dets = [_det(0.0, 0.0, 10.0, 10.0)]
    georeference_detections(dets, transform=transform, src_crs=UTM33N)

    west, south, east, north = dets[0].geometry.bounds
    assert west == pytest.approx(UTM_ORIGIN_LON, abs=1e-6)
    assert north == pytest.approx(UTM_ORIGIN_LAT, abs=1e-6)
    # 100 m east and south of the origin, so a small positive step in lon and a
    # small negative step in lat.
    assert 0.0 < east - west < 0.01
    assert 0.0 < north - south < 0.01
    # The classic pyproj axis-order bug would report latitude as longitude.
    assert abs(west) < 90.0 and 0.0 < north < 90.0


def test_georeference_detections_accepts_a_rasterio_crs_object():
    from rasterio.crs import CRS as RioCRS

    transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
    dets = [_det(0.0, 0.0, 10.0, 10.0)]
    georeference_detections(dets, transform=transform, src_crs=RioCRS.from_epsg(32633))
    assert dets[0].geometry.bounds[0] == pytest.approx(UTM_ORIGIN_LON, abs=1e-6)


def test_georeference_detections_skips_reprojection_when_already_wgs84():
    transform = affine.Affine(0.001, 0.0, 10.0, 0.0, -0.001, 50.0)
    a, b = [_det(0, 0, 100, 100)], [_det(0, 0, 100, 100)]
    georeference_detections(a, transform=transform)
    georeference_detections(b, transform=transform, src_crs="EPSG:4326")
    assert a[0].geometry.bounds == pytest.approx(b[0].geometry.bounds)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"tile": ROOT_TILE, "transform": affine.Affine.identity()}],
)
def test_georeference_detections_requires_exactly_one_frame(kwargs):
    with pytest.raises(ValueError, match="exactly one of tile= or transform="):
        georeference_detections([_det(0, 0, 1, 1)], **kwargs)


# ---------------------------------------------------------------- output ----


def test_write_geojson_creates_missing_parents(tmp_path):
    dets = [_det(0, 0, 64, 64), _det(64, 64, 128, 128)]
    georeference_detections(dets, tile=ROOT_TILE, tile_size=256)
    ds = DetectionSet(detections=dets, scene=SceneRef(scene_id="S2_TEST"))

    out = tmp_path / "deep" / "nested" / "dir" / "S2_TEST.geojson"
    written = write_geojson(ds, out)

    assert written == out
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 2
    assert payload["crs"]["properties"]["name"] == "urn:ogc:def:crs:OGC:1.3:CRS84"
    assert payload["properties"]["scene_id"] == "S2_TEST"


def test_write_geojson_of_an_empty_set_is_still_valid(tmp_path):
    out = write_geojson(DetectionSet(), tmp_path / "empty.geojson")
    payload = json.loads(out.read_text())
    assert payload["features"] == []


def test_write_geojson_compact_mode(tmp_path):
    dets = [_det(0, 0, 64, 64)]
    georeference_detections(dets, tile=ROOT_TILE, tile_size=256)
    out = write_geojson(DetectionSet(detections=dets), tmp_path / "c.geojson", indent=None)
    assert "\n" not in out.read_text()


def test_default_output_path_uses_settings_and_adds_the_suffix():
    assert default_output_path("scene_a").name == "scene_a.geojson"
    assert default_output_path("scene_a.geojson").name == "scene_a.geojson"


def test_detections_to_geodataframe():
    dets = [_det(0, 0, 64, 64, score=0.9), _det(64, 64, 128, 128, score=0.4)]
    georeference_detections(dets, tile=ROOT_TILE, tile_size=256)
    gdf = detections_to_geodataframe(DetectionSet(detections=dets))

    assert len(gdf) == 2
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf.columns) >= {"score", "label", "geometry"}
    assert gdf.geometry.iloc[0].bounds == pytest.approx(dets[0].geometry.bounds)


def test_detections_to_geodataframe_of_an_empty_set_keeps_the_schema():
    gdf = detections_to_geodataframe(DetectionSet())
    assert len(gdf) == 0
    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.name == "geometry"


def test_detections_to_geodataframe_drops_ungeoreferenced_detections():
    dets = [_det(0, 0, 64, 64), _det(64, 64, 128, 128)]
    georeference_detections(dets[:1], tile=ROOT_TILE, tile_size=256)
    assert len(detections_to_geodataframe(DetectionSet(detections=dets))) == 1
