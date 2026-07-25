"""Tests for tile addressing, slippy-map math and raster windowing."""

from __future__ import annotations

import itertools
import math

import mercantile
import numpy as np
import pytest

from mdebris.config import settings
from mdebris.geo.tiles import (
    WEB_MERCATOR_MAX_LAT,
    deg2num,
    num2deg,
    parse_tile_name,
    tile_affine,
    tile_bounds,
    tiles_for_bbox,
    windows_for_raster,
)
from mdebris.types import GeoBBox, TileRef

# ---------------------------------------------------------------- parsing ----


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("12-34-6", (12, 34, 6)),
        ("12-34-6.jpg", (12, 34, 6)),
        ("12-34-6.tif", (12, 34, 6)),
        ("/data/tiles/12-34-6.png", (12, 34, 6)),
        ("data\\tiles\\12-34-6.png", (12, 34, 6)),
        ("0-0-0", (0, 0, 0)),
        # The legacy split('-') died on any of these.
        ("S2A_MSIL2A_20190101-12-34-6.jpg", (12, 34, 6)),
        ("scene-abc-def-12-34-6", (12, 34, 6)),
        ("2019-06-01-12-34-6.jpg", (12, 34, 6)),
    ],
)
def test_parse_tile_name_accepts_prefixes_and_extensions(name, expected):
    tile = parse_tile_name(name)
    assert (tile.x, tile.y, tile.z) == expected


def test_parse_tile_name_roundtrips_through_str():
    for tile in (TileRef(0, 0, 0), TileRef(1, 2, 3), TileRef(70000, 45000, 17)):
        assert parse_tile_name(str(tile)) == tile
        assert parse_tile_name(f"{tile}.jpg") == tile


@pytest.mark.parametrize("bad", ["12-34", "tile", "", "12-34-x", "12--6", "a-b-c"])
def test_parse_tile_name_rejects_garbage(bad):
    with pytest.raises(ValueError, match="cannot parse tile name"):
        parse_tile_name(bad)


def test_parse_tile_name_error_names_the_input():
    with pytest.raises(ValueError) as exc:
        parse_tile_name("not_a_tile.jpg")
    assert "not_a_tile.jpg" in str(exc.value)


def test_parse_tile_name_rejects_out_of_range_coordinates():
    # Zoom 1 has a 2x2 grid, so x=9 cannot exist. TileRef enforces this.
    with pytest.raises(ValueError, match="out of range"):
        parse_tile_name("9-0-1.jpg")


def test_parse_tile_name_accepts_pathlike(tmp_path):
    path = tmp_path / "12-34-6.jpg"
    path.write_bytes(b"")
    assert parse_tile_name(path) == TileRef(12, 34, 6)


# ----------------------------------------------------------------- bounds ----


@pytest.mark.parametrize("xyz", [(0, 0, 0), (1, 1, 2), (12, 34, 6), (70000, 45000, 17)])
def test_tile_bounds_matches_mercantile(xyz):
    tile = TileRef(*xyz)
    got = tile_bounds(tile)
    want = mercantile.bounds(*xyz)
    assert got.as_tuple() == pytest.approx((want.west, want.south, want.east, want.north))


def test_tile_bounds_of_root_tile_is_the_whole_web_mercator_extent():
    b = tile_bounds(TileRef(0, 0, 0))
    assert b.west == pytest.approx(-180.0)
    assert b.east == pytest.approx(180.0)
    assert b.north == pytest.approx(WEB_MERCATOR_MAX_LAT)
    assert b.south == pytest.approx(-WEB_MERCATOR_MAX_LAT)


# ----------------------------------------------------------------- affine ----


def test_tile_affine_root_tile_hand_computed():
    """Root tile at 256 px: 360 degrees over 256 columns, origin at the NW corner."""
    a = tile_affine(TileRef(0, 0, 0), 256)
    assert a.a == pytest.approx(360.0 / 256.0)
    assert a.b == 0.0
    assert a.d == 0.0
    assert a.e == pytest.approx(-2.0 * WEB_MERCATOR_MAX_LAT / 256.0)
    assert a.xoff == pytest.approx(-180.0)
    assert a.yoff == pytest.approx(WEB_MERCATOR_MAX_LAT)

    # The centre pixel of the root tile is null island.
    assert a * (128.0, 128.0) == pytest.approx((0.0, 0.0))
    # The corners are the tile bounds.
    assert a * (0.0, 0.0) == pytest.approx((-180.0, WEB_MERCATOR_MAX_LAT))
    assert a * (256.0, 256.0) == pytest.approx((180.0, -WEB_MERCATOR_MAX_LAT))


@pytest.mark.parametrize("size", [64, 256, 512, 960])
def test_tile_affine_corners_match_mercantile_bounds_at_any_size(size):
    tile = TileRef(12, 34, 6)
    b = mercantile.bounds(12, 34, 6)
    a = tile_affine(tile, size)
    assert a * (0.0, 0.0) == pytest.approx((b.west, b.north))
    assert a * (float(size), float(size)) == pytest.approx((b.east, b.south))


def test_tile_affine_reproduces_the_legacy_coefficients():
    """Byte-for-byte parity with the TF1 script's inline transform at 256 px."""
    tile_x, tile_y, tile_z = 12, 34, 6
    b = mercantile.bounds(tile_x, tile_y, tile_z)
    width = b[2] - b[0]
    height = b[3] - b[1]
    legacy = (width / 256, 0.0, b[0], 0.0, (0 - height / 256), b[3])

    a = tile_affine(TileRef(tile_x, tile_y, tile_z), 256)
    assert (a.a, a.b, a.xoff, a.d, a.e, a.yoff) == pytest.approx(legacy)


def test_tile_affine_defaults_to_configured_tile_size():
    a = tile_affine(TileRef(12, 34, 6))
    b = tile_bounds(TileRef(12, 34, 6))
    assert a.a == pytest.approx((b.east - b.west) / settings.tile_size)


def test_tile_affine_rejects_non_positive_size():
    with pytest.raises(ValueError, match="tile_size must be positive"):
        tile_affine(TileRef(0, 0, 0), 0)


# ------------------------------------------------------------ slippy math ----


@pytest.mark.parametrize("z", [0, 1, 6, 12, 17])
def test_num2deg_agrees_with_mercantile_upper_left(z):
    limit = 1 << z
    for x, y in {(0, 0), (limit // 2, limit // 3), (limit - 1, limit - 1)}:
        want = mercantile.ul(x, y, z)
        got = num2deg(x, y, z)
        assert got == pytest.approx((want.lng, want.lat), abs=1e-9)


def test_num2deg_accepts_fractional_tile_coordinates():
    """Half a tile along each axis is the tile centre, not another tile's corner."""
    lon, lat = num2deg(0.5, 0.5, 1)
    assert lon == pytest.approx(-90.0)
    b = mercantile.bounds(0, 0, 1)
    # y = 0.5 of a 2x2 grid is the midpoint of tile row 0 in Mercator y, which is
    # not the arithmetic mean of the latitudes.
    assert b.south < lat < b.north


@pytest.mark.parametrize("z", [0, 1, 6, 12, 17])
def test_deg2num_agrees_with_mercantile(z):
    rng = np.random.default_rng(seed=z)
    lons = rng.uniform(-179.9, 179.9, size=25)
    lats = rng.uniform(-84.0, 84.0, size=25)
    for lon, lat in zip(lons, lats, strict=True):
        want = mercantile.tile(float(lon), float(lat), z)
        assert deg2num(float(lon), float(lat), z) == (want.x, want.y)


def test_deg2num_and_num2deg_are_inverse_within_a_tile():
    for x, y, z in [(12, 34, 6), (0, 0, 3), (70000, 45000, 17)]:
        # Aim at the tile centre so rounding cannot land in a neighbour.
        lon, lat = num2deg(x + 0.5, y + 0.5, z)
        assert deg2num(lon, lat, z) == (x, y)


def test_deg2num_clamps_at_the_projection_limits():
    assert deg2num(0.0, 89.9, 4) == deg2num(0.0, WEB_MERCATOR_MAX_LAT, 4)
    x, y = deg2num(180.0, -89.9, 4)
    assert (x, y) == (15, 15)


def test_web_mercator_max_lat_is_the_projection_cutoff():
    assert pytest.approx(math.degrees(math.atan(math.sinh(math.pi)))) == WEB_MERCATOR_MAX_LAT


# ------------------------------------------------------------ tile listing ----


def test_tiles_for_bbox_covers_the_box():
    bbox = GeoBBox(west=-1.0, south=50.0, east=1.0, north=51.0)
    tiles = list(tiles_for_bbox(bbox, 8))
    assert tiles
    assert all(isinstance(t, TileRef) for t in tiles)
    # Every corner of the AOI falls inside one of the returned tiles.
    for lon, lat in [(-1.0, 50.0), (1.0, 51.0), (0.0, 50.5)]:
        assert TileRef(*deg2num(lon, lat, 8), z=8) in tiles


def test_tiles_for_bbox_at_zoom_zero_is_the_single_root_tile():
    bbox = GeoBBox(west=-179.0, south=-80.0, east=179.0, north=80.0)
    assert list(tiles_for_bbox(bbox, 0)) == [TileRef(0, 0, 0)]


def test_tiles_for_bbox_rejects_absurd_zoom():
    bbox = GeoBBox(west=0.0, south=0.0, east=1.0, north=1.0)
    with pytest.raises(ValueError, match="outside supported range"):
        list(tiles_for_bbox(bbox, 99))


# ---------------------------------------------------------------- windows ----


@pytest.mark.parametrize(
    ("width", "height", "size", "overlap"),
    [
        (512, 512, 256, 0),  # exact division, no overlap
        (512, 512, 256, 64),  # exact division, overlapping
        (1000, 700, 256, 32),  # ragged on both axes
        (100, 100, 256, 16),  # raster smaller than one window
        (960, 960, 960, 96),  # exactly one window at the configured size
        (1, 1, 8, 2),  # degenerate but legal
        (1097, 13, 128, 127),  # maximum overlap, extreme aspect ratio
    ],
)
def test_windows_cover_every_pixel_and_stay_in_bounds(width, height, size, overlap):
    seen = np.zeros((height, width), dtype=bool)
    windows = list(windows_for_raster(width, height, size, overlap))
    assert windows, "at least one window is always required"

    for col_off, row_off, win_w, win_h in windows:
        assert 0 <= col_off < width
        assert 0 <= row_off < height
        assert 0 < win_w <= size
        assert 0 < win_h <= size
        assert col_off + win_w <= width, "window runs past the right edge"
        assert row_off + win_h <= height, "window runs past the bottom edge"
        seen[row_off : row_off + win_h, col_off : col_off + win_w] = True

    assert seen.all(), f"{(~seen).sum()} pixels were never covered"


def test_windows_for_small_raster_is_a_single_partial_window():
    assert list(windows_for_raster(100, 60, 256, 16)) == [(0, 0, 100, 60)]


def test_windows_have_the_requested_overlap():
    windows = list(windows_for_raster(1024, 256, 256, 64))
    cols = sorted({w[0] for w in windows})
    steps = {b - a for a, b in itertools.pairwise(cols)}
    assert steps == {256 - 64}


def test_windows_emit_a_partial_window_at_a_ragged_edge():
    # 300 wide, 256 window, 56 step: offsets 0 and 44 fit fully, then the tail.
    windows = list(windows_for_raster(300, 256, 256, 200))
    last = windows[-1]
    assert last[0] + last[2] == 300
    assert last[2] < 256, "the tail window must be partial, not clamped back"


def test_windows_default_to_settings():
    windows = list(windows_for_raster(4000, 4000))
    step = settings.tile_size - settings.tile_overlap
    assert windows[0] == (0, 0, settings.tile_size, settings.tile_size)
    assert windows[1][0] == step


@pytest.mark.parametrize(
    ("width", "height", "size", "overlap"),
    [(0, 10, 8, 0), (10, -1, 8, 0), (10, 10, 0, 0), (10, 10, 8, 8), (10, 10, 8, -1)],
)
def test_windows_reject_invalid_geometry(width, height, size, overlap):
    with pytest.raises(ValueError):
        list(windows_for_raster(width, height, size, overlap))
