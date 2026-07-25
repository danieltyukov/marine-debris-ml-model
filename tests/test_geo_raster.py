"""Tests for windowed raster reads, band alignment and RGB rendering.

Every raster is synthesized in ``tmp_path``, so the default run needs no
network and no sample data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from mdebris.geo.georef import georeference_detections
from mdebris.geo.raster import (
    SENTINEL2_RESOLUTION_M,
    raster_profile,
    read_bands,
    read_window,
    to_rgb,
    window_transform,
)
from mdebris.geo.tiles import windows_for_raster
from mdebris.types import BBox, Detection

# UTM zone 33N, an origin far from any zone edge. The two synthetic grids share
# this origin so a 10 m and a 20 m band cover exactly the same ground.
CRS = "EPSG:32633"
ORIGIN_X, ORIGIN_Y = 500000.0, 4000000.0
BLOCK_M = 80  # checkerboard block size, a common multiple of 10, 20 and 60


def _ground_checkerboard(cols: int, rows: int, gsd: float) -> np.ndarray:
    """A field defined by ground position, so any grid samples the same scene."""
    bx = (np.arange(cols) * gsd) // BLOCK_M
    by = (np.arange(rows) * gsd) // BLOCK_M
    return (((bx[None, :] + by[:, None]) % 2) * 1000).astype("uint16")


def _write(path: Path, array: np.ndarray, gsd: float, crs: str = CRS) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, gsd, gsd),
    ) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Path]:
    """One band per native Sentinel-2 resolution, all over the same 5120 m square."""
    return {
        "B04": _write(tmp_path / "B04.tif", _ground_checkerboard(512, 512, 10), 10),  # 10 m
        "B03": _write(tmp_path / "B03.tif", _ground_checkerboard(512, 512, 10), 10),
        "B02": _write(tmp_path / "B02.tif", _ground_checkerboard(512, 512, 10), 10),
        "B11": _write(tmp_path / "B11.tif", _ground_checkerboard(256, 256, 20), 20),  # 20 m
        "B01": _write(tmp_path / "B01.tif", _ground_checkerboard(86, 86, 60), 60),  # 60 m
    }


# ------------------------------------------------------------ read_window ----


def test_read_window_returns_the_requested_region(scene):
    got = read_window(scene["B04"], (10, 20, 64, 48))
    assert got.shape == (48, 64)
    expected = _ground_checkerboard(512, 512, 10)[20:68, 10:74]
    assert np.array_equal(got, expected)


def test_read_window_accepts_a_rasterio_window(scene):
    tup = read_window(scene["B04"], (0, 0, 32, 32))
    win = read_window(scene["B04"], Window(0, 0, 32, 32))
    assert np.array_equal(tup, win)


def test_read_window_without_a_window_reads_the_whole_raster(scene):
    assert read_window(scene["B04"]).shape == (512, 512)


def test_read_window_resamples_to_out_shape(scene):
    got = read_window(scene["B04"], (0, 0, 256, 256), out_shape=(64, 64), resampling="nearest")
    assert got.shape == (64, 64)


def test_read_window_rejects_an_unknown_resampling_method(scene):
    with pytest.raises(ValueError, match="unknown resampling method"):
        read_window(scene["B04"], (0, 0, 8, 8), out_shape=(4, 4), resampling="magic")


# ------------------------------------------------------------- read_bands ----


def test_read_bands_puts_every_resolution_on_one_grid(scene):
    """The correctness requirement: 10 m, 20 m and 60 m bands come back aligned.

    A 512x512 window at 10 m is 256x256 at 20 m and about 85x85 at 60 m. Reading
    the same offsets from each asset would return three different shapes over
    three different footprints, making any index that mixes them meaningless.
    """
    bands = read_bands(scene, (0, 0, 512, 512), resampling="nearest")
    assert set(bands) == set(scene)
    assert {a.shape for a in bands.values()} == {(512, 512)}


def test_read_bands_aligns_20m_onto_the_10m_grid_exactly(scene):
    """Same ground field sampled at two resolutions must land pixel for pixel."""
    bands = read_bands(
        {"B04": scene["B04"], "B11": scene["B11"]}, (0, 0, 256, 256), resampling="nearest"
    )
    assert np.array_equal(bands["B04"], bands["B11"])


def test_read_bands_alignment_holds_for_an_offset_window(scene):
    bands = read_bands(
        {"B04": scene["B04"], "B11": scene["B11"]}, (64, 32, 128, 128), resampling="nearest"
    )
    assert bands["B04"].shape == bands["B11"].shape == (128, 128)
    assert np.array_equal(bands["B04"], bands["B11"])


def test_read_bands_alignment_is_by_ground_not_by_pixel_index(scene):
    """Reading the raw offsets from the 20 m band would grab the wrong ground.

    This is the failure mode the function exists to prevent, shown explicitly.
    """
    naive = read_window(scene["B11"], (64, 32, 128, 128))
    aligned = read_bands(
        {"B04": scene["B04"], "B11": scene["B11"]}, (64, 32, 128, 128), resampling="nearest"
    )["B11"]
    assert not np.array_equal(naive, aligned)


def test_read_bands_honours_an_explicit_target_shape(scene):
    bands = read_bands(scene, (0, 0, 512, 512), target_shape=(128, 128), resampling="nearest")
    assert {a.shape for a in bands.values()} == {(128, 128)}


def test_read_bands_reference_band_defines_the_window_grid(scene):
    """The same window means different ground on a 10 m and a 20 m reference."""
    on_10m = read_bands(
        {"B04": scene["B04"], "B11": scene["B11"]},
        (0, 0, 128, 128),
        reference="B04",
        resampling="nearest",
    )
    on_20m = read_bands(
        {"B04": scene["B04"], "B11": scene["B11"]},
        (0, 0, 128, 128),
        reference="B11",
        resampling="nearest",
    )
    assert on_10m["B04"].shape == on_20m["B04"].shape == (128, 128)
    # A 128 px window at 20 m covers twice the ground of one at 10 m, so the
    # checkerboard comes back at half the period.
    assert not np.array_equal(on_10m["B04"], on_20m["B04"])


def test_read_bands_defaults_the_reference_to_the_first_key(scene):
    hrefs = {"B11": scene["B11"], "B04": scene["B04"]}
    bands = read_bands(hrefs, (0, 0, 100, 100), resampling="nearest")
    assert {a.shape for a in bands.values()} == {(100, 100)}


def test_read_bands_without_a_window_reads_the_reference_extent(scene):
    bands = read_bands({"B04": scene["B04"], "B11": scene["B11"]}, resampling="nearest")
    assert {a.shape for a in bands.values()} == {(512, 512)}


def test_read_bands_pads_where_a_band_stops_short(tmp_path):
    """Scene edges: a band that does not cover the window is padded, not stretched."""
    big = _write(tmp_path / "big.tif", _ground_checkerboard(512, 512, 10), 10)
    short = _write(tmp_path / "short.tif", _ground_checkerboard(200, 200, 20), 20)

    bands = read_bands(
        {"B04": big, "B11": short}, (0, 0, 512, 512), resampling="nearest", fill_value=7
    )
    assert bands["B11"].shape == (512, 512)
    # The 20 m band covers 4000 m of the 5120 m window, so the far edge is fill.
    assert bands["B11"][-1, -1] == 7
    assert bands["B11"][0, 0] == bands["B04"][0, 0]


def test_read_bands_rejects_mixed_crs(tmp_path):
    a = _write(tmp_path / "a.tif", _ground_checkerboard(64, 64, 10), 10, crs="EPSG:32633")
    b = _write(tmp_path / "b.tif", _ground_checkerboard(64, 64, 10), 10, crs="EPSG:32634")
    with pytest.raises(ValueError, match="reproject the assets"):
        read_bands({"B04": a, "B03": b}, (0, 0, 32, 32))


def test_read_bands_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="hrefs is empty"):
        read_bands({}, (0, 0, 8, 8))


def test_read_bands_rejects_an_unknown_reference(scene):
    with pytest.raises(KeyError, match="B99"):
        read_bands(scene, (0, 0, 8, 8), reference="B99")


def test_sentinel2_resolution_table_matches_the_mission_spec():
    assert {b for b, r in SENTINEL2_RESOLUTION_M.items() if r == 10} == {
        "B02",
        "B03",
        "B04",
        "B08",
    }
    assert {b for b, r in SENTINEL2_RESOLUTION_M.items() if r == 60} == {"B01", "B09", "B10"}
    assert SENTINEL2_RESOLUTION_M["B11"] == 20
    assert SENTINEL2_RESOLUTION_M["B8A"] == 20


# -------------------------------------------------------- window_transform ----


def test_window_transform_shifts_the_origin_by_the_offset(scene):
    tf = window_transform((64, 32, 128, 128), scene["B04"])
    assert tf.xoff == pytest.approx(ORIGIN_X + 64 * 10)
    assert tf.yoff == pytest.approx(ORIGIN_Y - 32 * 10)
    assert tf.a == pytest.approx(10.0)
    assert tf.e == pytest.approx(-10.0)


def test_window_transform_accepts_an_affine_directly(scene):
    with rasterio.open(scene["B04"]) as src:
        base = src.transform
    assert window_transform((10, 20, 4, 4), base) == window_transform((10, 20, 4, 4), scene["B04"])


def test_window_transform_of_the_full_extent_is_the_raster_transform(scene):
    with rasterio.open(scene["B04"]) as src:
        assert window_transform(Window(0, 0, src.width, src.height), src.transform) == src.transform


# ------------------------------------------------------------------ to_rgb ----


@pytest.fixture
def water_bands() -> dict[str, np.ndarray]:
    """Sentinel-2 style reflectance: dark water with a bright slick.

    L2A values are reflectance scaled by 10000, so open water sits near 200 to
    500 out of a uint16 range of 65535.
    """
    rng = np.random.default_rng(0)
    base = rng.integers(200, 500, size=(64, 64))
    bands = {
        "B04": base.astype("uint16"),
        "B03": (base + 60).astype("uint16"),
        "B02": (base + 140).astype("uint16"),
    }
    for arr in bands.values():
        arr[20:28, 20:28] = 1800  # a debris slick
    return bands


def test_to_rgb_shape_and_dtype(water_bands):
    rgb = to_rgb(water_bands)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_naive_scaling_of_dark_water_really_is_black(water_bands):
    """Motivates the percentile default: the obvious approach is unusable."""
    naive = to_rgb(water_bands, stretch="none")
    assert naive.max() < 10


def test_percentile_stretch_makes_dark_water_legible(water_bands):
    rgb = to_rgb(water_bands)
    assert rgb.max() == 255
    assert rgb.min() == 0
    assert rgb.std() > 20
    # Water fills most of the frame, so the median must be well off the floor.
    assert np.median(rgb) > 40


def test_percentile_bounds_are_configurable(water_bands):
    tight = to_rgb(water_bands, percentiles=(40.0, 60.0))
    wide = to_rgb(water_bands, percentiles=(0.0, 100.0))
    assert tight.std() > wide.std(), "a tighter window must increase contrast"


def test_linear_stretch_is_dominated_by_the_bright_slick(water_bands):
    linear = to_rgb(water_bands, stretch="linear")
    percentile = to_rgb(water_bands)
    assert linear.mean() < percentile.mean()


def test_gamma_above_one_darkens(water_bands):
    assert to_rgb(water_bands, gamma=2.0).mean() < to_rgb(water_bands, gamma=1.0).mean()


def test_gamma_below_one_brightens(water_bands):
    assert to_rgb(water_bands, gamma=0.5).mean() > to_rgb(water_bands, gamma=1.0).mean()


def test_to_rgb_channel_order_is_red_green_blue(water_bands):
    """B02 is the brightest of the three here, so it must land in the blue plane."""
    rgb = to_rgb(water_bands, stretch="none")
    assert rgb[..., 2].mean() > rgb[..., 1].mean() > rgb[..., 0].mean()


def test_to_rgb_accepts_named_and_lowercase_bands(water_bands):
    named = {
        "red": water_bands["B04"],
        "green": water_bands["B03"],
        "blue": water_bands["B02"],
    }
    lowered = {k.lower(): v for k, v in water_bands.items()}
    assert np.array_equal(to_rgb(named), to_rgb(water_bands))
    assert np.array_equal(to_rgb(lowered), to_rgb(water_bands))


def test_to_rgb_ignores_a_nodata_value(water_bands):
    padded = {k: v.copy() for k, v in water_bands.items()}
    for arr in padded.values():
        arr[:, :32] = 0  # half the frame is nodata
    without = to_rgb(padded)
    with_ignore = to_rgb(padded, ignore_value=0.0)
    # Ignoring nodata restores contrast across the half that carries data.
    assert with_ignore[:, 32:].std() > without[:, 32:].std()


def test_to_rgb_handles_a_constant_band():
    flat = {k: np.full((8, 8), 300, dtype="uint16") for k in ("B04", "B03", "B02")}
    rgb = to_rgb(flat)
    assert rgb.shape == (8, 8, 3)
    assert len(np.unique(rgb)) == 1


def test_to_rgb_handles_nan_and_all_nodata():
    nan_bands = {k: np.full((8, 8), np.nan) for k in ("B04", "B03", "B02")}
    assert to_rgb(nan_bands).max() == 0

    partial = {k: np.full((8, 8), 0.2) for k in ("B04", "B03", "B02")}
    partial["B04"] = partial["B04"].copy()
    partial["B04"][0, 0] = np.nan
    assert np.isfinite(to_rgb(partial)).all()


def test_to_rgb_rejects_mismatched_shapes(water_bands):
    bad = dict(water_bands)
    bad["B02"] = np.zeros((32, 32), dtype="uint16")
    with pytest.raises(ValueError, match="differing shapes"):
        to_rgb(bad)


def test_to_rgb_reports_a_missing_channel(water_bands):
    with pytest.raises(KeyError, match="b02"):
        to_rgb({"B04": water_bands["B04"], "B03": water_bands["B03"]})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stretch": "histogram"}, "stretch must be"),
        ({"gamma": 0.0}, "gamma must be positive"),
        ({"percentiles": (98.0, 2.0)}, "percentiles must satisfy"),
    ],
)
def test_to_rgb_rejects_bad_parameters(water_bands, kwargs, match):
    with pytest.raises(ValueError, match=match):
        to_rgb(water_bands, **kwargs)


def test_to_rgb_round_trips_through_read_bands(scene):
    """The two halves of the module compose: aligned read then render."""
    bands = read_bands(scene, (0, 0, 128, 128), resampling="bilinear")
    rgb = to_rgb(bands)
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8


# --------------------------------------------------------- raster_profile ----


def test_windowed_scene_detection_lands_at_the_right_ground_position(scene):
    """The whole production path: window the scene, detect, georeference.

    A detection is placed at a known pixel inside a known window. Its longitude
    and latitude are checked against the ground coordinate computed straight
    from the raster origin, so an off-by-one in the window offsets or a dropped
    offset in ``window_transform`` would show up as a shifted polygon.
    """
    with rasterio.open(scene["B04"]) as src:
        base_transform, width, height = src.transform, src.width, src.height

    windows = list(windows_for_raster(width, height, 128, 32))
    col_off, row_off, win_w, win_h = windows[3]
    assert (win_w, win_h) == (128, 128)

    tf = window_transform((col_off, row_off, win_w, win_h), base_transform)
    dets = [Detection(bbox=BBox(xmin=10.0, ymin=20.0, xmax=14.0, ymax=26.0), score=0.9)]
    georeference_detections(dets, transform=tf, src_crs=CRS)

    from pyproj import Transformer

    # Independent computation: the four box corners straight from the raster
    # origin. The envelope is taken over all four because meridian convergence
    # means a constant easting is not a constant longitude, so the two western
    # corners differ by a few centimetres.
    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    corners = [
        to_wgs84.transform(ORIGIN_X + (col_off + dc) * 10.0, ORIGIN_Y - (row_off + dr) * 10.0)
        for dc in (10, 14)
        for dr in (20, 26)
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    assert dets[0].geometry.bounds == pytest.approx(
        (min(lons), min(lats), max(lons), max(lats)), abs=1e-12
    )


def test_raster_profile(scene):
    profile = raster_profile(scene["B11"])
    assert profile["width"] == 256
    assert profile["height"] == 256
    assert profile["count"] == 1
    assert profile["dtype"] == "uint16"
    assert profile["crs"] == CRS
    assert profile["gsd"] == pytest.approx((20.0, 20.0))
