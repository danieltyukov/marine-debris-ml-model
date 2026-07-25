"""Tests for mdebris.indices.masks.

The centrepiece is a synthetic scene with a debris patch planted at a known location.
It is built by linearly mixing a plastic endmember into a water background rather than
by writing arbitrary bright numbers, because the mixing fraction is exactly what decides
whether the ``water AND high-FDI`` conjunction can fire at all: a pixel fully covered by
bright material stops looking like water. Encoding that assumption in the fixture keeps
it visible instead of buried in the mask code.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mdebris.indices.masks import (
    CLOUD_SCL_CLASSES,
    DEFAULT_FDI_THRESHOLD,
    SCL_CLASSES,
    candidate_regions,
    cloud_mask_from_scl,
    debris_candidate_mask,
    water_mask,
)
from mdebris.types import BBox

# Open-water surface reflectance, the magnitudes typical of a coastal L2A scene.
WATER = {"green": 0.030, "red": 0.012, "rededge2": 0.012, "nir": 0.010, "swir1": 0.008}

# Floating-plastic endmember: bright across the visible and NIR, and retaining more
# SWIR1 than a water-laden algal mat would.
PLASTIC = {"green": 0.25, "red": 0.25, "rededge2": 0.28, "nir": 0.30, "swir1": 0.12}

# Sub-pixel coverage of the planted patch. At 10 m, a debris windrow rarely fills a
# pixel; a fifth is a generous but realistic fraction.
DEBRIS_FRACTION = 0.20

PATCH_ROWS = slice(10, 15)
PATCH_COLS = slice(20, 24)
PATCH_BBOX = BBox(xmin=20.0, ymin=10.0, xmax=24.0, ymax=15.0)
SCENE_SHAPE = (32, 32)


def synthetic_scene(
    *, fraction: float = DEBRIS_FRACTION, shape: tuple[int, int] = SCENE_SHAPE
) -> dict[str, np.ndarray]:
    """Water background with a linearly mixed debris patch at ``PATCH_ROWS/COLS``.

    Resulting values, for the default fraction: water NDWI 0.50 and FDI 0.0051, patch
    NDWI 0.042 and FDI 0.065. Both are water by NDWI, and only the patch clears any
    sensible FDI threshold.
    """
    bands = {
        name: np.full(shape, value, dtype=np.float32) for name, value in WATER.items()
    }
    for name, array in bands.items():
        array[PATCH_ROWS, PATCH_COLS] = (
            fraction * PLASTIC[name] + (1.0 - fraction) * WATER[name]
        )
    return bands


def clear_scl(shape: tuple[int, int] = SCENE_SHAPE) -> np.ndarray:
    """An SCL raster classifying the whole scene as water (class 6)."""
    return np.full(shape, 6, dtype=np.uint8)


# ---------------------------------------------------------------------------
# water_mask
# ---------------------------------------------------------------------------


def test_water_mask_separates_water_from_land():
    bands = {
        "green": np.array([0.03, 0.05], dtype=np.float32),
        "nir": np.array([0.01, 0.30], dtype=np.float32),
    }
    np.testing.assert_array_equal(water_mask(bands), [True, False])


def test_water_mask_returns_bool_dtype_and_shape():
    mask = water_mask(synthetic_scene())
    assert mask.dtype == np.bool_
    assert mask.shape == SCENE_SHAPE


def test_water_mask_honours_the_threshold():
    # NDWI = (0.03 - 0.01) / 0.04 = 0.5
    bands = {"green": np.float32([0.03]), "nir": np.float32([0.01])}
    assert water_mask(bands, ndwi_threshold=0.4)[0]
    assert not water_mask(bands, ndwi_threshold=0.6)[0]


def test_water_mask_treats_nodata_as_not_water():
    bands = {
        "green": np.array([np.nan, 0.03], dtype=np.float32),
        "nir": np.array([0.01, np.nan], dtype=np.float32),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = water_mask(bands)
    assert not caught
    np.testing.assert_array_equal(mask, [False, False])


def test_water_mask_accepts_esa_band_ids():
    esa = {"B03": np.float32([0.03]), "B08": np.float32([0.01])}
    assert water_mask(esa)[0]


def test_water_mask_reports_the_missing_band():
    with pytest.raises(KeyError, match="nir"):
        water_mask({"B03": np.float32([0.03])})


def test_whole_synthetic_scene_reads_as_water():
    """Including the patch. If this ever fails, the AND cascade cannot work."""
    assert water_mask(synthetic_scene()).all()


# ---------------------------------------------------------------------------
# cloud_mask_from_scl
# ---------------------------------------------------------------------------


def test_scl_class_table_is_complete_and_correctly_labelled():
    assert sorted(SCL_CLASSES) == list(range(12))
    assert SCL_CLASSES[0] == "NO_DATA"
    assert SCL_CLASSES[1] == "SATURATED_OR_DEFECTIVE"
    assert SCL_CLASSES[2] == "CAST_SHADOWS"
    assert SCL_CLASSES[3] == "CLOUD_SHADOWS"
    assert SCL_CLASSES[4] == "VEGETATION"
    assert SCL_CLASSES[5] == "NOT_VEGETATED"
    assert SCL_CLASSES[6] == "WATER"
    assert SCL_CLASSES[7] == "UNCLASSIFIED"
    assert SCL_CLASSES[8] == "CLOUD_MEDIUM_PROBABILITY"
    assert SCL_CLASSES[9] == "CLOUD_HIGH_PROBABILITY"
    assert SCL_CLASSES[10] == "THIN_CIRRUS"
    assert SCL_CLASSES[11] == "SNOW_OR_ICE"


def test_default_cloud_classes_are_the_documented_set():
    assert sorted(CLOUD_SCL_CLASSES) == [0, 1, 3, 8, 9, 10]


def test_cloud_mask_flags_exactly_the_default_classes():
    scl = np.arange(12, dtype=np.uint8)
    expected = [code in CLOUD_SCL_CLASSES for code in range(12)]
    np.testing.assert_array_equal(cloud_mask_from_scl(scl), expected)


def test_cloud_mask_keeps_water_and_cast_shadow_usable_by_default():
    """Class 6 is the target surface; class 2 suppresses rather than fakes detections."""
    np.testing.assert_array_equal(
        cloud_mask_from_scl(np.array([2, 4, 5, 6, 7, 11], dtype=np.uint8)), [False] * 6
    )


def test_cloud_mask_accepts_a_custom_class_set():
    scl = np.array([2, 6, 11], dtype=np.uint8)
    np.testing.assert_array_equal(
        cloud_mask_from_scl(scl, classes={2, 11}), [True, False, True]
    )
    np.testing.assert_array_equal(cloud_mask_from_scl(scl, classes=frozenset()), [False] * 3)


def test_cloud_mask_treats_unknown_codes_as_usable():
    """A future processing baseline adding a class must not break the pipeline."""
    unknown = np.array([12, 200], dtype=np.uint8)
    np.testing.assert_array_equal(cloud_mask_from_scl(unknown), [False, False])


def test_cloud_mask_returns_bool_and_shape():
    mask = cloud_mask_from_scl(clear_scl())
    assert mask.dtype == np.bool_
    assert mask.shape == SCENE_SHAPE
    assert not mask.any()


def test_cloud_mask_handles_a_float_scl_with_nodata():
    """A rasterio read with a float nodata fill produces this; NaN is no-data."""
    scl = np.array([6.0, 9.0, np.nan, 4.0], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = cloud_mask_from_scl(scl)
    assert not caught
    np.testing.assert_array_equal(mask, [False, True, True, False])


# ---------------------------------------------------------------------------
# debris_candidate_mask
# ---------------------------------------------------------------------------


def test_debris_candidate_mask_finds_the_planted_patch():
    mask = debris_candidate_mask(synthetic_scene(), fdi_threshold=0.02, ndwi_threshold=0.0)
    expected = np.zeros(SCENE_SHAPE, dtype=bool)
    expected[PATCH_ROWS, PATCH_COLS] = True
    np.testing.assert_array_equal(mask, expected)


def test_debris_candidate_mask_works_with_the_configured_default_threshold():
    """Water sits at FDI 0.0051 and the patch at 0.065, either side of the 0.006 default."""
    mask = debris_candidate_mask(synthetic_scene())
    assert mask[PATCH_ROWS, PATCH_COLS].all()
    assert mask.sum() == mask[PATCH_ROWS, PATCH_COLS].size
    assert DEFAULT_FDI_THRESHOLD == 0.006


def test_debris_candidate_mask_finds_nothing_in_plain_water():
    water = {name: np.full(SCENE_SHAPE, value, dtype=np.float32) for name, value in WATER.items()}
    assert not debris_candidate_mask(water, fdi_threshold=0.02).any()


def test_debris_candidate_mask_excludes_clouded_pixels():
    """Same scene, same patch, but the SCL calls it cloud. Nothing survives."""
    scl = clear_scl()
    scl[PATCH_ROWS, PATCH_COLS] = 9  # CLOUD_HIGH_PROBABILITY
    mask = debris_candidate_mask(synthetic_scene(), fdi_threshold=0.02, scl=scl)
    assert not mask.any()


def test_debris_candidate_mask_keeps_the_patch_under_a_clear_scl():
    mask = debris_candidate_mask(synthetic_scene(), fdi_threshold=0.02, scl=clear_scl())
    assert mask[PATCH_ROWS, PATCH_COLS].all()


def test_debris_candidate_mask_falls_back_to_the_b04_fdi_variant():
    """A scene without B06 must still be screened rather than skipped entirely."""
    bands = synthetic_scene()
    del bands["rededge2"]
    mask = debris_candidate_mask(bands, fdi_threshold=0.02)
    assert mask[PATCH_ROWS, PATCH_COLS].all()
    assert mask.sum() == mask[PATCH_ROWS, PATCH_COLS].size


def test_debris_candidate_mask_needs_swir():
    with pytest.raises(KeyError, match="rededge2"):
        debris_candidate_mask({"green": np.float32([0.03]), "nir": np.float32([0.01])})


def test_debris_candidate_mask_ignores_nodata_without_warning():
    bands = synthetic_scene()
    for array in bands.values():
        array[0, 0] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = debris_candidate_mask(bands, fdi_threshold=0.02)
    assert not caught
    assert not mask[0, 0]
    assert mask[PATCH_ROWS, PATCH_COLS].all()


def test_debris_candidate_mask_returns_bool_dtype():
    mask = debris_candidate_mask(synthetic_scene(), fdi_threshold=0.02)
    assert mask.dtype == np.bool_
    assert mask.shape == SCENE_SHAPE


def test_a_fully_covered_pixel_fails_the_water_test():
    """Documents the known limit of an AND cascade rather than pretending it away."""
    saturated = synthetic_scene(fraction=1.0)
    assert not water_mask(saturated)[PATCH_ROWS, PATCH_COLS].any()
    assert not debris_candidate_mask(saturated, fdi_threshold=0.02).any()


# ---------------------------------------------------------------------------
# candidate_regions
# ---------------------------------------------------------------------------


def test_candidate_regions_recovers_the_planted_box():
    mask = debris_candidate_mask(synthetic_scene(), fdi_threshold=0.02)
    boxes = candidate_regions(mask)
    assert len(boxes) == 1
    assert boxes[0].as_xyxy() == PATCH_BBOX.as_xyxy()
    assert boxes[0].iou(PATCH_BBOX) == pytest.approx(1.0)


def test_candidate_regions_box_area_matches_the_patch():
    boxes = candidate_regions(debris_candidate_mask(synthetic_scene(), fdi_threshold=0.02))
    assert boxes[0].area == pytest.approx(5 * 4)


def test_candidate_regions_on_an_empty_mask():
    assert candidate_regions(np.zeros((8, 8), dtype=bool)) == []


def test_candidate_regions_drops_components_below_min_pixels():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:5, 2:5] = True  # 9 pixels
    mask[10, 10] = True  # 1 pixel
    assert len(candidate_regions(mask, min_pixels=4)) == 1
    assert len(candidate_regions(mask, min_pixels=1)) == 2
    assert candidate_regions(mask, min_pixels=10) == []


def test_candidate_regions_uses_half_open_pixel_edges():
    """A single pixel is a unit box, so a one-pixel component has area 1, not 0."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 5] = True
    boxes = candidate_regions(mask, min_pixels=1)
    assert boxes == [BBox(xmin=5.0, ymin=3.0, xmax=6.0, ymax=4.0)]
    assert boxes[0].area == 1.0


def test_candidate_regions_defaults_to_eight_connectivity():
    """Debris windrows are thin and diagonal; 4-connectivity shatters them."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    assert len(candidate_regions(mask, min_pixels=1, connectivity=2)) == 1
    assert len(candidate_regions(mask, min_pixels=1, connectivity=1)) == 2


def test_candidate_regions_separates_disjoint_patches():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:4, 2:4] = True
    mask[12:15, 14:18] = True
    boxes = candidate_regions(mask, min_pixels=4)
    assert len(boxes) == 2
    # Raster order: the top-left component comes first.
    assert boxes[0].as_xyxy() == (2.0, 2.0, 4.0, 4.0)
    assert boxes[1].as_xyxy() == (14.0, 12.0, 18.0, 15.0)


def test_candidate_regions_accepts_a_non_boolean_mask():
    boxes = candidate_regions(np.eye(4, dtype=np.uint8), min_pixels=1)
    assert len(boxes) == 1


def test_candidate_regions_rejects_a_non_2d_mask():
    with pytest.raises(ValueError, match="2D"):
        candidate_regions(np.zeros((2, 3, 4), dtype=bool))


def test_candidate_regions_rejects_min_pixels_below_one():
    with pytest.raises(ValueError, match="min_pixels"):
        candidate_regions(np.zeros((4, 4), dtype=bool), min_pixels=0)


def test_full_prescreen_round_trip():
    """The cascade decision end to end: bands in, boxes worth a detector call out."""
    scene = synthetic_scene()
    boxes = candidate_regions(
        debris_candidate_mask(scene, fdi_threshold=0.02, scl=clear_scl()), min_pixels=4
    )
    assert len(boxes) == 1
    centre_x, centre_y = boxes[0].centroid
    assert PATCH_COLS.start <= centre_x <= PATCH_COLS.stop
    assert PATCH_ROWS.start <= centre_y <= PATCH_ROWS.stop
