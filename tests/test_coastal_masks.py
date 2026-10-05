"""The masks a season run applies: land, mangroves and the vegetation check.

The mangrove mask exists because the OpenStreetMap coastline at Lac Bay runs round the
landward edge of the mangrove forest, so the canopy sat on the "water side" and was
classified. These tests pin which OpenStreetMap wetland counts as vegetation (seagrass
must not), how the polygons land on a raster, and the per-pixel vegetation rule.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Point, box

from mdebris.coastal.islands import load_island
from mdebris.coastal.masks import (
    is_vegetated_wetland,
    load_polygons,
    near,
    persistent_vegetation,
    polygon_mask,
    vegetated,
)

CRS = "EPSG:32619"
# A 50 x 50 grid of 10 m pixels, north-up, near Bonaire.
ORIGIN_X, ORIGIN_Y = 500_000.0, 1_340_500.0
TRANSFORM = Affine(10.0, 0.0, ORIGIN_X, 0.0, -10.0, ORIGIN_Y)
SHAPE = (50, 50)


def _lonlat_box(x0, y0, x1, y1):
    """A UTM 19N rectangle (metres) as an EPSG:4326 polygon."""
    from pyproj import Transformer
    from shapely.ops import transform

    back = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
    return transform(back, box(x0, y0, x1, y1))


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"natural": "wetland", "wetland": "mangrove"}, True),
        ({"natural": "wetland", "wetland": "swamp"}, True),
        ({"natural": "wetland", "wetland": "marsh;mangrove"}, True),
        ({"natural": "wetland", "wetland": "seagrass"}, False),
        ({"natural": "wetland", "wetland": "seagrass_bed"}, False),
        ({"natural": "wetland", "wetland": "tidalflat"}, False),
        ({"natural": "wetland", "wetland": "saltern"}, False),
        ({"natural": "wetland"}, False),
        ({"natural": "wetland", "wetland": "marsh", "name": "Seagrass meadow"}, False),
        ({"natural": "scrub", "wetland": "mangrove"}, False),
    ],
)
def test_only_vegetation_above_the_water_is_masked(tags, expected):
    assert is_vegetated_wetland(tags) is expected


def test_polygon_mask_lands_on_the_grid_and_grows_with_the_buffer():
    # 100 m square in the middle: 10 x 10 pixels.
    poly = _lonlat_box(ORIGIN_X + 200, ORIGIN_Y - 300, ORIGIN_X + 300, ORIGIN_Y - 200)
    tight = polygon_mask([poly], TRANSFORM, CRS, SHAPE, all_touched=False)
    assert 90 <= tight.sum() <= 110
    assert tight[25, 25] and not tight[0, 0]
    grown = polygon_mask([poly], TRANSFORM, CRS, SHAPE, buffer_m=20.0)
    assert grown.sum() > tight.sum() + 4 * 10 * 2
    assert (grown | tight).sum() == grown.sum()


def test_no_polygons_masks_nothing():
    assert not polygon_mask([], TRANSFORM, CRS, SHAPE, buffer_m=20.0).any()


def test_vegetated_needs_both_a_red_edge_and_a_bright_nir():
    red = np.array([0.02, 0.005, 0.05, 0.029], dtype=np.float32)
    nir = np.array([0.30, 0.05, 0.06, 0.09], dtype=np.float32)
    # canopy; dark water with a high ratio; bright but flat; just over both lines
    assert vegetated({"B04": red, "B08": nir}).tolist() == [True, False, False, True]


def test_vegetated_treats_zero_reflectance_as_not_vegetated():
    zeros = np.zeros(3, dtype=np.float32)
    assert not vegetated({"B04": zeros, "B08": zeros}).any()


def test_persistent_vegetation_needs_enough_looks_and_a_high_share():
    seen = np.array([10, 10, 9, 20, 0])
    leafy = np.array([9, 8, 9, 18, 0])
    assert persistent_vegetation(leafy, seen).tolist() == [True, False, False, True, False]


def test_near_grows_by_whole_pixels():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    assert near(mask, 0).sum() == 1
    assert near(mask, 1).sum() == 9
    assert near(mask, 2).sum() == 25


def test_bonaire_mask_covers_the_lac_mangroves_and_not_the_open_lagoon():
    island = load_island("bonaire")
    polygons = load_polygons(island.mangroves_path)
    assert len(polygons) > 50
    from shapely.ops import unary_union

    union = unary_union(polygons)
    # The open lagoon water where the 17 June 2025 candidate sat, about 400 m from
    # any mangrove, must stay classifiable even after the 20 m buffer.
    candidate = Point(-68.2256, 12.1023)
    assert not union.contains(candidate)
    assert union.distance(candidate) > 0.002  # degrees, roughly 200 m
    # And the mask is on the bay's north shore, inside the Lac Bay segment's zone.
    lac = box(-68.255, 12.095, -68.205, 12.130)
    assert union.intersection(lac).area > 0
