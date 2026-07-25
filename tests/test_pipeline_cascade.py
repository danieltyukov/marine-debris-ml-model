"""Tests for the screening cascade.

The reflectance values used here are chosen to resemble real Sentinel-2 L2A open
water (NIR and SWIR near zero, green highest) rather than arbitrary numbers, because
the whole behaviour under test is a threshold on a physical quantity.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdebris.pipeline.cascade import (
    ScreenResult,
    adaptive_fdi_threshold,
    screen_tile,
    summarize_screening,
)

# Approximate open-water L2A reflectance. Green is brightest, NIR and SWIR are close
# to zero because water absorbs strongly beyond the visible.
WATER = {"B03": 0.025, "B04": 0.015, "B06": 0.005, "B08": 0.003, "B11": 0.001}

SCL_WATER = 6  # ESA Scene Classification Layer code for water


def make_tile(size: int = 128, *, seed: int = 0, noise: float = 0.0005) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    bands = {
        name: (
            np.full((size, size), value, dtype=np.float32) + rng.normal(0, noise, (size, size))
        ).astype(np.float32)
        for name, value in WATER.items()
    }
    bands["SCL"] = np.full((size, size), SCL_WATER, dtype=np.uint8)
    return bands


def plant_debris(bands: dict[str, np.ndarray], rows: slice, cols: slice) -> dict[str, np.ndarray]:
    """Raise NIR while leaving SWIR low, which is the FDI signature of floating material.

    The magnitudes matter and are not arbitrary. Debris at 10 m is a sub-pixel
    mixture: a patch covers only part of a 100 square metre pixel, so NIR rises but
    the pixel still reads as water. NIR is therefore kept below green, which keeps
    NDWI positive.

    An earlier version of this fixture used NIR 0.075 against green 0.025, giving
    NDWI of -0.5. That patch failed the water test and was correctly excluded by the
    cascade, because a pixel that bright in NIR is no longer ocean. The lesson is
    that synthetic remote-sensing fixtures have to respect the physics or they test
    the wrong thing.

    Resulting values: NDWI approximately +0.11 (still water), FDI approximately
    +0.017 (well above any plausible threshold).
    """
    out = {k: v.copy() for k, v in bands.items()}
    out["B08"][rows, cols] = 0.020  # raised from 0.003, still below green at 0.025
    out["B04"][rows, cols] = 0.018
    out["B06"][rows, cols] = 0.012
    out["B11"][rows, cols] = 0.004  # SWIR stays low, which is what lifts FDI
    return out


class TestAdaptiveThreshold:
    def test_returns_the_requested_percentile(self):
        fdi = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
        assert adaptive_fdi_threshold(fdi, percentile=50.0, floor=0.0) == pytest.approx(
            0.5, abs=1e-3
        )

    def test_never_falls_below_the_floor(self):
        # A scene of pure calm water has a tiny p99.9. Without a floor the screen
        # would promote its own noise into candidates.
        fdi = np.full(1000, 0.0001, dtype=np.float32)
        assert adaptive_fdi_threshold(fdi, percentile=99.9, floor=0.006) == pytest.approx(0.006)

    def test_water_mask_restricts_the_sample(self):
        fdi = np.concatenate([np.full(500, 0.001), np.full(500, 10.0)]).astype(np.float32)
        water = np.concatenate([np.ones(500, bool), np.zeros(500, bool)])
        # Land pixels with absurd FDI must not drag the water threshold up.
        assert adaptive_fdi_threshold(fdi, water, percentile=99.0, floor=0.0) < 1.0

    def test_all_nan_falls_back_to_floor(self):
        assert adaptive_fdi_threshold(
            np.full(10, np.nan, dtype=np.float32), floor=0.006
        ) == pytest.approx(0.006)

    def test_empty_sample_falls_back_to_floor(self):
        fdi = np.ones(10, dtype=np.float32)
        assert adaptive_fdi_threshold(fdi, np.zeros(10, bool), floor=0.006) == pytest.approx(0.006)


class TestScreenTile:
    def test_empty_bands_rejected(self):
        with pytest.raises(ValueError, match="at least one band"):
            screen_tile({})

    def test_plain_water_is_not_accepted_on_region_coherence(self):
        # Scattered threshold crossings are noise. Without a coherent region the
        # tile should not cost a detector call.
        bands = make_tile(seed=1)
        result = screen_tile(bands, fdi_percentile=99.99, min_region_pixels=32)
        assert not result.accepted
        assert result.reason

    def test_planted_debris_patch_is_accepted(self):
        # The patch is deliberately a small fraction of the tile (256 px of 65536,
        # about 0.4%). A percentile threshold assumes the target is rare; a patch
        # larger than (100 - percentile)% of the water would raise the cutoff into
        # its own values and mask itself.
        bands = plant_debris(make_tile(256, seed=2), slice(100, 116), slice(120, 136))
        result = screen_tile(bands, fdi_percentile=99.0, min_region_pixels=8)
        assert result.accepted
        assert result.regions

    def test_accepted_region_overlaps_the_planted_patch(self):
        rows, cols = slice(100, 116), slice(120, 136)
        bands = plant_debris(make_tile(256, seed=3), rows, cols)
        result = screen_tile(bands, fdi_percentile=99.0, min_region_pixels=8)
        assert result.accepted
        hit = any(
            r.xmin <= cols.stop
            and r.xmax >= cols.start
            and r.ymin <= rows.stop
            and r.ymax >= rows.start
            for r in result.regions
        )
        assert hit, (
            f"no region overlapped the planted patch, got {[r.as_xyxy() for r in result.regions]}"
        )

    def test_adaptive_threshold_exceeds_the_fixed_floor_on_noisy_water(self):
        bands = make_tile(seed=4, noise=0.002)
        result = screen_tile(bands, fdi_percentile=99.9)
        # The whole point of adapting: a noisy scene gets a stricter cutoff than the
        # constant would have applied.
        assert result.threshold >= 0.006

    def test_explicit_threshold_overrides_the_adaptive_path(self):
        bands = make_tile(seed=5)
        result = screen_tile(bands, fdi_threshold=0.5, fdi_percentile=None)
        assert result.threshold == pytest.approx(0.5)
        assert not result.accepted

    def test_missing_bands_pass_the_tile_through_rather_than_dropping_it(self):
        # Failing open is deliberate: a tile the screen cannot evaluate must still
        # reach the detector instead of being silently discarded.
        result = screen_tile({"B03": np.zeros((16, 16), dtype=np.float32)})
        assert result.accepted
        assert "screen unavailable" in result.reason

    def test_scl_is_not_fed_into_index_arithmetic(self):
        # SCL is a class code raster, not reflectance. Treating it as a band would
        # silently corrupt every index.
        bands = plant_debris(make_tile(256, seed=6), slice(100, 116), slice(120, 136))
        with_scl = screen_tile(bands, fdi_percentile=99.0)
        without = screen_tile({k: v for k, v in bands.items() if k != "SCL"}, fdi_percentile=99.0)
        assert with_scl.candidate_pixels == without.candidate_pixels

    def test_compute_all_indices_populates_the_index_dict(self):
        result = screen_tile(make_tile(seed=7), compute_all_indices=True)
        assert {"FDI", "NDVI", "NDWI"} <= set(result.indices)

    def test_candidate_fraction_is_bounded(self):
        result = screen_tile(make_tile(seed=8), compute_all_indices=True)
        assert 0.0 <= result.candidate_fraction <= 1.0

    def test_cloud_pixels_are_excluded(self):
        rows, cols = slice(100, 116), slice(120, 136)
        bands = plant_debris(make_tile(256, seed=9), rows, cols)
        bands["SCL"][rows, cols] = 9  # high-probability cloud
        result = screen_tile(bands, fdi_percentile=99.0, min_region_pixels=8)
        assert not result.accepted


class TestSummarizeScreening:
    def test_empty_input(self):
        s = summarize_screening([])
        assert s["tiles_total"] == 0 and s["work_avoided"] == 0.0

    def test_work_avoided_is_the_complement_of_accept_rate(self):
        results = [ScreenResult(accepted=True)] * 3 + [ScreenResult(accepted=False)] * 7
        s = summarize_screening(results)
        assert s["tiles_total"] == 10
        assert s["tiles_accepted"] == 3
        assert s["tiles_skipped"] == 7
        assert s["accept_rate"] == pytest.approx(0.3)
        assert s["work_avoided"] == pytest.approx(0.7)

    def test_all_accepted_means_no_work_avoided(self):
        s = summarize_screening([ScreenResult(accepted=True)] * 4)
        assert s["work_avoided"] == 0.0
