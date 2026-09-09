"""Tests for the sensor band ablation.

The ablation trains the same classifier on a subset of bands. Two things have to be
right for any number it produces to mean anything: the columns it hides must be the
columns of the bands it claims to hide, and an index that needs a hidden band must be
hidden with it. Everything else is bookkeeping around scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdebris.eval.ablation import (
    SENSOR_SCENARIOS,
    binary_metrics,
    index_threshold_baseline,
    mask_features,
    select_threshold,
    surviving_features,
)
from mdebris.models.spectral import FEATURE_BANDS, feature_names


class TestScenarios:
    def test_every_scenario_names_only_real_bands(self):
        for name, bands in SENSOR_SCENARIOS.items():
            unknown = set(bands) - set(FEATURE_BANDS)
            assert not unknown, f"{name} lists bands that do not exist: {unknown}"

    def test_baseline_keeps_every_band(self):
        assert tuple(SENSOR_SCENARIOS["sentinel2"]) == FEATURE_BANDS

    def test_dove4_is_blue_green_red_nir(self):
        assert tuple(SENSOR_SCENARIOS["dove4"]) == ("B02", "B03", "B04", "B08")

    def test_indices_only_keeps_no_band(self):
        assert SENSOR_SCENARIOS["indices_only"] == ()


class TestSurvivingFeatures:
    def test_full_band_set_keeps_every_feature(self):
        assert surviving_features(FEATURE_BANDS) == feature_names()

    def test_dropping_swir_drops_the_indices_that_need_it(self):
        kept = [b for b in FEATURE_BANDS if b not in ("B11", "B12")]
        survivors = surviving_features(kept)
        for gone in ("B11", "B12", "FDI", "FAI", "MNDWI"):
            assert gone not in survivors
        for stays in ("NDVI", "NDWI", "PI", "KNDVI"):
            assert stays in survivors

    def test_dropping_b06_drops_fdi_only(self):
        kept = [b for b in FEATURE_BANDS if b != "B06"]
        survivors = surviving_features(kept)
        assert "FDI" not in survivors
        assert "FAI" in survivors

    def test_dropping_nir_drops_every_nir_index(self):
        kept = [b for b in FEATURE_BANDS if b != "B08"]
        survivors = surviving_features(kept)
        for gone in ("FDI", "FAI", "NDVI", "NDWI", "PI", "KNDVI"):
            assert gone not in survivors
        assert "MNDWI" in survivors

    def test_indices_only_keeps_all_seven_indices_and_no_band(self):
        # The indices need bands to be *computed*, but as a feature set they are
        # the physical features alone. Passing the sentinel "indices_only" spelling
        # keeps the seven index columns and hides every raw band.
        survivors = surviving_features((), indices_from=FEATURE_BANDS)
        assert survivors == ["FDI", "FAI", "NDVI", "NDWI", "PI", "KNDVI", "MNDWI"]

    def test_order_matches_feature_names(self):
        kept = ("B04", "B08", "B11")
        survivors = surviving_features(kept)
        order = feature_names()
        assert survivors == [f for f in order if f in survivors]


class TestMaskFeatures:
    def test_hidden_columns_become_nan_and_others_are_untouched(self):
        rng = np.random.default_rng(0)
        x = rng.random((5, len(feature_names()))).astype(np.float32)
        masked = mask_features(x, ("B02", "B03", "B04", "B08"))
        names = feature_names()
        for j, name in enumerate(names):
            if name in ("B02", "B03", "B04", "B08", "NDVI", "NDWI", "PI", "KNDVI"):
                assert np.array_equal(masked[:, j], x[:, j]), name
            else:
                assert np.isnan(masked[:, j]).all(), name

    def test_input_is_not_modified(self):
        x = np.ones((3, len(feature_names())), dtype=np.float32)
        mask_features(x, ("B04",))
        assert not np.isnan(x).any()

    def test_wrong_width_is_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            mask_features(np.ones((3, 4), dtype=np.float32), ("B04",))


class TestThresholdSelection:
    def test_picks_the_threshold_with_the_best_f1(self):
        truth = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        scores = np.array([0.1, 0.2, 0.6, 0.55, 0.8, 0.9])
        # Perfect separation exists only between 0.2 and 0.55 (0.6 is a negative
        # above 0.55), so best F1 is 1.0 only for thresholds in (0.6, 0.8]? No: the
        # negative at 0.6 sits above the positive at 0.55, so no threshold is
        # perfect. Best is 3 TP, 1 FP at t <= 0.55 (F1 0.857) versus 2 TP, 0 FP at
        # t in (0.6, 0.8] (F1 0.8).
        t = select_threshold(truth, scores)
        assert 0.2 < t <= 0.55
        m = binary_metrics(truth, scores, t)
        assert m["tp"] == 3 and m["fp"] == 1
        assert m["f1"] == pytest.approx(6 / 7)

    def test_metrics_report_counts_and_rates(self):
        truth = np.array([1, 1, 0, 0], dtype=bool)
        scores = np.array([0.9, 0.4, 0.6, 0.1])
        m = binary_metrics(truth, scores, 0.5)
        assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["threshold"] == pytest.approx(0.5)

    def test_no_positives_yields_zero_not_nan(self):
        truth = np.zeros(4, dtype=bool)
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        m = binary_metrics(truth, scores, 0.25)
        assert m["recall"] == 0.0 and m["f1"] == 0.0


class TestIndexBaseline:
    def test_threshold_chosen_on_validation_and_scored_on_test(self):
        # Validation says t in (0.3, 0.7] separates; the test set carries one
        # positive below that band, which the baseline must miss.
        val_truth = np.array([0, 0, 1, 1], dtype=bool)
        val_index = np.array([0.1, 0.3, 0.7, 0.9])
        test_truth = np.array([0, 1, 1], dtype=bool)
        test_index = np.array([0.2, 0.5, 0.8])
        result = index_threshold_baseline(val_truth, val_index, test_truth, test_index)
        assert 0.3 < result["threshold"] <= 0.7
        assert result["tp"] == 1 and result["fn"] == 1 and result["fp"] == 0

    def test_nan_index_values_never_count_as_positive(self):
        val_truth = np.array([0, 1], dtype=bool)
        val_index = np.array([0.1, 0.9])
        test_truth = np.array([1, 0], dtype=bool)
        test_index = np.array([np.nan, np.nan])
        result = index_threshold_baseline(val_truth, val_index, test_truth, test_index)
        assert result["tp"] == 0 and result["fp"] == 0


class TestSelectFeatures:
    def test_returns_only_the_surviving_columns_in_order(self):
        from mdebris.eval.ablation import select_features

        rng = np.random.default_rng(0)
        x = rng.random((4, len(feature_names()))).astype(np.float32)
        out, names = select_features(x, ("B02", "B03", "B04", "B08"))
        assert names == ["B02", "B03", "B04", "B08", "NDVI", "NDWI", "PI", "KNDVI"]
        cols = [feature_names().index(n) for n in names]
        assert np.array_equal(out, x[:, cols])

    def test_indices_only_has_seven_columns(self):
        from mdebris.eval.ablation import select_features

        x = np.ones((2, len(feature_names())), dtype=np.float32)
        out, names = select_features(x, (), indices_from=FEATURE_BANDS)
        assert out.shape == (2, 7) and len(names) == 7


class TestClusterBootstrap:
    def test_identical_clusters_give_a_zero_width_interval(self):
        from mdebris.eval.ablation import cluster_bootstrap_f1

        # Four patches, each with 2 TP, 1 FP, 1 FN at threshold 0.5.
        truth = np.array([1, 1, 0, 1] * 4, dtype=bool)
        scores = np.array([0.9, 0.8, 0.7, 0.2] * 4)
        groups = np.repeat(np.arange(4), 4)
        ci = cluster_bootstrap_f1(truth, scores, 0.5, groups, n_boot=50)
        expected = 2 * 2 / (2 * 2 + 1 + 1)
        assert ci["f1"] == pytest.approx(expected)
        assert ci["low"] == pytest.approx(expected) and ci["high"] == pytest.approx(expected)

    def test_interval_brackets_the_point_estimate_on_random_data(self):
        from mdebris.eval.ablation import cluster_bootstrap_f1

        rng = np.random.default_rng(3)
        n = 5000
        truth = rng.random(n) < 0.1
        scores = np.clip(truth * 0.5 + rng.random(n) * 0.6, 0, 1)
        groups = rng.integers(0, 40, size=n)
        ci = cluster_bootstrap_f1(truth, scores, 0.6, groups, n_boot=200, seed=1)
        assert ci["low"] <= ci["f1"] <= ci["high"]
        assert ci["high"] - ci["low"] > 0.0
        assert ci["n_groups"] == 40

    def test_group_length_mismatch_is_rejected(self):
        from mdebris.eval.ablation import cluster_bootstrap_f1

        with pytest.raises(ValueError, match="groups"):
            cluster_bootstrap_f1(np.array([True, False]), np.array([0.9, 0.1]), 0.5, np.array([0]))
