"""Tests for probability calibration and the abstention band.

Synthetic data with a known relationship between score and outcome is the only way
to test a calibration metric: on real data there is no ground truth for "the model
is calibrated", only the metric itself.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mdebris.eval.calibration import (
    IsotonicCalibrator,
    abstention_band,
    brier_score,
    expected_calibration_error,
    reliability_table,
)


def _calibrated(n: int = 20_000, seed: int = 0):
    """Scores drawn uniformly, outcomes Bernoulli(score): calibrated by construction."""
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    truth = rng.random(n) < scores
    return truth, scores


class TestReliabilityTable:
    def test_has_one_row_per_bin_with_counts_that_sum_to_n(self):
        truth, scores = _calibrated(1000)
        rows = reliability_table(truth, scores, n_bins=10)
        assert len(rows) == 10
        assert sum(r["count"] for r in rows) == 1000
        assert rows[0]["lower"] == 0.0 and rows[-1]["upper"] == 1.0

    def test_calibrated_scores_track_the_diagonal(self):
        truth, scores = _calibrated()
        for row in reliability_table(truth, scores, n_bins=10):
            assert row["fraction_positive"] == pytest.approx(row["mean_score"], abs=0.03)

    def test_empty_bin_reports_zero_count_and_nan_rates(self):
        truth = np.array([True, False])
        scores = np.array([0.95, 0.05])
        rows = reliability_table(truth, scores, n_bins=10)
        assert rows[5]["count"] == 0
        assert np.isnan(rows[5]["fraction_positive"])

    def test_score_of_exactly_one_lands_in_the_top_bin(self):
        rows = reliability_table(np.array([True]), np.array([1.0]), n_bins=10)
        assert rows[-1]["count"] == 1


class TestExpectedCalibrationError:
    def test_near_zero_when_calibrated(self):
        truth, scores = _calibrated()
        assert expected_calibration_error(truth, scores) < 0.02

    def test_large_when_scores_are_confidently_wrong(self):
        truth = np.zeros(1000, dtype=bool)
        scores = np.full(1000, 0.9)
        assert expected_calibration_error(truth, scores) == pytest.approx(0.9)

    def test_rejects_scores_outside_unit_interval(self):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            expected_calibration_error(np.array([True]), np.array([1.5]))


class TestBrier:
    def test_perfect_scores_give_zero(self):
        assert brier_score(np.array([True, False]), np.array([1.0, 0.0])) == 0.0

    def test_worst_scores_give_one(self):
        assert brier_score(np.array([True, False]), np.array([0.0, 1.0])) == 1.0


class TestIsotonicCalibrator:
    def test_fixes_a_systematically_overconfident_model(self):
        rng = np.random.default_rng(1)
        raw = rng.random(20_000)
        # The true probability is raw squared: the model overstates every score.
        truth = rng.random(20_000) < raw**2
        before = expected_calibration_error(truth, raw)
        cal = IsotonicCalibrator().fit(raw, truth)
        after = expected_calibration_error(truth, cal.apply(raw))
        assert before > 0.1
        assert after < before / 4

    def test_output_is_monotone_in_the_input(self):
        truth, scores = _calibrated(5000)
        cal = IsotonicCalibrator().fit(scores, truth)
        grid = np.linspace(0, 1, 101)
        out = cal.apply(grid)
        assert np.all(np.diff(out) >= -1e-12)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_round_trips_through_json(self, tmp_path):
        truth, scores = _calibrated(5000)
        cal = IsotonicCalibrator().fit(scores, truth)
        path = tmp_path / "cal.json"
        cal.save(path)
        loaded = IsotonicCalibrator.load(path)
        grid = np.linspace(0, 1, 50)
        assert np.allclose(loaded.apply(grid), cal.apply(grid))
        blob = json.loads(path.read_text())
        assert blob["kind"] == "isotonic" and "x" in blob and "y" in blob

    def test_apply_before_fit_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit"):
            IsotonicCalibrator().apply(np.array([0.5]))


class TestAbstentionBand:
    def test_edges_come_from_precision_and_recall_targets(self):
        # Ten positives at 0.60..0.90 plus one at 0.05; negatives at 0.1..0.5 plus
        # three sitting among the positives at 0.62, 0.65 and 0.70.
        truth = np.array([True] * 11 + [False] * 8)
        positives = [0.60, 0.63, 0.66, 0.70, 0.73, 0.76, 0.80, 0.83, 0.86, 0.90, 0.05]
        negatives = [0.10, 0.20, 0.30, 0.40, 0.50, 0.62, 0.65, 0.70]
        scores = np.array([*positives, *negatives])
        band = abstention_band(truth, scores, high_precision=0.9, miss_rate=0.10)
        # Cut at 0.73: 6 TP, 0 FP. Cut at 0.70: 7 TP, 1 FP (0.875). Cut at 0.66:
        # 8 TP, 1 FP (0.889). Cut at 0.60: 10 TP, 3 FP. Only 0.73 and above reach
        # 90%, and 0.73 has the most recall among them, so it is the upper edge.
        assert band["high"] == pytest.approx(0.73)
        # Lower edge: keep at least 90% of the 11 positives above it, i.e. 10 of 11.
        # The positive at 0.05 may be dropped; the cut is the lowest score among
        # the 10 retained positives, 0.60.
        assert band["low"] == pytest.approx(0.60)
        assert band["low"] <= band["high"]

    def test_falls_back_when_precision_target_is_unreachable(self):
        truth = np.array([True, False, True, False])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        band = abstention_band(truth, scores, high_precision=0.9, miss_rate=0.1)
        assert band["high"] is None
        assert band["low"] == pytest.approx(0.5)

    def test_band_membership_counts(self):
        truth = np.array([True, True, False, False, False])
        scores = np.array([0.9, 0.5, 0.5, 0.2, 0.95])
        band = abstention_band(truth, scores, high_precision=0.5, miss_rate=0.0)
        assert band["low"] == pytest.approx(0.5)
        # Everything with score in [low, high) is uncertain.
        assert band["n_uncertain"] == int(((scores >= band["low"]) & (scores < band["high"])).sum())
        assert 0.0 <= band["fraction_uncertain"] <= 1.0
