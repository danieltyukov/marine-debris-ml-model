"""Tests for turning a season of per-pass segment rows into gap statistics.

The rows are what ``append_history`` writes: one per segment per pass. The
statistics are the ones a monitoring proposal has to state up front: how often a
beach could be seen at all, and how long it went unseen.
"""

from __future__ import annotations

import pytest

from mdebris.coastal.season import SegmentSeason, summarize_history


def _row(seg, date, obs, dets=0, front=0.0, cloud=0.0, scene="S"):
    return {
        "segment_id": seg,
        "name": seg.title(),
        "observed_on": date,
        "observability": obs,
        "detection_count": dets,
        "affected_front_m": front,
        "cloud_fraction": cloud,
        "coverage": 0.0,
        "scene_id": scene,
    }


class TestSummarizeHistory:
    def test_counts_passes_by_observability(self):
        rows = [
            _row("a", "2025-01-01", "observed"),
            _row("a", "2025-01-06", "partial"),
            _row("a", "2025-01-11", "blind"),
            _row("a", "2025-01-16", "blind"),
        ]
        (season,) = summarize_history(rows)
        assert isinstance(season, SegmentSeason)
        assert season.segment_id == "a"
        assert (season.n_passes, season.n_observed, season.n_partial, season.n_blind) == (
            4,
            1,
            1,
            2,
        )
        assert season.usable_fraction == pytest.approx(0.5)

    def test_longest_gap_is_measured_between_usable_passes(self):
        rows = [
            _row("a", "2025-01-01", "observed"),
            _row("a", "2025-01-06", "blind"),
            _row("a", "2025-01-11", "blind"),
            _row("a", "2025-01-16", "blind"),
            _row("a", "2025-01-21", "partial"),
            _row("a", "2025-01-26", "observed"),
        ]
        (season,) = summarize_history(rows)
        assert season.longest_gap_days == 20
        assert season.median_gap_days == pytest.approx(12.5)

    def test_rows_are_sorted_by_date_before_gaps_are_measured(self):
        rows = [
            _row("a", "2025-01-26", "observed"),
            _row("a", "2025-01-01", "observed"),
            _row("a", "2025-01-11", "observed"),
        ]
        (season,) = summarize_history(rows)
        assert season.longest_gap_days == 15

    def test_single_usable_pass_has_no_gap(self):
        rows = [_row("a", "2025-01-01", "observed"), _row("a", "2025-01-06", "blind")]
        (season,) = summarize_history(rows)
        assert season.longest_gap_days is None
        assert season.median_gap_days is None

    def test_detections_only_count_on_usable_passes(self):
        rows = [
            _row("a", "2025-03-01", "observed", dets=3, front=420.0),
            _row("a", "2025-03-06", "blind", dets=1, front=50.0),
            _row("a", "2025-03-11", "partial", dets=2, front=150.0),
            _row("a", "2025-03-16", "observed", dets=0),
        ]
        (season,) = summarize_history(rows)
        assert season.detection_dates == ["2025-03-01", "2025-03-11"]
        assert season.max_affected_front_m == pytest.approx(420.0)
        assert season.n_detection_passes == 2

    def test_segments_come_back_in_first_seen_order(self):
        rows = [
            _row("b", "2025-01-01", "observed"),
            _row("a", "2025-01-01", "observed"),
            _row("b", "2025-01-06", "observed"),
        ]
        seasons = summarize_history(rows)
        assert [s.segment_id for s in seasons] == ["b", "a"]

    def test_duplicate_pass_of_the_same_date_counts_once(self):
        # append_history never deduplicates; a rerun appends again. The summary
        # must not count that as two looks at the beach.
        rows = [
            _row("a", "2025-01-01", "observed"),
            _row("a", "2025-01-01", "observed"),
            _row("a", "2025-01-06", "observed"),
        ]
        (season,) = summarize_history(rows)
        assert season.n_passes == 2

    def test_empty_history_is_an_error(self):
        with pytest.raises(ValueError, match="no rows"):
            summarize_history([])

    def test_to_row_is_flat_and_json_safe(self):
        rows = [_row("a", "2025-01-01", "observed", dets=1, front=10.0)]
        (season,) = summarize_history(rows)
        flat = season.to_row()
        assert flat["segment_id"] == "a"
        assert flat["n_passes"] == 1
        assert flat["longest_gap_days"] is None
