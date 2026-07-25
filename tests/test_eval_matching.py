"""Tests for greedy detection matching.

The matching rule is the definition every metric downstream inherits, so these
tests pin the rule itself: who gets first pick, what a tie does, and what happens
when one side of the comparison is empty.
"""

from __future__ import annotations

import random

import pytest

from mdebris.eval.matching import match_detections
from mdebris.types import BBox, Detection, SurfaceClass


def det(
    x: float,
    y: float = 0.0,
    *,
    size: float = 10.0,
    score: float = 0.9,
    label: SurfaceClass = SurfaceClass.DEBRIS,
) -> Detection:
    """A square detection of side ``size`` with its top-left corner at ``(x, y)``."""
    return Detection(bbox=BBox(x, y, x + size, y + size), score=score, label=label)


def shifted(x: float, *, dx: float, score: float = 0.9) -> Detection:
    """A 10x10 box offset from ``x`` by ``dx``, for building a known IoU."""
    return det(x + dx, score=score)


# ---------------------------------------------------------------------------
# the basic rule
# ---------------------------------------------------------------------------


def test_exact_overlap_is_a_true_positive() -> None:
    result = match_detections([det(0)], [det(0)])
    assert result.n_tp == 1
    assert result.n_fp == 0
    assert result.n_fn == 0
    assert result.tp == (True,)
    assert result.gt_index == (0,)
    assert result.iou == (1.0,)


def test_disjoint_boxes_are_fp_and_fn() -> None:
    result = match_detections([det(0)], [det(500)])
    assert (result.n_tp, result.n_fp, result.n_fn) == (0, 1, 1)
    assert result.tp == (False,)
    assert result.gt_index == (-1,)
    assert result.unmatched_gt == (0,)


def test_counts_are_internally_consistent() -> None:
    preds = [det(0), det(5), det(500), det(1000)]
    gts = [det(0), det(20), det(40)]
    result = match_detections(preds, gts)
    assert result.n_tp + result.n_fp == result.n_pred == len(preds)
    assert result.n_tp + result.n_fn == result.n_gt == len(gts)
    assert len(result.tp) == len(result.scores) == len(result.order) == len(preds)


def test_a_ground_truth_is_claimed_at_most_once() -> None:
    """Two predictions on one ground truth: the better-scoring one is the TP.

    This is the duplicate-detection penalty. Without it a detector could spam boxes
    on every object and score perfect recall with no precision cost.
    """
    strong = det(0, score=0.9)
    weak = det(0, score=0.4)
    result = match_detections([weak, strong], [det(0)])
    assert result.n_tp == 1
    assert result.n_fp == 1
    # Descending-score order, so the strong prediction is first and it is the TP.
    assert result.order == (1, 0)
    assert result.tp == (True, False)
    assert result.tp_flags_in_input_order() == (False, True)


def test_first_pick_goes_to_the_higher_score_not_the_tighter_box() -> None:
    """Score ordering, not IoU ordering, decides who chooses first.

    ``high`` overlaps the ground truth less well than ``low`` does, but it scores
    higher, so it takes the only available ground truth. Legacy sorted candidate
    pairs by IoU and would have credited ``low`` instead, which is what makes legacy
    counts inconsistent with a precision-recall curve.
    """
    gt = det(0)
    high = shifted(0, dx=3.0, score=0.95)  # IoU 0.7 / 1.3 = 0.538
    low = det(0, score=0.10)  # IoU 1.0
    result = match_detections([high, low], [gt])
    assert result.order == (0, 1)
    assert result.tp == (True, False)
    assert result.gt_index == (0, -1)


def test_prediction_takes_its_best_available_ground_truth() -> None:
    """Among admissible ground truths a prediction claims the highest IoU one."""
    pred = det(0, size=10)
    near = det(1)  # IoU 81/119 = 0.68
    exact = det(0)  # IoU 1.0
    result = match_detections([pred], [near, exact])
    assert result.gt_index == (1,)
    assert result.iou[0] == pytest.approx(1.0)


def test_second_prediction_falls_back_to_the_remaining_ground_truth() -> None:
    first = det(0, score=0.9)
    second = det(1, score=0.8)
    result = match_detections([first, second], [det(0), det(1)])
    assert result.tp == (True, True)
    # The strong prediction takes its exact match, the weaker one takes what is left.
    assert result.gt_index == (0, 1)


# ---------------------------------------------------------------------------
# threshold behaviour
# ---------------------------------------------------------------------------


def test_iou_at_exactly_the_threshold_matches() -> None:
    """The test is ``iou >= threshold``, matching pycocotools; legacy used ``>``.

    Two 10x10 boxes offset by dx overlap on ``(10 - dx) * 10`` and union to
    ``200 - (10 - dx) * 10``. Setting the threshold to that exact IoU must still
    produce a match.
    """
    dx = 4.0
    inter = (10.0 - dx) * 10.0
    iou = inter / (200.0 - inter)
    assert iou == pytest.approx(0.42857142857142855)

    assert match_detections([shifted(0, dx=dx)], [det(0)], iou_threshold=iou).n_tp == 1
    # Nudging the threshold above that IoU drops the match.
    assert match_detections([shifted(0, dx=dx)], [det(0)], iou_threshold=iou + 1e-9).n_tp == 0


def test_raising_the_threshold_can_only_lose_matches() -> None:
    preds = [det(0, score=0.9), shifted(20, dx=2.0, score=0.8), shifted(40, dx=6.0, score=0.7)]
    gts = [det(0), det(20), det(40)]
    counts = [match_detections(preds, gts, iou_threshold=t).n_tp for t in (0.1, 0.5, 0.9, 1.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts == [3, 2, 1, 1]


def test_zero_threshold_admits_touching_boxes() -> None:
    """At ``iou_threshold=0.0`` a zero-overlap candidate is still a legal match.

    Guards the sentinel in the search loop: initialising the best IoU to 0.0 instead
    of below zero would silently reject every candidate at this threshold.
    """
    result = match_detections([det(500)], [det(0)], iou_threshold=0.0)
    assert result.n_tp == 1
    assert result.iou == (0.0,)


def test_threshold_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([det(0)], [det(0)], iou_threshold=1.5)
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([det(0)], [det(0)], iou_threshold=-0.1)


# ---------------------------------------------------------------------------
# class awareness
# ---------------------------------------------------------------------------


def test_matching_is_class_aware_by_default() -> None:
    """A perfectly localised box with the wrong label is a false positive.

    This is the whole point of adding confuser classes: calling a sargassum mat
    "marine debris" must cost precision, not be quietly credited as a hit.
    """
    pred = det(0, label=SurfaceClass.DEBRIS)
    gt = det(0, label=SurfaceClass.SARGASSUM)
    result = match_detections([pred], [gt])
    assert (result.n_tp, result.n_fp, result.n_fn) == (0, 1, 1)


def test_class_agnostic_matching_pairs_across_labels() -> None:
    pred = det(0, label=SurfaceClass.DEBRIS)
    gt = det(0, label=SurfaceClass.SARGASSUM)
    result = match_detections([pred], [gt], class_agnostic=True)
    assert (result.n_tp, result.n_fp, result.n_fn) == (1, 0, 0)
    assert result.class_agnostic is True


def test_class_aware_matching_skips_to_the_same_class_ground_truth() -> None:
    """A prediction passes over a better-overlapping box of the wrong class."""
    pred = det(0, label=SurfaceClass.DEBRIS)
    wrong_class_exact = det(0, label=SurfaceClass.SARGASSUM)
    right_class_offset = det(2, label=SurfaceClass.DEBRIS)
    result = match_detections([pred], [wrong_class_exact, right_class_offset])
    assert result.gt_index == (1,)


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------


def test_no_predictions_and_no_ground_truths() -> None:
    result = match_detections([], [])
    assert (result.n_tp, result.n_fp, result.n_fn) == (0, 0, 0)
    assert result.order == () and result.tp == () and result.scores == ()
    assert result.unmatched_gt == ()
    assert result.tp_flags_in_input_order() == ()
    assert result.pairs() == ()


def test_predictions_but_no_ground_truths() -> None:
    result = match_detections([det(0), det(20)], [])
    assert (result.n_tp, result.n_fp, result.n_fn) == (0, 2, 0)
    assert result.tp == (False, False)
    assert result.unmatched_gt == ()


def test_ground_truths_but_no_predictions() -> None:
    result = match_detections([], [det(0), det(20), det(40)])
    assert (result.n_tp, result.n_fp, result.n_fn) == (0, 0, 3)
    assert result.unmatched_gt == (0, 1, 2)
    assert result.matched_gt == frozenset()


def test_zero_score_predictions_still_participate() -> None:
    """A score of 0.0 is a legal prediction, not an absent one.

    Filtering by score is the caller's decision; matching must not quietly drop the
    tail of the ranking, because that tail is what the low-precision end of the PR
    curve is made of.
    """
    result = match_detections([det(0, score=0.0)], [det(0)])
    assert result.n_tp == 1
    assert result.scores == (0.0,)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_result_is_independent_of_input_order() -> None:
    """Shuffling the inputs cannot change the outcome, only its indexing.

    Matching sorts by score internally, so the function is a pure function of the
    input as a set. The per-prediction flags follow the shuffle, which is why they
    are compared after re-indexing to input order.
    """
    preds = [
        det(0, score=0.90),
        shifted(20, dx=1.0, score=0.80),
        det(400, score=0.70),
        shifted(40, dx=3.0, score=0.60),
        det(800, score=0.50),
    ]
    gts = [det(0), det(20), det(40), det(60)]

    baseline = match_detections(preds, gts)
    baseline_flags = dict(zip(range(len(preds)), baseline.tp_flags_in_input_order(), strict=True))

    rng = random.Random(20240725)
    for _ in range(25):
        pred_perm = list(range(len(preds)))
        gt_perm = list(range(len(gts)))
        rng.shuffle(pred_perm)
        rng.shuffle(gt_perm)

        result = match_detections([preds[i] for i in pred_perm], [gts[i] for i in gt_perm])
        assert (result.n_tp, result.n_fp, result.n_fn) == (
            baseline.n_tp,
            baseline.n_fp,
            baseline.n_fn,
        )
        flags = result.tp_flags_in_input_order()
        assert {pred_perm[i]: flags[i] for i in range(len(preds))} == baseline_flags
        # The same ground truths get claimed, whatever order they arrived in.
        assert {gt_perm[i] for i in result.matched_gt} == set(baseline.matched_gt)


def test_equal_scores_break_ties_by_input_order() -> None:
    """Deterministic, and deliberately stable: first listed wins.

    A stable rule here is what lets the PR curve keep the true positive ahead of its
    equally scored twin instead of pessimising the curve at that rank.
    """
    a = det(0, score=0.5)
    b = det(0, score=0.5)
    result = match_detections([a, b], [det(0)])
    assert result.order == (0, 1)
    assert result.tp == (True, False)


def test_equal_iou_breaks_ties_by_lowest_ground_truth_index() -> None:
    pred = det(0)
    left = det(-5)
    right = det(5)
    assert pred.bbox.iou(left.bbox) == pytest.approx(pred.bbox.iou(right.bbox))
    assert match_detections([pred], [left, right], iou_threshold=0.3).gt_index == (0,)


def test_tied_scores_are_the_one_case_input_order_can_matter() -> None:
    """Documents a real limit of greedy matching rather than pretending it away.

    Two predictions with identical scores over two overlapping ground truths.
    ``reaches_both`` clears the threshold against either box; ``reaches_one`` only
    clears it against the first. Both prefer the first box, so whoever moves first
    takes it, and the loser either falls back to the second box or has nowhere to go.
    Ordering is still deterministic (input order breaks the tie) and pycocotools
    behaves the same way, but the invariant asserted elsewhere in this file, that
    shuffling cannot change the counts, holds only when scores are distinct.
    """
    gts = [det(0), det(4)]
    reaches_both = det(1, score=0.5)  # IoU 0.82 with gt0, 0.54 with gt1
    reaches_one = det(-1, score=0.5)  # IoU 0.82 with gt0, 0.33 with gt1
    assert reaches_both.bbox.iou(gts[1].bbox) == pytest.approx(0.5384615384615384)
    assert reaches_one.bbox.iou(gts[1].bbox) == pytest.approx(0.3333333333333333)

    assert match_detections([reaches_both, reaches_one], gts).n_tp == 1
    assert match_detections([reaches_one, reaches_both], gts).n_tp == 2


def test_repeated_calls_are_identical() -> None:
    preds = [det(0, score=0.5), det(1, score=0.5), det(2, score=0.5)]
    gts = [det(0), det(1)]
    first = match_detections(preds, gts)
    second = match_detections(preds, gts)
    assert first == second


# ---------------------------------------------------------------------------
# result accessors
# ---------------------------------------------------------------------------


def test_pairs_reports_accepted_matches_only() -> None:
    preds = [det(0, score=0.9), det(500, score=0.8), det(20, score=0.7)]
    gts = [det(0), det(20)]
    result = match_detections(preds, gts)
    assert result.pairs() == ((0, 0, 1.0), (2, 1, 1.0))


def test_matched_and_unmatched_ground_truths_partition_the_set() -> None:
    preds = [det(0), det(40)]
    gts = [det(0), det(20), det(40), det(60)]
    result = match_detections(preds, gts)
    assert result.matched_gt == frozenset({0, 2})
    assert result.unmatched_gt == (1, 3)
    assert len(result.matched_gt) + len(result.unmatched_gt) == result.n_gt
