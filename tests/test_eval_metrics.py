"""Tests for detection metrics, reporting, and parity with the legacy evaluation.

Three groups of tests carry most of the weight:

* a fully hand-computed average-precision example, worked out in the test docstring
  so the expected number is derived rather than recorded from a previous run,
* a cross-check of mAP against ``torchmetrics.detection.MeanAveragePrecision``,
  which routes through a COCO evaluator and is therefore an independent
  implementation of the same definition,
* a legacy-parity check reproducing the exact TP/FP/FN counts and the precision,
  recall and F1 published in the original README.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mdebris.eval.matching import match_detections
from mdebris.eval.metrics import (
    IOU_THRESHOLDS_50_95,
    ap_from_flags,
    average_precision,
    average_precision_per_class,
    confusion_matrix,
    confusion_matrix_labels,
    evaluate,
    map_50_95,
    mean_average_precision,
    pr_curve,
    precision_recall_f1,
)
from mdebris.eval.report import format_json, format_markdown, to_csv, write_report
from mdebris.types import BBox, Detection, DetectionSet, SurfaceClass

SIZE = 20.0


def box(index: int, *, dx: float = 0.0, size: float = SIZE) -> BBox:
    """A ``size``-square box on a 100 px grid cell, optionally nudged right by ``dx``.

    The grid spacing is five times the box size, so boxes in different cells can
    never overlap and every scenario's IoU structure is exactly what it looks like.
    """
    x = (index % 10) * 100.0 + dx
    y = (index // 10) * 100.0
    return BBox(x, y, x + size, y + size)


def det(
    index: int,
    *,
    dx: float = 0.0,
    score: float = 1.0,
    label: SurfaceClass = SurfaceClass.DEBRIS,
    size: float = SIZE,
) -> Detection:
    return Detection(bbox=box(index, dx=dx, size=size), score=score, label=label)


def iou_for_shift(dx: float, size: float = SIZE) -> float:
    """IoU of two ``size``-squares offset by ``dx`` along x."""
    inter = max(0.0, size - dx) * size
    return inter / (2.0 * size * size - inter)


# ---------------------------------------------------------------------------
# precision, recall, F1 and the zero-denominator convention
# ---------------------------------------------------------------------------


def test_precision_recall_f1_basic_arithmetic() -> None:
    scores = precision_recall_f1(tp=3, fp=1, fn=1)
    assert scores["precision"] == pytest.approx(0.75)
    assert scores["recall"] == pytest.approx(0.75)
    assert scores["f1"] == pytest.approx(0.75)
    assert (scores["tp"], scores["fp"], scores["fn"]) == (3.0, 1.0, 1.0)


def test_precision_recall_f1_matches_sklearn_on_the_defined_cases() -> None:
    """Cross-check the arithmetic against scikit-learn.

    A detection tally maps onto a binary classification tally exactly: a true
    positive is (1, 1), a false positive is (0, 1), a false negative is (1, 0). True
    negatives have no analogue in detection and, correctly, do not enter precision,
    recall or F1 for the positive class.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    for tp, fp, fn in [(3, 1, 1), (38, 11, 16), (1, 9, 0), (7, 0, 3), (0, 4, 5)]:
        y_true = [1] * tp + [0] * fp + [1] * fn
        y_pred = [1] * tp + [1] * fp + [0] * fn
        expected = sklearn_metrics.precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0.0
        )
        got = precision_recall_f1(tp, fp, fn)
        assert got["precision"] == pytest.approx(expected[0])
        assert got["recall"] == pytest.approx(expected[1])
        assert got["f1"] == pytest.approx(expected[2])


@pytest.mark.parametrize(
    ("tp", "fp", "fn", "precision", "recall", "f1"),
    [
        pytest.param(0, 0, 0, 1.0, 1.0, 1.0, id="nothing-predicted-nothing-there"),
        pytest.param(0, 5, 0, 0.0, 1.0, 0.0, id="predictions-but-no-ground-truth"),
        pytest.param(0, 0, 5, 1.0, 0.0, 0.0, id="ground-truth-but-no-predictions"),
        pytest.param(0, 5, 5, 0.0, 0.0, 0.0, id="both-present-nothing-matched"),
    ],
)
def test_zero_denominator_convention(
    tp: int, fp: int, fn: int, precision: float, recall: float, f1: float
) -> None:
    """Pins the vacuous-truth rule documented in the module docstring.

    The awkward case is "predictions but no ground truth": recall is 1.0 because
    nothing was missed, which is true but useless, while precision is 0.0 because
    every prediction is a false alarm. F1 is 0.0, so the summary number does not
    reward the detector for hallucinating over an empty scene.
    """
    scores = precision_recall_f1(tp, fp, fn)
    assert scores["precision"] == precision
    assert scores["recall"] == recall
    assert scores["f1"] == f1


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        precision_recall_f1(tp=-1, fp=0, fn=0)


# ---------------------------------------------------------------------------
# precision-recall curve
# ---------------------------------------------------------------------------


def test_pr_curve_values_at_each_rank() -> None:
    """Precision at rank k is (true positives so far) / k; recall is over all ground truths."""
    recall, precision = pr_curve(
        scores=[0.9, 0.8, 0.7, 0.6], tp_flags=[True, False, True, True], n_gt=5
    )
    assert recall == pytest.approx([1 / 5, 1 / 5, 2 / 5, 3 / 5])
    assert precision == pytest.approx([1 / 1, 1 / 2, 2 / 3, 3 / 4])


def test_pr_curve_sorts_by_descending_score() -> None:
    ordered = pr_curve([0.9, 0.5], [True, False], n_gt=2)
    shuffled = pr_curve([0.5, 0.9], [False, True], n_gt=2)
    assert ordered[0] == pytest.approx(shuffled[0])
    assert ordered[1] == pytest.approx(shuffled[1])


def test_pr_curve_recall_is_non_decreasing() -> None:
    rng = np.random.default_rng(20240725)
    scores = rng.random(50)
    flags = rng.random(50) > 0.4
    recall, _ = pr_curve(scores, flags, n_gt=int(flags.sum()) + 7)
    assert np.all(np.diff(recall) >= 0.0)


def test_pr_curve_requires_ground_truth() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        pr_curve([0.9], [False], n_gt=0)


def test_pr_curve_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        pr_curve([0.9, 0.5], [True], n_gt=2)


# ---------------------------------------------------------------------------
# average precision, hand-computed
# ---------------------------------------------------------------------------

# Ranked outcome shared by the two hand-computed AP tests below.
_WORKED_SCORES = [0.95, 0.85, 0.75, 0.65, 0.55]
_WORKED_FLAGS = [True, False, True, True, False]
_WORKED_N_GT = 4


def test_average_precision_all_points_hand_computed() -> None:
    """AP by hand for five ranked predictions against four ground truths.

    Ranking (descending score), with tp/fp per rank::

        rank  score  outcome  tp_cum  fp_cum  recall=tp/4  precision=tp/rank
        1     0.95   TP       1       0       0.25         1/1 = 1.0000
        2     0.85   FP       1       1       0.25         1/2 = 0.5000
        3     0.75   TP       2       1       0.50         2/3 = 0.6667
        4     0.65   TP       3       1       0.75         3/4 = 0.7500
        5     0.55   FP       3       2       0.75         3/5 = 0.6000

    One ground truth is never found, so recall stops at 0.75.

    The precision envelope is the running maximum taken from the bottom up, that is
    the best precision still reachable at that recall or beyond::

        rank  5      4      3      2      1
        env   0.60   0.75   0.75   0.75   1.00

    All-points AP integrates that envelope exactly over recall. Recall only moves at
    ranks 1, 3 and 4, and the region above recall 0.75 contributes nothing because no
    prediction reaches it::

        (0.25 - 0.00) * env(rank 1) = 0.25 * 1.00  = 0.2500
        (0.50 - 0.25) * env(rank 3) = 0.25 * 0.75  = 0.1875
        (0.75 - 0.50) * env(rank 4) = 0.25 * 0.75  = 0.1875
        (1.00 - 0.75) * 0.0                        = 0.0000
                                            AP     = 0.6250
    """
    ap = ap_from_flags(_WORKED_SCORES, _WORKED_FLAGS, _WORKED_N_GT, method="all-points")
    assert ap == pytest.approx(0.625, abs=1e-12)


def test_average_precision_101_point_hand_computed() -> None:
    """The same ranking under COCO's 101-point rule, also worked out by hand.

    COCO samples the same precision envelope at recall 0.00, 0.01, ..., 1.00 and
    averages the 101 samples. Each sample takes the envelope at the first rank whose
    recall reaches that level, or 0.0 if no rank does::

        recall levels    first rank reaching it   envelope   count
        0.00 .. 0.25     rank 1                   1.00       26
        0.26 .. 0.50     rank 3                   0.75       25
        0.51 .. 0.75     rank 4                   0.75       25
        0.76 .. 1.00     none                     0.00       25

        AP = (26 * 1.00 + 25 * 0.75 + 25 * 0.75 + 25 * 0.00) / 101
           = 63.5 / 101
           = 0.628713

    Note it exceeds the exact area (0.625): the level 0.00 is sampled at full
    precision even though it encloses zero area, which biases the 101-point figure
    slightly upward on short curves. This is the discretisation the COCO numbers in
    the literature carry.
    """
    ap = ap_from_flags(_WORKED_SCORES, _WORKED_FLAGS, _WORKED_N_GT, method="101-point")
    assert ap == pytest.approx(63.5 / 101.0, abs=1e-12)
    assert ap == pytest.approx(0.6287128712871287, abs=1e-12)


def test_the_two_ap_conventions_disagree_and_the_default_is_all_points() -> None:
    """The reason both are implemented: they are different numbers for the same data."""
    all_points = ap_from_flags(_WORKED_SCORES, _WORKED_FLAGS, _WORKED_N_GT, method="all-points")
    coco = ap_from_flags(_WORKED_SCORES, _WORKED_FLAGS, _WORKED_N_GT, method="101-point")
    assert all_points != coco
    assert abs(all_points - coco) == pytest.approx(0.0037128712871287)
    default = ap_from_flags(_WORKED_SCORES, _WORKED_FLAGS, _WORKED_N_GT)
    assert default == all_points


def test_perfect_detector_scores_one_under_both_conventions() -> None:
    preds = [det(i, score=0.9 - 0.01 * i) for i in range(6)]
    gts = [det(i) for i in range(6)]
    assert average_precision(preds, gts) == pytest.approx(1.0)
    assert average_precision(preds, gts, method="101-point") == pytest.approx(1.0)


def test_all_false_positives_score_zero() -> None:
    preds = [det(i, score=0.9) for i in range(5, 10)]
    gts = [det(i) for i in range(5)]
    assert average_precision(preds, gts) == 0.0
    assert average_precision(preds, gts, method="101-point") == 0.0


def test_no_ground_truth_makes_ap_undefined() -> None:
    """NaN, not 0.0: a class that never occurs cannot be scored, only excluded."""
    assert math.isnan(average_precision([det(0, score=0.9)], []))
    assert math.isnan(ap_from_flags([0.9], [False], 0))


def test_no_predictions_scores_zero_ap() -> None:
    assert average_precision([], [det(0)]) == 0.0


def test_ap_rewards_ranking_true_positives_first() -> None:
    """AP is a ranking metric: the same counts score differently if ordered worse."""
    gts = [det(i) for i in range(4)]
    good = [det(i, score=0.9 - 0.1 * i) for i in range(4)] + [det(9, score=0.2)]
    bad = [det(i, score=0.5 - 0.1 * i) for i in range(4)] + [det(9, score=0.95)]
    assert average_precision(good, gts) > average_precision(bad, gts)
    assert match_detections(good, gts).n_tp == match_detections(bad, gts).n_tp


def test_ap_is_independent_of_input_order() -> None:
    rng = np.random.default_rng(7)
    gts = [det(i) for i in range(8)]
    preds = [det(i, dx=1.0, score=float(rng.random())) for i in range(8)] + [
        det(i, score=float(rng.random())) for i in range(20, 25)
    ]
    baseline = average_precision(preds, gts)
    for _ in range(10):
        shuffled = list(preds)
        rng.shuffle(shuffled)
        assert average_precision(shuffled, gts) == pytest.approx(baseline, abs=1e-12)


def test_sklearn_average_precision_is_a_third_convention_and_is_not_interchangeable() -> None:
    """Pins how this module relates to ``sklearn.metrics.average_precision_score``.

    sklearn sums the precision-recall steps with no interpolation at all. When
    precision is already non-increasing the envelope is a no-op and the two agree
    exactly. When precision rises with recall, which happens whenever a false
    positive is ranked above true positives, sklearn reports the lower, uninterpolated
    area. Neither is a bug; they are different definitions, and reaching for sklearn
    because it is already a dependency would silently change every AP in the project.

    (sklearn also cannot express a ground truth that no prediction reached, so these
    cases are built with every ground truth found.)
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    monotone_scores, monotone_flags = [0.9, 0.8, 0.7, 0.6], [True, True, False, True]
    assert ap_from_flags(monotone_scores, monotone_flags, n_gt=3) == pytest.approx(
        sklearn_metrics.average_precision_score(monotone_flags, monotone_scores)
    )

    # A false positive ranked first makes precision climb, so the envelope bites.
    rising_scores, rising_flags = [0.9, 0.8, 0.7, 0.6, 0.5], [False, True, True, True, True]
    ours = ap_from_flags(rising_scores, rising_flags, n_gt=4)
    theirs = sklearn_metrics.average_precision_score(rising_flags, rising_scores)
    assert ours == pytest.approx(0.8)
    assert theirs == pytest.approx(0.6791666666666667)
    assert ours > theirs


def test_unknown_ap_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown AP method"):
        ap_from_flags([0.9], [True], 1, method="11-point")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# per-class AP, mAP and the IoU sweep
# ---------------------------------------------------------------------------


def _two_class_scenario() -> tuple[list[Detection], list[Detection]]:
    """A deliberately imperfect two-class scene.

    Debris: 4 ground truths, one found exactly, one found loosely (IoU 0.82), one
    found too loosely to count at 0.5 (IoU 0.43), one missed, plus a lone false
    positive and a duplicate box on the exact match.

    Sargassum: 3 ground truths, two found, one missed, plus a false positive.
    """
    debris_gt = [det(i) for i in range(4)]
    debris_pred = [
        det(0, score=0.95),
        det(1, dx=2.0, score=0.90),
        det(2, dx=8.0, score=0.70),
        det(0, dx=1.0, score=0.60),  # duplicate on ground truth 0
        det(7, score=0.55),  # nothing there
    ]
    sarg = SurfaceClass.SARGASSUM
    sarg_gt = [det(10 + i, label=sarg) for i in range(3)]
    sarg_pred = [
        det(10, score=0.88, label=sarg),
        det(11, dx=4.0, score=0.66, label=sarg),
        det(17, score=0.44, label=sarg),  # nothing there
    ]
    return debris_pred + sarg_pred, debris_gt + sarg_gt


def test_average_precision_per_class_splits_by_label() -> None:
    preds, gts = _two_class_scenario()
    per_class = average_precision_per_class(preds, gts)
    assert set(per_class) == {SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM}
    assert 0.0 < per_class[SurfaceClass.DEBRIS] < 1.0
    assert 0.0 < per_class[SurfaceClass.SARGASSUM] < 1.0
    # Sargassum is the easier half of this scene, so it must score higher.
    assert per_class[SurfaceClass.SARGASSUM] > per_class[SurfaceClass.DEBRIS]


def test_mean_average_precision_is_the_mean_of_the_defined_class_aps() -> None:
    preds, gts = _two_class_scenario()
    per_class = average_precision_per_class(preds, gts)
    expected = float(np.mean(list(per_class.values())))
    assert mean_average_precision(preds, gts) == pytest.approx(expected)


def test_classes_without_ground_truth_are_excluded_from_the_mean() -> None:
    """A class that never appears must not dilute mAP toward zero.

    With nine classes in the taxonomy and two in a typical scene, counting the absent
    seven as zero would quarter every score reported.
    """
    preds, gts = _two_class_scenario()
    present_only = mean_average_precision(preds, gts)
    with_absent = mean_average_precision(preds, gts, classes=list(SurfaceClass))
    assert with_absent == pytest.approx(present_only)

    per_class = average_precision_per_class(preds, gts, classes=list(SurfaceClass))
    assert math.isnan(per_class[SurfaceClass.CLOUD])


def test_mean_average_precision_is_nan_when_there_is_no_ground_truth() -> None:
    assert math.isnan(mean_average_precision([det(0, score=0.5)], []))
    assert math.isnan(map_50_95([det(0, score=0.5)], []))


def test_iou_sweep_is_the_coco_grid() -> None:
    assert IOU_THRESHOLDS_50_95 == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    assert len(IOU_THRESHOLDS_50_95) == 10


def test_map_50_95_penalises_loose_boxes_that_map_50_forgives() -> None:
    """The point of the sweep: mAP@0.5 barely notices box quality, mAP@[.5:.95] does."""
    gts = [det(i) for i in range(6)]
    tight = [det(i, score=0.9) for i in range(6)]
    loose = [det(i, dx=6.0, score=0.9) for i in range(6)]
    assert iou_for_shift(6.0) == pytest.approx(0.5384615384615384)

    assert mean_average_precision(tight, gts) == pytest.approx(1.0)
    assert mean_average_precision(loose, gts) == pytest.approx(1.0)
    assert map_50_95(tight, gts) == pytest.approx(1.0)
    # Loose boxes clear 0.5 but fail every threshold above it, so only one of the ten
    # sweep terms survives.
    assert map_50_95(loose, gts) == pytest.approx(0.1)


def test_map_50_95_equals_the_mean_of_the_per_threshold_maps() -> None:
    preds, gts = _two_class_scenario()
    per_threshold = [
        mean_average_precision(preds, gts, iou_threshold=t) for t in IOU_THRESHOLDS_50_95
    ]
    assert map_50_95(preds, gts) == pytest.approx(float(np.mean(per_threshold)))


# ---------------------------------------------------------------------------
# cross-check against torchmetrics
# ---------------------------------------------------------------------------


def _to_torchmetrics(preds: list[Detection], gts: list[Detection]):  # type: ignore[no-untyped-def]
    """Convert one scene into the single-image batch torchmetrics expects."""
    import torch

    class_ids = {c: i for i, c in enumerate(SurfaceClass)}

    def boxes(dets: list[Detection]) -> torch.Tensor:
        if not dets:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor([d.bbox.as_xyxy() for d in dets], dtype=torch.float32)

    def labels(dets: list[Detection]) -> torch.Tensor:
        return torch.tensor([class_ids[d.label] for d in dets], dtype=torch.int64)

    pred_batch = [
        {
            "boxes": boxes(preds),
            "scores": torch.tensor([d.score for d in preds], dtype=torch.float32),
            "labels": labels(preds),
        }
    ]
    target_batch = [{"boxes": boxes(gts), "labels": labels(gts)}]
    return pred_batch, target_batch


def _single_class_scenario() -> tuple[list[Detection], list[Detection]]:
    """One class, mixed box quality, a duplicate, two false positives, two misses."""
    gts = [det(i) for i in range(6)]
    preds = [
        det(0, score=0.99),
        det(1, dx=1.0, score=0.92),
        det(2, dx=4.0, score=0.81),
        det(3, dx=6.0, score=0.77),
        det(1, dx=2.0, score=0.61),  # duplicate on ground truth 1
        det(20, score=0.50),
        det(21, score=0.33),
    ]
    return preds, gts


def _ranking_scenario() -> tuple[list[Detection], list[Detection]]:
    """False positives deliberately ranked above true positives."""
    gts = [det(i) for i in range(5)]
    preds = [
        det(30, score=0.98),
        det(31, score=0.94),
        det(0, score=0.90),
        det(1, dx=2.0, score=0.71),
        det(2, dx=1.0, score=0.55),
        det(32, score=0.40),
        det(3, score=0.21),
    ]
    return preds, gts


def _coco_backend() -> str:
    """The COCO evaluator torchmetrics can delegate to here, or skip the test."""
    for module, backend in (
        ("pycocotools", "pycocotools"),
        ("faster_coco_eval", "faster_coco_eval"),
    ):
        try:
            __import__(module)
        except ImportError:
            continue
        return backend
    pytest.skip("torchmetrics MeanAveragePrecision needs pycocotools or faster-coco-eval")


# COCO's recall sample points in double precision, exactly as pycocotools builds them.
# torchmetrics' own default comes from ``torch.linspace`` in float32; see
# ``test_torchmetrics_float32_recall_thresholds_explain_the_only_disagreement``.
_COCO_REC_THRESHOLDS = np.linspace(0.0, 1.0, 101).tolist()


def _reference_metric(**kwargs):  # type: ignore[no-untyped-def]
    from torchmetrics.detection import MeanAveragePrecision

    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        backend=_coco_backend(),
        **kwargs,
    )


@pytest.mark.parametrize(
    "scenario",
    [_single_class_scenario, _two_class_scenario, _ranking_scenario],
    ids=["single-class", "two-class", "adversarial-ranking"],
)
def test_map_agrees_with_torchmetrics(scenario) -> None:  # type: ignore[no-untyped-def]
    """Cross-check mAP against an independent COCO evaluator.

    torchmetrics delegates to pycocotools or faster-coco-eval, so this compares this
    module's matching, PR accumulation and interpolation against a separate
    implementation of the same definitions end to end. It is the single most
    load-bearing test here.

    Two alignments are needed for the comparison to be about the definitions rather
    than about incidental differences:

    * ``method="101-point"``, because that is the interpolation COCO computes.
      Comparing the all-points default would fail by a fraction of a point through no
      fault of either implementation, which is exactly the mismatch that makes
      published mAP numbers hard to compare and the reason both conventions exist here.
    * ``rec_thresholds`` supplied in float64. torchmetrics' default sample points come
      from ``torch.linspace`` in float32, which places 0.6 at 0.6000000238; the next
      test isolates that.
    """
    _coco_backend()
    preds, gts = scenario()
    metric = _reference_metric(class_metrics=True, rec_thresholds=_COCO_REC_THRESHOLDS)
    pred_batch, target_batch = _to_torchmetrics(preds, gts)
    metric.update(pred_batch, target_batch)
    reference = metric.compute()

    ours_50 = mean_average_precision(preds, gts, iou_threshold=0.5, method="101-point")
    ours_75 = mean_average_precision(preds, gts, iou_threshold=0.75, method="101-point")
    ours_sweep = map_50_95(preds, gts, method="101-point")

    assert ours_50 == pytest.approx(float(reference["map_50"]), abs=1e-6)
    assert ours_75 == pytest.approx(float(reference["map_75"]), abs=1e-6)
    assert ours_sweep == pytest.approx(float(reference["map"]), abs=1e-6)

    # And per class, so a two-class scene cannot pass by averaging two errors away.
    per_class = average_precision_per_class(preds, gts, iou_threshold=0.5, method="101-point")
    class_ids = {c: i for i, c in enumerate(SurfaceClass)}
    reference_classes = np.atleast_1d(reference["classes"].numpy())
    reference_per_class = np.atleast_1d(reference["map_per_class"].numpy())
    for label, ap in per_class.items():
        position = int(np.flatnonzero(reference_classes == class_ids[label])[0])
        # torchmetrics reports map_per_class over the full IoU sweep, so compare our
        # sweep for that one class rather than its 0.5 figure.
        ours_class_sweep = float(
            np.mean(
                [
                    average_precision_per_class(
                        preds, gts, iou_threshold=float(t), method="101-point"
                    )[label]
                    for t in IOU_THRESHOLDS_50_95
                ]
            )
        )
        assert ours_class_sweep == pytest.approx(float(reference_per_class[position]), abs=1e-6)
        assert not math.isnan(ap)


def _random_scene(seed: int) -> tuple[list[Detection], list[Detection]]:
    """A messy multi-class scene: jittered duplicates, wrong labels, stray boxes.

    Boxes overlap partially and at arbitrary IoUs, which is what exercises the
    matching order and the threshold sweep. Three classes, so per-class splitting and
    the class-aware rule are both in play.
    """
    rng = np.random.default_rng(seed)
    labels = [SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM, SurfaceClass.SHIP]
    preds: list[Detection] = []
    gts: list[Detection] = []

    for _ in range(int(rng.integers(2, 12))):
        label = labels[int(rng.integers(0, len(labels)))]
        x, y = float(rng.integers(0, 40)) * 60.0, float(rng.integers(0, 40)) * 60.0
        w, h = float(rng.integers(5, 40)), float(rng.integers(5, 40))
        gts.append(Detection(bbox=BBox(x, y, x + w, y + h), score=1.0, label=label))
        for _ in range(int(rng.integers(0, 3))):
            jx, jy = float(rng.normal(0.0, w / 4.0)), float(rng.normal(0.0, h / 4.0))
            guess = label if rng.random() < 0.8 else labels[int(rng.integers(0, len(labels)))]
            preds.append(
                Detection(
                    bbox=BBox(x + jx, y + jy, x + jx + w, y + jy + h),
                    score=float(rng.random()),
                    label=guess,
                )
            )
    for _ in range(int(rng.integers(0, 5))):
        x, y = 3000.0 + float(rng.integers(0, 40)) * 60.0, float(rng.integers(0, 40)) * 60.0
        preds.append(
            Detection(
                bbox=BBox(x, y, x + 20.0, y + 20.0),
                score=float(rng.random()),
                label=labels[int(rng.integers(0, len(labels)))],
            )
        )
    return preds, gts


@pytest.mark.parametrize("seed", range(12))
def test_map_agrees_with_torchmetrics_on_random_scenes(seed: int) -> None:
    """The same cross-check over messy generated scenes, not just hand-built ones.

    Hand-built cases only test the structures someone thought to build. These have
    arbitrary IoUs, duplicate predictions at random jitter, mislabelled boxes and
    stray detections, which is where an off-by-one in the matching order or the
    recall accumulation would show up.
    """
    _coco_backend()
    preds, gts = _random_scene(seed)
    metric = _reference_metric(rec_thresholds=_COCO_REC_THRESHOLDS)
    pred_batch, target_batch = _to_torchmetrics(preds, gts)
    metric.update(pred_batch, target_batch)
    reference = metric.compute()

    assert mean_average_precision(preds, gts, iou_threshold=0.5, method="101-point") == (
        pytest.approx(float(reference["map_50"]), abs=1e-6)
    )
    assert map_50_95(preds, gts, method="101-point") == pytest.approx(
        float(reference["map"]), abs=1e-6
    )


def test_torchmetrics_float32_recall_thresholds_explain_the_only_disagreement() -> None:
    """Isolates the one systematic gap against torchmetrics, and shows it is not ours.

    torchmetrics builds its 101 recall sample points with ``torch.linspace(0, 1, 101)``,
    which is float32 by default, so 0.6 is stored as 0.6000000238418579 and 0.8 as
    0.800000011920929. pycocotools builds the same points with ``np.linspace`` in
    float64. The difference only shows up when an achieved recall lands exactly on a
    sample point, which needs a ground-truth count whose reciprocal is a non-dyadic
    multiple of 0.01: five ground truths give recalls 0.2, 0.4, 0.6, 0.8, and the
    float32 threshold then sits a hair above the recall it is meant to select, so the
    sample is taken from the next rank down (or dropped entirely at the top of the
    curve).

    In :func:`_ranking_scenario` that costs one of the 101 samples, 0.6 / 101 = 0.0059
    of mAP. Supplying float64 thresholds makes the two implementations agree to 6e-9,
    which is the actual verdict of the cross-check: the definitions match, and the
    residue is torchmetrics' dtype choice.
    """
    _coco_backend()
    preds, gts = _ranking_scenario()
    ours = mean_average_precision(preds, gts, iou_threshold=0.5, method="101-point")

    default_metric = _reference_metric()
    assert default_metric.rec_thresholds[60] > 0.6  # float32 residue, the whole cause
    pred_batch, target_batch = _to_torchmetrics(preds, gts)
    default_metric.update(pred_batch, target_batch)
    with_float32 = float(default_metric.compute()["map_50"])

    exact_metric = _reference_metric(rec_thresholds=_COCO_REC_THRESHOLDS)
    exact_metric.update(pred_batch, target_batch)
    with_float64 = float(exact_metric.compute()["map_50"])

    assert ours == pytest.approx(with_float64, abs=1e-6)
    # Any residual gap is whole 101-point samples, never a disagreement about the
    # metric. Written as a bound rather than an equality so that an upstream dtype fix
    # makes this test pass trivially instead of failing.
    assert abs(ours - with_float32) < 3.0 / 101.0


def test_all_points_and_coco_ap_stay_close_on_realistic_input() -> None:
    """The two conventions differ, but not by enough to change a conclusion.

    Worth pinning: if a refactor ever made the gap large, one of the two
    implementations would be wrong.
    """
    for scenario in (_single_class_scenario, _two_class_scenario, _ranking_scenario):
        preds, gts = scenario()
        all_points = mean_average_precision(preds, gts, method="all-points")
        coco = mean_average_precision(preds, gts, method="101-point")
        assert abs(all_points - coco) < 0.02


# ---------------------------------------------------------------------------
# confusion matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_single_class_layout() -> None:
    """Rows are ground truth, columns are predicted, background last.

    For one class the matrix is ``[[TP, FN], [FP, 0]]``, which is the shape the
    legacy README table printed.
    """
    gts = [det(0), det(1), det(2)]
    preds = [det(0, score=0.9), det(1, score=0.8), det(20, score=0.7), det(21, score=0.6)]
    cm = confusion_matrix(preds, gts, classes=[SurfaceClass.DEBRIS])
    assert cm.shape == (2, 2)
    assert cm.tolist() == [[2, 1], [2, 0]]
    assert confusion_matrix_labels([SurfaceClass.DEBRIS]) == ["marine_debris", "background"]


def test_confusion_matrix_rows_and_columns_are_not_transposed() -> None:
    """An asymmetric case, so a transposed implementation cannot pass by symmetry.

    Three ground truths of which one is found, and four predictions of which three
    hit nothing: the false-negative cell must hold 2 and the false-positive cell 3.
    """
    gts = [det(0), det(1), det(2)]
    preds = [det(0, score=0.9)] + [det(i, score=0.5) for i in (20, 21, 22)]
    cm = confusion_matrix(preds, gts, classes=[SurfaceClass.DEBRIS])
    assert cm[0, 0] == 1  # true debris, predicted debris
    assert cm[0, 1] == 2  # true debris, predicted nothing
    assert cm[1, 0] == 3  # nothing there, predicted debris
    assert cm[1, 1] == 0


def test_confusion_matrix_records_class_confusion_off_the_diagonal() -> None:
    """The reason the matrix matches class-agnostically by default.

    A debris box sitting exactly on a sargassum ground truth is one object that was
    given the wrong name. Class-agnostic matching records that as a single
    off-diagonal cell; class-aware matching would split it into an unrelated false
    positive and an unrelated false negative and lose the connection.
    """
    classes = [SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM]
    gts = [det(0, label=SurfaceClass.SARGASSUM)]
    preds = [det(0, score=0.9, label=SurfaceClass.DEBRIS)]

    agnostic = confusion_matrix(preds, gts, classes=classes)
    assert agnostic.tolist() == [[0, 0, 0], [1, 0, 0], [0, 0, 0]]

    aware = confusion_matrix(preds, gts, classes=classes, class_agnostic_matching=False)
    assert aware.tolist() == [[0, 0, 0], [0, 0, 1], [1, 0, 0]]


def test_confusion_matrix_totals_account_for_every_box() -> None:
    preds, gts = _two_class_scenario()
    classes = [SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM]
    cm = confusion_matrix(preds, gts, classes=classes)
    k = len(classes)
    # Every ground truth lands in exactly one row cell, every prediction in one column.
    assert cm[:k, :].sum() == len(gts)
    assert cm[:, :k].sum() == len(preds)
    assert cm[k, k] == 0


def test_confusion_matrix_rejects_labels_outside_the_class_list() -> None:
    with pytest.raises(ValueError, match="not in classes"):
        confusion_matrix(
            [det(0, score=0.9, label=SurfaceClass.SHIP)],
            [det(0)],
            classes=[SurfaceClass.DEBRIS],
        )
    with pytest.raises(ValueError, match="not in classes"):
        confusion_matrix(
            [det(0, score=0.9)],
            [det(0, label=SurfaceClass.FOAM)],
            classes=[SurfaceClass.DEBRIS],
        )


def test_confusion_matrix_rejects_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        confusion_matrix([], [], classes=[SurfaceClass.DEBRIS, SurfaceClass.DEBRIS])


def test_confusion_matrix_of_an_empty_scene_is_all_zeros() -> None:
    cm = confusion_matrix([], [], classes=[SurfaceClass.DEBRIS])
    assert cm.tolist() == [[0, 0], [0, 0]]


# ---------------------------------------------------------------------------
# legacy parity
# ---------------------------------------------------------------------------

LEGACY_TP = 38
LEGACY_FP = 11
LEGACY_FN = 16


def _legacy_scenario() -> tuple[DetectionSet, DetectionSet]:
    """Synthesise exactly the tally the original README published.

    38 predictions land on a ground truth, 11 land on empty water, and 16 ground
    truths are missed. Boxes sit on a 100 px grid so no unintended overlap can shift
    the counts.
    """
    preds: list[Detection] = []
    gts: list[Detection] = []
    for i in range(LEGACY_TP):
        preds.append(det(i, score=0.99 - 0.001 * i))
        gts.append(det(i))
    for i in range(LEGACY_TP, LEGACY_TP + LEGACY_FN):
        gts.append(det(i))
    for j in range(LEGACY_FP):
        preds.append(det(200 + j, score=0.60 - 0.001 * j))
    return DetectionSet(preds), DetectionSet(gts)


def test_legacy_parity_counts_and_scores() -> None:
    """Regression guard against the numbers published for the 2019 model.

    The original README reported, for class ``marine_debris`` at IoU 0.5:
    TP 38, FP 11, FN 16, precision 0.78, recall 0.70, F1 0.74. Those must still be
    what this implementation produces from the same tally, otherwise old and new
    results are not comparable and every claim in the README about improvement is
    meaningless.
    """
    pred_set, gt_set = _legacy_scenario()
    result = evaluate(pred_set, gt_set, iou_threshold=0.5)

    assert (result.tp, result.fp, result.fn) == (LEGACY_TP, LEGACY_FP, LEGACY_FN)

    precision = LEGACY_TP / (LEGACY_TP + LEGACY_FP)
    recall = LEGACY_TP / (LEGACY_TP + LEGACY_FN)
    f1 = 2 * precision * recall / (precision + recall)
    assert precision == pytest.approx(38 / 49)
    assert recall == pytest.approx(38 / 54)

    assert result.precision == pytest.approx(precision)
    assert result.recall == pytest.approx(recall)
    assert result.f1 == pytest.approx(f1)

    # The published figures, to the two decimals the README printed.
    assert round(result.precision, 2) == 0.78
    assert round(result.recall, 2) == 0.70
    assert round(result.f1, 2) == 0.74


def test_legacy_parity_confusion_matrix() -> None:
    """The legacy README's table, cell for cell::

    |                | Predicted debris | Predicted none |
    | True debris    | 38               | 16             |
    | True none      | 11               | 0              |
    """
    pred_set, gt_set = _legacy_scenario()
    result = evaluate(pred_set, gt_set)
    assert result.confusion.tolist() == [[LEGACY_TP, LEGACY_FN], [LEGACY_FP, 0]]


def test_legacy_map_column_was_precision_in_disguise() -> None:
    """Explains why the legacy mAP cannot be reproduced, and should not be.

    The legacy ``display()`` computed its "mAP" from two scalars::

        for recall_level in np.linspace(0.0, 1.0, 11):
            args = np.argwhere(recall >= recall_level).flatten()
            for row in args:
                prec_at_rec.append(precision)
        avg_prec = np.mean(np.array(prec_at_rec))

    ``precision`` and ``recall`` there are single floats for the whole dataset, not
    per-threshold arrays, so the loop appends the same scalar once per satisfied
    recall level and the mean of a constant list is that constant. Legacy mAP was
    identically precision, which is why the old README shows 0.78 for both. A real
    average precision integrates a precision-recall curve and needs the ranking,
    which those two scalars have already thrown away.

    (``np.argwhere`` on a scalar returned ``array([0])`` under the numpy 1.x that
    TensorFlow 1.x pinned; on numpy 2 it returns an empty array, hence the explicit
    ``atleast_1d`` here to reproduce the original behaviour.)
    """
    precision = LEGACY_TP / (LEGACY_TP + LEGACY_FP)
    recall = LEGACY_TP / (LEGACY_TP + LEGACY_FN)

    prec_at_rec: list[float] = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        args = np.argwhere(np.atleast_1d(recall >= recall_level)).flatten()
        for _row in args:
            prec_at_rec.append(precision)
    legacy_map = float(np.mean(np.array(prec_at_rec)))

    assert legacy_map == pytest.approx(precision)
    assert round(legacy_map, 2) == 0.78  # the figure printed as "map_@0.5IOU"

    # A real AP on the same scenario is a different quantity and does not match it.
    pred_set, gt_set = _legacy_scenario()
    result = evaluate(pred_set, gt_set)
    assert result.mean_ap != pytest.approx(legacy_map)
    assert result.mean_ap == pytest.approx(recall)  # all 38 hits outrank all 11 misses


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_end_to_end() -> None:
    preds, gts = _two_class_scenario()
    result = evaluate(DetectionSet(preds), DetectionSet(gts))

    assert result.classes == (SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM)
    assert result.n_pred == len(preds)
    assert result.n_gt == len(gts)
    assert result.iou_threshold == 0.5
    assert result.ap_method == "all-points"
    assert result.confusion.shape == (3, 3)
    assert result.mean_ap_50_95 is not None
    assert result.mean_ap_50_95 < result.mean_ap

    pooled = sum(m.tp for m in result.per_class.values())
    assert result.tp == pooled
    assert result.tp + result.fn == result.n_gt
    assert result.tp + result.fp == result.n_pred


def test_evaluate_orders_classes_by_taxonomy_not_by_appearance() -> None:
    """Report column order must not depend on which detection happened to come first."""
    sarg = det(10, score=0.9, label=SurfaceClass.SARGASSUM)
    debris = det(0, score=0.8)
    result = evaluate(
        DetectionSet([sarg, debris]), DetectionSet([det(10, label=SurfaceClass.SARGASSUM), det(0)])
    )
    assert result.classes == (SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM)


def test_evaluate_applies_the_score_threshold_before_matching() -> None:
    gts = [det(0), det(1)]
    preds = [det(0, score=0.9), det(1, score=0.2)]
    kept = evaluate(DetectionSet(preds), DetectionSet(gts), score_threshold=0.5)
    assert kept.n_pred == 1
    assert (kept.tp, kept.fp, kept.fn) == (1, 0, 1)

    unthresholded = evaluate(DetectionSet(preds), DetectionSet(gts))
    assert unthresholded.n_pred == 2
    assert (unthresholded.tp, unthresholded.fp, unthresholded.fn) == (2, 0, 0)
    # Thresholding truncates the PR curve, so it can only lower AP. This is why the
    # default threshold is 0.0 and the legacy hardcoded 0.5 is not reproduced.
    assert kept.mean_ap < unthresholded.mean_ap


def test_evaluate_on_empty_sets() -> None:
    result = evaluate(DetectionSet([]), DetectionSet([]))
    assert result.classes == ()
    assert (result.tp, result.fp, result.fn) == (0, 0, 0)
    assert result.precision == 1.0 and result.recall == 1.0
    assert math.isnan(result.mean_ap)
    assert result.confusion.tolist() == [[0]]


def test_evaluate_with_predictions_against_an_empty_ground_truth() -> None:
    result = evaluate(DetectionSet([det(0, score=0.9), det(1, score=0.8)]), DetectionSet([]))
    assert (result.tp, result.fp, result.fn) == (0, 2, 0)
    assert result.precision == 0.0
    assert result.recall == 1.0
    assert result.f1 == 0.0
    assert math.isnan(result.mean_ap)


def test_evaluate_with_ground_truth_but_no_predictions() -> None:
    result = evaluate(DetectionSet([]), DetectionSet([det(0), det(1)]))
    assert (result.tp, result.fp, result.fn) == (0, 0, 2)
    assert result.precision == 1.0
    assert result.recall == 0.0
    assert result.mean_ap == 0.0


def test_evaluate_can_skip_the_iou_sweep() -> None:
    preds, gts = _two_class_scenario()
    result = evaluate(DetectionSet(preds), DetectionSet(gts), include_map_50_95=False)
    assert result.mean_ap_50_95 is None


def test_evaluate_carries_the_scene_id_through() -> None:
    from mdebris.types import SceneRef

    scene = SceneRef(scene_id="S2A_MSIL2A_20240101T000000")
    result = evaluate(DetectionSet([]), DetectionSet([], scene=scene))
    assert result.scene_id == "S2A_MSIL2A_20240101T000000"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_markdown_reproduces_the_legacy_table_shapes() -> None:
    pred_set, gt_set = _legacy_scenario()
    md = format_markdown(evaluate(pred_set, gt_set))

    assert "| **True marine_debris** | 38 | 16 |" in md
    assert "| **True none** | 11 | 0 |" in md
    assert "| True Positive | False Positive | False Negative |" in md
    assert "| 38 | 11 | 16 |" in md
    assert "| category | precision_@0.5IOU | recall_@0.5IOU | map_@0.5IOU | f1_@0.5IOU |" in md
    assert "| marine_debris | 0.78 | 0.70 |" in md
    assert md.endswith("\n")


def test_markdown_warns_that_the_two_views_of_a_class_confusion_differ() -> None:
    """A mislabelled object is one off-diagonal cell but two counting errors.

    The confusion matrix matches on overlap alone, so it shows the object once, in
    the "truly sargassum, called debris" cell. The counts table matches within a
    class, so the same object is a false positive for debris and a false negative for
    sargassum. Both are correct for their purpose, and the report says so rather than
    leaving a reader to reconcile a matrix with no background entries against a
    non-zero FP/FN row.
    """
    gts = [det(0, label=SurfaceClass.SARGASSUM)]
    preds = [det(0, score=0.9, label=SurfaceClass.DEBRIS)]
    result = evaluate(DetectionSet(preds), DetectionSet(gts))

    assert result.confusion.tolist() == [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)

    md = format_markdown(result)
    assert "off the diagonal" in md
    # The single-class report stays clean, so legacy diffs are unaffected.
    single = format_markdown(evaluate(*_legacy_scenario()))
    assert "off the diagonal" not in single


def test_markdown_column_names_track_the_iou_threshold() -> None:
    preds, gts = _single_class_scenario()
    md = format_markdown(evaluate(DetectionSet(preds), DetectionSet(gts), iou_threshold=0.75))
    assert "precision_@0.75IOU" in md
    assert "precision_@0.5IOU" not in md


def test_markdown_shows_undefined_metrics_as_not_available() -> None:
    md = format_markdown(
        evaluate(
            DetectionSet([det(0, score=0.9)]),
            DetectionSet([]),
            classes=[SurfaceClass.DEBRIS],
        )
    )
    assert "n/a" in md


def test_json_is_strictly_serialisable_even_with_undefined_metrics() -> None:
    """NaN is not JSON. ``allow_nan=False`` fails loudly if an NaN ever leaks through."""
    result = evaluate(
        DetectionSet([det(0, score=0.9)]), DetectionSet([]), classes=[SurfaceClass.DEBRIS]
    )
    payload = format_json(result)
    assert payload["mean_ap"] is None
    assert payload["per_class"][0]["ap"] is None
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))
    assert round_tripped["counts"] == {"tp": 0, "fp": 1, "fn": 0}


def test_json_carries_the_matrix_and_its_orientation() -> None:
    pred_set, gt_set = _legacy_scenario()
    payload = format_json(evaluate(pred_set, gt_set))
    assert payload["confusion_matrix"]["labels"] == ["marine_debris", "background"]
    assert payload["confusion_matrix"]["matrix"] == [[38, 16], [11, 0]]
    assert payload["confusion_matrix"]["orientation"] == "rows=ground_truth, columns=predicted"
    assert payload["counts"] == {"tp": 38, "fp": 11, "fn": 16}


def test_confusion_matrix_csv_layout() -> None:
    pred_set, gt_set = _legacy_scenario()
    csv_text = to_csv(evaluate(pred_set, gt_set))
    assert csv_text == ",marine_debris,background\nmarine_debris,38,16\nbackground,11,0\n"


def test_scores_csv_matches_the_legacy_column_layout() -> None:
    """The legacy script wrote ``df.to_csv(path)``: unnamed index column, then scores."""
    pred_set, gt_set = _legacy_scenario()
    csv_text = to_csv(evaluate(pred_set, gt_set), table="scores")
    header, row = csv_text.strip().split("\n")
    assert header == ",category,precision_@0.5IOU,recall_@0.5IOU,map_@0.5IOU,f1_@0.5IOU"
    fields = row.split(",")
    assert fields[0] == "0"
    assert fields[1] == "marine_debris"
    assert float(fields[2]) == pytest.approx(38 / 49)
    assert float(fields[3]) == pytest.approx(38 / 54)


def test_unknown_csv_table_is_rejected() -> None:
    pred_set, gt_set = _legacy_scenario()
    with pytest.raises(ValueError, match="unknown table"):
        to_csv(evaluate(pred_set, gt_set), table="everything")  # type: ignore[arg-type]


def test_write_report_writes_all_four_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pred_set, gt_set = _legacy_scenario()
    paths = write_report(evaluate(pred_set, gt_set), tmp_path)
    assert set(paths) == {"markdown", "json", "confusion_csv", "scores_csv"}
    for path in paths.values():
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["counts"]["tp"] == 38
