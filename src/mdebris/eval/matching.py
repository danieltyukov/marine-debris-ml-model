"""Greedy matching of predicted boxes onto ground-truth boxes.

Every detection metric in this package is a tally over the output of this module,
so the matching rule *is* the metric definition. The rule implemented here is the
one COCO and the TensorFlow Object Detection API use:

1. Sort predictions by descending score (ties broken by input order, stably).
2. Walk that ranking. Each prediction claims the highest-IoU ground truth that is
   still unclaimed, has IoU at or above the threshold, and (unless matching is
   class-agnostic) carries the same label.
3. A prediction that claims a ground truth is a true positive. A prediction that
   claims nothing is a false positive. A ground truth nobody claimed is a false
   negative.

Score ordering is what makes the result meaningful for a precision-recall curve:
a confident prediction gets first pick, so lowering the score cut-off can only add
detections to the tail of the ranking and never re-shuffle the decisions already
made above it. That monotonicity is exactly the property an AP integral assumes.

Two deliberate departures from the legacy ``eval_cmatrix_f1_map.py``:

* Legacy sorted candidate pairs by descending **IoU**, not by score. Under that rule
  a low-confidence prediction with a slightly tighter box outranks a high-confidence
  one, so the true-positive set can change when the score threshold moves, and the
  resulting counts do not sit on a valid PR curve.
* Legacy required ``iou > threshold`` (strict) and computed IoU with the Pascal VOC
  ``+1`` pixel convention (``max(0, xb - xa + 1)``), which inflates IoU for small
  boxes: two 10 px boxes offset by 5 px score 0.38 under the ``+1`` convention versus
  0.33 without it. Here the threshold test is ``iou >= threshold`` (matching
  pycocotools, which compares against ``min(t, 1 - 1e-10)``) and IoU comes from
  :meth:`mdebris.types.BBox.iou`, which uses plain continuous coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

from mdebris.types import Detection

__all__ = ["MatchResult", "match_detections"]


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of matching one list of predictions against one list of ground truths.

    All per-prediction tuples (:attr:`order`, :attr:`tp`, :attr:`scores`,
    :attr:`gt_index`, :attr:`iou`) are aligned with each other and are in
    **descending-score order**, not input order, because that is the order a PR
    curve is accumulated in. :meth:`tp_flags_in_input_order` converts back.

    Attributes:
        order: Indices into the original ``preds`` list, sorted by descending score.
        tp: ``True`` where that prediction matched a ground truth.
        scores: The prediction scores, descending.
        gt_index: Index into the original ``gts`` list, or ``-1`` for a false positive.
        iou: IoU of the accepted match, or ``0.0`` for a false positive.
        n_pred: Number of predictions considered.
        n_gt: Number of ground truths considered.
        iou_threshold: Threshold the matching was run at.
        class_agnostic: Whether labels were ignored when matching.
    """

    order: tuple[int, ...]
    tp: tuple[bool, ...]
    scores: tuple[float, ...]
    gt_index: tuple[int, ...]
    iou: tuple[float, ...]
    n_pred: int
    n_gt: int
    iou_threshold: float
    class_agnostic: bool

    @property
    def n_tp(self) -> int:
        """Predictions that matched a ground truth."""
        return sum(self.tp)

    @property
    def n_fp(self) -> int:
        """Predictions that matched nothing."""
        return self.n_pred - self.n_tp

    @property
    def n_fn(self) -> int:
        """Ground truths nothing matched.

        Each ground truth can be claimed at most once, so this is simply the
        ground-truth count minus the number of true positives.
        """
        return self.n_gt - self.n_tp

    @property
    def matched_gt(self) -> frozenset[int]:
        """Indices into ``gts`` that were claimed."""
        return frozenset(g for g in self.gt_index if g >= 0)

    @property
    def unmatched_gt(self) -> tuple[int, ...]:
        """Indices into ``gts`` that were not claimed, in ascending index order."""
        claimed = self.matched_gt
        return tuple(i for i in range(self.n_gt) if i not in claimed)

    def tp_flags_in_input_order(self) -> tuple[bool, ...]:
        """True-positive flags re-indexed to the original ``preds`` order."""
        flags = [False] * self.n_pred
        for rank, pred_index in enumerate(self.order):
            flags[pred_index] = self.tp[rank]
        return tuple(flags)

    def pairs(self) -> tuple[tuple[int, int, float], ...]:
        """Accepted ``(pred_index, gt_index, iou)`` triples, in descending-score order."""
        return tuple(
            (pred_index, gt, iou)
            for pred_index, gt, iou in zip(self.order, self.gt_index, self.iou, strict=True)
            if gt >= 0
        )


def match_detections(
    preds: list[Detection],
    gts: list[Detection],
    *,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> MatchResult:
    """Greedily match predictions to ground truths by descending score.

    Args:
        preds: Predicted detections. Order is irrelevant to the result: the function
            sorts by score internally, so shuffling the input cannot change the
            true-positive count (it can only change which of two equally scored,
            equally overlapping predictions is credited).
        gts: Ground-truth detections. Their ``score`` field is ignored.
        iou_threshold: Minimum IoU for a match. The test is ``iou >= iou_threshold``.
        class_agnostic: When ``False`` (the default, and what AP requires) a
            prediction may only claim a ground truth carrying the same
            :class:`~mdebris.types.SurfaceClass`. When ``True`` labels are ignored
            during matching, which is what a confusion matrix needs: it is the only
            way an off-diagonal cell such as "truly sargassum, called debris" can
            ever be populated.

    Returns:
        A :class:`MatchResult`. Degenerate inputs are handled without special-casing
        the caller:

        * no predictions and no ground truths: everything is empty, all counts zero.
        * predictions but no ground truths: every prediction is a false positive.
        * ground truths but no predictions: every ground truth is a false negative.

        Turning those counts into precision and recall requires a convention for
        division by zero; that convention lives in
        :func:`mdebris.eval.metrics.precision_recall_f1` and is documented there
        rather than baked in here.

    Ties are resolved deterministically so the function is a pure function of the
    input *set*: equal scores keep input order, and equal IoUs prefer the
    lower-indexed ground truth.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold {iou_threshold} outside [0, 1]")

    order = sorted(range(len(preds)), key=lambda i: (-preds[i].score, i))
    claimed_by: list[int] = [-1] * len(gts)

    tp: list[bool] = []
    gt_index: list[int] = []
    ious: list[float] = []

    for pred_index in order:
        pred = preds[pred_index]
        best_gt = -1
        # Start below zero so that an iou_threshold of 0.0 still admits a
        # zero-overlap match rather than silently rejecting it.
        best_iou = -1.0
        for gi, gt in enumerate(gts):
            if claimed_by[gi] >= 0:
                continue
            if not class_agnostic and gt.label != pred.label:
                continue
            iou = pred.bbox.iou(gt.bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_gt, best_iou = gi, iou

        if best_gt >= 0:
            claimed_by[best_gt] = pred_index
            tp.append(True)
            gt_index.append(best_gt)
            ious.append(best_iou)
        else:
            tp.append(False)
            gt_index.append(-1)
            ious.append(0.0)

    return MatchResult(
        order=tuple(order),
        tp=tuple(tp),
        scores=tuple(preds[i].score for i in order),
        gt_index=tuple(gt_index),
        iou=tuple(ious),
        n_pred=len(preds),
        n_gt=len(gts),
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
    )
