"""Detection metrics: precision, recall, F1, average precision and confusion matrices.

This module grades every other module, so each convention it picks is spelled out
rather than assumed. The three that actually bite:

**Zero-denominator convention.** Precision and recall are ratios that can be 0/0.
The rule here is vacuous truth: precision is 1.0 when nothing was predicted (no
false alarm was raised), recall is 1.0 when there was nothing to find (nothing was
missed), and F1 is 0.0 whenever precision + recall is 0. So an empty prediction set
against an empty ground truth scores a perfect 1/1/1, while predictions against an
empty ground truth score precision 0.0 because every one of them is a false
positive. This differs from scikit-learn, which returns 0.0 with a warning for
every zero denominator; that choice makes "found nothing, there was nothing" look
like a failure. AP uses a *different* rule (see below) because averaging a vacuous
1.0 into a mAP would inflate it.

**AP interpolation.** Two conventions are implemented and they do not agree:

* ``"all-points"`` (default) is the VOC2010-and-later rule: take the precision
  envelope (precision made monotonically non-increasing as recall grows) and
  integrate it exactly over recall. This is the true area under the interpolated
  PR curve.
* ``"101-point"`` is the COCO rule: sample that same envelope at 101 evenly spaced
  recall levels 0.00, 0.01, ..., 1.00 and average. It is a quantized approximation
  of the same integral and is what pycocotools, faster-coco-eval, torchmetrics,
  Ultralytics and most published COCO numbers report.

The gap is small but real (order 0.5-1 point, larger when there are few ground
truths, because each recall step is then a coarse fraction). Papers routinely fail
to say which they used, which is why both are available and why the choice is a
named argument on every entry point instead of a hidden constant. The default is
all-points because it is the quantity the numbers are meant to approximate; pass
``method="101-point"`` when comparing against a COCO-reported figure.

A third convention exists and is deliberately *not* implemented here:
``sklearn.metrics.average_precision_score`` sums the precision-recall steps without
taking the envelope at all. It agrees with all-points whenever precision happens to
be non-increasing already and reports a lower number whenever it is not, so it is
not interchangeable with a detection AP even though the name matches. It also has no
way to represent a ground truth that no prediction reached, which is most of what a
detector gets wrong.

**Confusion-matrix orientation.** Rows are ground truth, columns are prediction,
with a trailing background row and column. That is the layout of the legacy
``eval_cmatrix_f1_map.py`` output and of its README table, so old and new reports
can be diffed cell by cell.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mdebris.eval.matching import MatchResult, match_detections
from mdebris.types import Detection, DetectionSet, SurfaceClass

__all__ = [
    "BACKGROUND",
    "DEFAULT_AP_METHOD",
    "IOU_THRESHOLDS_50_95",
    "APMethod",
    "ClassMetrics",
    "EvaluationResult",
    "ap_from_flags",
    "average_precision",
    "average_precision_per_class",
    "confusion_matrix",
    "confusion_matrix_labels",
    "evaluate",
    "map_50_95",
    "match_result_counts",
    "mean_average_precision",
    "pr_curve",
    "precision_recall_f1",
]

APMethod = Literal["all-points", "101-point"]

DEFAULT_AP_METHOD: APMethod = "all-points"

#: Label used for the background row and column of a confusion matrix.
BACKGROUND = "background"

#: COCO's IoU sweep, 0.50 to 0.95 in steps of 0.05. Built with ``linspace`` rather
#: than ``arange`` because ``arange(0.5, 1.0, 0.05)`` accumulates float error and
#: yields 0.6000000000000001; this is the same construction pycocotools uses.
IOU_THRESHOLDS_50_95: tuple[float, ...] = tuple(
    round(float(t), 2) for t in np.linspace(0.5, 0.95, 10)
)

# 101 recall sample points, COCO's ``recThrs``.
_REC_THRESHOLDS = np.linspace(0.0, 1.0, 101)


# ---------------------------------------------------------------------------
# counting metrics
# ---------------------------------------------------------------------------


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 from raw counts, with the zero-denominator rule applied.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        ``{"tp", "fp", "fn", "precision", "recall", "f1"}``. Counts come back as
        floats so the dict is uniformly typed and JSON-safe.

    The conventions, restated because they are the part people get wrong:

    * ``tp + fp == 0`` (nothing predicted): precision is **1.0**. No prediction was
      wrong because no prediction was made.
    * ``tp + fn == 0`` (nothing to find): recall is **1.0**. Nothing was missed.
    * ``precision + recall == 0``: F1 is **0.0**.

    Together these give the four degenerate cases:

    ===================== ========= ====== ===
    situation             precision recall F1
    ===================== ========= ====== ===
    no preds, no GT       1.0       1.0    1.0
    preds, no GT          0.0       1.0    0.0
    no preds, GT          1.0       0.0    0.0
    preds and GT, no hits 0.0       0.0    0.0
    ===================== ========= ====== ===
    """
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError(f"counts must be non-negative, got tp={tp}, fp={fp}, fn={fn}")

    n_pred = tp + fp
    n_gt = tp + fn
    precision = 1.0 if n_pred == 0 else tp / n_pred
    recall = 1.0 if n_gt == 0 else tp / n_gt
    denom = precision + recall
    f1 = 0.0 if denom == 0.0 else 2.0 * precision * recall / denom
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# precision-recall curve and average precision
# ---------------------------------------------------------------------------


def pr_curve(
    scores: Sequence[float] | np.ndarray,
    tp_flags: Sequence[bool] | np.ndarray,
    n_gt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recall and precision at every rank of the score-sorted prediction list.

    Args:
        scores: Prediction scores, in any order.
        tp_flags: True-positive flag per prediction, aligned with ``scores``.
        n_gt: Number of ground truths, including those nothing matched. Must be > 0;
            recall is undefined otherwise.

    Returns:
        ``(recall, precision)``, each of length ``len(scores)``, ordered by
        descending score. Recall is non-decreasing; precision is not monotone and is
        deliberately returned raw so callers can see the sawtooth before any
        interpolation is applied.

    Ties in score are broken stably, preserving the order the flags arrived in. That
    matters: :func:`~mdebris.eval.matching.match_detections` credits the first of two
    equally scored predictions, and a stable sort here keeps that true positive ahead
    of its twin instead of pessimising the curve.
    """
    if n_gt <= 0:
        raise ValueError("pr_curve needs at least one ground truth; recall is undefined at n_gt=0")
    scores_arr = np.asarray(scores, dtype=np.float64)
    tp_arr = np.asarray(tp_flags, dtype=bool)
    if scores_arr.shape != tp_arr.shape:
        raise ValueError(f"scores {scores_arr.shape} and tp_flags {tp_arr.shape} differ in length")

    order = np.argsort(-scores_arr, kind="stable")
    tp_sorted = tp_arr[order]
    tp_cum = np.cumsum(tp_sorted, dtype=np.float64)
    fp_cum = np.cumsum(~tp_sorted, dtype=np.float64)

    recall = tp_cum / float(n_gt)
    # The denominator is the rank (1, 2, 3, ...), so it is never zero.
    precision = tp_cum / (tp_cum + fp_cum)
    return recall, precision


def _precision_envelope(precision: np.ndarray) -> np.ndarray:
    """Make precision monotonically non-increasing in recall.

    ``envelope[i] = max(precision[i:])``: the best precision achievable at recall
    ``recall[i]`` or beyond. Both AP conventions integrate this envelope rather than
    the raw sawtooth, on the reasoning that a detector could always have used a
    lower score cut-off to reach that better operating point.
    """
    return np.maximum.accumulate(precision[::-1])[::-1]


def _ap_all_points(recall: np.ndarray, precision: np.ndarray) -> float:
    """Exact area under the interpolated PR curve (VOC2010 and later).

    Recall is padded with 0 at the start and 1 at the end, precision with 0 at both
    ends, so the region above the highest achieved recall contributes exactly zero
    area instead of being extrapolated.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = _precision_envelope(mpre)
    steps = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


def _ap_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO's 101-point interpolated AP.

    Reproduces ``pycocotools.cocoeval.COCOeval.accumulate``: smooth precision into an
    envelope, then for each of the 101 recall thresholds take the precision at the
    first rank whose recall reaches that threshold, using 0.0 where the threshold is
    never reached. The mean of those 101 samples is AP.
    """
    envelope = _precision_envelope(precision)
    samples = np.zeros_like(_REC_THRESHOLDS)
    idx = np.searchsorted(recall, _REC_THRESHOLDS, side="left")
    reachable = idx < len(envelope)
    samples[reachable] = envelope[idx[reachable]]
    return float(samples.mean())


def ap_from_flags(
    scores: Sequence[float] | np.ndarray,
    tp_flags: Sequence[bool] | np.ndarray,
    n_gt: int,
    *,
    method: APMethod = DEFAULT_AP_METHOD,
) -> float:
    """Average precision from a ranked list of true-positive flags.

    This is the AP kernel; the box-level entry points are thin wrappers around it.
    It is public because pooling matches across many images (evaluate each image
    separately, concatenate the flags and sum the ground-truth counts) is the correct
    way to compute a dataset-level AP, and that requires reaching the kernel directly.

    Args:
        scores: Prediction scores, any order.
        tp_flags: True-positive flag per prediction, aligned with ``scores``.
        n_gt: Total ground truths for this class.
        method: ``"all-points"`` or ``"101-point"``. See the module docstring.

    Returns:
        AP in [0, 1], or ``nan`` when ``n_gt == 0``. NaN rather than 0.0 or COCO's
        -1 sentinel: a class with no ground truth has no defined AP, and NaN
        propagates loudly through any average that forgets to exclude it.
        ``mean_average_precision`` excludes it explicitly.
    """
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    recall, precision = pr_curve(scores, tp_flags, n_gt)
    if method == "all-points":
        return _ap_all_points(recall, precision)
    if method == "101-point":
        return _ap_101_point(recall, precision)
    raise ValueError(f"unknown AP method {method!r}; expected 'all-points' or '101-point'")


def average_precision(
    preds: list[Detection],
    gts: list[Detection],
    iou_threshold: float = 0.5,
    *,
    method: APMethod = DEFAULT_AP_METHOD,
) -> float:
    """Average precision for one class, from raw detections.

    Matching is class-aware, so a debris prediction cannot claim a sargassum ground
    truth even when the boxes coincide. The recall denominator is ``len(gts)``,
    which means this returns a genuine per-class AP only when ``gts`` holds a single
    class. For multi-class input use :func:`average_precision_per_class` or
    :func:`mean_average_precision`; passing a mixed list here computes a pooled AP
    (correct localisation, but recall measured against every class at once), which
    is a different and usually not the intended quantity.

    Args:
        preds: Predicted detections for the scene.
        gts: Ground-truth detections for the scene.
        iou_threshold: IoU at or above which a prediction counts as a match.
        method: AP interpolation convention.

    Returns:
        AP in [0, 1], or ``nan`` when there are no ground truths.
    """
    result = match_detections(preds, gts, iou_threshold=iou_threshold)
    return ap_from_flags(result.scores, result.tp, result.n_gt, method=method)


def _present_classes(
    preds: Iterable[Detection], gts: Iterable[Detection]
) -> tuple[SurfaceClass, ...]:
    """Classes appearing in either list, in :class:`SurfaceClass` declaration order.

    Declaration order rather than alphabetical or first-seen order, so a report's
    column order does not depend on which detections happened to be produced.
    """
    seen = {d.label for d in preds} | {d.label for d in gts}
    return tuple(c for c in SurfaceClass if c in seen)


def average_precision_per_class(
    preds: list[Detection],
    gts: list[Detection],
    *,
    iou_threshold: float = 0.5,
    classes: Sequence[SurfaceClass] | None = None,
    method: APMethod = DEFAULT_AP_METHOD,
) -> dict[SurfaceClass, float]:
    """AP for each class, computed independently on that class's detections.

    Args:
        preds: Predicted detections.
        gts: Ground-truth detections.
        iou_threshold: IoU at or above which a prediction counts as a match.
        classes: Classes to report. Defaults to those present in either input.
        method: AP interpolation convention.

    Returns:
        ``{class: ap}``, with ``nan`` for any class carrying no ground truth.
    """
    labels = tuple(classes) if classes is not None else _present_classes(preds, gts)
    out: dict[SurfaceClass, float] = {}
    for label in labels:
        class_preds = [p for p in preds if p.label == label]
        class_gts = [g for g in gts if g.label == label]
        out[label] = average_precision(
            class_preds, class_gts, iou_threshold=iou_threshold, method=method
        )
    return out


def mean_average_precision(
    preds: list[Detection],
    gts: list[Detection],
    *,
    iou_threshold: float = 0.5,
    classes: Sequence[SurfaceClass] | None = None,
    method: APMethod = DEFAULT_AP_METHOD,
) -> float:
    """Mean of the per-class APs at a single IoU threshold.

    Classes with no ground truth are **excluded** from the mean rather than counted
    as zero. This is COCO behaviour (pycocotools stores -1 and filters it out) and it
    matters here: the taxonomy has nine classes and a typical scene contains two, so
    counting absent classes as zero would divide every score by four or five.

    Args:
        preds: Predicted detections.
        gts: Ground-truth detections.
        iou_threshold: IoU at or above which a prediction counts as a match.
        classes: Classes to average over. Defaults to those present in either input.
        method: AP interpolation convention.

    Returns:
        The mean AP, or ``nan`` if no class has any ground truth.
    """
    per_class = average_precision_per_class(
        preds, gts, iou_threshold=iou_threshold, classes=classes, method=method
    )
    defined = [ap for ap in per_class.values() if not np.isnan(ap)]
    return float(np.mean(defined)) if defined else float("nan")


def map_50_95(
    preds: list[Detection],
    gts: list[Detection],
    *,
    classes: Sequence[SurfaceClass] | None = None,
    method: APMethod = DEFAULT_AP_METHOD,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS_50_95,
) -> float:
    """COCO's primary metric: mAP averaged over IoU 0.50 to 0.95 in steps of 0.05.

    Averaging over thresholds is what stops a detector from being rewarded for boxes
    that are merely in the right neighbourhood. mAP@0.5 alone is nearly a localisation-
    free score; the 0.75 and above terms are where box quality shows up.

    Args:
        preds: Predicted detections.
        gts: Ground-truth detections.
        classes: Classes to average over. Defaults to those present in either input.
        method: AP interpolation convention.
        iou_thresholds: Thresholds to average over. Defaults to :data:`IOU_THRESHOLDS_50_95`.

    Returns:
        The mean over thresholds of the per-threshold mAP, or ``nan`` if no class has
        any ground truth. Class set is fixed across thresholds, so every threshold
        contributes the same classes.
    """
    labels = tuple(classes) if classes is not None else _present_classes(preds, gts)
    values = [
        mean_average_precision(preds, gts, iou_threshold=float(t), classes=labels, method=method)
        for t in iou_thresholds
    ]
    defined = [v for v in values if not np.isnan(v)]
    return float(np.mean(defined)) if defined else float("nan")


# ---------------------------------------------------------------------------
# confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix_labels(classes: Sequence[SurfaceClass]) -> list[str]:
    """Row and column names for :func:`confusion_matrix`, background last."""
    return [str(c) for c in classes] + [BACKGROUND]


def confusion_matrix(
    preds: list[Detection],
    gts: list[Detection],
    *,
    iou_threshold: float = 0.5,
    classes: Sequence[SurfaceClass],
    class_agnostic_matching: bool = True,
) -> np.ndarray:
    """Detection confusion matrix with an explicit background row and column.

    Orientation, stated once and relied on everywhere: **rows are ground truth,
    columns are prediction**, and index ``len(classes)`` is background. This is the
    layout the legacy ``eval_cmatrix_f1_map.py`` wrote and the layout of the legacy
    README table, so for the single-class case the matrix is::

        [[TP, FN],
         [FP,  0]]

    Cell meanings:

    * ``cm[i, j]`` for ``i, j < K``: a ground truth of class ``i`` was matched by a
      prediction of class ``j``. Off-diagonal entries are class confusions.
    * ``cm[i, K]``: a ground truth of class ``i`` that nothing matched (false negative).
    * ``cm[K, j]``: a prediction of class ``j`` that matched no ground truth (false
      positive against background).
    * ``cm[K, K]``: always 0. True negatives are not enumerable in detection, since
      the set of boxes that correctly were not predicted is unbounded. The legacy
      table printed a 0 in that corner for the same reason.

    Args:
        preds: Predicted detections.
        gts: Ground-truth detections.
        iou_threshold: IoU at or above which a prediction counts as a match.
        classes: Class order for rows and columns. Required, not inferred, so that
            two reports over different scenes stay comparable.
        class_agnostic_matching: When ``True`` (default) boxes are matched on IoU
            alone and the labels are only read afterwards to pick the cell. This is
            what the TF Object Detection API did and it is the only setting under
            which off-diagonal class confusions can appear at all: with class-aware
            matching a debris prediction sitting exactly on a sargassum ground truth
            is scored as one false positive plus one false negative, and the fact
            that the two are the same object is lost. Set ``False`` to make the
            matrix consistent with the per-class AP numbers instead.

    Returns:
        An integer array of shape ``(len(classes) + 1, len(classes) + 1)``.

    Raises:
        ValueError: If any detection carries a label outside ``classes``. Silently
            dropping them would quietly shrink the ground-truth count and inflate
            recall, so it is refused rather than tolerated.
    """
    index = {c: i for i, c in enumerate(classes)}
    if len(index) != len(classes):
        raise ValueError(f"duplicate entries in classes: {classes}")
    for kind, dets in (("prediction", preds), ("ground truth", gts)):
        unknown = {d.label for d in dets} - index.keys()
        if unknown:
            raise ValueError(
                f"{kind} labels {sorted(str(u) for u in unknown)} are not in classes "
                f"{[str(c) for c in classes]}"
            )

    bg = len(classes)
    cm = np.zeros((bg + 1, bg + 1), dtype=np.int64)

    result = match_detections(
        preds, gts, iou_threshold=iou_threshold, class_agnostic=class_agnostic_matching
    )
    for pred_index, gt_i, _iou in result.pairs():
        cm[index[gts[gt_i].label], index[preds[pred_index].label]] += 1
    for gt_i in result.unmatched_gt:
        cm[index[gts[gt_i].label], bg] += 1
    for rank, pred_index in enumerate(result.order):
        if not result.tp[rank]:
            cm[bg, index[preds[pred_index].label]] += 1
    return cm


# ---------------------------------------------------------------------------
# the tie-it-together result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Per-class scores at a single IoU threshold.

    ``ap`` is ``nan`` when ``n_gt`` is 0. ``precision``, ``recall`` and ``f1`` follow
    the zero-denominator convention of :func:`precision_recall_f1`, so they stay
    finite in that case.
    """

    label: SurfaceClass
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap: float
    n_pred: int
    n_gt: int

    @property
    def name(self) -> str:
        return str(self.label)


@dataclass(slots=True, eq=False)
class EvaluationResult:
    """Everything :func:`evaluate` computed, in one object the report layer can render.

    ``eq`` is disabled because the dataclass holds a numpy array, and the generated
    ``__eq__`` would compare it elementwise and then raise on the ambiguous truth
    value. Compare the fields you care about instead.

    Attributes:
        classes: Class order used for rows, columns and tables.
        per_class: Per-class metrics, keyed by class, in ``classes`` order.
        confusion: ``(K+1, K+1)`` matrix from :func:`confusion_matrix`, rows = truth.
        micro: Pooled counts and scores over all classes, from
            :func:`precision_recall_f1`. Pooled ("micro") rather than averaged
            ("macro"), matching the legacy single-class report where the two coincide.
        mean_ap: Mean of the defined per-class APs at ``iou_threshold``.
        mean_ap_50_95: mAP averaged over IoU 0.50:0.05:0.95, or ``None`` if not requested.
        iou_threshold: Primary IoU threshold for every non-sweep number here.
        ap_method: Which AP interpolation was used.
        score_threshold: Predictions below this score were dropped before scoring.
        n_pred: Predictions kept after the score threshold.
        n_gt: Ground truths.
        scene_id: Scene the detections came from, when known.
    """

    classes: tuple[SurfaceClass, ...]
    per_class: dict[SurfaceClass, ClassMetrics]
    confusion: np.ndarray
    micro: dict[str, float]
    mean_ap: float
    mean_ap_50_95: float | None
    iou_threshold: float
    ap_method: APMethod
    score_threshold: float
    n_pred: int
    n_gt: int
    scene_id: str | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def tp(self) -> int:
        return int(self.micro["tp"])

    @property
    def fp(self) -> int:
        return int(self.micro["fp"])

    @property
    def fn(self) -> int:
        return int(self.micro["fn"])

    @property
    def precision(self) -> float:
        return self.micro["precision"]

    @property
    def recall(self) -> float:
        return self.micro["recall"]

    @property
    def f1(self) -> float:
        return self.micro["f1"]


def evaluate(
    pred_set: DetectionSet,
    gt_set: DetectionSet,
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    classes: Sequence[SurfaceClass] | None = None,
    method: APMethod = DEFAULT_AP_METHOD,
    include_map_50_95: bool = True,
) -> EvaluationResult:
    """Score one prediction set against one ground-truth set.

    Both sets are treated as a single image: matching is global across the set, so a
    prediction on one tile could in principle claim a ground truth on another. That
    is correct for the intended use (one :class:`~mdebris.types.DetectionSet` is one
    scene or AOI, with boxes in a shared coordinate frame) and wrong if you
    concatenate unrelated images into one set. To score a multi-image test set,
    evaluate per image and pool the match flags through
    :func:`ap_from_flags`; averaging per-image APs is not the same number.

    Args:
        pred_set: Predictions.
        gt_set: Ground truth. Its detections' ``score`` fields are ignored.
        iou_threshold: Primary IoU threshold.
        score_threshold: Drop predictions scoring below this before matching. The
            legacy script hardcoded 0.5 here; the default is 0.0 because thresholding
            before computing AP truncates the PR curve and lowers AP artificially.
        classes: Class order. Defaults to those present in either set, in
            :class:`SurfaceClass` declaration order.
        method: AP interpolation convention.
        include_map_50_95: Compute the IoU sweep. It costs ten matching passes, so it
            can be switched off in a hot loop.

    Returns:
        An :class:`EvaluationResult`. A class passed explicitly in ``classes`` but
        absent from both sets gets the vacuous row (precision 1.0, recall 1.0, F1 1.0,
        AP ``nan``); it is excluded from ``mean_ap`` because its AP is undefined.
    """
    preds = [d for d in pred_set if d.score >= score_threshold]
    gts = list(gt_set)
    labels = tuple(classes) if classes is not None else _present_classes(preds, gts)

    per_class: dict[SurfaceClass, ClassMetrics] = {}
    for label in labels:
        class_preds = [p for p in preds if p.label == label]
        class_gts = [g for g in gts if g.label == label]
        match = match_detections(class_preds, class_gts, iou_threshold=iou_threshold)
        counts = precision_recall_f1(match.n_tp, match.n_fp, match.n_fn)
        per_class[label] = ClassMetrics(
            label=label,
            tp=match.n_tp,
            fp=match.n_fp,
            fn=match.n_fn,
            precision=counts["precision"],
            recall=counts["recall"],
            f1=counts["f1"],
            ap=ap_from_flags(match.scores, match.tp, match.n_gt, method=method),
            n_pred=match.n_pred,
            n_gt=match.n_gt,
        )

    micro = precision_recall_f1(
        tp=sum(m.tp for m in per_class.values()),
        fp=sum(m.fp for m in per_class.values()),
        fn=sum(m.fn for m in per_class.values()),
    )
    defined_ap = [m.ap for m in per_class.values() if not np.isnan(m.ap)]
    mean_ap = float(np.mean(defined_ap)) if defined_ap else float("nan")

    return EvaluationResult(
        classes=labels,
        per_class=per_class,
        confusion=confusion_matrix(preds, gts, iou_threshold=iou_threshold, classes=labels),
        micro=micro,
        mean_ap=mean_ap,
        mean_ap_50_95=(
            map_50_95(preds, gts, classes=labels, method=method) if include_map_50_95 else None
        ),
        iou_threshold=iou_threshold,
        ap_method=method,
        score_threshold=score_threshold,
        n_pred=len(preds),
        n_gt=len(gts),
        scene_id=(gt_set.scene or pred_set.scene).scene_id
        if (gt_set.scene or pred_set.scene)
        else None,
    )


def match_result_counts(result: MatchResult) -> dict[str, float]:
    """Convenience bridge from a :class:`MatchResult` to the scalar scores."""
    return precision_recall_f1(result.n_tp, result.n_fp, result.n_fn)
