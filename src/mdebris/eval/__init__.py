"""Detection evaluation: matching, metrics and reporting.

This subpackage is the scoreboard for everything else in the project, so its
conventions are stated where they are implemented rather than left implicit:

* :mod:`mdebris.eval.matching` defines what counts as a hit (greedy, by descending
  score, one ground truth per prediction).
* :mod:`mdebris.eval.metrics` defines how hits become numbers, including the two
  average-precision interpolation conventions and the zero-denominator rule.
* :mod:`mdebris.eval.report` renders a result in the shape the legacy report used,
  so 2019 numbers and current numbers can be compared cell by cell.

Typical use::

    from mdebris.eval import evaluate, format_markdown

    result = evaluate(predictions, ground_truth, iou_threshold=0.5)
    print(format_markdown(result))
"""

from __future__ import annotations

from mdebris.eval.matching import MatchResult, match_detections
from mdebris.eval.metrics import (
    BACKGROUND,
    DEFAULT_AP_METHOD,
    IOU_THRESHOLDS_50_95,
    APMethod,
    ClassMetrics,
    EvaluationResult,
    ap_from_flags,
    average_precision,
    average_precision_per_class,
    confusion_matrix,
    confusion_matrix_labels,
    evaluate,
    map_50_95,
    match_result_counts,
    mean_average_precision,
    pr_curve,
    precision_recall_f1,
)
from mdebris.eval.report import format_json, format_markdown, to_csv, write_report

__all__ = [
    "BACKGROUND",
    "DEFAULT_AP_METHOD",
    "IOU_THRESHOLDS_50_95",
    "APMethod",
    "ClassMetrics",
    "EvaluationResult",
    "MatchResult",
    "ap_from_flags",
    "average_precision",
    "average_precision_per_class",
    "confusion_matrix",
    "confusion_matrix_labels",
    "evaluate",
    "format_json",
    "format_markdown",
    "map_50_95",
    "match_detections",
    "match_result_counts",
    "mean_average_precision",
    "pr_curve",
    "precision_recall_f1",
    "to_csv",
    "write_report",
]
