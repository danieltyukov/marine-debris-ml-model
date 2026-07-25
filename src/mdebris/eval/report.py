"""Rendering of an :class:`~mdebris.eval.metrics.EvaluationResult`.

The table shapes here are copied from the legacy report so a new run can be diffed
against the numbers published in the old README: a confusion matrix with a
background ("none") row and column, a TP/FP/FN row, and a per-category row of
precision, recall, mAP and F1 with the legacy ``*_@0.5IOU`` column names.

Two notes on legacy fidelity:

* The legacy ``eval_cmatrix_f1_map.py`` (removed from the tree; readable with
  ``git show 9509b0e:evaluation_utils/eval_cmatrix_f1_map.py``) printed its confusion
  matrix to stdout and wrote only the per-category score table to CSV, even though
  ``evaluation.md`` described the CSV as containing "the confusion matrix and scores".
  :func:`to_csv` can emit either table; ``table="scores"`` reproduces the file that
  script actually wrote, index column and all.
* The legacy ``map_@0.5IOU`` column did not contain an average precision. Its
  computation collapsed to the scalar precision (see
  :func:`mdebris.eval.metrics.average_precision` for what AP actually is), which is
  why the old README reports mAP 0.78 and precision 0.78 for the same class. The
  column is kept for shape parity and now holds a real AP, so that one cell is
  expected to differ from the legacy output.

The background class is spelled ``none`` in the Markdown table, which is what the
legacy README called it, and ``background`` in the JSON and CSV output, which is
what :func:`mdebris.eval.metrics.confusion_matrix_labels` returns. Same row, same
column, two audiences.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Literal

from mdebris.eval.metrics import EvaluationResult, confusion_matrix_labels

__all__ = ["format_json", "format_markdown", "to_csv", "write_report"]

CsvTable = Literal["confusion", "scores"]


def _fmt_iou(threshold: float) -> str:
    """Render an IoU threshold the way the legacy column names did: 0.5, not 0.50."""
    return f"{threshold:g}"


def _fmt(value: float, decimals: int) -> str:
    """Fixed-point float for tables, with NaN shown as ``n/a`` instead of ``nan``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [_row(header), _row(["---"] * len(header))]
    lines.extend(_row(r) for r in rows)
    return "\n".join(lines)


def _json_safe(value: float | None) -> float | None:
    """NaN is not valid JSON, so undefined metrics serialise as null."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def format_markdown(result: EvaluationResult, *, decimals: int = 2) -> str:
    """Render a result as Markdown, in the legacy report's table order.

    Args:
        result: The evaluation to render.
        decimals: Digits for the score table. Two by default, matching the legacy
            README, which is also about as much precision as a few dozen test boxes
            can support.

    Returns:
        A Markdown document: heading, confusion matrix, TP/FP/FN counts, per-category
        scores, then a summary block carrying the information the legacy report had
        no field for (mAP@[.5:.95], the AP interpolation used, thresholds).
    """
    iou = _fmt_iou(result.iou_threshold)
    names = [str(c) for c in result.classes]

    lines: list[str] = ["# Detection evaluation", ""]
    if result.scene_id:
        lines += [f"Scene: `{result.scene_id}`", ""]

    lines += ["## Confusion matrix", ""]
    lines += [
        "Rows are ground truth, columns are predicted. `none` is background: the last "
        "column counts missed ground truths and the last row counts predictions that "
        "matched nothing. The bottom-right cell is always 0 because true negatives are "
        "not countable in detection.",
        "",
    ]
    if len(names) > 1:
        lines += [
            "Boxes are matched here on overlap alone, so an object given the wrong name "
            "appears once, off the diagonal. The counts and per-category tables below "
            "match within a class, so that same object is one false positive for the "
            "predicted class and one false negative for the true one. The two views "
            "coincide whenever there is a single class.",
            "",
        ]
    header = [""] + [f"Predicted {n}" for n in names] + ["Predicted none"]
    rows = []
    for i, name in enumerate([*names, "none"]):
        cells = [f"**True {name}**"] + [str(int(v)) for v in result.confusion[i]]
        rows.append(cells)
    lines += [_table(header, rows), ""]

    lines += ["## Counts", ""]
    lines += [
        _table(
            ["True Positive", "False Positive", "False Negative"],
            [[str(result.tp), str(result.fp), str(result.fn)]],
        ),
        "",
    ]

    lines += ["## Per-category scores", ""]
    score_header = [
        "category",
        f"precision_@{iou}IOU",
        f"recall_@{iou}IOU",
        f"map_@{iou}IOU",
        f"f1_@{iou}IOU",
    ]
    score_rows = []
    for label in result.classes:
        m = result.per_class[label]
        score_rows.append(
            [
                m.name,
                _fmt(m.precision, decimals),
                _fmt(m.recall, decimals),
                _fmt(m.ap, decimals),
                _fmt(m.f1, decimals),
            ]
        )
    lines += [_table(score_header, score_rows), ""]

    lines += ["## Summary", ""]
    summary = [
        ["detections scored", str(result.n_pred)],
        ["ground truths", str(result.n_gt)],
        ["IoU threshold", iou],
        ["score threshold", _fmt(result.score_threshold, decimals)],
        ["AP interpolation", result.ap_method],
        [f"mAP@{iou}", _fmt(result.mean_ap, decimals + 2)],
    ]
    if result.mean_ap_50_95 is not None:
        summary.append(["mAP@[.5:.95]", _fmt(result.mean_ap_50_95, decimals + 2)])
    summary.append([f"micro F1_@{iou}IOU", _fmt(result.f1, decimals + 2)])
    lines += [_table(["metric", "value"], summary), ""]

    return "\n".join(lines)


def format_json(result: EvaluationResult) -> dict[str, Any]:
    """Render a result as a plain dict that :func:`json.dumps` accepts as strict JSON.

    Undefined metrics (an AP for a class with no ground truth) become ``null`` rather
    than ``NaN``, which ``json.dumps`` would happily write and every strict parser
    would then reject.
    """
    return {
        "iou_threshold": result.iou_threshold,
        "score_threshold": result.score_threshold,
        "ap_method": result.ap_method,
        "scene_id": result.scene_id,
        "n_pred": result.n_pred,
        "n_gt": result.n_gt,
        "counts": {"tp": result.tp, "fp": result.fp, "fn": result.fn},
        "micro": {
            "precision": result.micro["precision"],
            "recall": result.micro["recall"],
            "f1": result.micro["f1"],
        },
        "mean_ap": _json_safe(result.mean_ap),
        "mean_ap_50_95": _json_safe(result.mean_ap_50_95),
        "classes": [str(c) for c in result.classes],
        "per_class": [
            {
                "category": result.per_class[label].name,
                "tp": result.per_class[label].tp,
                "fp": result.per_class[label].fp,
                "fn": result.per_class[label].fn,
                "n_pred": result.per_class[label].n_pred,
                "n_gt": result.per_class[label].n_gt,
                "precision": result.per_class[label].precision,
                "recall": result.per_class[label].recall,
                "f1": result.per_class[label].f1,
                "ap": _json_safe(result.per_class[label].ap),
            }
            for label in result.classes
        ],
        "confusion_matrix": {
            "orientation": "rows=ground_truth, columns=predicted",
            "labels": confusion_matrix_labels(result.classes),
            "matrix": result.confusion.astype(int).tolist(),
        },
        "meta": dict(result.meta),
    }


def to_csv(result: EvaluationResult, *, table: CsvTable = "confusion") -> str:
    """Render one table as CSV text.

    Args:
        result: The evaluation to render.
        table: ``"confusion"`` writes the matrix with a leading label column and a
            leading empty header cell (pandas' ``to_csv`` layout for a labelled
            frame). ``"scores"`` writes the per-category table the legacy script
            produced, including the unnamed integer index column pandas emits by
            default, and the legacy ``*_@0.5IOU`` column names.

    Returns:
        CSV text with ``\\n`` line endings and a trailing newline.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    if table == "confusion":
        labels = confusion_matrix_labels(result.classes)
        writer.writerow(["", *labels])
        for name, row in zip(labels, result.confusion, strict=True):
            writer.writerow([name, *(int(v) for v in row)])
    elif table == "scores":
        iou = _fmt_iou(result.iou_threshold)
        writer.writerow(
            [
                "",
                "category",
                f"precision_@{iou}IOU",
                f"recall_@{iou}IOU",
                f"map_@{iou}IOU",
                f"f1_@{iou}IOU",
            ]
        )
        for i, label in enumerate(result.classes):
            m = result.per_class[label]
            writer.writerow([i, m.name, m.precision, m.recall, m.ap, m.f1])
    else:
        raise ValueError(f"unknown table {table!r}; expected 'confusion' or 'scores'")

    return buf.getvalue()


def write_report(
    result: EvaluationResult, out_dir: Path | str, *, stem: str = "eval"
) -> dict[str, Path]:
    """Write the Markdown, JSON and both CSV renderings into ``out_dir``.

    Args:
        result: The evaluation to write.
        out_dir: Directory, created if missing.
        stem: Filename stem shared by all four files.

    Returns:
        ``{"markdown": ..., "json": ..., "confusion_csv": ..., "scores_csv": ...}``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / f"{stem}.md",
        "json": out / f"{stem}.json",
        "confusion_csv": out / f"{stem}_confusion_matrix.csv",
        "scores_csv": out / f"{stem}_scores.csv",
    }
    paths["markdown"].write_text(format_markdown(result), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(format_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    paths["confusion_csv"].write_text(to_csv(result, table="confusion"), encoding="utf-8")
    paths["scores_csv"].write_text(to_csv(result, table="scores"), encoding="utf-8")
    return paths
