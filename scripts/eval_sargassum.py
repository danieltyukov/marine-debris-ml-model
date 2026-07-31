"""Score the MARIDA-trained classifier as a sargassum detector.

    python scripts/eval_sargassum.py

The repository leads with the *Marine Debris* class, where the honest answer is a
negative result: best F1 0.515, below the MARIDA paper's Random Forest baseline.
That is the right thing to lead with for a debris detector, and it buried a
second fact that matters for anyone asking what this is good for.

MARIDA labels Dense and Sparse Sargassum as separate classes, and the same model,
trained in the same run on the same scene-grouped split, scores far better on
them than on debris. That is not surprising once stated: sargassum floats in
mats tens of metres across with a chlorophyll red edge, so it fills pixels and
has a spectral signature the indices key on directly. Debris is thin filaments
of low contrast at 10 m. Same model, same features, different physical target.

This script reports the sargassum result on its own terms rather than as one row
of a fifteen-class table:

* per-class scores for Dense and Sparse Sargassum
* the binary "any sargassum vs everything else" task, which is what an operational
  user actually asks, and its full precision-recall curve
* the confusion that remains, so the failure modes are visible

Nothing is retrained. The model file produced by ``train_marida.py`` is loaded and
re-scored, so these numbers cannot drift from the ones in the MARIDA report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from mdebris.models.spectral import SpectralClassifier

log = logging.getLogger("sargassum")

# MARIDA's two sargassum classes. Kept as a tuple rather than inlined because the
# binary task is defined entirely by this membership, and a reader should be able
# to see exactly what "sargassum" means here without reading the code below.
SARGASSUM_CLASSES: tuple[str, ...] = ("Dense Sargassum", "Sparse Sargassum")


def _binary_scores(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> np.ndarray:
    """Total probability mass assigned to any sargassum class.

    Summing the two class probabilities is correct rather than a shortcut: the
    classes are mutually exclusive in the label space, so the probability that a
    pixel is sargassum of either density is the sum. Taking the max instead would
    understate every pixel the model splits between dense and sparse, which is
    exactly the ambiguous middle of a mat.
    """
    idx = [classes.index(c) for c in SARGASSUM_CLASSES if c in classes]
    if not idx:
        raise RuntimeError(
            f"model has no sargassum classes; it knows {classes}. "
            "Retrain with scripts/train_marida.py."
        )
    return proba[:, idx].sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/sargassum_report.md"))
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--min-confidence", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.model.exists():
        raise SystemExit(
            f"no model at {args.model}. Run scripts/train_marida.py first, which "
            "downloads MARIDA and fits the classifier."
        )

    # Imported here rather than at module scope so --help works without MARIDA on disk.
    from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

    # Running this file puts scripts/ on sys.path, so the sibling imports flat.
    from train_marida import collect

    log.info("loading the held-out MARIDA test split")
    x_test, y_test = collect(
        "test", max_patches=args.max_patches, min_confidence=args.min_confidence
    )

    clf = SpectralClassifier.load(args.model)
    classes = list(clf._model.classes_)
    proba = clf.predict_proba(x_test)
    predicted = np.array(classes)[proba.argmax(axis=1)]

    per_class: dict[str, dict[str, float]] = {}
    for name in SARGASSUM_CLASSES:
        if name not in classes:
            continue
        precision, recall, f1, _support = precision_recall_fscore_support(
            y_test == name, predicted == name, average="binary", zero_division=0
        )
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((y_test == name).sum()),
        }
        log.info(
            "%-18s P %.3f  R %.3f  F1 %.3f  on %d pixels",
            name,
            precision,
            recall,
            f1,
            per_class[name]["support"],
        )

    truth = np.isin(y_test, SARGASSUM_CLASSES)
    scores = _binary_scores(y_test, proba, classes)
    predicted_binary = np.isin(predicted, SARGASSUM_CLASSES)

    argmax_p, argmax_r, argmax_f1, _ = precision_recall_fscore_support(
        truth, predicted_binary, average="binary", zero_division=0
    )
    log.info(
        "any sargassum, argmax:  P %.3f  R %.3f  F1 %.3f  on %d of %d pixels",
        argmax_p,
        argmax_r,
        argmax_f1,
        int(truth.sum()),
        len(truth),
    )

    precision, recall, thresholds = precision_recall_curve(truth.astype(int), scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    best_threshold = float(thresholds[min(best, len(thresholds) - 1)])
    log.info(
        "any sargassum, best F1: P %.3f  R %.3f  F1 %.3f  at probability %.3f",
        precision[best],
        recall[best],
        f1[best],
        best_threshold,
    )

    # The operating point a dispatcher would pick: the highest recall still holding
    # 90% precision. Sending a crew to a clean beach costs a shift, so precision is
    # the constraint and recall is what is maximised under it.
    usable = precision >= 0.90
    if usable.any():
        pick = int(np.argmax(np.where(usable, recall, -1.0)))
        high_precision = {
            "precision": float(precision[pick]),
            "recall": float(recall[pick]),
            "f1": float(f1[pick]),
            "threshold": float(thresholds[min(pick, len(thresholds) - 1)]),
        }
        log.info(
            "at 90%% precision:      P %.3f  R %.3f  F1 %.3f  at probability %.3f",
            high_precision["precision"],
            high_precision["recall"],
            high_precision["f1"],
            high_precision["threshold"],
        )
    else:
        high_precision = {}
        log.warning("the model never reaches 90% precision on sargassum")

    # What the misses actually are. A sargassum pixel called Marine Water is a
    # different problem from one called Dense Sargassum-adjacent Natural Organic
    # Material, and only the confusion tells them apart.
    missed = predicted[truth & ~predicted_binary]
    confusion = {
        str(name): int(count)
        for name, count in zip(*np.unique(missed, return_counts=True), strict=True)
    }
    false_alarms = predicted_binary & ~truth
    alarm_sources = {
        str(name): int(count)
        for name, count in zip(*np.unique(y_test[false_alarms], return_counts=True), strict=True)
    }

    step = max(1, len(precision) // 200)
    curve = [
        {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i in range(0, len(precision), step)
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _markdown(
            per_class=per_class,
            n_positive=int(truth.sum()),
            n_total=len(truth),
            argmax=(float(argmax_p), float(argmax_r), float(argmax_f1)),
            best=(float(precision[best]), float(recall[best]), float(f1[best]), best_threshold),
            high_precision=high_precision,
            confusion=confusion,
            alarm_sources=alarm_sources,
        ),
        encoding="utf-8",
    )
    args.report.with_suffix(".json").write_text(
        json.dumps(
            {
                "per_class": per_class,
                "binary": {
                    "n_positive": int(truth.sum()),
                    "n_total": len(truth),
                    "argmax": {
                        "precision": float(argmax_p),
                        "recall": float(argmax_r),
                        "f1": float(argmax_f1),
                    },
                    "best_f1": {
                        "precision": float(precision[best]),
                        "recall": float(recall[best]),
                        "f1": float(f1[best]),
                        "threshold": best_threshold,
                    },
                    "high_precision": high_precision,
                },
                "missed_as": confusion,
                "false_alarms_from": alarm_sources,
                "pr_curve": curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", args.report)


def _markdown(
    *,
    per_class: dict[str, dict[str, float]],
    n_positive: int,
    n_total: int,
    argmax: tuple[float, float, float],
    best: tuple[float, float, float, float],
    high_precision: dict[str, float],
    confusion: dict[str, int],
    alarm_sources: dict[str, int],
) -> str:
    """Render the report, written to be readable by someone who is not a remote-sensing person."""
    lines = [
        "# Sargassum detection on MARIDA",
        "",
        "Produced by `python scripts/eval_sargassum.py`. No retraining happens here:",
        "the classifier fitted by `scripts/train_marida.py` is loaded and re-scored on",
        "MARIDA's own scene-grouped held-out test split, so these numbers and the ones",
        "in `marida_report.md` come from the same model.",
        "",
        "## Why this is reported separately",
        "",
        "The headline result in this repository is a negative one about *marine debris*:",
        "at 10 m ground sampling a debris filament is a few low-contrast pixels, and the",
        "best F1 is 0.515. Sargassum is a different physical target. It floats in mats",
        "tens of metres across and carries a chlorophyll red edge, so it fills pixels and",
        "the spectral indices key on it directly. Same model, same 18 features, same split.",
        "",
        "## Per class",
        "",
        "| class | precision | recall | F1 | test pixels |",
        "|---|---|---|---|---|",
    ]
    for name, m in per_class.items():
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"**{m['f1']:.3f}** | {m['support']:,} |"
        )

    lines += [
        "",
        "## Any sargassum vs everything else",
        "",
        f"The operational question is not which density it is, it is whether there is any. "
        f"{n_positive:,} of {n_total:,} held-out pixels are sargassum of either class.",
        "",
        "| operating point | precision | recall | F1 |",
        "|---|---|---|---|",
        f"| `argmax` default | {argmax[0]:.3f} | {argmax[1]:.3f} | {argmax[2]:.3f} |",
        f"| best F1, threshold {best[3]:.3f} | {best[0]:.3f} | {best[1]:.3f} | **{best[2]:.3f}** |",
    ]
    if high_precision:
        lines.append(
            f"| 90% precision, threshold {high_precision['threshold']:.3f} | "
            f"{high_precision['precision']:.3f} | {high_precision['recall']:.3f} | "
            f"{high_precision['f1']:.3f} |"
        )
    lines += [
        "",
        "The last row is the one that matters for dispatch. Sending a crew to a clean",
        "beach costs a shift, so precision is the constraint and recall is whatever can",
        "be had under it.",
        "",
        "## What it gets wrong",
        "",
        "Sargassum pixels the model missed, by what it called them instead:",
        "",
        "| called | pixels |",
        "|---|---|",
    ]
    for name, count in sorted(confusion.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"| {name} | {count:,} |")
    lines += [
        "",
        "False alarms, by what they actually were:",
        "",
        "| actually | pixels |",
        "|---|---|",
    ]
    for name, count in sorted(alarm_sources.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"| {name} | {count:,} |")

    lines += [
        "",
        "## What this does not say",
        "",
        "MARIDA's sargassum labels are Sentinel-2 pixels annotated by remote-sensing",
        "researchers, not field observations, and its scenes are not the Mexican Caribbean.",
        "A per-pixel score on a benchmark is not a validated landfall forecast, and nothing",
        "here measures whether material detected offshore reaches a particular beach. Those",
        "are separate claims and this file supports neither of them.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
