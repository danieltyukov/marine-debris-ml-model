"""Train the spectral pixel classifier on MARIDA and report held-out accuracy.

    python scripts/train_marida.py --out models/marida_spectral.joblib

MARIDA (Kikaki et al. 2022, PLoS ONE 17(1): e0262247) is the public Sentinel-2
marine-debris benchmark. It labels 15 sea-surface classes, including the confusers
that matter: Sargassum, foam, wakes, sediment-laden water and ships.

This exists because measurement said it should. The zero-shot open-vocabulary path
does not transfer to 10 m imagery (0.13 to 0.21 confidence with whole-chip boxes,
against 0.67 to 0.79 on natural photographs). The spectrum, not the picture, carries
the signal at this resolution, so the model is trained on labelled pixels.

The split is MARIDA's own train/val/test partition, which is grouped by scene rather
than sampled at random. That matters: neighbouring pixels in one patch are highly
correlated, so a random pixel split would leak and report an inflated score.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from mdebris.data import MARIDA_BANDS, MARIDA_CLASSES, load_marida_split
from mdebris.models.spectral import SpectralClassifier, build_features, feature_names

log = logging.getLogger("train")

# MARIDA encodes 0 as unlabelled; classes run 1..15 in MARIDA_CLASSES order.
UNLABELLED = 0


def collect(split: str, *, max_patches: int | None = None, min_confidence: int = 0):
    """Build a feature matrix and label vector from every labelled pixel in a split.

    Args:
        split: One of train, val, test.
        max_patches: Cap the number of patches, for quick runs.
        min_confidence: Drop pixels below this MARIDA confidence level. The dataset
            grades annotations, and low-confidence pixels are exactly the ambiguous
            ones a model should not be graded against.

    Returns:
        (features, labels) where labels are MARIDA class name strings.
    """
    patches = load_marida_split(split)
    if max_patches:
        patches = patches[:max_patches]

    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for i, patch in enumerate(patches):
        image, classes, confidence = patch.read()
        labelled = classes != UNLABELLED
        if min_confidence:
            labelled &= confidence >= min_confidence
        if not labelled.any():
            continue
        bands = {name: image[j] for j, name in enumerate(MARIDA_BANDS)}
        features = build_features(bands).reshape(*classes.shape, -1)
        feature_rows.append(features[labelled])
        label_rows.append(classes[labelled])
        if (i + 1) % 150 == 0:
            log.info("  %s: %d/%d patches", split, i + 1, len(patches))

    if not feature_rows:
        raise RuntimeError(f"no labelled pixels found in the {split} split")

    x = np.concatenate(feature_rows).astype(np.float32)
    codes = np.concatenate(label_rows)
    names = np.array([MARIDA_CLASSES[c - 1] for c in codes])
    return x, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/marida_report.md"))
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--class-weight",
        default="balanced",
        help="'balanced' or 'none'. Balanced trades precision for recall on rare classes.",
    )
    parser.add_argument("--tag", default="", help="Suffix for output filenames.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("loading MARIDA splits")
    x_train, y_train = collect(
        "train", max_patches=args.max_patches, min_confidence=args.min_confidence
    )
    x_test, y_test = collect(
        "test", max_patches=args.max_patches, min_confidence=args.min_confidence
    )
    log.info(
        "train %s pixels, test %s pixels, %d features",
        f"{len(x_train):,}",
        f"{len(x_test):,}",
        x_train.shape[1],
    )

    counts = {c: int((y_train == c).sum()) for c in sorted(set(y_train))}
    log.info("class balance in train:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        log.info("  %-28s %8d  %5.2f%%", name, n, 100 * n / len(y_train))

    log.info("fitting")
    started = time.perf_counter()
    weight = None if args.class_weight.lower() in {"none", "null", ""} else args.class_weight
    clf = SpectralClassifier(max_iter=args.max_iter, class_weight=weight).fit(
        x_train, y_train, class_names=list(MARIDA_CLASSES)
    )
    elapsed = time.perf_counter() - started
    log.info("fit in %.1fs", elapsed)

    report = clf.evaluate(x_test, y_test)
    report.n_train = len(x_train)
    report.train_seconds = round(elapsed, 1)
    try:
        from sklearn.inspection import permutation_importance

        # Permutation importance on a subsample: HistGradientBoosting exposes no
        # native importances, and the full test set would take far too long.
        idx = np.random.default_rng(0).choice(
            len(x_test), size=min(4000, len(x_test)), replace=False
        )
        imp = permutation_importance(
            clf._model, x_test[idx], y_test[idx], n_repeats=3, random_state=0, n_jobs=-1
        )
        report.feature_importance = dict(
            zip(feature_names(), (float(v) for v in imp.importances_mean), strict=True)
        )
    except Exception as exc:
        log.warning("skipping feature importance: %s", exc)

    # A single F1 hides that the model is a family of operating points. Debris is
    # 0.2% of labelled pixels, so the precision/recall balance is a deployment
    # decision (wide-area screening versus cleanup tasking), not a model property.
    debris_curve = []
    try:
        from sklearn.metrics import precision_recall_curve

        classes = list(clf._model.classes_)
        if "Marine Debris" in classes:
            scores = clf.predict_proba(x_test)[:, classes.index("Marine Debris")]
            precision, recall, thresholds = precision_recall_curve(
                (y_test == "Marine Debris").astype(int), scores
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) > 0,
            )
            best = int(np.argmax(f1))
            log.info(
                "best debris F1 %.3f at probability threshold %.3f (P %.3f, R %.3f)",
                f1[best],
                thresholds[min(best, len(thresholds) - 1)],
                precision[best],
                recall[best],
            )
            step = max(1, len(precision) // 200)
            debris_curve = [
                {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                for i in range(0, len(precision), step)
            ]
            report.per_class.setdefault("Marine Debris", {})["best_f1"] = float(f1[best])
            report.per_class["Marine Debris"]["best_f1_threshold"] = float(
                thresholds[min(best, len(thresholds) - 1)]
            )
    except Exception as exc:
        log.warning("skipping the debris PR curve: %s", exc)

    print()
    print(report.to_markdown())

    if args.tag:
        args.out = args.out.with_name(f"{args.out.stem}_{args.tag}{args.out.suffix}")
        args.report = args.report.with_name(f"{args.report.stem}_{args.tag}{args.report.suffix}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)
    log.info("\nsaved model to %s", args.out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# MARIDA benchmark results\n\n"
        "Produced by `python scripts/train_marida.py`. MARIDA's own scene-grouped\n"
        "train/test split is used, so no pixels from a test scene appear in training.\n\n"
        + report.to_markdown()
        + "\n",
        encoding="utf-8",
    )
    (args.report.with_suffix(".json")).write_text(
        json.dumps(
            {
                "n_train": report.n_train,
                "n_test": report.n_test,
                "accuracy": report.accuracy,
                "macro_f1": report.macro_f1,
                "per_class": report.per_class,
                "train_seconds": report.train_seconds,
                "class_weight": args.class_weight,
                "debris_pr_curve": debris_curve,
                "feature_importance": report.feature_importance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
