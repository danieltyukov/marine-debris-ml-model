"""Train the same classifier on each sensor's band set and score both targets.

    python scripts/eval_band_ablation.py

Every optical satellite carries a different band set, and the NASA IMPACT debris
labels were drawn on 4-band PlanetScope imagery with no red edge and no shortwave
infrared. Before comparing this model at 10 m against that data at 3 m, the question
that has to be answered is how much of the 10 m score comes from bands the 3 m sensor
does not have. Otherwise a resolution comparison is silently a band comparison too.

So this fits one model per scenario in ``mdebris.eval.ablation.SENSOR_SCENARIOS``,
with the hidden bands and the indices that need them set to NaN, and scores the
any-sargassum and marine-debris tasks on the held-out split. Two physical baselines,
FDI and FAI thresholded with no learning, sit beside the learned rows.

Thresholds are chosen on MARIDA's validation split and applied to the test split.
That is stricter than the sargassum report, whose 0.948 was the best F1 over test
thresholds, and the baseline row here is correspondingly a little lower. Both
numbers are reported so nobody has to guess which protocol produced which.

Runtime is about a minute per scenario on CPU; MARIDA must already be on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from mdebris.eval.ablation import (
    SENSOR_SCENARIOS,
    binary_metrics,
    cluster_bootstrap_f1,
    index_threshold_baseline,
    select_features,
    select_threshold,
)
from mdebris.models.spectral import FEATURE_BANDS, SpectralClassifier, feature_names

log = logging.getLogger("ablation")

SARGASSUM_CLASSES: tuple[str, ...] = ("Dense Sargassum", "Sparse Sargassum")
DEBRIS_CLASS = "Marine Debris"
TASKS: dict[str, tuple[str, ...]] = {
    "sargassum": SARGASSUM_CLASSES,
    "debris": (DEBRIS_CLASS,),
}


def _task_scores(proba: np.ndarray, classes: list[str], members: tuple[str, ...]) -> np.ndarray:
    idx = [classes.index(c) for c in members if c in classes]
    if not idx:
        raise RuntimeError(f"model knows none of {members}; it has {classes}")
    return proba[:, idx].sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/band_ablation.md"))
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--scenarios",
        default=",".join(SENSOR_SCENARIOS),
        help="Comma-separated subset of scenario names, in the order to run them.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Running this file puts scripts/ on sys.path, so the sibling imports flat.
    from train_marida import collect

    log.info("loading MARIDA splits")
    x_train, y_train = collect("train", max_patches=args.max_patches)
    x_val, y_val = collect("val", max_patches=args.max_patches)
    x_test, y_test, groups_test = collect("test", max_patches=args.max_patches, return_groups=True)
    log.info(
        "train %s, val %s, test %s pixels",
        f"{len(x_train):,}",
        f"{len(x_val):,}",
        f"{len(x_test):,}",
    )
    truth_val = {t: np.isin(y_val, m) for t, m in TASKS.items()}
    truth_test = {t: np.isin(y_test, m) for t, m in TASKS.items()}
    names = feature_names()

    results: dict[str, dict] = {}
    for scenario in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        bands = SENSOR_SCENARIOS[scenario]
        indices_from = FEATURE_BANDS if scenario == "indices_only" else None
        xt, kept = select_features(x_train, bands, indices_from=indices_from)
        xv, _ = select_features(x_val, bands, indices_from=indices_from)
        xs, _ = select_features(x_test, bands, indices_from=indices_from)
        log.info("scenario %-13s %2d features: %s", scenario, len(kept), " ".join(kept))

        started = time.perf_counter()
        clf = SpectralClassifier(max_iter=args.max_iter).fit(xt, y_train)
        seconds = time.perf_counter() - started
        classes = list(clf._model.classes_)
        proba_val = clf.predict_proba(xv)
        proba_test = clf.predict_proba(xs)

        row: dict = {
            "bands": list(bands),
            "features": kept,
            "n_features": len(kept),
            "fit_seconds": round(seconds, 1),
        }
        for task, members in TASKS.items():
            sv = _task_scores(proba_val, classes, members)
            st = _task_scores(proba_test, classes, members)
            threshold = select_threshold(truth_val[task], sv)
            metrics = binary_metrics(truth_test[task], st, threshold)
            # The optimistic number, for comparison with the older reports only.
            optimistic = binary_metrics(
                truth_test[task], st, select_threshold(truth_test[task], st)
            )
            metrics["f1_test_optimal_threshold"] = optimistic["f1"]
            ci = cluster_bootstrap_f1(truth_test[task], st, threshold, groups_test)
            metrics["f1_ci_low"], metrics["f1_ci_high"] = ci["low"], ci["high"]
            row[task] = metrics
            log.info(
                "  %-9s P %.3f  R %.3f  F1 %.3f [%.3f, %.3f]  (threshold %.3f from val; %.3f if chosen on test)",
                task,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                ci["low"],
                ci["high"],
                threshold,
                optimistic["f1"],
            )
        results[scenario] = row

    # The physical rows: one index, one cut, no learning. Sargassum and debris both
    # raise NIR above the red-SWIR baseline, so "greater than" is the right sign.
    baselines: dict[str, dict] = {}
    for index_name in ("FDI", "FAI"):
        col = names.index(index_name)
        entry: dict = {"features": [index_name], "n_features": 1}
        for task in TASKS:
            entry[task] = index_threshold_baseline(
                truth_val[task], x_val[:, col], truth_test[task], x_test[:, col]
            )
            ci = cluster_bootstrap_f1(
                truth_test[task], x_test[:, col], entry[task]["threshold"], groups_test
            )
            entry[task]["f1_ci_low"], entry[task]["f1_ci_high"] = ci["low"], ci["high"]
            log.info(
                "baseline %-4s %-9s P %.3f  R %.3f  F1 %.3f  at %s > %.4f",
                index_name,
                task,
                entry[task]["precision"],
                entry[task]["recall"],
                entry[task]["f1"],
                index_name,
                entry[task]["threshold"],
            )
        baselines[f"{index_name.lower()}_threshold"] = entry

    payload = {
        "protocol": {
            "train_pixels": len(x_train),
            "val_pixels": len(x_val),
            "test_pixels": len(x_test),
            "threshold_selection": "best F1 on the MARIDA validation split, applied to test",
            "interval": "95% percentile interval from a 500-resample cluster bootstrap over test patches",
            "test_patches": len(np.unique(groups_test)),
            "positives_test": {t: int(v.sum()) for t, v in truth_test.items()},
            "positives_val": {t: int(v.sum()) for t, v in truth_val.items()},
        },
        "scenarios": results,
        "baselines": baselines,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(_markdown(payload), encoding="utf-8")
    log.info("wrote %s", args.report)


def _markdown(payload: dict) -> str:
    proto = payload["protocol"]
    lines = [
        "# Which bands carry the signal: a sensor ablation",
        "",
        "Produced by `python scripts/eval_band_ablation.py`. The same gradient-boosting",
        "classifier is fitted on MARIDA with only the bands a given sensor has, the",
        "rest set to missing, and an index that needs a missing band is dropped with",
        "it. Thresholds are chosen on the validation split and applied to the test",
        "split, so every row is scored on pixels its threshold never saw.",
        "",
        f"Train {proto['train_pixels']:,} pixels, validation {proto['val_pixels']:,}, "
        f"test {proto['test_pixels']:,} in {proto.get('test_patches', 0)} patches. Test positives: "
        f"{proto['positives_test']['sargassum']:,} sargassum, "
        f"{proto['positives_test']['debris']:,} debris. Intervals are 95% percentile",
        "intervals from a cluster bootstrap over test patches, because pixels in one",
        "patch share a scene and an annotator and are not independent.",
        "",
        "## Learned model per band set",
        "",
        "| scenario | stands for | features | sargassum P | R | F1 (95% CI) | debris P | R | F1 (95% CI) | fit s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    stands_for = {
        "sentinel2": "all 11 Sentinel-2 bands",
        "no_swir": "no shortwave infrared",
        "no_red_edge": "Landsat-like, no red edge",
        "superdove": "PlanetScope 8-band, nearest S2 bands",
        "dove4": "PlanetScope 4-band, the NASA IMPACT imagery",
        "rgb": "visible only",
        "indices_only": "the seven physical indices, no raw band",
    }
    for name, row in payload["scenarios"].items():
        s, d = row["sargassum"], row["debris"]
        lines.append(
            f"| `{name}` | {stands_for.get(name, '')} | {row['n_features']} | "
            f"{s['precision']:.3f} | {s['recall']:.3f} | **{s['f1']:.3f}** "
            f"({s['f1_ci_low']:.2f} to {s['f1_ci_high']:.2f}) | "
            f"{d['precision']:.3f} | {d['recall']:.3f} | **{d['f1']:.3f}** "
            f"({d['f1_ci_low']:.2f} to {d['f1_ci_high']:.2f}) | {row['fit_seconds']:.0f} |"
        )
    lines += [
        "",
        "## One physical index, one threshold, no learning",
        "",
        "| index | sargassum P | R | F1 (95% CI) | cut | debris P | R | F1 (95% CI) | cut |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["baselines"].values():
        s, d = row["sargassum"], row["debris"]
        lines.append(
            f"| `{row['features'][0]} >` | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"**{s['f1']:.3f}** ({s['f1_ci_low']:.2f} to {s['f1_ci_high']:.2f}) | {s['threshold']:.4f} | "
            f"{d['precision']:.3f} | {d['recall']:.3f} | **{d['f1']:.3f}** "
            f"({d['f1_ci_low']:.2f} to {d['f1_ci_high']:.2f}) | {d['threshold']:.4f} |"
        )
    base = payload["scenarios"].get("sentinel2")
    if base:
        lines += [
            "",
            "## Protocol note",
            "",
            f"The full-band sargassum F1 here is {base['sargassum']['f1']:.3f}. The 0.948 in",
            "`sargassum_report.md` is the same model at the threshold that maximises F1 on",
            f"the test split itself, which this protocol reproduces as "
            f"{base['sargassum']['f1_test_optimal_threshold']:.3f} and does not report as a",
            "result. Choosing a threshold on the pixels you then score is optimistic by",
            "construction; the gap between the two numbers is the size of that optimism.",
        ]
    lines += [
        "",
        "## Reading it",
        "",
        "A scenario close to the full band set means the score does not depend on the",
        "bands it lacks, and a resolution comparison against a sensor with that band set",
        "is a resolution comparison. A scenario far below it means the comparison would",
        "be confounded and has to be reported as bands plus resolution.",
        "",
        "The physical rows are the other end of the axis this repository keeps returning",
        "to: a single index with a single cut is what most of the operational sargassum",
        "literature uses, and the learned rows show what the extra bands buy over it at",
        "10 m, on this benchmark, for each target.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
