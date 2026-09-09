"""Measure whether the sargassum probability is an uncertainty, and fix it if not.

    python scripts/eval_calibration.py

A decision maker who asks for uncertainties alongside a map is asking whether "0.9"
means nine in ten. This script answers that for the any-sargassum probability of
the MARIDA-trained classifier: a reliability table, the expected calibration error
and the Brier score on the held-out split, before and after an isotonic remap
fitted on the validation split. The remap is written to
``models/sargassum_calibration.json`` as breakpoints, so it can be applied without
pickle and without scikit-learn.

It also fixes the abstention band: the probability interval in which the honest
output is "uncertain". The upper edge is the 90% precision operating point already
used for dispatch; the lower edge keeps 98% of true sargassum pixels above it. Both
edges are chosen on validation and the band's population is then counted on test.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from mdebris.eval.calibration import (
    IsotonicCalibrator,
    abstention_band,
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from mdebris.models.spectral import SpectralClassifier

log = logging.getLogger("calibration")

SARGASSUM_CLASSES: tuple[str, ...] = ("Dense Sargassum", "Sparse Sargassum")
DEBRIS_CLASS = "Marine Debris"


def _scores(proba: np.ndarray, classes: list[str], members: tuple[str, ...]) -> np.ndarray:
    idx = [classes.index(c) for c in members if c in classes]
    if not idx:
        raise RuntimeError(f"model knows none of {members}; it has {classes}")
    return proba[:, idx].sum(axis=1)


def _summary(truth: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "ece": expected_calibration_error(truth, scores),
        "brier": brier_score(truth, scores),
        "reliability": reliability_table(truth, scores),
    }


def _band_on_test(truth: np.ndarray, scores: np.ndarray, band: dict) -> dict:
    """Population of a validation-chosen band when applied to the test split."""
    low, high = band["low"], band["high"]
    uncertain = (scores >= low) & (scores < high) if high is not None else scores >= low
    confident_positive = scores >= high if high is not None else np.zeros_like(truth)
    confident_negative = scores < low
    out = {
        "n_uncertain": int(uncertain.sum()),
        "fraction_uncertain": float(uncertain.mean()),
        "uncertain_positive": int((uncertain & truth).sum()),
        "uncertain_negative": int((uncertain & ~truth).sum()),
        "positives_called_negative": int((confident_negative & truth).sum()),
        "negatives_called_positive": int((confident_positive & ~truth).sum()),
        "n_positive": int(truth.sum()),
    }
    if high is not None and confident_positive.any():
        out["precision_above_high"] = float(
            (confident_positive & truth).sum() / confident_positive.sum()
        )
    out["recall_at_or_above_low"] = float(((scores >= low) & truth).sum() / max(1, truth.sum()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument(
        "--calibrator", type=Path, default=Path("models/sargassum_calibration.json")
    )
    parser.add_argument("--report", type=Path, default=Path("docs/calibration.md"))
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--high-precision", type=float, default=0.90)
    parser.add_argument("--miss-rate", type=float, default=0.02)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.model.exists():
        raise SystemExit(f"no model at {args.model}; run scripts/train_marida.py first")

    from train_marida import collect

    log.info("loading MARIDA validation and test splits")
    x_val, y_val = collect("val", max_patches=args.max_patches)
    x_test, y_test = collect("test", max_patches=args.max_patches)
    clf = SpectralClassifier.load(args.model)
    classes = list(clf._model.classes_)
    proba_val = clf.predict_proba(x_val)
    proba_test = clf.predict_proba(x_test)

    payload: dict = {"model": str(args.model), "tasks": {}}
    for task, members in {"sargassum": SARGASSUM_CLASSES, "debris": (DEBRIS_CLASS,)}.items():
        truth_val = np.isin(y_val, members)
        truth_test = np.isin(y_test, members)
        raw_val = _scores(proba_val, classes, members)
        raw_test = _scores(proba_test, classes, members)

        cal = IsotonicCalibrator().fit(raw_val, truth_val)
        cal.meta = {
            "task": task,
            "classes": list(members),
            "fitted_on": "MARIDA validation split",
            "model": str(args.model),
        }
        cal_test = cal.apply(raw_test)
        cal_val = cal.apply(raw_val)

        band_raw = abstention_band(
            truth_val, raw_val, high_precision=args.high_precision, miss_rate=args.miss_rate
        )
        band_cal = abstention_band(
            truth_val, cal_val, high_precision=args.high_precision, miss_rate=args.miss_rate
        )
        # The operating point the Bonaire runner uses: a calibrated probability is
        # its own statement, so "at least 0.9" is "nine in ten" by construction, and
        # this records what that cut actually achieves on pixels it never saw.
        called = cal_test >= 0.9
        nine_in_ten = {
            "cut": 0.9,
            "n_called": int(called.sum()),
            "precision": float((called & truth_test).sum() / called.sum()) if called.any() else 0.0,
            "recall": float((called & truth_test).sum() / max(1, truth_test.sum())),
            "n_uncertain": int(((cal_test >= band_cal["low"]) & ~called).sum()),
            "uncertain_positive": int(((cal_test >= band_cal["low"]) & ~called & truth_test).sum()),
        }
        entry = {
            "n_val": len(truth_val),
            "nine_in_ten": nine_in_ten,
            "n_test": len(truth_test),
            "positives_val": int(truth_val.sum()),
            "positives_test": int(truth_test.sum()),
            "raw": _summary(truth_test, raw_test),
            "calibrated": _summary(truth_test, cal_test),
            "band_raw": {
                "chosen_on_val": band_raw,
                "on_test": _band_on_test(truth_test, raw_test, band_raw),
            },
            "band_calibrated": {
                "chosen_on_val": band_cal,
                "on_test": _band_on_test(truth_test, cal_test, band_cal),
            },
        }
        payload["tasks"][task] = entry
        log.info(
            "%-9s ECE %.4f -> %.4f   Brier %.4f -> %.4f   (test, %d positives)",
            task,
            entry["raw"]["ece"],
            entry["calibrated"]["ece"],
            entry["raw"]["brier"],
            entry["calibrated"]["brier"],
            entry["positives_test"],
        )
        log.info(
            "%-9s calibrated >= 0.9: P %.3f  R %.3f on %d called; %d in [%.3f, 0.9) of which %d positive",
            "",
            nine_in_ten["precision"],
            nine_in_ten["recall"],
            nine_in_ten["n_called"],
            nine_in_ten["n_uncertain"],
            band_cal["low"],
            nine_in_ten["uncertain_positive"],
        )
        log.info(
            "%-9s abstain in [%.3f, %s) raw: %.2f%% of test pixels; calibrated [%.3f, %s): %.2f%%",
            "",
            band_raw["low"],
            "none" if band_raw["high"] is None else f"{band_raw['high']:.3f}",
            100 * entry["band_raw"]["on_test"]["fraction_uncertain"],
            band_cal["low"],
            "none" if band_cal["high"] is None else f"{band_cal['high']:.3f}",
            100 * entry["band_calibrated"]["on_test"]["fraction_uncertain"],
        )
        if task == "sargassum":
            cal.save(args.calibrator)
            log.info("wrote %s", args.calibrator)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(_markdown(payload, args), encoding="utf-8")
    log.info("wrote %s", args.report)


def _reliability_rows(rows: list[dict]) -> list[str]:
    out = ["| bin | pixels | mean probability | observed fraction |", "|---|---|---|---|"]
    for r in rows:
        if r["count"] == 0:
            out.append(f"| {r['lower']:.1f} to {r['upper']:.1f} | 0 | - | - |")
        else:
            out.append(
                f"| {r['lower']:.1f} to {r['upper']:.1f} | {r['count']:,} | "
                f"{r['mean_score']:.3f} | {r['fraction_positive']:.3f} |"
            )
    return out


def _markdown(payload: dict, args: argparse.Namespace) -> str:
    s = payload["tasks"]["sargassum"]
    d = payload["tasks"]["debris"]
    lines = [
        "# Is the probability an uncertainty? Calibration of the spectral classifier",
        "",
        "Produced by `python scripts/eval_calibration.py`. The classifier from",
        "`train_marida.py` is loaded, its any-sargassum probability is scored on the",
        "held-out test split, and an isotonic remap fitted on the validation split is",
        "applied to see how much of the miscalibration is fixable without retraining.",
        "",
        "## Summary",
        "",
        "| task | test positives | ECE raw | ECE calibrated | Brier raw | Brier calibrated |",
        "|---|---|---|---|---|---|",
        f"| any sargassum | {s['positives_test']:,} | {s['raw']['ece']:.4f} | "
        f"**{s['calibrated']['ece']:.4f}** | {s['raw']['brier']:.4f} | {s['calibrated']['brier']:.4f} |",
        f"| marine debris | {d['positives_test']:,} | {d['raw']['ece']:.4f} | "
        f"**{d['calibrated']['ece']:.4f}** | {d['raw']['brier']:.4f} | {d['calibrated']['brier']:.4f} |",
        "",
        "Expected calibration error is the count-weighted gap between what the model",
        "says and what happened, over ten equal-width probability bins; zero is a model",
        "whose 0.9 means nine in ten. The remap is monotone, so it does not change which",
        "pixels rank above which, only what number is attached to them.",
        "",
        "## Reliability, any sargassum, raw",
        "",
        *_reliability_rows(s["raw"]["reliability"]),
        "",
        "## Reliability, any sargassum, after isotonic calibration on validation",
        "",
        *_reliability_rows(s["calibrated"]["reliability"]),
        "",
        "## The abstention band",
        "",
        "Two edges, chosen on the validation split. Above the upper edge, precision is at",
        f"least {100 * args.high_precision:.0f}%, so a pixel there is called sargassum. Below the lower edge,",
        f"at most {100 * args.miss_rate:.0f}% of true sargassum pixels are lost, so a pixel there is called",
        "clear. In between, the model says so.",
        "",
        "| probabilities | lower edge | upper edge | test pixels in band | of which sargassum | sargassum called clear | non-sargassum called sargassum |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in (("raw", "band_raw"), ("calibrated", "band_calibrated")):
        chosen = s[key]["chosen_on_val"]
        on_test = s[key]["on_test"]
        high = "none reachable" if chosen["high"] is None else f"{chosen['high']:.3f}"
        lines.append(
            f"| {label} | {chosen['low']:.3f} | {high} | "
            f"{on_test['n_uncertain']:,} ({100 * on_test['fraction_uncertain']:.2f}%) | "
            f"{on_test['uncertain_positive']:,} | {on_test['positives_called_negative']:,} of "
            f"{on_test['n_positive']:,} | {on_test['negatives_called_positive']:,} |"
        )
    lines += [
        "",
        "## The operating point the Bonaire runner uses",
        "",
        "Once the probability is calibrated, the number is its own statement, and a cut",
        "at 0.9 means the model's estimate for that pixel is at least nine in ten. On the",
        "test split that cut gives:",
        "",
        "| task | called | precision | recall | uncertain, low edge to 0.9 | of which positive |",
        "|---|---|---|---|---|---|",
        f"| any sargassum | {s['nine_in_ten']['n_called']:,} | **{s['nine_in_ten']['precision']:.3f}** | "
        f"{s['nine_in_ten']['recall']:.3f} | {s['nine_in_ten']['n_uncertain']:,} | {s['nine_in_ten']['uncertain_positive']:,} |",
        f"| marine debris | {d['nine_in_ten']['n_called']:,} | **{d['nine_in_ten']['precision']:.3f}** | "
        f"{d['nine_in_ten']['recall']:.3f} | {d['nine_in_ten']['n_uncertain']:,} | {d['nine_in_ten']['uncertain_positive']:,} |",
        "",
        "The band is what the beach brief should carry as a third state next to",
        '"detected" and "clear": not as a smaller number, but as a different kind of',
        'answer, the same way `observability` already separates "not seen" from "seen',
        'and clean".',
        "",
        "## What this does not say",
        "",
        "Calibration on MARIDA pixels is calibration against MARIDA's annotators over",
        "MARIDA's scenes. A calibrated probability over Bonaire in 2025 is a hope, not a",
        "measurement, until pixels from that coast have been checked against the ground.",
        "The remap is stored as breakpoints in `models/sargassum_calibration.json` so it",
        "can be refitted the moment such pixels exist.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
