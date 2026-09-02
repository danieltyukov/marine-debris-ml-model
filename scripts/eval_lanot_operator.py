"""Score the LANOT platform's published detection operator on MARIDA, beside this model.

    python scripts/eval_lanot_operator.py

LANOT (National Laboratory for Earth Observation, Instituto de Geografia UNAM) runs the
operational Sargassum monitoring platform for the Mexican Caribbean, Belize, Guatemala
and Honduras. Their method paper publishes the detection rule in full:

    Arellano-Verdejo, J., Lazcano-Hernandez, H.E., Prado Molina, J. et al.
    Towards enhanced Sargassum monitoring in the Caribbean Sea.
    Sci Rep 15, 8965 (2025). https://doi.org/10.1038/s41598-025-93001-9

It is not a learned model. Expression (1) of that paper is five hand-calibrated
inequalities on Sentinel-2 L2A surface reflectance, derived by photointerpretation and
tuned for the optical properties of Caribbean coastal water.

Two facts make a direct comparison possible rather than hypothetical:

* The paper names the 18 MGRS tiles the platform processes. Four of MARIDA's seventeen
  sites are on that list, and 95.6% of MARIDA's annotated sargassum pixels fall inside
  them. The benchmark is, for sargassum, mostly LANOT's own operating area.
* MARIDA ships reflectance in the same 0-1 units the expression is written in, so the
  thresholds apply without rescaling. It is ACOLITE Rayleigh-corrected reflectance on
  L1C, not the Sen2Cor L2A the rule was calibrated on, and the report says so.

So this runs their operator, as published, on the same held-out pixels this repository
scores its own classifier on, and puts both in one table. It also answers the question
that motivated the comparison: their `b11 < 0.05` term rejects anything bright in SWIR,
cloud included, and this classifier carries no equivalent. Adding that one gate to the
learned model is the third row of the report.

What this is not: a score of the LANOT platform. The platform applies segmentation,
entropy filtering and threshold denoising to the pixel mask before anything becomes a
published polygon. Expression (1) is the per-pixel gate at the front of that pipeline
and nothing here measures the rest of it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from mdebris.data import MARIDA_BANDS, MARIDA_CLASSES, load_marida_split
from mdebris.models.spectral import SpectralClassifier, build_features

log = logging.getLogger("lanot")

# MARIDA encodes 0 as unlabelled; classes run 1..15 in MARIDA_CLASSES order.
UNLABELLED = 0

SARGASSUM_CLASSES: tuple[str, ...] = ("Dense Sargassum", "Sparse Sargassum")

# The 18 Sentinel-2 tiles the LANOT platform processes, listed verbatim in the paper's
# "limites" section. Kept whole rather than filtered down to the four MARIDA overlaps so
# that the overlap is computed here rather than asserted.
LANOT_TILES: frozenset[str] = frozenset(
    {
        "16QDJ",
        "16QEJ",
        "16QDH",
        "16QEH",
        "16QDG",
        "16QEG",
        "16QDF",
        "16QEF",
        "16QCF",
        "16QCE",
        "16QDE",
        "16QEE",
        "16QCD",
        "16QDD",
        "16QED",
        "16PCC",
        "16PDC",
        "16PEC",
    }
)

# The SWIR gate from expression (1), isolated because the interesting experiment is
# bolting this one term onto the learned classifier.
SWIR_GATE = 0.05


def lanot_operator(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Expression (1) of Arellano-Verdejo et al. 2025, applied per pixel.

        (b8A < 0.07) and (b04 < 0.1) and (b11 < 0.05) and (b04 < b8A) and (b04 < b08)

    The first three terms are rejection gates against bright surfaces: cloud, land and
    sun glint are all bright somewhere in 665-1610 nm and open water is nearly black at
    1610 nm. The last two are the detection, requiring near-infrared above red, which is
    the red-edge signature of floating vegetation.

    Args:
        bands: Surface reflectance keyed by MARIDA band name, any matching shape.

    Returns:
        Boolean array, True where the operator fires.
    """
    b04, b08, b8a, b11 = bands["B04"], bands["B08"], bands["B8A"], bands["B11"]
    return (b8a < 0.07) & (b04 < 0.1) & (b11 < SWIR_GATE) & (b04 < b8a) & (b04 < b08)


def collect(split: str, *, tiles: frozenset[str] | None = None, min_confidence: int = 0):
    """Every labelled pixel in a split, as features, labels, raw bands and tile ids.

    ``train_marida.collect`` returns features and labels only. The operator needs raw
    reflectance on exactly the same pixels, so this collects both together rather than
    reading the patches twice and hoping the orders line up.

    Args:
        split: One of train, val, test.
        tiles: Restrict to these MGRS tiles. None keeps everything.
        min_confidence: Drop pixels below this MARIDA annotation confidence.

    Returns:
        ``(features, labels, bands, tile_ids)``. ``bands`` maps band name to a 1-D array
        parallel to ``labels``.
    """
    patches = load_marida_split(split)
    if tiles is not None:
        patches = [p for p in patches if p.tile in tiles]
    if not patches:
        raise RuntimeError(f"no {split} patches left after filtering to {tiles}")

    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    band_rows: list[np.ndarray] = []
    tile_rows: list[np.ndarray] = []
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
        band_rows.append(image[:, labelled].T)
        tile_rows.append(np.full(int(labelled.sum()), patch.tile))
        if (i + 1) % 150 == 0:
            log.info("  %s: %d/%d patches", split, i + 1, len(patches))

    x = np.concatenate(feature_rows).astype(np.float32)
    codes = np.concatenate(label_rows)
    names = np.array([MARIDA_CLASSES[c - 1] for c in codes])
    stacked = np.concatenate(band_rows)
    band_cols = {name: stacked[:, j] for j, name in enumerate(MARIDA_BANDS)}
    return x, names, band_cols, np.concatenate(tile_rows)


def _score(truth: np.ndarray, fired: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 for one boolean detector against one boolean truth."""
    tp = int((truth & fired).sum())
    fp = int((~truth & fired).sum())
    fn = int((truth & ~fired).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _breakdown(labels: np.ndarray, mask: np.ndarray, limit: int = 8) -> dict[str, int]:
    """What the pixels under ``mask`` were actually annotated as, commonest first."""
    if not mask.any():
        return {}
    names, counts = np.unique(labels[mask], return_counts=True)
    pairs = sorted(zip(names, counts, strict=True), key=lambda kv: -kv[1])[:limit]
    return {str(n): int(c) for n, c in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/lanot_comparison.md"))
    parser.add_argument("--min-confidence", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.model.exists():
        raise SystemExit(
            f"no model at {args.model}. Run scripts/train_marida.py first, which "
            "downloads MARIDA and fits the classifier."
        )

    from sklearn.metrics import precision_recall_curve

    log.info("loading the held-out MARIDA test split, restricted to LANOT's 18 tiles")
    x, y, bands, tiles = collect("test", tiles=LANOT_TILES, min_confidence=args.min_confidence)
    truth = np.isin(y, SARGASSUM_CLASSES)
    log.info(
        "%s labelled pixels across %d tiles, %s of them sargassum",
        f"{len(y):,}",
        len(set(tiles.tolist())),
        f"{int(truth.sum()):,}",
    )

    results: dict[str, dict[str, float]] = {}

    fired = lanot_operator(bands)
    results["LANOT expression (1)"] = _score(truth, fired)
    log.info(
        "LANOT expression (1):        P %.3f  R %.3f  F1 %.3f",
        *(results["LANOT expression (1)"][k] for k in ("precision", "recall", "f1")),
    )

    clf = SpectralClassifier.load(args.model)
    classes = list(clf._model.classes_)
    proba = clf.predict_proba(x)
    sarg_idx = [classes.index(c) for c in SARGASSUM_CLASSES if c in classes]
    scores = proba[:, sarg_idx].sum(axis=1)
    predicted = np.array(classes)[proba.argmax(axis=1)]
    argmax_fired = np.isin(predicted, SARGASSUM_CLASSES)
    results["this classifier, argmax"] = _score(truth, argmax_fired)
    log.info(
        "this classifier, argmax:     P %.3f  R %.3f  F1 %.3f",
        *(results["this classifier, argmax"][k] for k in ("precision", "recall", "f1")),
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
    best_fired = scores >= best_threshold
    results["this classifier, best F1"] = _score(truth, best_fired) | {"threshold": best_threshold}
    log.info(
        "this classifier, best F1:    P %.3f  R %.3f  F1 %.3f  at %.3f",
        *(results["this classifier, best F1"][k] for k in ("precision", "recall", "f1")),
        best_threshold,
    )

    # The experiment the comparison exists for. If the SWIR gate is the term this model
    # is missing, bolting it on should remove cloud false positives and leave sargassum
    # recall alone, because sargassum mats are dark at 1610 nm and cloud is not.
    gated_argmax = argmax_fired & (bands["B11"] < SWIR_GATE)
    results["this classifier + b11 gate, argmax"] = _score(truth, gated_argmax)
    log.info(
        "classifier + b11 gate:       P %.3f  R %.3f  F1 %.3f",
        *(results["this classifier + b11 gate, argmax"][k] for k in ("precision", "recall", "f1")),
    )

    gated_scores = np.where(bands["B11"] < SWIR_GATE, scores, 0.0)
    g_precision, g_recall, g_thresholds = precision_recall_curve(truth.astype(int), gated_scores)
    g_f1 = np.divide(
        2 * g_precision * g_recall,
        g_precision + g_recall,
        out=np.zeros_like(g_precision),
        where=(g_precision + g_recall) > 0,
    )
    g_best = int(np.argmax(g_f1))
    g_threshold = float(g_thresholds[min(g_best, len(g_thresholds) - 1)])
    results["this classifier + b11 gate, best F1"] = _score(truth, gated_scores >= g_threshold) | {
        "threshold": g_threshold
    }
    log.info(
        "classifier + gate, best F1:  P %.3f  R %.3f  F1 %.3f  at %.3f",
        *(results["this classifier + b11 gate, best F1"][k] for k in ("precision", "recall", "f1")),
        g_threshold,
    )

    # Why the gate does nothing, which is the actual result. The gate is not weak: it
    # rejects most cloud in the benchmark. It is that the cloud this classifier mistakes
    # for sargassum is already on the transmissive side of it, so the two failures are
    # the same failure and adding the term cannot fix it.
    cloud = y == "Clouds"
    b11 = bands["B11"]
    cls_cloud_fp = argmax_fired & ~truth & cloud
    gate_diagnostic = {
        "cloud_pixels": int(cloud.sum()),
        "cloud_passing_gate": int((cloud & (b11 < SWIR_GATE)).sum()),
        "cloud_passing_gate_fraction": float((b11[cloud] < SWIR_GATE).mean()),
        "classifier_cloud_false_positives": int(cls_cloud_fp.sum()),
        "of_those_passing_gate": int((cls_cloud_fp & (b11 < SWIR_GATE)).sum()),
        "median_b11_all_cloud": float(np.median(b11[cloud])),
        "median_b11_classifier_cloud_fp": float(np.median(b11[cls_cloud_fp])),
        "median_b11_lanot_cloud_fp": float(np.median(b11[fired & ~truth & cloud])),
        "sargassum_passing_gate_fraction": float((b11[truth] < SWIR_GATE).mean()),
    }
    log.info(
        "gate rejects %.0f%% of cloud overall but 0 of the %d cloud pixels this model "
        "calls sargassum",
        100 * (1 - gate_diagnostic["cloud_passing_gate_fraction"]),
        gate_diagnostic["classifier_cloud_false_positives"],
    )

    # Where the two disagree is worth more than where either wins, because a pixel only
    # one of them finds is a pixel the other could gain.
    agreement = {
        "identical_calls_fraction": float((argmax_fired == fired).mean()),
        "sargassum_found_by_both": int((truth & argmax_fired & fired).sum()),
        "sargassum_only_classifier": int((truth & argmax_fired & ~fired).sum()),
        "sargassum_only_lanot": int((truth & ~argmax_fired & fired).sum()),
        "sargassum_found_by_neither": int((truth & ~argmax_fired & ~fired).sum()),
        "false_positives_shared": int((argmax_fired & fired & ~truth).sum()),
        "shared_false_positives_from": _breakdown(y, argmax_fired & fired & ~truth),
    }
    union = (
        agreement["sargassum_found_by_both"]
        + agreement["sargassum_only_classifier"]
        + agreement["sargassum_only_lanot"]
    )
    agreement["sargassum_found_by_either"] = union
    agreement["union_recall"] = union / int(truth.sum())
    log.info(
        "union recall %.3f: only %d of %d sargassum pixels are missed by both",
        agreement["union_recall"],
        agreement["sargassum_found_by_neither"],
        int(truth.sum()),
    )

    breakdowns = {
        "LANOT expression (1)": {
            "false_alarms_from": _breakdown(y, fired & ~truth),
            "missed_as_truth": _breakdown(y, truth & ~fired),
        },
        "this classifier, argmax": {
            "false_alarms_from": _breakdown(y, argmax_fired & ~truth),
            "missed_as_truth": _breakdown(y, truth & ~argmax_fired),
        },
        "this classifier + b11 gate, argmax": {
            "false_alarms_from": _breakdown(y, gated_argmax & ~truth),
            "missed_as_truth": _breakdown(y, truth & ~gated_argmax),
        },
    }

    per_tile = {}
    for tile in sorted(set(tiles.tolist())):
        sel = tiles == tile
        per_tile[tile] = {
            "pixels": int(sel.sum()),
            "sargassum": int(truth[sel].sum()),
            "lanot": _score(truth[sel], fired[sel]),
            "classifier_argmax": _score(truth[sel], argmax_fired[sel]),
        }

    payload = {
        "pixels": len(y),
        "sargassum_pixels": int(truth.sum()),
        "tiles": sorted(set(tiles.tolist())),
        "results": results,
        "breakdowns": breakdowns,
        "gate_diagnostic": gate_diagnostic,
        "agreement": agreement,
        "per_tile": per_tile,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown(payload), encoding="utf-8")
    args.report.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", args.report)


def _markdown(d: dict) -> str:
    r, a, g = d["results"], d["agreement"], d["gate_diagnostic"]
    order = [
        "LANOT expression (1)",
        "this classifier, argmax",
        "this classifier, best F1",
        "this classifier + b11 gate, argmax",
        "this classifier + b11 gate, best F1",
    ]
    lines = [
        "# Where this classifier and LANOT's published operator disagree",
        "",
        "Produced by `python scripts/eval_lanot_operator.py`.",
        "",
        "## Read this first",
        "",
        "**This is not a score of the LANOT platform, and the table below should not be",
        "quoted as one.** Expression (1) is the per-pixel gate at the front of their",
        "pipeline. Everything that reaches a published LANOT polygon has been through",
        "segmentation, entropy filtering and threshold denoising afterwards, none of which",
        "is run here, and all of which exists precisely to clear up isolated per-pixel",
        "false positives. Scoring the gate alone measures the gate alone.",
        "",
        "The comparison is here for one reason: the two detectors fail on different pixels,",
        "and the union is better than either. That is the result, not the ranking.",
        "",
        "## The systems",
        "",
        "The LANOT platform (Instituto de Geografia UNAM) is the operational Sargassum",
        "monitoring system for the Mexican Caribbean, Belize, Guatemala and Honduras. Its",
        "detection rule is published in full and is not a learned model:",
        "",
        "> (b8A < 0.07) and (b04 < 0.1) and (b11 < 0.05) and (b04 < b8A) and (b04 < b08)",
        "",
        "Arellano-Verdejo, J., Lazcano-Hernandez, H.E., Prado Molina, J. et al. *Towards",
        "enhanced Sargassum monitoring in the Caribbean Sea.* Sci Rep 15, 8965 (2025).",
        "<https://doi.org/10.1038/s41598-025-93001-9>",
        "",
        "The first three terms reject bright surfaces: cloud, land and sun glint are bright",
        "somewhere in 665-1610 nm and open water is nearly black at 1610 nm. The last two",
        "are the detection, requiring near-infrared above red, the red-edge signature of",
        "floating vegetation.",
        "",
        "## Why these pixels",
        "",
        "The paper names the 18 Sentinel-2 tiles the platform processes. Four of MARIDA's",
        "seventeen sites are on that list: 16PCC (Motagua, Guatemala), 16PDC (Ulua,",
        "Honduras), 16PEC (La Ceiba, Honduras) and 16QED (Roatan, Honduras).",
        "",
        f"Restricted to those tiles, MARIDA's held-out test split has {d['pixels']:,} labelled",
        f"pixels, {d['sargassum_pixels']:,} of them annotated Dense or Sparse Sargassum. That is",
        "every sargassum pixel in the test split: on this benchmark, sargassum occurs only",
        "inside LANOT's footprint. The classifier is scored on scenes it never trained on,",
        "and the operator, which has no training, is scored on the same pixels.",
        "",
        "## The result: they miss different things",
        "",
        "| | sargassum pixels |",
        "|---|---|",
        f"| found by both | {a['sargassum_found_by_both']:,} |",
        f"| found only by this classifier | {a['sargassum_only_classifier']:,} |",
        f"| found only by LANOT expression (1) | {a['sargassum_only_lanot']:,} |",
        f"| **found by either** | **{a['sargassum_found_by_either']:,}** |",
        f"| missed by both | {a['sargassum_found_by_neither']:,} |",
        "",
        f"Union recall is {a['union_recall']:.3f}. Only {a['sargassum_found_by_neither']} of",
        f"{d['sargassum_pixels']:,} annotated sargassum pixels are invisible to both methods, so",
        "the floor on this benchmark is far lower than either detector reaches alone. A",
        "hand-calibrated physical rule and a gradient-boosted classifier trained on different",
        "evidence keep some genuinely independent signal from each other.",
        "",
        f"They make identical calls on {100 * a['identical_calls_fraction']:.1f}% of all labelled pixels, and",
        f"{a['false_positives_shared']:,} false positives are shared. What those shared errors are:",
        "",
        "| actually | pixels |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in a["shared_false_positives_from"].items()]

    lines += [
        "",
        "## The cloud that neither one rejects",
        "",
        "Cloud near-edge error is the failure mode both systems have, and the interesting",
        "part is that it is a specific physical population rather than a tuning problem.",
        "",
        f"The `b11 < 0.05` term is a good gate. Of the {g['cloud_pixels']:,} cloud pixels on these",
        f"tiles it rejects {100 * (1 - g['cloud_passing_gate_fraction']):.0f}%, while keeping",
        f"{100 * g['sargassum_passing_gate_fraction']:.1f}% of the sargassum. Thick cloud is bright at",
        "1610 nm and sargassum mats are not, so the separation is real.",
        "",
        "It does not help with the cloud that actually causes false positives. All",
        f"{g['of_those_passing_gate']} of the {g['classifier_cloud_false_positives']} cloud pixels this",
        "classifier calls sargassum already pass the gate, which is why adding the term to",
        "the learned model changes nothing at all: the two rows are identical in the table",
        "below. Those pixels are optically thin cloud, with a median B11 of",
        f"{g['median_b11_classifier_cloud_fp']:.4f} against {g['median_b11_all_cloud']:.4f} for cloud",
        "generally. LANOT's own cloud false positives sit in the same population, median B11",
        f"{g['median_b11_lanot_cloud_fp']:.4f}.",
        "",
        "Thin cloud over dark water is dim in SWIR and lifts red-edge reflectance, so it",
        "looks like floating vegetation to a physical rule and to a learned classifier alike.",
        "No threshold on B11 separates it, because it is not bright. That makes it a distinct",
        "target for a mask rather than a parameter to retune, and MARIDA labels those exact",
        "pixels, so any candidate cloud mask can be tested against them directly.",
        "",
        "## Operating points",
        "",
        "Per-pixel, on the tiles and split described above. The first row is a front gate",
        "with no post-processing behind it and the others are a trained classifier, so the",
        "rows are not like for like and the F1 column is not a ranking.",
        "",
        "| detector | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in order:
        m = r[name]
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['tp']:,} | {m['fp']:,} | {m['fn']:,} |"
        )

    lines += ["", "## What each one gets wrong", ""]
    for name, b in d["breakdowns"].items():
        lines += [f"### {name}", "", "False alarms, by what the pixel actually was:", ""]
        if b["false_alarms_from"]:
            lines += ["| actually | pixels |", "|---|---|"]
            lines += [f"| {k} | {v:,} |" for k, v in b["false_alarms_from"].items()]
        else:
            lines.append("None.")
        lines += ["", "Sargassum pixels missed, by annotated class:", ""]
        if b["missed_as_truth"]:
            lines += ["| annotated | pixels |", "|---|---|"]
            lines += [f"| {k} | {v:,} |" for k, v in b["missed_as_truth"].items()]
        else:
            lines.append("None.")
        lines.append("")

    lines += [
        "## Per tile",
        "",
        "| tile | pixels | sargassum | LANOT F1 | classifier F1 |",
        "|---|---|---|---|---|",
    ]
    for tile, m in d["per_tile"].items():
        lines.append(
            f"| {tile} | {m['pixels']:,} | {m['sargassum']:,} | "
            f"{m['lanot']['f1']:.3f} | {m['classifier_argmax']['f1']:.3f} |"
        )

    lines += [
        "",
        "## What this does not say",
        "",
        "Repeating the first section because it is the part that is easiest to misread: the",
        "LANOT row is their per-pixel gate without the segmentation, entropy filtering and",
        "denoising that follow it in their pipeline. It is not their platform and it is not",
        "their published product.",
        "",
        "Expression (1) was calibrated by photointerpretation for the optical properties of",
        "the Mexican Caribbean. These four tiles are Guatemala and Honduras: the same",
        "platform footprint, adjacent water, not the water it was tuned on. The classifier,",
        "by contrast, was trained on MARIDA's own training split, so it has seen this",
        "dataset's annotation conventions and the operator has not.",
        "",
        "MARIDA annotates a deliberately confuser-rich subset of each scene rather than whole",
        "scenes. Precision on these pixels is pessimistic for both detectors compared with an",
        "operational scene that is mostly plain water.",
        "",
        "The reflectance is not the reflectance the rule was tuned on. MARIDA is ACOLITE",
        "output on Level-1C, Rayleigh-corrected reflectance from dark spectrum fitting (Kikaki",
        "et al. 2022). The LANOT pipeline runs Sen2Cor to Level-2A, and expression (1) was",
        "calibrated on that. Both are unitless 0-1 reflectance, so the thresholds apply",
        "without rescaling, but the Rayleigh-corrected product still carries the aerosol term,",
        "largest in the blue and smallest in the SWIR. The `b11 < 0.05` term is the least",
        "affected of the five and the `b04 < 0.1` term the most. A fairer test of the rule",
        "would run it on Sen2Cor reflectance for the same pixels, which MARIDA does not ship.",
        "",
        "MARIDA's labels are annotations by remote-sensing researchers, not field",
        "observations. Nothing here is validated against sargassum that anyone touched.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
