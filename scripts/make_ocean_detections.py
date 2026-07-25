"""Highlight detected debris on open-ocean Sentinel-2 imagery, with per-scene accuracy.

    python scripts/make_ocean_detections.py

Selects MARIDA test patches that are predominantly open water and contain annotated
Marine Debris, runs the trained classifier, and draws the detection as a highlight
over the true-colour image. Each panel reports precision, recall and F1 for that
scene, computed only against the pixels a human actually annotated.

Open water is the selection criterion on purpose. Coastal scenes make detection look
easier than it is, because land and surf give a model obvious structure to key on.
Debris on open ocean, with nothing else in frame, is the honest case.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mdebris.data import MARIDA_BANDS, MARIDA_CLASSES, load_marida_split
from mdebris.models.spectral import SpectralClassifier
from mdebris.viz.plots import save_figure, stretch_to_uint8

log = logging.getLogger("ocean")

DEBRIS = "Marine Debris"
DEBRIS_CODE = MARIDA_CLASSES.index(DEBRIS) + 1
WATER_CODES = {
    MARIDA_CLASSES.index(n) + 1
    for n in ("Marine Water", "Turbid Water", "Shallow Water", "Mixed Water")
}
HIGHLIGHT = "#FF9500"
TRUTH_EDGE = "#00E5FF"


def true_colour(image: np.ndarray) -> np.ndarray:
    """Joint-stretched RGB. Per-channel stretching turns open water into colour static."""
    idx = {name: i for i, name in enumerate(MARIDA_BANDS)}
    stack = np.dstack([image[idx["B04"]], image[idx["B03"]], image[idx["B02"]]])
    return stretch_to_uint8(stack, low=1.0, high=99.0, per_channel=False)


def pick_open_ocean(split: str, limit: int) -> list:
    """Patches whose annotation is debris on water, ranked by how much debris they hold."""
    chosen = []
    for patch in load_marida_split(split):
        _, classes, _ = patch.read()
        debris_px = int((classes == DEBRIS_CODE).sum())
        if debris_px < 8:
            continue
        annotated = classes != 0
        water_px = int(np.isin(classes, list(WATER_CODES)).sum())
        # Reject scenes whose annotation is dominated by anything that is not water
        # or debris, which in practice means cloud, land-adjacent or ship-heavy frames.
        other = int(annotated.sum()) - water_px - debris_px
        if other > 0.35 * max(1, int(annotated.sum())):
            continue
        chosen.append((debris_px, patch))
    chosen.sort(key=lambda kv: -kv[0])
    return [p for _, p in chosen[:limit]]


def crop_around(mask: np.ndarray, *, pad: int = 30, minimum: int = 110) -> tuple[slice, slice]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return slice(None), slice(None)
    h, w = mask.shape
    r0, r1 = int(rows.min()) - pad, int(rows.max()) + pad + 1
    c0, c1 = int(cols.min()) - pad, int(cols.max()) + pad + 1
    if r1 - r0 < minimum:
        mid = (r0 + r1) // 2
        r0, r1 = mid - minimum // 2, mid + minimum // 2
    if c1 - c0 < minimum:
        mid = (c0 + c1) // 2
        c0, c1 = mid - minimum // 2, mid + minimum // 2
    return slice(max(0, r0), min(h, r1)), slice(max(0, c0), min(w, c1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument("--out", type=Path, default=Path("assets/ocean_detections.png"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Debris probability cutoff. Defaults to the best-F1 threshold from training.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    threshold = args.threshold
    if threshold is None:
        report = Path("docs/marida_report_balanced.json")
        if report.exists():
            import json

            m = json.loads(report.read_text()).get("per_class", {}).get(DEBRIS, {})
            threshold = float(m.get("best_f1_threshold", 0.5))
        else:
            threshold = 0.5
    log.info("debris probability threshold %.3f", threshold)

    clf = SpectralClassifier.load(args.model)
    patches = pick_open_ocean(args.split, args.n)
    if not patches:
        raise SystemExit("no open-ocean test patch with annotated debris was found")
    log.info("rendering %d open-ocean patches", len(patches))

    fig, axes = plt.subplots(len(patches), 2, figsize=(12.2, 5.9 * len(patches)))
    axes = np.atleast_2d(axes)

    for row, patch in enumerate(patches):
        image, truth, _ = patch.read()
        bands = {name: image[i] for i, name in enumerate(MARIDA_BANDS)}
        probability = clf.debris_probability(bands)

        rs, cs = crop_around(truth == DEBRIS_CODE)
        image, truth, probability = image[:, rs, cs], truth[rs, cs], probability[rs, cs]
        rgb = true_colour(image)

        detected = probability >= threshold
        actual = truth == DEBRIS_CODE
        annotated = truth != 0

        # Score only where a human annotated. Counting confident predictions on
        # unlabelled pixels as false positives would penalise the model for pixels
        # nobody ever checked.
        tp = int((detected & actual).sum())
        fp = int((detected & annotated & ~actual).sum())
        fn = int((~detected & actual).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_axis_off()
        axes[row, 0].set_title(
            f"{patch.roi}   Sentinel-2 true colour, 10 m\n{rgb.shape[1]}x{rgb.shape[0]} px "
            f"= {rgb.shape[1] * 10 / 1000:.1f}x{rgb.shape[0] * 10 / 1000:.1f} km of open ocean",
            fontsize=9.5,
            fontfamily="monospace",
            pad=8,
        )

        axes[row, 1].imshow(rgb)
        # Filled highlight for the detection, plus a contour so single pixels stay visible.
        overlay = np.zeros((*detected.shape, 4), dtype=np.float32)
        overlay[..., :3] = matplotlib.colors.to_rgb(HIGHLIGHT)
        overlay[..., 3] = np.where(detected, 0.55, 0.0)
        axes[row, 1].imshow(overlay)
        if detected.any():
            axes[row, 1].contour(
                detected.astype(float), levels=[0.5], colors=HIGHLIGHT, linewidths=1.4
            )
        if actual.any():
            axes[row, 1].contour(
                actual.astype(float), levels=[0.5], colors=TRUTH_EDGE, linewidths=0.9
            )
        axes[row, 1].set_axis_off()
        axes[row, 1].set_title(
            f"detected debris   precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}\n"
            f"{tp} true positive, {fp} false positive, {fn} missed  "
            f"({tp * 100:,} m2 detected)",
            fontsize=9.5,
            fontfamily="monospace",
            pad=8,
        )
        log.info(
            "  %-22s P %.2f  R %.2f  F1 %.2f   tp %3d fp %3d fn %3d",
            patch.roi,
            precision,
            recall,
            f1,
            tp,
            fp,
            fn,
        )

    fig.legend(
        handles=[
            Line2D([0], [0], color=HIGHLIGHT, lw=6, alpha=0.75, label="model detection"),
            Line2D([0], [0], color=TRUTH_EDGE, lw=2, label="human annotation outline"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=9.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.008),
    )
    fig.subplots_adjust(bottom=0.055, hspace=0.14, wspace=0.03)
    log.info("wrote %s", save_figure(fig, args.out, tight=False))


if __name__ == "__main__":
    main()
