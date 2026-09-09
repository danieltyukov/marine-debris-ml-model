"""Figures for the band ablation and the calibration report, from their JSON outputs.

    python scripts/plot_post_meeting_figures.py

Kept separate from the scripts that compute the numbers so a figure can be redrawn
without refitting eight models. Reads ``docs/band_ablation.json`` and
``docs/calibration.json``; writes ``assets/band_ablation.png`` and
``assets/calibration.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdebris.viz.plots import save_figure

SARGASSUM = "#1f77b4"
DEBRIS = "#d95f02"
INK = "#333333"
MUTED = "#777777"

STANDS_FOR = {
    "sentinel2": "Sentinel-2, all 11 bands",
    "no_swir": "no SWIR",
    "no_red_edge": "no red edge (Landsat-like)",
    "superdove": "PlanetScope SuperDove, 6 of 8",
    "dove4": "PlanetScope Dove, 4 bands",
    "rgb": "RGB only",
    "indices_only": "7 physical indices, no raw band",
    "fdi_threshold": "FDI > cut, no learning",
    "fai_threshold": "FAI > cut, no learning",
}


def band_ablation(payload: dict, path: Path) -> None:
    rows = [*payload["scenarios"].items(), *payload["baselines"].items()]
    labels = [STANDS_FOR.get(k, k) for k, _ in rows]
    y = np.arange(len(rows))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
    for offset, task, colour in (
        (-height / 2, "sargassum", SARGASSUM),
        (height / 2, "debris", DEBRIS),
    ):
        f1 = np.array([r[task]["f1"] for _, r in rows])
        lo = np.array([r[task].get("f1_ci_low", r[task]["f1"]) for _, r in rows])
        hi = np.array([r[task].get("f1_ci_high", r[task]["f1"]) for _, r in rows])
        ax.barh(
            y + offset, f1, height=height, color=colour, edgecolor="white", linewidth=1, label=task
        )
        ax.errorbar(
            f1, y + offset, xerr=[f1 - lo, hi - f1], fmt="none", ecolor=INK, elinewidth=1, capsize=2
        )
        for yi, v, h in zip(y + offset, f1, hi, strict=True):
            ax.text(
                max(h, v) + 0.015, yi, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(
        "F1 on the MARIDA test split, threshold chosen on validation (95% patch bootstrap)"
    )
    ax.axhline(len(payload["scenarios"]) - 0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(
        0.01,
        len(payload["scenarios"]) - 0.62,
        "learned model above, one index below",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    ax.grid(axis="x", color="0.92")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Which bands carry the signal: the same classifier on each sensor's band set")
    save_figure(fig, path)


def calibration(payload: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True, sharey=True)
    for ax, task, colour in ((axes[0], "sargassum", SARGASSUM), (axes[1], "debris", DEBRIS)):
        entry = payload["tasks"][task]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        for key, style, label in (("raw", "o", "raw"), ("calibrated", "s", "after isotonic remap")):
            rel = [r for r in entry[key]["reliability"] if r["count"] > 0]
            xs = [r["mean_score"] for r in rel]
            ys = [r["fraction_positive"] for r in rel]
            sizes = [18 + 6 * np.log10(r["count"]) ** 2 for r in rel]
            ax.plot(xs, ys, color=colour, linewidth=1.2, alpha=0.9 if key == "calibrated" else 0.45)
            ax.scatter(
                xs,
                ys,
                s=sizes,
                marker=style,
                color=colour,
                alpha=0.9 if key == "calibrated" else 0.45,
                edgecolor="white",
                linewidth=0.8,
                label=f"{label}, ECE {entry[key]['ece']:.4f}",
                zorder=3,
            )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("mean predicted probability in bin")
        ax.set_title(f"any {task}" if task == "sargassum" else "marine debris")
        ax.grid(color="0.92")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("observed fraction that was the class")
    fig.suptitle(
        "Is 0.9 nine in ten? Reliability of the spectral classifier on the held-out split",
        fontsize=11,
    )
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, default=Path("docs/band_ablation.json"))
    parser.add_argument("--calibration", type=Path, default=Path("docs/calibration.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()
    band_ablation(json.loads(args.ablation.read_text()), args.out_dir / "band_ablation.png")
    calibration(json.loads(args.calibration.read_text()), args.out_dir / "calibration.png")


if __name__ == "__main__":
    main()
