"""Render real MARIDA patches with ground truth beside model predictions.

    python scripts/make_classification_samples.py --model models/marida_spectral.joblib

Produces assets/classification_samples.png: for each patch, the true-colour image,
the human annotation, and what the trained classifier predicted. Patches are chosen
because they contain annotated Marine Debris, so the figure shows the actual task
rather than easy open water.

Unannotated pixels are left blank in both the truth and prediction panels. MARIDA is
sparsely labelled, and colouring a prediction where no ground truth exists would
invite reading agreement into pixels nobody checked.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from mdebris.data import MARIDA_BANDS, MARIDA_CLASSES, load_marida_split
from mdebris.models.spectral import SpectralClassifier
from mdebris.viz.plots import save_figure, stretch_to_uint8

log = logging.getLogger("samples")

# Okabe-Ito based, matching mdebris.viz.plots.CLASS_COLORS where the classes align.
# Debris is the vivid amber; the confusers get distinct hues because telling them
# apart is the entire point of the figure.
CLASS_COLORS: dict[str, str] = {
    "Marine Debris": "#E69F00",
    "Dense Sargassum": "#009E73",
    "Sparse Sargassum": "#66C2A5",
    "Natural Organic Material": "#8C6D31",
    "Ship": "#CC79A7",
    "Clouds": "#FFFFFF",
    "Marine Water": "#0072B2",
    "Sediment-Laden Water": "#D55E00",
    "Foam": "#F0E442",
    "Turbid Water": "#7C5B8F",
    "Shallow Water": "#56B4E9",
    "Waves": "#9ED1F0",
    "Cloud Shadows": "#4D4D4D",
    "Wakes": "#B3E2F5",
    "Mixed Water": "#3B7A9E",
}


def rgb_from_cube(image: np.ndarray) -> np.ndarray:
    """True colour from the MARIDA band cube.

    The stretch is deliberately JOINT across the three bands rather than per channel.
    Open water sits in a narrow dark slice of every band, so normalising R, G and B
    independently rescales each band's sensor noise to full range and decorrelates
    them, turning an ocean scene into colour static. A joint stretch keeps the ratios
    between bands, which is what makes the result look like water.
    """
    idx = {name: i for i, name in enumerate(MARIDA_BANDS)}
    stack = np.dstack([image[idx["B04"]], image[idx["B03"]], image[idx["B02"]]])
    return stretch_to_uint8(stack, low=1.0, high=99.0, per_channel=False)


def crop_to_labels(truth: np.ndarray, *, pad: int = 24, minimum: int = 96) -> tuple[slice, slice]:
    """Window around the annotated pixels.

    MARIDA is sparsely annotated: a patch of 65,536 pixels may carry only ~100 labels.
    At full extent those are invisible, so the figure would show a pretty picture and
    prove nothing. Cropping to the annotation is what makes the comparison readable.
    """
    rows, cols = np.nonzero(truth)
    if rows.size == 0:
        return slice(None), slice(None)
    h, w = truth.shape
    r0, r1 = int(rows.min()) - pad, int(rows.max()) + pad + 1
    c0, c1 = int(cols.min()) - pad, int(cols.max()) + pad + 1
    # grow a too-small window about its centre so tiny annotations stay in context
    if r1 - r0 < minimum:
        mid = (r0 + r1) // 2
        r0, r1 = mid - minimum // 2, mid + minimum // 2
    if c1 - c0 < minimum:
        mid = (c0 + c1) // 2
        c0, c1 = mid - minimum // 2, mid + minimum // 2
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h, r1), min(w, c1)
    return slice(r0, r1), slice(c0, c1)


def label_image(codes: np.ndarray, names: list[str]) -> tuple[np.ndarray, ListedColormap]:
    """Map class codes to a colour image, leaving unlabelled pixels transparent."""
    lut = np.zeros((len(names) + 1, 4), dtype=np.float32)
    for i, name in enumerate(names, start=1):
        rgb = matplotlib.colors.to_rgb(CLASS_COLORS.get(name, "#999999"))
        lut[i] = (*rgb, 1.0)
    return lut[codes], ListedColormap(lut[1:, :3])


def pick_patches(split: str, limit: int) -> list:
    """Patches containing annotated Marine Debris, largest first."""
    debris_code = MARIDA_CLASSES.index("Marine Debris") + 1
    scored = []
    for patch in load_marida_split(split):
        _, classes, _ = patch.read()
        n = int((classes == debris_code).sum())
        if n:
            scored.append((n, patch))
    scored.sort(key=lambda kv: -kv[0])
    return [p for _, p in scored[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument("--out", type=Path, default=Path("assets/classification_samples.png"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    clf = SpectralClassifier.load(args.model)
    patches = pick_patches(args.split, args.n)
    if not patches:
        raise SystemExit(f"no {args.split} patch contains annotated Marine Debris")
    log.info("rendering %d patches from the %s split", len(patches), args.split)

    names = list(MARIDA_CLASSES)
    code_of = {name: i + 1 for i, name in enumerate(names)}

    fig, axes = plt.subplots(len(patches), 3, figsize=(11.5, 3.9 * len(patches)))
    axes = np.atleast_2d(axes)
    present: set[str] = set()

    for row, patch in enumerate(patches):
        image, truth, _ = patch.read()
        bands = {name: image[i] for i, name in enumerate(MARIDA_BANDS)}
        predicted_names = clf.predict_image(bands)
        predicted = np.vectorize(lambda s: code_of.get(str(s), 0))(predicted_names).astype(np.uint8)

        # Classify on the full patch, then show only the annotated neighbourhood.
        rs, cs = crop_to_labels(truth)
        image = image[:, rs, cs]
        truth = truth[rs, cs]
        predicted = predicted[rs, cs]

        # Score only where a human actually annotated, and show only those pixels.
        annotated = truth != 0
        predicted_shown = np.where(annotated, predicted, 0).astype(np.uint8)
        agreement = (
            float((predicted[annotated] == truth[annotated]).mean()) if annotated.any() else 0.0
        )

        present.update(names[c - 1] for c in np.unique(truth) if c)
        present.update(names[c - 1] for c in np.unique(predicted_shown) if c)

        axes[row, 0].imshow(rgb_from_cube(image))
        axes[row, 0].set_title(
            f"{patch.roi}\n{truth.shape[0]}x{truth.shape[1]} px crop",
            fontsize=9,
            fontfamily="monospace",
            pad=6,
        )
        axes[row, 1].imshow(rgb_from_cube(image))
        axes[row, 1].imshow(label_image(truth, names)[0])
        axes[row, 1].set_title(
            f"human annotation\n{int(annotated.sum()):,} labelled px",
            fontsize=9,
            fontfamily="monospace",
            pad=6,
        )
        axes[row, 2].imshow(rgb_from_cube(image))
        axes[row, 2].imshow(label_image(predicted_shown, names)[0])
        axes[row, 2].set_title(
            f"predicted\n{agreement * 100:.1f}% agreement on labelled px",
            fontsize=9,
            fontfamily="monospace",
            pad=6,
        )
        for ax in axes[row]:
            ax.set_axis_off()

    handles = [
        mpatches.Patch(color=CLASS_COLORS.get(n, "#999999"), label=n) for n in names if n in present
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, max(1, len(handles))),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
    )
    fig.subplots_adjust(bottom=0.09, hspace=0.12, wspace=0.03)
    path = save_figure(fig, args.out, tight=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
