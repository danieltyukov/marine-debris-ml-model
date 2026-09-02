"""Package the MARIDA pixels a cloud or shallow-water mask can be tested against.

    python scripts/package_lanot_subset.py

Writes ``docs/lanot_subset.csv.gz`` with one row per labelled pixel, ``docs/lanot_subset.md``
describing it, and ``docs/lanot_subset.json`` with the counts.

Why this exists
---------------
Uriel Mendoza, technical lead of the LANOT sargassum platform at UNAM, named the three
sources of false detections the operational system actually sees (2 September 2026):
isolated pixels along the edges of thin cloud, cloud shadows, and shallow water near the
coast where bottom reflectance comes through. MARIDA annotates all three as classes
(Clouds, Cloud Shadows, Shallow Water), and four of its seventeen sites are on the 18
Sentinel-2 tiles the platform processes. A labelled test set for exactly those failures,
inside their own footprint, therefore already exists. This extracts it.

What is in it
-------------
Every annotated pixel on the four overlap tiles, in every split, whose class is one of

    Dense Sargassum, Sparse Sargassum     the positives a mask must keep
    Clouds, Cloud Shadows, Shallow Water  the confusers a mask should reject

with the 11 MARIDA bands, the annotation confidence, pixel-centre coordinates in the
patch CRS and in WGS84, and two booleans: whether expression (1) of Arellano-Verdejo
et al. 2025 fires on the pixel, and whether the pixel passes that expression's
``b11 < 0.05`` term on its own.

All three splits are included and tagged. The use here is testing a mask, which has no
training split to keep clean. Anyone scoring a learned model on these rows should keep
to ``split == "test"``: the classifier in this repository has seen the train split.

What the reflectance is
-----------------------
MARIDA is ACOLITE output on Level-1C: Rayleigh-corrected reflectance from the dark
spectrum fitting processor, with the 20 m and 60 m bands replicated to 10 m (Kikaki et
al. 2022). It is not Sen2Cor Level-2A, which is what LANOT's pipeline runs and what
expression (1) was calibrated on. Both are unitless 0-1 reflectance, but the
Rayleigh-corrected product still carries the aerosol term, which is largest in the blue
and smallest in the SWIR. The expression (1) column is therefore the published rule
applied to reflectance it was not tuned for. Read it as a baseline on the confuser
classes, not as a measurement of the platform, which also segments and filters the
pixel mask before anything is published.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy

from eval_lanot_operator import LANOT_TILES, SWIR_GATE, lanot_operator
from mdebris.data.marida import (
    MARIDA_BANDS,
    MARIDA_CITATION,
    MARIDA_CLASSES,
    MARIDA_CONFIDENCE,
    MARIDA_LICENSE,
    MARIDA_SITES,
    MaridaPatch,
    load_marida_split,
)

log = logging.getLogger("lanot-subset")

SPLITS: tuple[str, ...] = ("train", "val", "test")

POSITIVE_CLASSES: tuple[str, ...] = ("Dense Sargassum", "Sparse Sargassum")
CONFUSER_CLASSES: tuple[str, ...] = ("Clouds", "Cloud Shadows", "Shallow Water")
SUBSET_CLASSES: tuple[str, ...] = POSITIVE_CLASSES + CONFUSER_CLASSES

# Column order in the CSV. Kept explicit so the file is stable across pandas versions.
COLUMNS: tuple[str, ...] = (
    "roi",
    "split",
    "tile",
    "date",
    "row",
    "col",
    "epsg",
    "easting",
    "northing",
    "lon",
    "lat",
    "label",
    "confidence",
    *MARIDA_BANDS,
    "lanot_expr1",
    "b11_gate",
)

COLUMN_NOTES: dict[str, str] = {
    "roi": "MARIDA patch id, `d-m-yy_TILE_index`, as in the split files",
    "split": "train, val or test, MARIDA's published split",
    "tile": "Sentinel-2 MGRS tile",
    "date": "acquisition date, ISO",
    "row": "pixel row inside the 256x256 patch, 0 at the top",
    "col": "pixel column inside the patch, 0 at the left",
    "epsg": "EPSG code of the patch CRS (UTM zone 16N here)",
    "easting": "pixel-centre easting in that CRS, metres",
    "northing": "pixel-centre northing in that CRS, metres",
    "lon": "pixel-centre longitude, WGS84",
    "lat": "pixel-centre latitude, WGS84",
    "label": "MARIDA class name",
    "confidence": "annotator confidence, High / Moderate / Low",
    "B01 .. B12": "ACOLITE Rayleigh-corrected reflectance, 0-1, the 11 MARIDA bands",
    "lanot_expr1": "True where expression (1) of Arellano-Verdejo et al. 2025 fires",
    "b11_gate": "True where B11 < 0.05, the SWIR term of expression (1) on its own",
}


# The five terms of expression (1), separately, so the report can say which one rejects
# each class. ``lanot_operator`` is their conjunction and stays the source of truth.
EXPRESSION_TERMS: dict[str, str] = {
    "b8A < 0.07": "B8A < 0.07",
    "b04 < 0.1": "B04 < 0.1",
    "b11 < 0.05": "B11 < 0.05",
    "b04 < b8A": "B04 < B8A",
    "b04 < b08": "B04 < B08",
}


def expression_terms(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Each term of expression (1) evaluated per row, keyed as in :data:`EXPRESSION_TERMS`."""
    return {
        "b8A < 0.07": df["B8A"] < 0.07,
        "b04 < 0.1": df["B04"] < 0.1,
        "b11 < 0.05": df["B11"] < SWIR_GATE,
        "b04 < b8A": df["B04"] < df["B8A"],
        "b04 < b08": df["B04"] < df["B08"],
    }


def overlap_tiles() -> frozenset[str]:
    """MARIDA sites that are also LANOT tiles, computed rather than asserted."""
    return frozenset(MARIDA_SITES) & LANOT_TILES


def patch_rows(
    patch: MaridaPatch, split: str, classes: tuple[str, ...] = SUBSET_CLASSES
) -> pd.DataFrame | None:
    """One row per pixel of ``patch`` annotated with one of ``classes``.

    Pixels with a NaN in any band are dropped: MARIDA carries NaN where the source
    scene had no data, and a reflectance a rule cannot evaluate is not a test case.

    Returns:
        A frame in :data:`COLUMNS` order, or None when nothing in the patch qualifies.
    """
    image, cls, conf = patch.read()
    keep = np.isin(cls, [MARIDA_CLASSES.index(c) + 1 for c in classes])
    keep &= ~np.isnan(image).any(axis=0)
    if not keep.any():
        return None

    rows, cols = np.nonzero(keep)
    with rasterio.open(patch.image) as src:
        transform, crs = src.transform, src.crs
    easting, northing = xy(transform, rows, cols, offset="center")
    easting, northing = np.asarray(easting), np.asarray(northing)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(easting, northing)

    class_names = np.array(("", *MARIDA_CLASSES))
    confidence_names = np.array(("", *MARIDA_CONFIDENCE))
    bands = {name: image[j][keep] for j, name in enumerate(MARIDA_BANDS)}

    frame = pd.DataFrame(
        {
            "roi": patch.roi,
            "split": split,
            "tile": patch.tile,
            "date": patch.date,
            "row": rows.astype(np.int16),
            "col": cols.astype(np.int16),
            "epsg": crs.to_epsg(),
            "easting": np.round(easting).astype(np.int64),
            "northing": np.round(northing).astype(np.int64),
            "lon": np.round(lon, 6),
            "lat": np.round(lat, 6),
            "label": class_names[cls[keep]],
            "confidence": confidence_names[conf[keep]],
            **{name: np.round(values, 5) for name, values in bands.items()},
            "lanot_expr1": lanot_operator(bands),
            "b11_gate": bands["B11"] < SWIR_GATE,
        }
    )
    return frame[list(COLUMNS)]


def build_subset(
    root: Path | str | None = None,
    *,
    tiles: frozenset[str] | None = None,
    classes: tuple[str, ...] = SUBSET_CLASSES,
    splits: tuple[str, ...] = SPLITS,
) -> pd.DataFrame:
    """Every qualifying pixel across ``splits``, restricted to ``tiles``.

    Args:
        root: MARIDA directory. None uses the configured cache.
        tiles: MGRS tiles to keep. None means the MARIDA sites inside LANOT's footprint.
        classes: MARIDA classes to keep.
        splits: Which published splits to read.

    Raises:
        RuntimeError: when no pixel qualifies, which means the wrong tiles or classes.
    """
    tiles = overlap_tiles() if tiles is None else tiles
    frames: list[pd.DataFrame] = []
    for split in splits:
        patches = [p for p in load_marida_split(split, root) if p.tile in tiles]
        log.info("%s: %d patches on %s", split, len(patches), ", ".join(sorted(tiles)))
        for patch in patches:
            frame = patch_rows(patch, split, classes)
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise RuntimeError(f"no pixels of {classes} on tiles {sorted(tiles)} in {splits}")
    return pd.concat(frames, ignore_index=True)


def _fraction(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def summarise(df: pd.DataFrame) -> dict:
    """Counts and the expression (1) baseline per class, split and tile."""
    per_class = {}
    for name in [c for c in SUBSET_CLASSES if c in set(df["label"])]:
        sel = df["label"] == name
        per_class[name] = {
            "pixels": int(sel.sum()),
            "patches": int(df.loc[sel, "roi"].nunique()),
            "expr1_fires": int(df.loc[sel, "lanot_expr1"].sum()),
            "expr1_fraction": _fraction(df.loc[sel, "lanot_expr1"]),
            "b11_gate_passes": int(df.loc[sel, "b11_gate"].sum()),
            "b11_gate_fraction": _fraction(df.loc[sel, "b11_gate"]),
            "median_b11": float(df.loc[sel, "B11"].median()),
        }

    cloud = df["label"] == "Clouds"
    thin = cloud & df["b11_gate"]
    thin_cloud = {
        "pixels": int(thin.sum()),
        "fraction_of_cloud": _fraction(df.loc[cloud, "b11_gate"]),
        "expr1_fires": int(df.loc[thin, "lanot_expr1"].sum()),
        "expr1_fraction": _fraction(df.loc[thin, "lanot_expr1"]),
        "median_b11": float(df.loc[thin, "B11"].median()) if thin.any() else 0.0,
    }

    terms = expression_terms(df)
    per_term = {
        name: {term: _fraction(passes[df["label"] == name]) for term, passes in terms.items()}
        for name in per_class
    }
    for name in per_class:
        sel = df["label"] == name
        per_class[name]["median_b8a"] = float(df.loc[sel, "B8A"].median())

    per_split = {
        split: {label: int(n) for label, n in group["label"].value_counts().items()}
        for split, group in df.groupby("split", sort=False)
    }
    per_tile = {
        tile: {
            "site": MARIDA_SITES.get(tile, tile),
            "patches": int(group["roi"].nunique()),
            **{label: int(n) for label, n in group["label"].value_counts().items()},
        }
        for tile, group in df.groupby("tile", sort=True)
    }
    return {
        "pixels": len(df),
        "patches": int(df["roi"].nunique()),
        "dates": int(df["date"].nunique()),
        "date_range": [str(df["date"].min()), str(df["date"].max())],
        "tiles": sorted(set(df["tile"])),
        "splits": [s for s in SPLITS if s in set(df["split"])],
        "classes": list(per_class),
        "per_class": per_class,
        "per_term": per_term,
        "thin_cloud": thin_cloud,
        "per_split": per_split,
        "per_tile": per_tile,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _markdown(summary: dict, csv_name: str) -> str:
    c, t = summary["per_class"], summary["thin_cloud"]
    positives = [n for n in POSITIVE_CLASSES if n in c]
    confusers = [n for n in CONFUSER_CLASSES if n in c]
    lines = [
        "# MARIDA pixels for testing a cloud, shadow or shallow-water mask inside LANOT's footprint",
        "",
        f"Produced by `python scripts/package_lanot_subset.py`. The data is `{csv_name}`",
        "next to this file, one row per pixel, gzip-compressed CSV.",
        "",
        "## What it is",
        "",
        "The LANOT sargassum platform (Instituto de Geografia, UNAM) reports three sources",
        "of false detections in operation: isolated pixels along the edges of thin cloud,",
        "cloud shadows, and shallow water near the coast where the bottom shows through.",
        "MARIDA (Kikaki et al. 2022) annotates all three as classes, and four of its",
        "seventeen sites are on the 18 Sentinel-2 tiles the platform processes. This file is",
        "every MARIDA pixel on those four tiles annotated as one of those three confusers or",
        "as sargassum, so a candidate mask can be scored on both halves of its job: reject",
        "the confusers, keep the sargassum.",
        "",
        "| | |",
        "|---|---|",
        f"| pixels | {summary['pixels']:,} |",
        f"| patches | {summary['patches']:,} |",
        f"| acquisition dates | {summary['dates']} ({summary['date_range'][0]} to {summary['date_range'][1]}) |",
        f"| tiles | {', '.join(summary['tiles'])} |",
        f"| splits | {', '.join(summary['splits'])} |",
        "",
        "## Baseline: what expression (1) does on each class",
        "",
        "Expression (1) of Arellano-Verdejo et al. 2025 is the platform's per-pixel detection",
        "rule, `(b8A < 0.07) and (b04 < 0.1) and (b11 < 0.05) and (b04 < b8A) and (b04 < b08)`.",
        "For the sargassum rows the `fires` column is recall, and higher is better. For the",
        "confuser rows it is the leak a mask would have to close, and lower is better. The",
        "`b11 < 0.05` column is that one term alone.",
        "",
        "| class | pixels | patches | expression (1) fires | passes b11 < 0.05 | median B11 |",
        "|---|---|---|---|---|---|",
    ]
    for name in positives + confusers:
        r = c[name]
        lines.append(
            f"| {name} | {r['pixels']:,} | {r['patches']} | {r['expr1_fires']:,} "
            f"({_pct(r['expr1_fraction'])}) | {r['b11_gate_passes']:,} "
            f"({_pct(r['b11_gate_fraction'])}) | {r['median_b11']:.4f} |"
        )
    lines += [
        "",
        "The cloud the platform actually sees as noise is the thin kind, and the SWIR term",
        "does not remove it because it is not bright. On these rows:",
        "",
        "| thin cloud (Clouds with B11 < 0.05) | |",
        "|---|---|",
        f"| pixels | {t['pixels']:,} ({_pct(t['fraction_of_cloud'])} of all cloud) |",
        f"| expression (1) fires on | {t['expr1_fires']:,} ({_pct(t['expr1_fraction'])}) |",
        f"| median B11 | {t['median_b11']:.4f} |",
        "",
        "Those are the rows a thin-cloud mask is for. `df[(df.label == 'Clouds') & df.b11_gate]`",
        "selects them.",
        "",
        "## Which term does the work",
        "",
        "Fraction of each class that passes each term of expression (1) on its own. A",
        "confuser is rejected by whichever term has the lowest number in its row; a",
        "sargassum class is lost to it.",
        "",
        "| class | " + " | ".join(EXPRESSION_TERMS) + " | median B8A |",
        "|---|" + "---|" * (len(EXPRESSION_TERMS) + 1),
    ]
    for name in positives + confusers:
        lines.append(
            f"| {name} | "
            + " | ".join(_pct(summary["per_term"][name][term]) for term in EXPRESSION_TERMS)
            + f" | {c[name]['median_b8a']:.3f} |"
        )
    lines += [
        "",
        "Two things follow. Cloud shadow and shallow water are rejected by the red-edge",
        "terms, `b04 < b08` above all, not by the SWIR gate, which nearly all of them pass.",
        "And what MARIDA annotates as Dense Sargassum is lost almost entirely to the",
        "`b8A < 0.07` ceiling: those mats are bright in the near-infrared, well above the",
        "cap, while passing the other four terms. Some of that gap is the processor, since",
        "Rayleigh-corrected reflectance sits above Sen2Cor's in the NIR, but not a doubling.",
        "Whether the ceiling is a deliberate choice against land and foam, or whether dense",
        "mats in Sen2Cor L2A really do stay under it, is a question for the people who",
        "calibrated the rule.",
        "",
        "## Per split",
        "",
        "All three of MARIDA's published splits are included and tagged. A mask has no",
        "training split to keep clean, so use everything. A learned model scored on these",
        "rows should keep to `test`; the classifier in this repository was trained on `train`.",
        "",
        "| split | " + " | ".join(positives + confusers) + " |",
        "|---|" + "---|" * len(positives + confusers),
    ]
    for split in ("train", "val", "test"):
        if split not in summary["per_split"]:
            continue
        counts = summary["per_split"][split]
        lines.append(
            f"| {split} | "
            + " | ".join(f"{counts.get(n, 0):,}" for n in positives + confusers)
            + " |"
        )
    lines += [
        "",
        "## Per tile",
        "",
        "| tile | site | patches | " + " | ".join(positives + confusers) + " |",
        "|---|---|---|" + "---|" * len(positives + confusers),
    ]
    for tile, r in summary["per_tile"].items():
        lines.append(
            f"| {tile} | {r['site']} | {r['patches']} | "
            + " | ".join(f"{r.get(n, 0):,}" for n in positives + confusers)
            + " |"
        )
    lines += [
        "",
        "## Columns",
        "",
        "| column | meaning |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {v} |" for k, v in COLUMN_NOTES.items()]
    lines += [
        "",
        "Coordinates are pixel centres. `row` and `col` index the 256x256 MARIDA patch named",
        "in `roi`, so any row can be traced back to the original GeoTIFF.",
        "",
        "## What the reflectance is, and why it matters here",
        "",
        "MARIDA is ACOLITE output on Level-1C: Rayleigh-corrected reflectance from the dark",
        "spectrum fitting processor, with the 20 m and 60 m bands replicated to 10 m. It is",
        "not Sen2Cor Level-2A, which is what the LANOT pipeline runs and what expression (1)",
        "was calibrated on. Both are unitless 0-1 reflectance, but the Rayleigh-corrected",
        "product still carries the aerosol term, which is largest in the blue and smallest in",
        "the SWIR. So the `lanot_expr1` column is the published rule applied to reflectance it",
        "was not tuned for. It is a baseline for these classes, not a measurement of the",
        "platform, which segments and filters the pixel mask before anything is published.",
        "",
        "The same fact cuts the other way: LANOT are evaluating ACOLITE for the platform, and",
        "these rows are already ACOLITE reflectance with labels attached.",
        "",
        "## Scoring a candidate mask",
        "",
        "```python",
        "import pandas as pd",
        "",
        f'df = pd.read_csv("{csv_name}")',
        "",
        "# A mask is a boolean per row, True where the pixel is kept for detection.",
        "# The SWIR term of expression (1) is the trivial baseline:",
        'keep = df["B11"] < 0.05',
        "",
        'positive = df["label"].isin(["Dense Sargassum", "Sparse Sargassum"])',
        'print("sargassum kept:", f"{keep[positive].mean():.3f}")',
        'for name in ["Clouds", "Cloud Shadows", "Shallow Water"]:',
        '    rejected = 1 - keep[df["label"] == name].mean()',
        '    print(f"{name} rejected: {rejected:.3f}")',
        "```",
        "",
        "A mask that uses more than the 11 bands, for instance the spatial context around a",
        "cloud edge, needs the patches themselves. `roi`, `row` and `col` locate every pixel",
        "in the MARIDA GeoTIFFs, which are on Zenodo.",
        "",
        "## Licence and citation",
        "",
        f"The pixels are MARIDA, {MARIDA_LICENSE}. Cite the dataset, not this repository:",
        "",
        f"> {MARIDA_CITATION}",
        "",
        "The detection rule is expression (1) of Arellano-Verdejo, J., Lazcano-Hernandez, H.E.,",
        "Prado Molina, J. et al. *Towards enhanced Sargassum monitoring in the Caribbean Sea.*",
        "Sci Rep 15, 8965 (2025). <https://doi.org/10.1038/s41598-025-93001-9>",
        "",
        "MARIDA's labels are annotations by remote-sensing researchers, not field",
        "observations, and MARIDA annotates a confuser-rich selection of each scene rather",
        "than whole scenes. Class proportions here say nothing about how common each",
        "confuser is in an operational scene.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marida-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/lanot_subset.csv.gz"))
    parser.add_argument("--report", type=Path, default=Path("docs/lanot_subset.md"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = build_subset(args.marida_root)
    summary = summarise(df)
    log.info(
        "%s pixels from %d patches on %s",
        f"{summary['pixels']:,}",
        summary["patches"],
        ", ".join(summary["tiles"]),
    )
    for name, r in summary["per_class"].items():
        log.info(
            "  %-18s %8s pixels, expression (1) fires on %5.1f%%, b11 gate passes %5.1f%%",
            name,
            f"{r['pixels']:,}",
            100 * r["expr1_fraction"],
            100 * r["b11_gate_fraction"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    args.report.write_text(_markdown(summary, args.out.name), encoding="utf-8")
    args.report.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s (%.1f MB), %s", args.out, args.out.stat().st_size / 1e6, args.report)


if __name__ == "__main__":
    main()
