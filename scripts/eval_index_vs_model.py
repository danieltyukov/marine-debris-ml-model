"""Physical indices against the classifier, pixel by pixel, on every pass of a season.

    python scripts/eval_index_vs_model.py --island bonaire --focus lac_bay
    python scripts/eval_index_vs_model.py --island bonaire --focus lac_bay --no-mangrove-mask
    python scripts/eval_index_vs_model.py --island bonaire --focus lac_bay --analyse-only

On every pass the season runner keeps (same reader, land and mangrove masks, surf zones,
cloud mask, classifier and calibrator), compare what the classifier flags as sargassum
with what one index and one cut flag:

- classifier: calibrated sargassum probability >= 0.90, the season run's edge;
- FAI >= 0.0443 and FDI >= 0.0240: the sargassum rows of ``docs/band_ablation.json``,
  chosen for best F1 on MARIDA validation and scored on MARIDA test;
- NDVI >= 0.2375: chosen on MARIDA validation with the same protocol, because the band
  ablation publishes no NDVI rule.

The index values are the classifier's own feature columns, so they are computed on
exactly the pixels it scored. Stage one reads each pass once and caches the usable
pixels' flat index, score, FAI, FDI and NDVI under
``~/.cache/mdebris/index-vs-model/<prefix>[-nomask]/`` (about 2 MB a pass; delete it
freely). Stage two writes ``docs/<prefix>_index_vs_model.csv`` (one row per segment per
pass) and ``.json`` (totals, overlaps, distinct and stationary pixels per rule, the focus
segment month by month, agreement on coarser blocks). ``--analyse-only`` skips the
network. ``docs/index_vs_model.md`` reports the 2.0 Bonaire run, which is
``--no-mangrove-mask``.

Counting: a pixel look is one visible water pixel on one pass. Group totals add the
segments' rows, as the season reports do, so a pixel where two surf zones overlap counts
in both. A pixel is stationary for a rule when that rule flags it on at least half of the
(at least five) passes that saw it; transient flags are the rest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mdebris.coastal import load_segments, surf_zone
from mdebris.coastal.islands import load_island
from mdebris.coastal.masks import load_polygons
from mdebris.coastal.runner import (
    STATIONARY_MIN_PASSES,
    STATIONARY_SHARE,
    area_of_interest,
    find_passes,
    load_operating_points,
    process_pass,
)
from mdebris.eval.calibration import IsotonicCalibrator
from mdebris.models.spectral import SpectralClassifier

log = logging.getLogger("index-vs-model")

INDICES = ("FAI", "FDI", "NDVI")
METHODS = ("model", *INDICES)
# Block sizes in pixels for the coarser comparison: 10 m, 30 m, 100 m, 300 m, 1 km.
BLOCKS = (1, 3, 10, 30, 100)
NDVI_CUT = 0.23753255605697632
CACHE_ROOT = Path.home() / ".cache" / "mdebris" / "index-vs-model"


def load_cuts(band_ablation: Path) -> dict[str, float]:
    published = json.loads(band_ablation.read_text(encoding="utf-8"))["baselines"]
    cuts = {
        name: float(published[f"{name.lower()}_threshold"]["sargassum"]["threshold"])
        for name in ("FAI", "FDI")
    }
    cuts["NDVI"] = NDVI_CUT
    return cuts


def extract(args, island, part, cache: Path) -> list[dict]:
    """Read every pass through the runner and cache the usable pixels (network)."""
    from types import SimpleNamespace

    cache.mkdir(parents=True, exist_ok=True)
    segments = load_segments(island.segments_path)
    if part.segment_ids:
        segments = [s for s in segments if s.segment_id in part.segment_ids]
    zones = {s.segment_id: surf_zone(s, 500.0) for s in segments}
    land = load_polygons(island.island_path)
    mangroves = None if args.no_mangrove_mask else load_polygons(island.mangroves_path)
    aoi = area_of_interest(SimpleNamespace(island=island, part=part), zones, land)
    points = load_operating_points(args.calibration_json, calibrated=True)
    clf = SpectralClassifier.load(args.model)
    calibrator = IsotonicCalibrator.load(args.calibrator)
    scenes, _dropped = find_passes(
        aoi,
        args.start,
        args.end,
        tiles_ok=part.accepts,
        zones=list(zones.values()),
        min_cover=part.min_cover,
    )
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    meta = []
    for i, scene in enumerate(scenes, 1):
        out = cache / f"{scene.scene_id}.npz"
        rows_path = cache / f"{scene.scene_id}.json"
        if not (out.exists() and rows_path.exists()):
            started = time.perf_counter()
            result = process_pass(
                scene,
                aoi=aoi,
                segments=segments,
                zones=zones,
                land=land,
                mangroves=mangroves,
                clf=clf,
                calibrator=calibrator,
                points=points,
                surf_zone_m=500.0,
                land_buffer_m=island.land_buffer_m,
                mangrove_buffer_m=island.mangrove_buffer_m,
                keep_scores=True,
            )
            kept = result.scores
            if not np.array_equal(
                kept["score"] >= points.high, result.hit_mask.ravel()[kept["flat"]]
            ):
                raise RuntimeError(f"{scene.scene_id}: kept scores disagree with the flag mask")
            zones_path = cache / "zones.npz"
            if not zones_path.exists():
                np.savez_compressed(
                    zones_path,
                    zone_ids=np.asarray(list(result.zones), dtype=str),
                    zones=np.stack(list(result.zones.values())),
                    shape=np.asarray(result.hit_mask.shape),
                )
            np.savez_compressed(out, **kept)
            rows_path.write_text(
                json.dumps(
                    {
                        "date": str(scene.datetime)[:10],
                        "platform": scene.platform or "",
                        "observability": {r["segment_id"]: r["observability"] for r in result.rows},
                    }
                ),
                encoding="utf-8",
            )
            log.info(
                "%2d/%d %s usable %7d  model %5d  (%.0fs)",
                i,
                len(scenes),
                str(scene.datetime)[:10],
                kept["flat"].size,
                int((kept["score"] >= points.high).sum()),
                time.perf_counter() - started,
            )
        meta.append({"scene_id": scene.scene_id, **json.loads(rows_path.read_text())})
    (cache / "passes.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _total(rows: list[dict]) -> dict:
    def s(col: str) -> int:
        return sum(r[col] for r in rows)

    def ratio(a, b):
        return a / b if b else None

    t = {
        "segment_passes": len(rows),
        "usable_pixels": s("usable_pixels"),
        "model_pixels": s("model_pixels"),
        "model_transient_pixels": s("model_transient_pixels"),
    }
    for k in INDICES:
        for c in (
            f"{k}_pixels",
            f"model_and_{k}",
            f"model_or_{k}",
            f"{k}_only_model_clear",
            f"{k}_transient_pixels",
            f"model_on_{k}_stationary",
        ):
            t[c] = s(c)
        t[f"jaccard_model_{k}"] = ratio(t[f"model_and_{k}"], t[f"model_or_{k}"])
        t[f"share_of_model_also_{k}"] = ratio(t[f"model_and_{k}"], t["model_pixels"])
        t[f"share_of_{k}_also_model"] = ratio(t[f"model_and_{k}"], t[f"{k}_pixels"])
        t[f"share_of_{k}_on_stationary"] = ratio(
            t[f"{k}_pixels"] - t[f"{k}_transient_pixels"], t[f"{k}_pixels"]
        )
        t[f"share_of_{k}_only_model_clear"] = ratio(
            t[f"{k}_only_model_clear"], t[f"{k}_pixels"] - t[f"model_and_{k}"]
        )
        t[f"share_of_model_on_{k}_stationary"] = ratio(
            t[f"model_on_{k}_stationary"], t["model_pixels"]
        )
    for n in (0, 1):
        t[f"model_within_{n}px_FAI_stationary"] = s(f"model_within_{n}px_FAI_stationary")
        t[f"usable_within_{n}px_FAI_stationary"] = s(f"usable_within_{n}px_FAI_stationary")
    for m in METHODS:
        t[f"{m}_per_1000"] = ratio(1000 * t[f"{m}_pixels"], t["usable_pixels"])
        t[f"{m}_transient_per_1000"] = ratio(1000 * t[f"{m}_transient_pixels"], t["usable_pixels"])
    return t


def analyse(args, island, part, cache: Path, meta: list[dict]) -> dict:
    from scipy import ndimage

    cuts = load_cuts(args.band_ablation)
    points = load_operating_points(args.calibration_json, calibrated=True)
    high, low = points.high, points.low
    z = np.load(cache / "zones.npz")
    zone_ids = z["zone_ids"].tolist()
    zones = z["zones"]
    shape = tuple(int(v) for v in z["shape"])
    n_pix, n_cols = shape[0] * shape[1], shape[1]
    membership = np.zeros(n_pix, dtype=np.uint32)
    for j, zone in enumerate(zones):
        membership[zone.ravel()] |= np.uint32(1 << j)
    exposure = {s.segment_id: s.exposure for s in island.segments}
    windward = [s for s in zone_ids if exposure.get(s) == "windward"]
    groups: dict[str, list[str]] = {}
    if args.focus:
        groups[args.focus] = [args.focus]
        groups["other windward"] = [s for s in windward if s != args.focus]
    else:
        groups["windward"] = windward
    if island.leeward_control in zone_ids:
        groups["leeward control"] = [island.leeward_control]
    group_bits = {
        g: np.uint32(sum(1 << zone_ids.index(s) for s in segs)) for g, segs in groups.items()
    }

    def load(scene_id: str):
        blob = np.load(cache / f"{scene_id}.npz")
        flags = {"model": blob["score"] >= high, **{k: blob[k] >= cuts[k] for k in INDICES}}
        return blob, flags

    seen = np.zeros(n_pix, np.int16)
    hits = {m: np.zeros(n_pix, np.int16) for m in METHODS}
    for p in meta:
        blob, flags = load(p["scene_id"])
        flat = blob["flat"]
        seen[flat] += 1
        for m in METHODS:
            hits[m][flat[flags[m]]] += 1
    enough = seen >= STATIONARY_MIN_PASSES
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = {
            m: enough & (np.where(enough, hits[m] / np.maximum(seen, 1), 0.0) >= STATIONARY_SHARE)
            for m in METHODS
        }
    fai_still = stationary["FAI"].reshape(shape)
    near_fai = {
        0: fai_still.ravel(),
        1: ndimage.binary_dilation(fai_still, structure=np.ones((3, 3), bool)).ravel(),
    }

    rows: list[dict] = []
    blocks = {g: {k_px: defaultdict(int) for k_px in BLOCKS} for g in groups}
    for p in meta:
        blob, flags = load(p["scene_id"])
        flat = blob["flat"]
        score = blob["score"]
        transient = {m: flags[m] & ~stationary[m][flat] for m in METHODS}
        clear = score < low
        bits = membership[flat]
        rr, cc = np.divmod(flat.astype(np.int64), n_cols)
        for g, gb in group_bits.items():
            sel = (bits & gb) > 0
            for k_px in BLOCKS:
                bid = (rr[sel] // k_px) * (n_cols // k_px + 1) + cc[sel] // k_px
                sets = {m: np.unique(bid[flags[m][sel]]) for m in METHODS}
                acc = blocks[g][k_px]
                for k in INDICES:
                    acc[f"model_and_{k}"] += np.intersect1d(sets["model"], sets[k]).size
                    acc[f"model_or_{k}"] += np.union1d(sets["model"], sets[k]).size
        for j, seg in enumerate(zone_ids):
            sel = (bits & np.uint32(1 << j)) > 0
            row = {
                "scene_id": p["scene_id"],
                "date": p["date"],
                "segment_id": seg,
                "observability": p["observability"][seg],
                "usable_pixels": int(sel.sum()),
                "model_pixels": int((flags["model"] & sel).sum()),
                "model_transient_pixels": int((transient["model"] & sel).sum()),
            }
            for k in INDICES:
                row[f"{k}_pixels"] = int((flags[k] & sel).sum())
                row[f"model_and_{k}"] = int((flags["model"] & flags[k] & sel).sum())
                row[f"model_or_{k}"] = int(((flags["model"] | flags[k]) & sel).sum())
                row[f"{k}_only_model_clear"] = int((flags[k] & ~flags["model"] & clear & sel).sum())
                row[f"{k}_transient_pixels"] = int((transient[k] & sel).sum())
                row[f"model_on_{k}_stationary"] = int(
                    (flags["model"] & stationary[k][flat] & sel).sum()
                )
            for n in (0, 1):
                near = near_fai[n][flat] & sel
                row[f"usable_within_{n}px_FAI_stationary"] = int(near.sum())
                row[f"model_within_{n}px_FAI_stationary"] = int((flags["model"] & near).sum())
            rows.append(row)

    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment_id"]].append(r)

    def group_rows(g: str, keep=lambda r: True) -> list[dict]:
        return [r for s in groups[g] for r in by_seg[s] if keep(r)]

    def not_blind(r: dict) -> bool:
        return r["observability"] != "blind"

    distinct = {}
    for g, segs in groups.items():
        zf = np.zeros(n_pix, bool)
        for s in segs:
            zf |= zones[zone_ids.index(s)].ravel()
        distinct[g] = {
            m: {
                "ever_flagged": int((zf & (hits[m] > 0)).sum()),
                "stationary": int((zf & stationary[m]).sum()),
            }
            for m in METHODS
        }
    monthly = {}
    for g in groups:
        months = defaultdict(list)
        for r in group_rows(g, not_blind):
            months[r["date"][:7]].append(r)
        monthly[g] = {mo: _total(rs) for mo, rs in sorted(months.items())}
    summary = {
        "island": island.key,
        "part": part.key,
        "mangrove_mask": not args.no_mangrove_mask,
        "passes": len(meta),
        "rules": {"model": high, **cuts},
        "model_uncertain_band": [low, high],
        "groups": groups,
        "group_totals": {g: _total(group_rows(g)) for g in groups},
        "segment_totals": {s: _total(by_seg[s]) for s in zone_ids},
        "distinct_pixels": distinct,
        "monthly_non_blind": monthly,
        "block_agreement": {
            g: {
                f"{10 * k_px} m": {
                    k: (
                        acc[f"model_and_{k}"] / acc[f"model_or_{k}"]
                        if acc[f"model_or_{k}"]
                        else None
                    )
                    for k in INDICES
                }
                for k_px, acc in per_k.items()
            }
            for g, per_k in blocks.items()
        },
    }
    prefix = part.prefix(island.key) + ("_nomask" if args.no_mangrove_mask else "")
    out_csv = args.docs_dir / f"{prefix}_index_vs_model.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("wrote %s and %s", out_csv, out_csv.with_suffix(".json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--island", default="bonaire")
    parser.add_argument("--part", default=None)
    parser.add_argument("--focus", default=None, help="A segment to report on its own.")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--no-mangrove-mask", action="store_true")
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument(
        "--calibrator", type=Path, default=Path("models/sargassum_calibration.json")
    )
    parser.add_argument("--calibration-json", type=Path, default=Path("docs/calibration.json"))
    parser.add_argument("--band-ablation", type=Path, default=Path("docs/band_ablation.json"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    island = load_island(args.island)
    part = island.part(args.part)
    name = part.prefix(island.key) + ("-nomask" if args.no_mangrove_mask else "")
    cache = args.cache or CACHE_ROOT / name
    if args.analyse_only:
        meta = json.loads((cache / "passes.json").read_text(encoding="utf-8"))
    else:
        meta = extract(args, island, part, cache)
    analyse(args, island, part, cache, meta)


if __name__ == "__main__":
    main()
