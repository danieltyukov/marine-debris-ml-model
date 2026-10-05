"""Sentinel-1 radar over Lac Bay: does it see anything floating while Sentinel-2 cannot?

    python scripts/eval_sentinel1_lac.py
    python scripts/eval_sentinel1_lac.py --no-wind      # skip the ERA5 wind lookup

Lists every Sentinel-1 RTC acquisition over a 7.6 by 5.25 km window round Lac Bay in the
season (Microsoft Planetary Computer ``sentinel-1-rtc``, no account), reads VV and VH
gamma0 on the same 10 m UTM grid as the Bonaire season run, and searches each pass for
radar-bright patches in the water of the bay, with the open sea more than 1.5 km offshore
as a control where nothing floating is expected.

The detector follows Biermann et al. (2024): VV + 3 VH in linear power, smoothed 3 by 3
over water pixels only, compared with a 410 m local mean of the same, kept where at least
3 dB brighter in a patch of at least 4 connected pixels (400 m2). Each patch is checked
against every pass: seen on half of them or more it is stationary (shore, structure,
mooring), on two or fewer it is transient, the candidate for floating material.

"The optical gap" is the longest stretch with no fully clear Sentinel-2 look at Lac Bay,
read from ``docs/bonaire_season.csv``. Radar is bright over canopy and land on every
pass, so both are left out of the search: land is the island polygon plus 15 m, and
canopy is the mapped mangrove plus 20 m together with every pixel that looked like leaf
canopy on at least half of its clear Sentinel-2 looks in the season run
(``docs/bonaire_persistence.npz``, whose grid this window sits on). OpenStreetMap's
mangrove outlines alone miss part of the canopy, and radar finds it. The lagoon interior
is the bay inside the line from Sorobon to Cai, more than 30 m from land and canopy and
60 m from that line. Writes ``docs/sentinel1_lac.json``; ``docs/sentinel1_lac.md``
explains the results and what this reduced version leaves out.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np

from mdebris.coastal import load_segments, surf_zone
from mdebris.coastal.islands import load_island
from mdebris.coastal.masks import load_polygons, polygon_mask
from mdebris.coastal.season import summarize_history

log = logging.getLogger("sentinel1")

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
CRS = "EPSG:32619"
RES = 10.0
# Lac Bay's zone plus one kilometre west, north and south, and 3.2 km of open sea east.
X0, X1 = 581_400.0, 589_000.0
Y0, Y1 = 1_335_500.0, 1_340_750.0
WIDTH, HEIGHT = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
CONTRAST_DB = 3.0
MIN_BLOB_PX = 4
BACKGROUND_PX = 41
STATIONARY_SHARE = 0.5
TRANSIENT_MAX = 2
THRESHOLDS = (3.0, 4.0, 5.0)


def _transform():
    from affine import Affine

    return Affine(RES, 0.0, X0, 0.0, -RES, Y1)


def _to_db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(x)


def _distance_to(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask) * RES


def _rasterize(geoms, *, all_touched: bool = False) -> np.ndarray:
    from rasterio.features import geometry_mask

    return geometry_mask(
        list(geoms),
        out_shape=(HEIGHT, WIDTH),
        transform=_transform(),
        invert=True,
        all_touched=all_touched,
    )


def optical_gap(season_csv: Path, segment: str) -> tuple[str, str]:
    with season_csv.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["segment_id"] == segment]
    (season,) = summarize_history(rows)
    if season.clear_gap_from is None:
        raise SystemExit(f"{segment} has no two fully clear passes in {season_csv}")
    return season.clear_gap_from, season.clear_gap_to


def season_canopy(persistence: Path, *, share: float = 0.5, min_looks: int = 5) -> np.ndarray:
    """Pixels vegetated on at least ``share`` of their clear looks, cut to this window."""
    from mdebris.coastal.runner import Persistence

    grid = Persistence.load(persistence)
    if grid.vegetated_total is None:
        raise SystemExit(f"{persistence} has no vegetation counts; rerun the 2.1 season")
    t = grid.transform
    col0, row0 = (X0 - t.c) / RES, (t.f - Y1) / RES
    if not (float(col0).is_integer() and float(row0).is_integer()):
        raise SystemExit(f"{persistence} is not on this window's 10 m grid")
    rows = slice(int(row0), int(row0) + HEIGHT)
    cols = slice(int(col0), int(col0) + WIDTH)
    seen = grid.usable_total[rows, cols].astype(np.int32)
    leafy = grid.vegetated_total[rows, cols].astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (seen >= min_looks) & (leafy / np.maximum(seen, 1) >= share)


def masks(island, persistence: Path) -> dict[str, np.ndarray]:
    from pyproj import Transformer
    from shapely.geometry import LineString, Polygon
    from shapely.ops import transform as shapely_transform

    land_polys = load_polygons(island.island_path)
    land0 = polygon_mask(land_polys, _transform(), CRS, (HEIGHT, WIDTH), all_touched=False)
    land15 = polygon_mask(
        land_polys, _transform(), CRS, (HEIGHT, WIDTH), buffer_m=island.land_buffer_m
    )
    mangrove = polygon_mask(
        load_polygons(island.mangroves_path),
        _transform(),
        CRS,
        (HEIGHT, WIDTH),
        buffer_m=island.mangrove_buffer_m,
    )
    fixed = land15 | mangrove | season_canopy(persistence)
    fwd = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    lac = next(s for s in load_segments(island.segments_path) if s.segment_id == "lac_bay")
    line = shapely_transform(fwd, lac.geometry)
    inside = _rasterize([Polygon([*line.coords, line.coords[0]])])
    zone = _rasterize([shapely_transform(fwd, surf_zone(lac, 500.0))])
    mouth = LineString([line.coords[0], line.coords[-1]])
    d_mouth = _distance_to(_rasterize([mouth.buffer(1.0)], all_touched=True))
    d_fixed = _distance_to(fixed)
    cols = np.arange(WIDTH)[None, :].repeat(HEIGHT, 0)
    water = (zone | inside) & ~fixed
    return {
        "canopy": (mangrove | season_canopy(persistence)) & (zone | inside) & ~land15,
        "bay_water": water,
        "lagoon": inside & water & (d_fixed > 30) & (d_mouth > 60),
        "seaward": zone & ~inside & ~fixed & (d_mouth > 100),
        "open_sea": (_distance_to(land0) > 1500) & ~land0 & (cols > 450),
    }


def _masked_mean(img: np.ndarray, domain: np.ndarray, size: int) -> np.ndarray:
    from scipy import ndimage

    good = domain & np.isfinite(img)
    num = ndimage.uniform_filter(np.where(good, img, 0.0), size)
    den = ndimage.uniform_filter(good.astype(np.float64), size)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def detect(
    index: np.ndarray, domain: np.ndarray, threshold_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bright patches in ``domain``: a 3x3 mean against its 410 m local mean."""
    from scipy import ndimage

    smooth = _masked_mean(index, domain, 3)
    contrast = _to_db(smooth / _masked_mean(smooth, domain, BACKGROUND_PX))
    hit = domain & (contrast >= threshold_db)
    labels, n = ndimage.label(hit)
    if n:
        sizes = ndimage.sum(hit, labels, index=np.arange(1, n + 1))
        hit &= np.isin(labels, np.flatnonzero(sizes >= MIN_BLOB_PX) + 1)
    return hit, contrast


def passes(start: str, end: str) -> list:
    import planetary_computer
    import pystac_client
    from pyproj import Transformer

    inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
    west, south = inv(X0, Y0)
    east, north = inv(X1, Y1)
    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    items = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=[west, south, east, north],
        datetime=f"{start}/{end}",
        limit=200,
    ).items()
    one_per_acquisition: dict[str, object] = {}
    for item in sorted(items, key=lambda i: (i.datetime, -len(i.id))):
        one_per_acquisition.setdefault(item.datetime.strftime("%Y-%m-%dT%H:%M"), item)
    return list(one_per_acquisition.values())


def read_rtc(item) -> dict[str, np.ndarray]:
    import rasterio
    from rasterio.windows import from_bounds

    out = {}
    for pol in ("vv", "vh"):
        with rasterio.open(item.assets[pol].href) as src:
            if str(src.crs) != CRS:
                raise ValueError(f"{item.id} is in {src.crs}, expected {CRS}")
            win = from_bounds(X0, Y0, X1, Y1, transform=src.transform)
            win = win.round_offsets().round_lengths()
            arr = src.read(1, window=win, boundless=True, fill_value=src.nodata).astype(np.float64)
            arr[(arr == src.nodata) | (arr <= 0)] = np.nan
        if arr.shape != (HEIGHT, WIDTH):
            raise ValueError(f"{item.id}: window read {arr.shape}")
        out[pol] = arr
    return out


def era5_wind(start: str, end: str) -> dict[str, float]:
    import requests

    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 12.10,
            "longitude": -68.20,
            "start_date": start,
            "end_date": end,
            "hourly": "wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "models": "era5",
        },
        timeout=120,
    )
    r.raise_for_status()
    hourly = r.json()["hourly"]
    return dict(zip(hourly["time"], hourly["wind_speed_10m"], strict=True))


def main() -> None:
    from scipy import ndimage

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--season-csv", type=Path, default=Path("docs/bonaire_season.csv"))
    parser.add_argument("--persistence", type=Path, default=Path("docs/bonaire_persistence.npz"))
    parser.add_argument("--out", type=Path, default=Path("docs/sentinel1_lac.json"))
    parser.add_argument("--no-wind", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    island = load_island("bonaire")
    gap_from, gap_to = optical_gap(args.season_csv, "lac_bay")
    m = masks(island, args.persistence)
    items = passes(args.start, args.end)
    log.info("%d Sentinel-1 RTC acquisitions; optical gap %s to %s", len(items), gap_from, gap_to)
    wind = {}
    if not args.no_wind:
        try:
            wind = era5_wind(args.start, args.end)
        except Exception as exc:  # the radar results do not depend on it
            log.warning("no ERA5 wind: %s", exc)

    meta, index = [], []
    for item in items:
        arrays = read_rtc(item)
        when = item.datetime
        date = when.strftime("%Y-%m-%d")
        vv_db = _to_db(arrays["vv"])
        meta.append(
            {
                "date": date,
                "time_local": (when - timedelta(hours=4)).strftime("%H:%M"),
                "orbit": item.properties.get("sat:orbit_state"),
                "relative_orbit": item.properties.get("sat:relative_orbit"),
                "in_optical_gap": gap_from < date < gap_to,
                "era5_wind_ms": wind.get(when.strftime("%Y-%m-%dT%H:00")),
                "vv_open_sea_db": round(float(np.nanmedian(vv_db[m["open_sea"]])), 1),
                "vv_lagoon_db": round(float(np.nanmedian(vv_db[m["lagoon"]])), 1),
            }
        )
        index.append(arrays["vv"] + 3.0 * arrays["vh"])
        log.info("%s %s %s", date, meta[-1]["time_local"], meta[-1]["orbit"])
    index = np.stack(index)
    domain = m["bay_water"] | m["open_sea"]
    regions = {
        "lagoon interior": m["lagoon"],
        "seaward of the mouth": m["seaward"],
        "open sea": m["open_sea"],
    }
    area_km2 = {r: round(float(mask.sum() * RES * RES / 1e6), 3) for r, mask in regions.items()}
    canopy_km2 = round(float((m["canopy"]).sum() * RES * RES / 1e6), 3)
    ascending = [i for i, p in enumerate(meta) if p["orbit"] == "ascending"]

    results: dict[str, object] = {}
    largest = None
    for t in THRESHOLDS:
        hits = np.stack([detect(index[i], domain, t)[0] for i in range(len(meta))])
        recurrence = np.stack([ndimage.binary_dilation(h) for h in hits]).sum(axis=0)
        any_kind = {f"{r}|{g}": 0 for r in regions for g in ("gap", "outside")}
        transient = {(r, g): [] for r in regions for g in ("gap", "outside")}
        for i, p in enumerate(meta):
            period = "gap" if p["in_optical_gap"] else "outside"
            labels, n = ndimage.label(hits[i])
            counts = dict.fromkeys(regions, 0)
            for b in range(1, n + 1):
                blob = labels == b
                rr, cc = np.nonzero(blob)
                r0, c0 = round(float(rr.mean())), round(float(cc.mean()))
                seen_on = int(recurrence[blob].max())
                for r, mask in regions.items():
                    if mask[r0, c0]:
                        any_kind[f"{r}|{period}"] += 1
                        if seen_on <= TRANSIENT_MAX:
                            counts[r] += 1
                # The candidate for floating material: the largest in-gap patch in the
                # bay's water that is not there on half the passes or more.
                if (
                    t == CONTRAST_DB
                    and p["in_optical_gap"]
                    and m["bay_water"][r0, c0]
                    and seen_on / len(meta) < STATIONARY_SHARE
                    and (largest is None or blob.sum() > largest["pixels"])
                ):
                    from pyproj import Transformer

                    inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
                    lon, lat = inv(X0 + (c0 + 0.5) * RES, Y1 - (r0 + 0.5) * RES)
                    largest = {
                        "date": p["date"],
                        "pixels": int(blob.sum()),
                        "area_m2": int(blob.sum() * RES * RES),
                        "max_contrast_db": round(
                            float(np.nanmax(detect(index[i], domain, t)[1][blob])), 1
                        ),
                        "passes_detected": seen_on,
                        "lon": round(lon, 4),
                        "lat": round(lat, 4),
                    }
            if i in ascending:
                for r in regions:
                    transient[(r, period)].append(counts[r] / max(area_km2[r], 1e-9))
        results[f"{t:.0f} dB"] = {
            "patches_any_kind": any_kind,
            "transient_per_km2_per_ascending_pass": {
                f"{r}|{g}": round(float(np.mean(v)), 2) if v else None
                for (r, g), v in transient.items()
            },
        }
    summary = {
        "window_utm19n": [X0, Y0, X1, Y1],
        "optical_gap": [gap_from, gap_to],
        "passes": meta,
        "passes_in_gap": sum(p["in_optical_gap"] for p in meta),
        "region_km2": area_km2,
        "canopy_in_bay_km2": canopy_km2,
        "detector": {
            "index": "VV + 3 VH, linear gamma0",
            "smoothing_px": 3,
            "background_px": BACKGROUND_PX,
            "min_patch_px": MIN_BLOB_PX,
            "transient_max_passes": TRANSIENT_MAX,
        },
        "by_threshold": results,
        "largest_non_stationary_in_gap_bay_patch_3db": largest,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
