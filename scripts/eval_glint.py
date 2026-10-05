"""How much sun glint each island's orbit sees on its windward coast, pass by pass.

    python scripts/eval_glint.py                 # every pass of every island run, writes docs/glint.csv
    python scripts/eval_glint.py --pair          # Sint Maarten, 4 June 2025, the same pixels from both orbits

In the morning the sun is in the east. An island on the east side of its orbit's swath
is seen from the west, looking towards the sun's mirror image on the sea, and in summer,
with the sun near the zenith, that is sun glint: the water brightens in every band, the
scene classification layer calls some of it cloud, and the classifier sees floating
vegetation against a bright background. Two measures per pass, both independent of the
classifier:

- The glint angle at a point on the windward coast: the angle between the direction to
  the sensor and the mirror direction of the sun, from the 5 km sun and view angle grids
  in the granule metadata (band B04). Under about 20 degrees a trade-wind sea glints.
- The median B11 (1.6 um) reflectance over a 2 km box of open water a few kilometres
  off the windward coast, on scene-classification water pixels only, counted when the
  box is over 90% water. Clear water reflects almost nothing at 1.6 um, so B11 there is
  mostly glint plus a small atmospheric residual. It is a lower bound: glint bright
  enough to be called cloud is not in the box's water pixels.

The passes are the ones in each part's ``docs/<prefix>_season.csv``; the point and the
box are ``glint`` in the island's ``island.toml``. Writes ``docs/glint.csv`` and prints a
summary table. ``docs/islands.md`` reports the results.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mdebris.coastal.islands import list_islands, load_island


def _grid(block: str) -> np.ndarray:
    return np.array(
        [[float(v) for v in r.split()] for r in re.findall(r"<VALUES>(.*?)</VALUES>", block)]
    )


def _section(text: str, tag: str) -> str:
    return text[text.find(f"<{tag}>") : text.find(f"</{tag}>")]


def glint_angle(
    sun_zenith: float, sun_azimuth: float, view_zenith: float, view_azimuth: float
) -> float:
    """Degrees between the sensor direction and the sun's mirror direction.

    Both azimuths point from the ground: towards the sun and towards the sensor. Zero
    when the sensor sits exactly in the sun's specular reflection.
    """
    r = math.radians
    cos_g = math.cos(r(sun_zenith)) * math.cos(r(view_zenith)) - math.sin(r(sun_zenith)) * math.sin(
        r(view_zenith)
    ) * math.cos(r(sun_azimuth - view_azimuth))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_g))))


def angles(scene_id: str, lon: float, lat: float) -> tuple[float, float, float, float, float]:
    """Sun zenith and azimuth, view zenith and azimuth (B04) and glint angle at a point."""
    import planetary_computer
    from pyproj import Transformer
    from pystac_client import Client

    from mdebris.config import settings

    item = (
        Client.open(settings.stac_endpoint)
        .get_collection(settings.stac_collection)
        .get_item(scene_id)
    )
    asset = item.assets.get("granule-metadata") or item.assets["granule_metadata"]
    href = planetary_computer.sign(asset.href)
    text = urllib.request.urlopen(href, timeout=120).read().decode()
    epsg = re.search(r"<HORIZONTAL_CS_CODE>EPSG:(\d+)</HORIZONTAL_CS_CODE>", text).group(1)
    geo = re.search(
        r'<Geoposition resolution="10">\s*<ULX>([\d.]+)</ULX>\s*<ULY>([\d.]+)</ULY>', text
    )
    ulx, uly = float(geo.group(1)), float(geo.group(2))
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    j, i = round((x - ulx) / 5000), round((uly - y) / 5000)
    sun = _section(text, "Sun_Angles_Grid")
    sz, sa = _grid(_section(sun, "Zenith"))[i, j], _grid(_section(sun, "Azimuth"))[i, j]
    vz = va = float("nan")
    for m in re.finditer(
        r'<Viewing_Incidence_Angles_Grids bandId="3" detectorId="\d+">(.*?)</Viewing_Incidence_Angles_Grids>',
        text,
        re.S,
    ):
        z = _grid(_section(m.group(1), "Zenith"))[i, j]
        if not np.isnan(z):
            vz, va = z, _grid(_section(m.group(1), "Azimuth"))[i, j]
            break
    return float(sz), float(sa), float(vz), float(va), glint_angle(sz, sa, vz, va)


def offshore_b11(scene_id: str, box) -> tuple[float, float | None]:
    """Share of scene-classification water in the box, and the median B11 over it."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    from mdebris.data import get_scene_assets
    from mdebris.data.scaling import BOA_OFFSET, to_reflectance

    hrefs = get_scene_assets(scene_id, ["B11", "SCL"])
    with rasterio.open(hrefs["B11"]) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        x0, y0 = t(box[0], box[1])
        x1, y1 = t(box[2], box[3])
        window = (
            from_bounds(x0, y0, x1, y1, transform=src.transform).round_lengths().round_offsets()
        )
        b11 = to_reflectance(src.read(1, window=window), offset=BOA_OFFSET)
    with rasterio.open(hrefs["SCL"]) as src:
        water = src.read(1, window=window) == 6
    return float(water.mean()), (float(np.median(b11[water])) if water.sum() > 50 else None)


def _retry(fn, *args, attempts: int = 5):
    """Call ``fn``, waiting and trying again when the archive rate-limits or drops us."""
    import time

    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = 15 * 2**attempt
            print(f"  {type(exc).__name__}; trying again in {wait} s")
            time.sleep(wait)
    return None


def survey(docs: Path, out: Path, threads: int) -> list[dict]:
    rows: list[dict] = []
    for key in list_islands():
        island = load_island(key)
        for part in island.parts:
            prefix = part.prefix(island.key)
            season_csv = docs / f"{prefix}_season.csv"
            if part.glint_point is None or part.glint_box is None or not season_csv.exists():
                print(f"{prefix}: skipped (no glint point and box, or no season CSV)")
                continue
            orbit = ", ".join(part.orbits)
            # "Curaçao west, R082", but "Sint Maarten, R139" for a part named after its orbit.
            named = part.key and part.key.upper() not in part.orbits
            label = f"{island.name}{' ' + part.key if named else ''}, {orbit}"
            with season_csv.open(encoding="utf-8", newline="") as fh:
                scenes = sorted(
                    {(r["observed_on"][:10], r["scene_id"]) for r in csv.DictReader(fh)}
                )

            def one(s, point=part.glint_point, box=part.glint_box, label=label, prefix=prefix):
                sz, sa, vz, va, g = _retry(angles, s[1], *point)
                share, b11 = _retry(offshore_b11, s[1], box)
                return {
                    "run": label,
                    "prefix": prefix,
                    "date": s[0],
                    "scene_id": s[1],
                    "sun_zenith": round(sz, 2),
                    "sun_azimuth": round(sa, 1),
                    "view_zenith": round(vz, 2),
                    "view_azimuth": round(va, 1),
                    "glint_angle": round(g, 2),
                    "box_water_share": round(share, 3),
                    "offshore_b11": "" if b11 is None else round(b11, 4),
                }

            with ThreadPoolExecutor(threads) as pool:
                got = list(pool.map(one, scenes))
            rows += got
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return rows


def summary(rows: list[dict]) -> str:
    """One line per run: view zenith, glint angle and offshore B11 before and after April."""
    lines = [
        "| island, orbit | view zenith at the coast | glint angle, median (min) "
        "| passes under 20 degrees | offshore B11, Jan-Mar | offshore B11, Apr-Aug |",
        "|---|---|---|---|---|---|",
    ]
    runs: dict[str, list[dict]] = {}
    for r in rows:
        runs.setdefault(r["run"], []).append(r)
    for label, got in sorted(
        runs.items(), key=lambda kv: np.median([float(r["glint_angle"]) for r in kv[1]])
    ):
        g = [float(r["glint_angle"]) for r in got]
        clear = [r for r in got if float(r["box_water_share"]) > 0.9 and r["offshore_b11"] != ""]
        early = [float(r["offshore_b11"]) for r in clear if r["date"] < "2025-04-01"]
        late = [float(r["offshore_b11"]) for r in clear if r["date"] >= "2025-04-01"]
        lines.append(
            f"| {label} | {np.median([float(r['view_zenith']) for r in got]):.1f} | "
            f"{np.median(g):.1f} ({min(g):.1f}) | {sum(x < 20 for x in g)} of {len(g)} | "
            f"{np.median(early):.3f} | {np.median(late):.3f} |"
        )
    return "\n".join(lines)


def pair() -> None:
    """The pixels R139 flagged on 4 June 2025 at Sint Maarten, as R039 saw them minutes later."""
    import rasterio

    from mdebris.coastal import load_segments, surf_zone
    from mdebris.coastal.masks import load_polygons
    from mdebris.coastal.runner import _to_crs, load_operating_points, process_pass
    from mdebris.data import get_scene_assets
    from mdebris.data.scaling import BOA_OFFSET, to_reflectance
    from mdebris.eval.calibration import IsotonicCalibrator
    from mdebris.geo.raster import read_bands
    from mdebris.models.spectral import SpectralClassifier
    from mdebris.types import SceneRef

    island = load_island("sint_maarten")
    part = island.part("r139")
    segments = [s for s in load_segments(island.segments_path) if s.segment_id in part.segment_ids]
    zones = {s.segment_id: surf_zone(s, 500.0) for s in segments}
    clf = SpectralClassifier.load(Path("models/marida_spectral.joblib"))
    calibrator = IsotonicCalibrator.load(Path("models/sargassum_calibration.json"))
    points = load_operating_points(Path("docs/calibration.json"), calibrated=True)
    bands = ("B02", "B03", "B04", "B05", "B06", "B08", "B11")
    got = {}
    for orbit, sid in (
        ("R139", "S2B_MSIL2A_20250604T144729_R139_T20QMF_20250604T200647"),
        ("R039", "S2A_MSIL2A_20250604T145741_R039_T20QMF_20250604T213715"),
    ):
        result = process_pass(
            SceneRef(scene_id=sid),
            aoi=part.aoi,
            segments=segments,
            zones=zones,
            land=load_polygons(island.island_path),
            mangroves=load_polygons(island.mangroves_path),
            clf=clf,
            calibrator=calibrator,
            points=points,
            surf_zone_m=500.0,
            keep_arrays=True,
        )
        hrefs = get_scene_assets(sid, list(bands))
        with rasterio.open(hrefs["B04"]) as src:
            window = (
                rasterio.windows.from_bounds(*_to_crs(part.aoi, src.crs), transform=src.transform)
                .round_lengths()
                .round_offsets()
            )
        arrays = read_bands({b: hrefs[b] for b in bands}, window, reference="B04")
        got[orbit] = (result, {b: to_reflectance(a, offset=BOA_OFFSET) for b, a in arrays.items()})
    (r139, f139), (r039, f039) = got["R139"], got["R039"]
    if tuple(r139.transform)[:6] != tuple(r039.transform)[:6]:
        raise SystemExit("the two passes are not on the same grid")
    hits = r139.hit_mask
    print(f"R139 flagged {int(hits.sum())} pixels in the windward surf zones on 4 June 2025.")
    print(
        f"The same pixels on R039: flagged {int((hits & r039.hit_mask).sum())}, uncertain "
        f"{int((hits & r039.uncertain_mask).sum())}, scene-classification cloud "
        f"{int((hits & r039.cloud_mask).sum())}, usable {int((hits & r039.usable_mask).sum())}."
    )
    water = r139.usable_mask & r039.usable_mask & ~hits & ~r139.uncertain_mask
    for label, f in (("R139", f139), ("R039", f039)):
        print(
            f"  {label}, mean over those pixels: "
            + " ".join(f"{b} {np.nanmean(f[b][hits]):.3f}" for b in bands)
        )
    for label, f in (("R139", f139), ("R039", f039)):
        print(
            f"  {label}, median over the other usable surf-zone water: "
            + " ".join(f"{b} {np.nanmedian(f[b][water]):.3f}" for b in bands)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pair", action="store_true", help="Only the 4 June 2025 comparison.")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--out", type=Path, default=Path("docs/glint.csv"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.pair:
        pair()
        return
    print(summary(survey(args.docs_dir, args.out, args.threads)))


if __name__ == "__main__":
    main()
