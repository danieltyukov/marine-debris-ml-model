"""Small real Sentinel-2 chips bundled with the package, so nothing needs the network.

Tests, the README examples and the demo all need imagery that behaves like the real thing:
correct dtype and scaling, a real CRS and affine transform, real cloud and land pixels,
and real spectral relationships between bands. Synthetic arrays satisfy none of that, and
a test suite built on them passes while the pipeline is broken.

Each bundled chip is a 512x512 window at 10 m over a documented marine-litter hotspot,
carrying only the bands the pipeline actually reads, plus a JSON sidecar recording the
source scene, date, bounds and licence. Total on-disk size is a few megabytes.

Resolution
----------
Sentinel-2 mixes 10 m and 20 m bands. Chips are written on a single 10 m grid so that a
pixel index means the same thing in every band, with the 20 m bands (B06, B11, SCL)
upsampled on read. SCL uses nearest-neighbour because it is a categorical mask and
averaging class codes produces classes that do not exist; the reflectance bands use
bilinear.

Radiometry
----------
L2A reflectance is stored as integers, and converting them back is not just a division.
From ESA processing baseline 04.00 (January 2022) onward the products carry
``BOA_ADD_OFFSET = -1000``, so reflectance is ``dn * 0.0001 - 0.1``, not ``dn / 10000``.
Skipping the offset shifts every band by +0.1, which is more than an order of magnitude
larger than the Floating Debris Index threshold the cascade screens on, so it would not
fail loudly, it would just fill the output with false positives. The scale and offset for
each chip are recorded in its sidecar and applied by :func:`to_reflectance`.

Copernicus Sentinel data is free and open. Chips here are derived from Copernicus
Sentinel-2 L2A and redistributed under the Copernicus terms, which require attribution
and permit modification and redistribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mdebris.data.scaling import (
    REFLECTANCE_SCALE,
    reflectance_params_for_item,
    scale_bands,
    to_reflectance,
)
from mdebris.data.stac import BAND_GSD_M, Band, StacClient, canonical_band
from mdebris.types import GeoBBox, SceneRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

    import numpy as np

__all__ = [
    "HOTSPOTS",
    "SAMPLES_DIR",
    "SAMPLE_BANDS",
    "Hotspot",
    "SampleError",
    "fetch_sample_chips",
    "list_samples",
    "load_sample",
    "reflectance_from_meta",
    "sample_bbox",
    "sample_meta",
    "sample_reflectance",
    "sample_scene",
]

SAMPLES_DIR = Path(__file__).parent / "samples"

COPERNICUS_NOTICE = (
    "Contains modified Copernicus Sentinel data. Sentinel-2 L2A is free and open under "
    "the Copernicus Sentinel Data Terms and Conditions "
    "(https://sentinels.copernicus.eu/web/sentinel/terms-conditions), which permit "
    "redistribution and modification with attribution."
)

# The bands the detection pipeline reads, and why each one is here:
#   B02, B03, B04  true-colour composite fed to the open-vocabulary detector
#   B06            red-edge baseline term in the Floating Debris Index
#   B08            NIR, the band floating plastic brightens in, used by FDI and NDWI
#   B11            SWIR, separates water from everything else and anchors the FDI baseline
#   SCL            scene classification, masks cloud and land before anything else runs
SAMPLE_BANDS: tuple[Band, ...] = (
    Band.B02,
    Band.B03,
    Band.B04,
    Band.B06,
    Band.B08,
    Band.B11,
    Band.SCL,
)

CHIP_PIXELS = 512
CHIP_GSD_M = 10


class SampleError(RuntimeError):
    """A bundled sample is missing or could not be read."""


@dataclass(frozen=True, slots=True)
class Hotspot:
    """A place with published marine-litter observations, used to source a sample chip."""

    name: str
    lon: float
    lat: float
    description: str
    start: str
    end: str

    @property
    def bbox(self) -> GeoBBox:
        """A search box around the point, roughly 0.3 degrees on a side."""
        d = 0.15
        return GeoBBox(self.lon - d, self.lat - d, self.lon + d, self.lat + d)


# Two sites chosen because both have independent ground evidence of floating litter, so a
# detection here is checkable against something other than the model's own confidence.
HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot(
        name="accra",
        lon=-0.20,
        lat=5.55,
        description=(
            "The Accra coastline, Ghana, where the Odaw river and Korle lagoon discharge "
            "into the Gulf of Guinea. River outfalls into the sea are the main pathway "
            "by which land plastic becomes marine plastic, which is why this coast "
            "recurs in the Sentinel-2 marine-litter literature."
        ),
        start="2024-01-01",
        end="2024-06-30",
    ),
    Hotspot(
        name="limassol",
        lon=33.00,
        lat=34.60,
        description=(
            "Akrotiri Bay off Limassol, Cyprus. This is the coastal water used by the "
            "Cyprus University of Technology Plastic Litter Project, which floated "
            "plastic targets of known size and measured how they appear to Sentinel-2."
        ),
        start="2024-04-01",
        end="2024-09-30",
    ),
)


def list_samples() -> list[str]:
    """Names of the chips bundled with the package, sorted."""
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.tif"))


def load_sample(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a bundled chip and its provenance.

    Args:
        name: A name from :func:`list_samples`, e.g. ``"accra"``.

    Returns:
        ``(bands, meta)``. ``bands`` maps canonical band names to 2-D arrays on a shared
        512x512 grid: reflectance bands are uint16 in Sentinel-2 L2A digital numbers
        (divide by ``meta["scale"]`` for reflectance) and ``SCL`` is uint8 class codes.
        ``meta`` carries ``scene_id``, ``datetime``, ``crs``, ``transform`` (a
        6-tuple in ``affine`` order), ``bounds``, ``bands``, ``gsd_m`` and ``license``.

    Raises:
        SampleError: if no chip of that name is bundled.

    Example:
        >>> bands, meta = load_sample("accra")
        >>> bands["B08"].shape
        (512, 512)
        >>> meta["gsd_m"]
        10
    """
    tif = SAMPLES_DIR / f"{name}.tif"
    if not tif.is_file():
        available = ", ".join(list_samples()) or "none bundled"
        raise SampleError(f"no sample named {name!r}; available: {available}")

    import rasterio

    meta = sample_meta(name)
    bands: dict[str, np.ndarray] = {}
    with rasterio.open(tif) as src:
        # Band order lives in the GeoTIFF's own band descriptions, so the file stays
        # self-describing even if the sidecar is lost.
        names = [d or f"band{i}" for i, d in enumerate(src.descriptions, start=1)]
        stack = src.read()  # one pass over the file rather than one per band
        for index, band_name in enumerate(names):
            data = stack[index]
            # SCL holds class codes 0..11 and is stored as uint16 only because a GeoTIFF
            # has one dtype for all bands. Hand it back in its real dtype.
            bands[band_name] = data.astype("uint8") if band_name == Band.SCL else data
        meta.setdefault("crs", str(src.crs))
        meta.setdefault("transform", tuple(src.transform)[:6])
        meta.setdefault("bounds", tuple(src.bounds))
        meta["bands"] = names
    return bands, meta


def sample_meta(name: str) -> dict[str, Any]:
    """The JSON sidecar for a bundled chip.

    Raises:
        SampleError: if the sidecar is missing.
    """
    sidecar = SAMPLES_DIR / f"{name}.json"
    if not sidecar.is_file():
        raise SampleError(f"missing provenance sidecar for sample {name!r} at {sidecar}")
    return json.loads(sidecar.read_text())


def sample_scene(name: str) -> SceneRef:
    """The source scene a bundled chip was cut from, so detections stay traceable."""
    meta = sample_meta(name)
    return SceneRef(
        scene_id=meta["scene_id"],
        collection=meta.get("collection", "sentinel-2-l2a"),
        datetime=meta.get("datetime"),
        platform=meta.get("platform"),
        cloud_cover=meta.get("cloud_cover"),
        source=meta.get("source", "planetary-computer"),
    )


def sample_bbox(name: str) -> GeoBBox:
    """Geographic bounds of a bundled chip in EPSG:4326."""
    west, south, east, north = sample_meta(name)["bbox_wgs84"]
    return GeoBBox(west, south, east, north)


def reflectance_from_meta(array: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Convert one band to surface reflectance using a chip's recorded scaling.

    A thin convenience over :func:`mdebris.data.scaling.to_reflectance` that reads the
    scale and offset out of a sidecar, so callers working with bundled chips never have to
    know which processing baseline the source scene used.

    Args:
        array: A band as returned by :func:`load_sample`. Do not pass ``SCL``, which is a
            class code rather than a measurement; use :func:`sample_reflectance` to
            convert a whole stack safely.
        meta: The metadata dict from :func:`load_sample`.

    Returns:
        A float32 array of surface reflectance, nominally in ``[0, 1]``.

    Example:
        >>> bands, meta = load_sample("accra")
        >>> nir = reflectance_from_meta(bands["B08"], meta)
        >>> bool(nir[bands["SCL"] == 6].mean() < 0.1)  # water is dark in the near infrared
        True
    """
    return to_reflectance(array, **_scaling_from_meta(meta))


def sample_reflectance(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a chip with its reflectance bands already converted to physical units.

    The usual entry point for anything doing spectral work on a sample: it applies the
    right scale and offset and leaves ``SCL`` alone as a class code.

    Example:
        >>> bands, meta = sample_reflectance("accra")
        >>> bool(bands["B08"][bands["SCL"] == 6].mean() < 0.1)
        True
    """
    bands, meta = load_sample(name)
    return scale_bands(bands, **_scaling_from_meta(meta)), meta


def _scaling_from_meta(meta: dict[str, Any]) -> dict[str, float]:
    """Pull scale and offset out of a sidecar.

    Older sidecars, and any hand-written metadata, may not record them. Defaulting the
    offset to 0.0 rather than -0.1 is deliberate: applying an offset that was never
    recorded is a silent 0.1 error, whereas omitting one is at worst the pre-2022 encoding.
    """
    return {
        "scale": float(meta.get("reflectance_scale", REFLECTANCE_SCALE)),
        "offset": float(meta.get("reflectance_offset", 0.0)),
    }


# -- fetching --------------------------------------------------------------------


def fetch_sample_chips(
    hotspots: Iterable[Hotspot] | None = None,
    *,
    dest: Path | str | None = None,
    bands: Sequence[str | Band] = SAMPLE_BANDS,
    size: int = CHIP_PIXELS,
    max_cloud: float = 25.0,
    min_water: float = 0.10,
    client: StacClient | None = None,
) -> list[Path]:
    """Cut fresh sample chips from Planetary Computer and write them into the package.

    This is the script that produced the bundled chips. It needs the network; the chips
    it writes do not, which is the entire point. Re-run it to refresh the samples or to
    add a site.

    Candidate scenes are tried least-cloudy first and each is checked before it is
    accepted. Sentinel-2 tiles overlap, so a scene can intersect a search box while the
    point of interest sits outside its footprint or hard against its edge; taking the
    first search hit produces a chip that is mostly nodata. A scene is used only when the
    whole window fits inside it, the pixels are valid, and enough of the chip is water.

    Args:
        hotspots: Sites to sample. Defaults to :data:`HOTSPOTS`.
        dest: Output directory. Defaults to the packaged ``samples/`` directory.
        bands: Bands to include, canonical or provider names.
        size: Chip width and height in 10 m pixels.
        max_cloud: Maximum scene cloud cover percent to accept.
        min_water: Minimum fraction of the chip that must be classified as water by SCL.
            These are marine samples; a chip of pure land is not useful here.
        client: A pre-built STAC client, mostly for tests.

    Returns:
        Paths of the GeoTIFFs written, one per hotspot.

    Raises:
        SampleError: if no scene at a site passes the checks.
    """
    out_dir = Path(dest) if dest is not None else SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stac = client or StacClient()
    wanted = [canonical_band(b) for b in bands]

    written: list[Path] = []
    for spot in hotspots or HOTSPOTS:
        scenes = stac.search(spot.bbox, spot.start, spot.end, max_cloud=max_cloud, limit=20)
        if not scenes:
            raise SampleError(
                f"no scene under {max_cloud}% cloud over {spot.name} "
                f"between {spot.start} and {spot.end}"
            )
        written.append(_cut_chip(out_dir, spot, scenes, stac, wanted, size, min_water))
    return written


# GDAL settings for reading COGs out of object storage. Without the first one, every open
# costs a directory listing that takes seconds and returns nothing useful.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
}

# SCL class code for water, from the Sentinel-2 L2A scene classification legend.
SCL_WATER = 6



def _cut_chip(
    out_dir: Path,
    spot: Hotspot,
    scenes: Sequence[SceneRef],
    stac: StacClient,
    bands: Sequence[Band],
    size: int,
    min_water: float,
) -> Path:
    """Try candidate scenes in order and write the first that yields a usable chip."""
    rejected: list[str] = []
    for scene in scenes:
        hrefs = stac.asset_hrefs(scene, bands)
        try:
            stack, grid_transform, crs, bounds = _read_stack(spot, hrefs, bands, size)
        except _WindowOutsideScene as exc:
            rejected.append(f"  {scene.scene_id}: {exc}")
            continue
        reason = _reject_reason(stack, bands, min_water)
        if reason:
            rejected.append(f"  {scene.scene_id}: {reason}")
            continue
        item = stac.get_item(scene.scene_id, collection=scene.collection)
        scaling = reflectance_params_for_item(item)
        return _write_chip(
            out_dir, spot, scene, bands, stack, grid_transform, crs, bounds, size, scaling
        )
    raise SampleError(
        f"no usable chip for {spot.name} from {len(scenes)} candidate scenes:\n"
        + "\n".join(rejected)
    )


class _WindowOutsideScene(Exception):
    """The requested window does not fit inside the candidate scene."""


def _read_stack(
    spot: Hotspot,
    hrefs: dict[str, str],
    bands: Sequence[Band],
    size: int,
) -> tuple[list[Any], Any, Any, tuple[float, float, float, float]]:
    """Read every band over one window onto a shared 10 m grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    # The 10 m bands define the reference grid; everything else is resampled onto it.
    anchor = next((b for b in bands if BAND_GSD_M[b] == CHIP_GSD_M), bands[0])

    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(hrefs[str(anchor)]) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [spot.lon], [spot.lat])
            row, col = src.index(xs[0], ys[0])
            col0, row0 = col - size // 2, row - size // 2
            if not (col0 >= 0 and col0 + size <= src.width):
                raise _WindowOutsideScene("point too close to the scene's east or west edge")
            if not (row0 >= 0 and row0 + size <= src.height):
                raise _WindowOutsideScene("point too close to the scene's north or south edge")
            window = Window(col0, row0, size, size)
            grid_transform = src.window_transform(window)
            crs = src.crs
            bounds = rasterio.windows.bounds(window, src.transform)

        stack = []
        for band in bands:
            factor = BAND_GSD_M[band] // CHIP_GSD_M
            # A 20 m band covers the same ground in half as many pixels, so its window is
            # the 10 m window scaled down, then read back out at the full 10 m shape.
            band_window = Window(
                window.col_off / factor,
                window.row_off / factor,
                window.width / factor,
                window.height / factor,
            )
            resampling = Resampling.nearest if band is Band.SCL else Resampling.bilinear
            with rasterio.open(hrefs[str(band)]) as src:
                data = src.read(
                    [1], window=band_window, out_shape=(1, size, size), resampling=resampling
                )[0]
            stack.append(data)
    return stack, grid_transform, crs, bounds


def _reject_reason(stack: Sequence[Any], bands: Sequence[Band], min_water: float) -> str | None:
    """Why this chip is unusable, or ``None`` if it passes."""
    by_band = dict(zip(bands, stack, strict=True))
    nodata = float((by_band[bands[0]] == 0).mean())
    if nodata > 0.02:
        return f"{nodata:.0%} of pixels are nodata"
    scl = by_band.get(Band.SCL)
    if scl is None:
        return None
    water = float((scl == SCL_WATER).mean())
    if water < min_water:
        return f"only {water:.0%} water, below the {min_water:.0%} minimum"
    # SCL cloud classes: 8 medium probability, 9 high probability, 10 thin cirrus.
    cloud = float(((scl >= 8) & (scl <= 10)).mean())
    if cloud > 0.30:
        return f"{cloud:.0%} of the chip is cloud"
    return None


def _write_chip(
    out_dir: Path,
    spot: Hotspot,
    scene: SceneRef,
    bands: Sequence[Band],
    stack: Sequence[Any],
    grid_transform: Any,
    crs: Any,
    bounds: tuple[float, float, float, float],
    size: int,
    scaling: tuple[float, float],
) -> Path:
    """Write one validated stack out as a compressed GeoTIFF plus a JSON sidecar."""
    import rasterio

    path = out_dir / f"{spot.name}.tif"
    # SCL is categorical uint8 while reflectance is uint16. One dtype has to win for a
    # single-file chip; uint16 is lossless for both, and load_sample narrows SCL back.
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=len(bands),
        dtype="uint16",
        crs=crs,
        transform=grid_transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for index, (band, data) in enumerate(zip(bands, stack, strict=True), start=1):
            dst.write(data.astype("uint16"), index)
            dst.set_band_description(index, str(band))
        dst.update_tags(scene_id=scene.scene_id, datetime=scene.datetime or "")

    _write_sidecar(out_dir, spot, scene, bands, bounds, crs, grid_transform, size, scaling)
    return path


def _write_sidecar(
    out_dir: Path,
    spot: Hotspot,
    scene: SceneRef,
    bands: Sequence[Band],
    bounds: tuple[float, float, float, float],
    crs: Any,
    transform: Any,
    size: int,
    scaling: tuple[float, float],
) -> None:
    from rasterio.warp import transform_bounds

    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
    payload = {
        "name": spot.name,
        "description": spot.description,
        "scene_id": scene.scene_id,
        "collection": scene.collection,
        "datetime": scene.datetime,
        "platform": scene.platform,
        "cloud_cover": scene.cloud_cover,
        "source": scene.source,
        "bands": [str(b) for b in bands],
        "width": size,
        "height": size,
        "gsd_m": CHIP_GSD_M,
        "crs": str(crs),
        "transform": list(transform)[:6],
        "bounds": list(bounds),
        "bbox_wgs84": [west, south, east, north],
        "centre_lonlat": [spot.lon, spot.lat],
        # reflectance = dn * reflectance_scale + reflectance_offset. The offset is -0.1
        # for anything processed under baseline 04.00 or later; see to_reflectance. SCL is
        # a class code and is not scaled at all.
        "reflectance_scale": scaling[0],
        "reflectance_offset": scaling[1],
        "processing_baseline_offset_applied": scaling[1] != 0.0,
        "license": COPERNICUS_NOTICE,
        "attribution": "Copernicus Sentinel data, processed by ESA, hosted by Microsoft "
        "Planetary Computer",
    }
    (out_dir / f"{spot.name}.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":  # pragma: no cover - operator entry point
    for written in fetch_sample_chips():
        print(f"wrote {written} ({written.stat().st_size / 1e6:.1f} MB)")
