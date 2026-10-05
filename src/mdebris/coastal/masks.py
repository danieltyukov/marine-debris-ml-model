"""Rasterised masks for a season run: land, mangroves, surf zones and vegetation.

Why mangroves are masked
------------------------
The classifier is trained on MARIDA, which has no land class and no mangrove class, and
whose only bright vegetation is sargassum. Anything left on the "water side" of the land
mask is therefore scored as if it were sea surface. On Bonaire the OpenStreetMap
coastline runs round the *landward* edge of Lac Bay's mangrove forest, so the 2.0
season run treated the canopy as water and classified it: 93% of the 1,167 Lac Bay
pixels it ever flagged, and all 25 of the ones it called stationary, sit on mangrove
canopy or within 20 m of it. A pixel on the canopy edge,
part leaf and part water, crosses the sargassum edge on clean passes and drops back on
hazy ones, so the flags hop around while the forest does not move. ``docs/lac_bay_mangroves.md``
has the evidence.

So the runner now removes OpenStreetMap's mapped mangrove and other vegetated wetland,
buffered 20 m, from the water side before anything is classified or counted. Seagrass
(``wetland=seagrass``) is under water, where sargassum floats over it, and is kept;
untagged ``natural=wetland`` polygons are kept too, because on these islands most of
them are salt ponds. The mask is on by default and can be switched off to reproduce
the 2.0 numbers.

OpenStreetMap is volunteer-mapped and its mangrove outlines are coarse at 10 m, so the
runner also keeps a per-pixel vegetation count (:func:`vegetated`) across the season.
It removes nothing; it is reported next to the detections, so a flag that sits on
vegetation the map missed is named rather than counted as sargassum.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PERSISTENT_MIN_PASSES",
    "PERSISTENT_SHARE",
    "VEGETATED_B08",
    "VEGETATED_NDVI",
    "VEGETATED_WETLAND",
    "is_vegetated_wetland",
    "load_polygons",
    "near",
    "persistent_vegetation",
    "polygon_mask",
    "vegetated",
    "zone_masks",
]

# natural=wetland values that are vegetation standing above the water. Seagrass, tidal
# flats and salt pans are not in the list: they are surfaces sargassum can float over.
VEGETATED_WETLAND = frozenset({"mangrove", "swamp", "marsh", "saltmarsh", "reedbed", "wet_meadow"})

# Vegetated on a pass: NDVI (B08, B04) above 0.5 and B08 reflectance above 0.08. The B08
# floor keeps dark water with a noisy red band from passing on NDVI alone.
VEGETATED_NDVI = 0.5
VEGETATED_B08 = 0.08
# Persistent vegetation: vegetated on at least 90% of at least 10 clear looks. Floating
# material is there on one or two passes and gone; a canopy is there on all of them.
PERSISTENT_SHARE = 0.9
PERSISTENT_MIN_PASSES = 10


def is_vegetated_wetland(tags: Mapping[str, str]) -> bool:
    """Whether an OpenStreetMap element is mangrove or other vegetated wetland.

    ``wetland`` may hold several values separated by ``;`` (``"marsh;mangrove"``).
    """
    if tags.get("natural") != "wetland":
        return False
    values = {v.strip() for v in str(tags.get("wetland", "")).split(";")}
    if "seagrass" in str(tags.get("name", "")).lower():
        return False
    return bool(values & VEGETATED_WETLAND)


def load_polygons(path: str | Path) -> list[Any]:
    """Every polygon in a GeoJSON FeatureCollection, in EPSG:4326."""
    from shapely.geometry import shape

    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return [shape(f["geometry"]) for f in blob.get("features", []) if f.get("geometry")]


def _projected(polygons: Iterable[Any], crs: Any) -> list[Any]:
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    fwd = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True
    ).transform
    return [shapely_transform(fwd, p) for p in polygons]


def polygon_mask(
    polygons: Sequence[Any],
    transform: Any,
    crs: Any,
    shape: tuple[int, int],
    *,
    buffer_m: float = 0.0,
    all_touched: bool = True,
) -> np.ndarray:
    """True inside any of ``polygons`` (EPSG:4326), grown by ``buffer_m``, on a raster grid.

    ``all_touched`` errs towards masking: a pixel the edge passes through is inside.
    That is the right side to err on for land and mangroves, whose edge pixels are the
    mixed ones the classifier mistakes for floating vegetation.
    """
    from rasterio.features import geometry_mask

    if not polygons:
        return np.zeros(shape, dtype=bool)
    projected = [p.buffer(buffer_m) if buffer_m else p for p in _projected(polygons, crs)]
    projected = [p for p in projected if not p.is_empty]
    if not projected:
        return np.zeros(shape, dtype=bool)
    return geometry_mask(
        projected, out_shape=shape, transform=transform, invert=True, all_touched=all_touched
    )


def zone_masks(
    zones: Mapping[str, Any], transform: Any, crs: Any, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Boolean raster of each surf zone (EPSG:4326 polygons, by segment id)."""
    from rasterio.features import geometry_mask

    out = {}
    for seg_id, poly in zip(zones, _projected(zones.values(), crs), strict=True):
        out[seg_id] = geometry_mask([poly], out_shape=shape, transform=transform, invert=True)
    return out


def vegetated(reflectance: Mapping[str, np.ndarray]) -> np.ndarray:
    """Pixels that look like leaf canopy on this pass: NDVI > 0.5 and B08 > 0.08."""
    nir = reflectance["B08"].astype(np.float32)
    red = reflectance["B04"].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    return (np.nan_to_num(ndvi, nan=-1.0) > VEGETATED_NDVI) & (nir > VEGETATED_B08)


def persistent_vegetation(
    vegetated_total: np.ndarray,
    seen_total: np.ndarray,
    *,
    share: float = PERSISTENT_SHARE,
    min_passes: int = PERSISTENT_MIN_PASSES,
) -> np.ndarray:
    """Pixels vegetated on at least ``share`` of at least ``min_passes`` clear looks."""
    seen = seen_total >= min_passes
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(seen, vegetated_total / np.maximum(seen_total, 1), 0.0)
    return seen & (frac >= share)


def near(mask: np.ndarray, pixels: int) -> np.ndarray:
    """``mask`` grown by ``pixels`` in every direction (a square neighbourhood)."""
    if pixels <= 0 or not mask.any():
        return mask.copy()
    from scipy import ndimage

    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=pixels)
