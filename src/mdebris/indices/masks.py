"""Boolean masks and candidate extraction, the cheap front half of the cascade.

The detector costs roughly 18 seconds per tile on CPU. Spectral arithmetic costs
microseconds. So the pipeline never runs the detector on a tile it can cheaply prove is
open water, cloud, or land. Everything in this module exists to make that decision, and
its bias should be towards recall: a false positive here costs one detector call, a
false negative here costs a missed debris patch that nothing downstream can recover.

Masks follow one convention throughout: ``True`` marks the pixels the name refers to.
``water_mask`` is True on water, ``cloud_mask_from_scl`` is True on cloud-contaminated
pixels (not on clear ones), ``debris_candidate_mask`` is True on candidates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdebris.indices.spectral import INDEX_REGISTRY, FloatArray, ndwi, normalize_bands
from mdebris.types import BBox

__all__ = [
    "CLOUD_SCL_CLASSES",
    "DEFAULT_FDI_THRESHOLD",
    "DEFAULT_NDWI_THRESHOLD",
    "SCL_CLASSES",
    "BoolArray",
    "candidate_regions",
    "cloud_mask_from_scl",
    "debris_candidate_mask",
    "water_mask",
]

BoolArray = NDArray[np.bool_]

# Mirrors mdebris.config.Settings.fdi_threshold and .ndwi_water_threshold. Duplicated as
# plain constants rather than imported so this module stays free of pydantic and of any
# environment-dependent state; the pipeline passes the configured values explicitly.
DEFAULT_FDI_THRESHOLD = 0.006
DEFAULT_NDWI_THRESHOLD = 0.0

# Sentinel-2 Level-2A Scene Classification Layer, produced by Sen2Cor. Codes and labels
# from the ESA Sentinel-2 Level-2A Algorithm Overview (Copernicus SentiWiki, S2
# processing). Class 2 was labelled DARK_FEATURES / DARK_AREA_PIXELS before processing
# baseline 05.11 and CAST_SHADOWS from that baseline on; the code is unchanged, so both
# names refer to the same value here.
SCL_CLASSES: dict[int, str] = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "CAST_SHADOWS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW_OR_ICE",
}

# Classes treated as unusable for debris detection by default.
#
# 0 and 1 are not cloud but are equally unusable, and excluding them here means the
# caller needs one mask rather than two. 3, 8, 9 and 10 are the cloud family. Thin
# cirrus (10) is included because bright thin cloud over dark water raises NIR without
# raising SWIR1 much, which is precisely the FDI signature of floating debris, making it
# one of the more expensive false positives.
#
# Deliberately excluded: class 2 (cast shadows) darkens rather than brightens, so it
# suppresses detections instead of faking them, and over water it is frequently assigned
# to real dark features; class 11 (snow/ice), because in polar scenes it is genuine
# surface and a caller studying ice-edge debris should decide for themselves. Both are
# available through the ``classes`` argument.
CLOUD_SCL_CLASSES: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})


def water_mask(
    bands: Mapping[str, ArrayLike], *, ndwi_threshold: float = DEFAULT_NDWI_THRESHOLD
) -> BoolArray:
    """Water pixels by McFeeters NDWI.

    Args:
        bands: Band arrays keyed by ESA id, STAC common name or canonical name. Must
            contain green (B03) and nir (B08).
        ndwi_threshold: NDWI strictly above this counts as water. Zero is the
            conventional land-water cut.

    Returns:
        Boolean array, True on water. NaN NDWI (no-data, or a pixel where green and nir
        sum to zero) yields False, so no-data is never treated as water.

    Raises:
        KeyError: if green or nir is absent.
    """
    resolved = normalize_bands(bands)
    missing = [b for b in ("green", "nir") if b not in resolved]
    if missing:
        raise KeyError(f"water_mask needs missing band(s): {', '.join(missing)}")
    index = ndwi(resolved["green"], resolved["nir"])
    # NaN > threshold is False and does not warn, which is the behaviour wanted here.
    return np.asarray(index > ndwi_threshold, dtype=bool)


def cloud_mask_from_scl(
    scl: ArrayLike, *, classes: frozenset[int] | set[int] | None = None
) -> BoolArray:
    """Cloud and no-data mask from the Sentinel-2 L2A Scene Classification Layer.

    The SCL is a per-pixel class raster shipped with every L2A product, at 20 m and
    60 m; resample it to the analysis grid with nearest-neighbour before calling, since
    any interpolating resampler will invent class codes that do not exist.

    Args:
        scl: Integer-valued class raster. See ``SCL_CLASSES`` for the code table.
        classes: Codes to treat as unusable. Defaults to ``CLOUD_SCL_CLASSES``, which is
            ``{0, 1, 3, 8, 9, 10}``: no-data, saturated/defective, cloud shadow, cloud
            medium and high probability, and thin cirrus.

    Returns:
        Boolean array, True where the pixel is unusable. Unrecognised class codes are
        treated as usable (False) rather than raising, because a future processing
        baseline adding a class should degrade to a slightly permissive mask rather than
        break the pipeline.

    Reference:
        ESA, Sentinel-2 Level-2A Algorithm Overview, scene classification.
        https://sentiwiki.copernicus.eu/web/s2-processing
    """
    selected = CLOUD_SCL_CLASSES if classes is None else classes
    codes = np.asarray(scl)
    if codes.dtype.kind == "f":
        # Float SCL turns up after a careless resample or a rasterio read with a float
        # nodata fill. Round to the nearest code and treat NaN as no-data.
        with np.errstate(invalid="ignore"):
            nan = np.isnan(codes)
            rounded = np.where(nan, 0, np.rint(codes)).astype(np.int32)
        return np.asarray(np.isin(rounded, list(selected)) | nan, dtype=bool)
    return np.asarray(np.isin(codes, list(selected)), dtype=bool)


def debris_candidate_mask(
    bands: Mapping[str, ArrayLike],
    *,
    fdi_threshold: float = DEFAULT_FDI_THRESHOLD,
    ndwi_threshold: float = DEFAULT_NDWI_THRESHOLD,
    scl: ArrayLike | None = None,
) -> BoolArray:
    """Prescreen mask: water AND high FDI AND not cloud.

    This is the gate that decides which tiles are worth an 18-second detector call.

    FDI comes from ``INDEX_REGISTRY["FDI"]`` when red edge 2 (B06) is available and from
    ``INDEX_REGISTRY["FDI_B04"]`` otherwise. The two are not on the same scale, so a
    threshold tuned against B06 data is only approximately right for the B04 fallback;
    prefer scenes with B06.

    On the water term: floating debris at 10 m is nearly always a sub-pixel mixture,
    typically well under half a pixel of material, so a debris pixel still reads as
    water under NDWI and the conjunction holds. A pixel fully covered by bright material
    would fail the water test and be dropped. That is a real limitation of an AND
    cascade rather than an oversight, and it only bites for targets large and bright
    enough that the surrounding partially covered pixels will flag anyway.

    Args:
        bands: Band arrays keyed by ESA id, STAC common name or canonical name. Needs
            green, nir, swir1 and one of rededge2 or red.
        fdi_threshold: FDI strictly above this marks a candidate.
        ndwi_threshold: Passed to ``water_mask``.
        scl: Optional Scene Classification Layer, same shape as the bands. When given,
            pixels flagged by ``cloud_mask_from_scl`` are excluded.

    Returns:
        Boolean array, True on candidate pixels. NaN in any input yields False for that
        pixel, so no-data never becomes a candidate.

    Raises:
        KeyError: if the bands needed for NDWI or for either FDI variant are absent.
    """
    resolved = normalize_bands(bands)
    index: FloatArray | None = None
    for name in ("FDI", "FDI_B04"):
        spec = INDEX_REGISTRY[name]
        if spec.available(resolved):
            index = spec.compute(resolved)
            break
    if index is None:
        raise KeyError(
            "debris_candidate_mask needs nir, swir1 and one of rededge2 (B06) or red (B04); "
            f"got {', '.join(sorted(resolved)) or 'no recognised bands'}"
        )

    candidate = water_mask(resolved, ndwi_threshold=ndwi_threshold) & (index > fdi_threshold)
    if scl is not None:
        candidate &= ~cloud_mask_from_scl(scl)
    return np.asarray(candidate, dtype=bool)


def candidate_regions(
    mask: ArrayLike, *, min_pixels: int = 4, connectivity: Literal[1, 2] = 2
) -> list[BBox]:
    """Connected components of a boolean mask as pixel-space bounding boxes.

    Args:
        mask: 2D boolean array, typically from ``debris_candidate_mask``.
        min_pixels: Components with fewer pixels than this are dropped. Single flagged
            pixels on Sentinel-2 are overwhelmingly sensor noise or sun glint; four
            10 m pixels is 400 square metres, which is around the smallest patch the
            literature reports as reliably detectable.
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity (the default). Debris
            windrows are thin and often diagonal, and 4-connectivity fragments them into
            unusable single-pixel pieces.

    Returns:
        Boxes in raster order (top-left first), with ``xmax`` and ``ymax`` as exclusive
        pixel edges, so a single-pixel component at row 3, column 5 gives
        ``BBox(5, 3, 6, 4)`` with area 1. Empty list for an empty mask.

    Raises:
        ImportError: if scipy is not installed.
        ValueError: if ``mask`` is not 2D or ``min_pixels`` is below 1.
    """
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be at least 1, got {min_pixels}")
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {binary.shape}")
    if not binary.any():
        return []

    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - scipy is a declared dependency
        raise ImportError("candidate_regions requires scipy; install scipy>=1.11") from exc

    structure = ndimage.generate_binary_structure(2, connectivity)
    labelled, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return []

    # find_objects returns slices indexed by label-1, so counting pixels per label with a
    # single bincount avoids re-scanning the array once per component.
    sizes = np.bincount(labelled.ravel(), minlength=count + 1)
    boxes: list[BBox] = []
    for label, extent in enumerate(ndimage.find_objects(labelled), start=1):
        if extent is None or sizes[label] < min_pixels:
            continue
        rows, cols = extent
        boxes.append(
            BBox(
                xmin=float(cols.start),
                ymin=float(rows.start),
                xmax=float(cols.stop),
                ymax=float(rows.stop),
            )
        )
    return boxes
