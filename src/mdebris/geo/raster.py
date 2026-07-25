"""Windowed raster reads, multi-resolution band alignment and RGB rendering.

The legacy pipeline only ever saw pre-rendered 256 pixel JPEG tiles, so none of
this existed: band alignment and contrast handling were somebody else's problem,
solved upstream by whatever produced the tiles. Reading Sentinel-2 COGs directly
means owning both.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio import windows as rio_windows
from rasterio.enums import Resampling
from rasterio.windows import Window

__all__ = [
    "SENTINEL2_RESOLUTION_M",
    "WindowLike",
    "raster_profile",
    "read_bands",
    "read_window",
    "to_rgb",
    "window_transform",
]

# A window is either a rasterio Window or (col_off, row_off, width, height),
# which is what geo.tiles.windows_for_raster yields.
WindowLike = Window | tuple[float, float, float, float]

# Native ground sample distance per Sentinel-2 band, in metres. Bands at
# different resolutions cover the same ground with different pixel counts, which
# is the whole reason read_bands has to resample onto a common grid.
SENTINEL2_RESOLUTION_M: dict[str, int] = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
}

# Offsets derived from geographic bounds land on integers up to floating point
# noise (rasterio returns -0.0 for a window starting at the origin). Snapping
# keeps the read on the pixel grid instead of triggering a sub-pixel resample.
_SNAP_TOLERANCE = 1e-6


def _as_window(window: WindowLike) -> Window:
    """Coerce a window tuple to a rasterio Window."""
    if isinstance(window, Window):
        return window
    col_off, row_off, width, height = window
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _snap_window(window: Window) -> Window:
    """Round near-integer window components to exact integers."""
    return Window(
        *(
            round(v) if abs(v - round(v)) < _SNAP_TOLERANCE else v
            for v in (window.col_off, window.row_off, window.width, window.height)
        )
    )


def _resampling(method: str | Resampling) -> Resampling:
    """Look up a rasterio Resampling enum from its name."""
    if isinstance(method, Resampling):
        return method
    try:
        return Resampling[str(method).lower()]
    except KeyError as exc:
        options = ", ".join(sorted(r.name for r in Resampling))
        raise ValueError(
            f"unknown resampling method {method!r}; expected one of {options}"
        ) from exc


def window_transform(window: WindowLike, source: Affine | str | Path) -> Affine:
    """Affine transform of a window's own pixel grid.

    Detections are found in window pixel coordinates, so georeferencing them
    needs the transform of that window, not of the full raster.

    Args:
        window: Window into ``source``.
        source: Either the full raster's affine transform, or a path or href to
            open and take the transform from.

    Returns:
        Affine mapping window ``(column, row)`` to the raster's world coordinates.
    """
    if not isinstance(source, Affine):
        with rasterio.open(source) as src:
            source = src.transform
    return rio_windows.transform(_as_window(window), source)


def read_window(
    href_or_path: str | Path,
    window: WindowLike | None = None,
    *,
    band_index: int = 1,
    out_shape: tuple[int, int] | None = None,
    resampling: str | Resampling = "bilinear",
) -> np.ndarray:
    """Read one band over one window.

    Args:
        href_or_path: Local path or remote href (rasterio handles ``/vsicurl``
            style URLs and signed hrefs directly).
        window: Region to read. None reads the whole raster.
        band_index: 1-based band index, matching rasterio's convention.
        out_shape: ``(height, width)`` to resample the read into. None keeps the
            window's native pixel count.
        resampling: Method used only when ``out_shape`` forces a resample.

    Returns:
        A 2-D array of the band's native dtype.
    """
    with rasterio.open(href_or_path) as src:
        win = _as_window(window) if window is not None else Window(0, 0, src.width, src.height)
        return src.read(
            band_index,
            window=win,
            out_shape=out_shape,
            resampling=_resampling(resampling),
        )


def read_bands(
    hrefs: Mapping[str, str | Path],
    window: WindowLike | None = None,
    *,
    target_shape: tuple[int, int] | None = None,
    resampling: str | Resampling = "bilinear",
    reference: str | None = None,
    band_index: int = 1,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """Read several bands onto one common pixel grid.

    Sentinel-2 bands are distributed at three ground sample distances: 10 m for
    B02/B03/B04/B08, 20 m for B05/B06/B07/B8A/B11/B12 and 60 m for B01/B09/B10.
    A 512x512 window on a 10 m band is 256x256 on a 20 m band, so reading the
    same window offsets from every asset returns arrays that are neither the
    same shape nor aligned to the same ground. Index arithmetic such as FDI,
    which mixes 10 m B08 with 20 m B11, is meaningless on unaligned arrays.

    The window is resolved to geographic bounds on the reference band's grid,
    each band's own window is derived from those bounds, and every read is
    resampled to a single output shape. Alignment is therefore by ground
    coordinates, not by pixel index.

    Args:
        hrefs: Band name to path or href. Insertion order matters only in that
            the first key is the default reference grid.
        window: Window expressed on the *reference* band's grid. None reads the
            reference band's full extent.
        target_shape: ``(height, width)`` every band is resampled to. Defaults
            to the reference window's own pixel shape, so 20 m and 60 m bands
            are upsampled to the 10 m grid when a 10 m band is the reference.
        resampling: Resampling method. ``"bilinear"`` suits reflectance;
            ``"nearest"`` preserves exact values for masks and classifications.
        reference: Band name whose grid defines ``window`` and the default
            ``target_shape``. Defaults to the first key in ``hrefs``.
        band_index: 1-based band index within each asset. Sentinel-2 COGs are
            single band, so the default is nearly always right.
        fill_value: Value written where a band does not cover the requested
            bounds, which happens at scene edges.

    Returns:
        Band name to 2-D array, every array of shape ``target_shape``. Keys and
        order match ``hrefs``.

    Raises:
        ValueError: If ``hrefs`` is empty or a band's CRS differs from the
            reference band's. Cross-CRS mosaicking is a warp, not a read, and is
            deliberately out of scope here.
        KeyError: If ``reference`` is not a key of ``hrefs``.
    """
    if not hrefs:
        raise ValueError("hrefs is empty; nothing to read")
    ref_key = reference if reference is not None else next(iter(hrefs))
    if ref_key not in hrefs:
        raise KeyError(f"reference band {ref_key!r} is not in hrefs (have {sorted(hrefs)})")

    method = _resampling(resampling)
    with rasterio.open(hrefs[ref_key]) as ref:
        ref_win = _as_window(window) if window is not None else Window(0, 0, ref.width, ref.height)
        ref_crs = ref.crs
        geo_bounds = rio_windows.bounds(ref_win, ref.transform)
    # Window dimensions can be numpy floats, so round through float to land on a
    # plain int that rasterio's out_shape accepts.
    shape = target_shape or (round(float(ref_win.height)), round(float(ref_win.width)))

    out: dict[str, np.ndarray] = {}
    for name, href in hrefs.items():
        with rasterio.open(href) as src:
            if ref_crs is not None and src.crs != ref_crs:
                raise ValueError(
                    f"band {name!r} is in {src.crs} but reference band {ref_key!r} is in "
                    f"{ref_crs}; reproject the assets before reading them together"
                )
            win = _snap_window(rio_windows.from_bounds(*geo_bounds, transform=src.transform))
            covered = (
                win.col_off >= -_SNAP_TOLERANCE
                and win.row_off >= -_SNAP_TOLERANCE
                and win.col_off + win.width <= src.width + _SNAP_TOLERANCE
                and win.row_off + win.height <= src.height + _SNAP_TOLERANCE
            )
            out[name] = src.read(
                band_index,
                window=win,
                out_shape=shape,
                resampling=method,
                # A plain read of a window hanging off the raster returns a
                # smaller array, which would then be stretched into target_shape
                # and quietly misregistered. Boundless padding keeps the grid.
                boundless=not covered,
                fill_value=None if covered else fill_value,
            )
    return out


def to_rgb(
    bands: Mapping[str, np.ndarray],
    *,
    stretch: str = "percentile",
    gamma: float = 1.0,
    percentiles: tuple[float, float] = (2.0, 98.0),
    ignore_value: float | None = None,
) -> np.ndarray:
    """Render red, green and blue bands as a displayable uint8 image.

    Sentinel-2 L2A surface reflectance is stored as uint16 scaled by 10000, so
    open water sits around 100 to 500 out of a 65535 range. Dividing by the
    dtype maximum, or stretching linearly from zero, yields a black rectangle,
    which is what makes naive RGB previews of marine scenes useless. The default
    2nd-to-98th percentile stretch is computed per band from the data actually
    present, so water scenes come out legible without any per-scene tuning.

    The stretch is per band rather than global. A single stretch driven by all
    three would be dominated by blue over water and leave red and green crushed,
    hiding exactly the contrast that separates a debris slick from open sea.

    Args:
        bands: Band name to 2-D array. Looked up as B04/B03/B02, case
            insensitively, with ``red``/``green``/``blue`` accepted as aliases.
            All three must already be on a common grid (see ``read_bands``).
        stretch: ``"percentile"``, ``"linear"`` (per-band min to max) or
            ``"none"`` (integers scaled by their dtype range, floats clipped to
            ``[0, 1]``).
        gamma: Applied after normalization as ``value ** gamma``, matching
            ``skimage.exposure.adjust_gamma``. Values above 1 darken, below 1
            brighten. 1.0 is a no-op.
        percentiles: Low and high percentile for the ``"percentile"`` stretch.
        ignore_value: Excluded when computing percentiles. Pass 0.0 for scenes
            with large nodata blocks, which otherwise drag the low percentile
            down and wash the image out.

    Returns:
        A ``(H, W, 3)`` uint8 array.

    Raises:
        KeyError: If a red, green or blue band cannot be found.
        ValueError: If ``stretch`` is unknown, ``gamma`` is not positive, the
            percentiles are not ordered, or the bands have differing shapes.
    """
    if stretch not in {"percentile", "linear", "none"}:
        raise ValueError(f"stretch must be 'percentile', 'linear' or 'none', got {stretch!r}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    lo_pct, hi_pct = percentiles
    if not 0.0 <= lo_pct < hi_pct <= 100.0:
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {percentiles}")

    aliases = (("b04", "red", "r"), ("b03", "green", "g"), ("b02", "blue", "b"))
    channels = [_pick_band(bands, group) for group in aliases]
    shapes = {c.shape for c in channels}
    if len(shapes) != 1:
        raise ValueError(f"RGB bands have differing shapes {shapes}; resample them first")

    scaled = [_stretch_band(c, stretch, lo_pct, hi_pct, ignore_value) for c in channels]
    rgb = np.stack(scaled, axis=-1)
    if gamma != 1.0:
        rgb = rgb**gamma
    return np.round(rgb * 255.0).astype(np.uint8)


def _pick_band(bands: Mapping[str, np.ndarray], candidates: tuple[str, ...]) -> np.ndarray:
    """Find one channel in a band dict, tolerating naming conventions."""
    lookup = {str(k).lower(): v for k, v in bands.items()}
    for key in candidates:
        if key in lookup:
            return np.asarray(lookup[key])
    raise KeyError(f"none of {candidates} found in bands (have {sorted(bands)})")


def _stretch_band(
    band: np.ndarray,
    stretch: str,
    lo_pct: float,
    hi_pct: float,
    ignore_value: float | None,
) -> np.ndarray:
    """Normalize one band to [0, 1] using the requested contrast stretch."""
    x = band.astype(np.float64)
    valid = np.isfinite(x)
    if ignore_value is not None:
        valid &= x != ignore_value

    if stretch == "none":
        if np.issubdtype(band.dtype, np.integer):
            info = np.iinfo(band.dtype)
            lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = 0.0, 1.0
    elif not valid.any():
        # An all-nodata window has no contrast to recover; return black rather
        # than letting nanpercentile raise.
        return np.zeros(x.shape, dtype=np.float64)
    elif stretch == "percentile":
        lo, hi = (float(v) for v in np.percentile(x[valid], [lo_pct, hi_pct]))
    else:
        lo, hi = float(x[valid].min()), float(x[valid].max())

    if hi <= lo:
        # Constant band. Mid grey carries as much information as anything else
        # and avoids a divide by zero.
        return np.full(x.shape, 0.5, dtype=np.float64)
    out = (np.nan_to_num(x, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def raster_profile(href_or_path: str | Path) -> dict[str, Any]:
    """Summarize a raster's grid, for logging and for sizing a tiling pass.

    Args:
        href_or_path: Local path or remote href.

    Returns:
        Width, height, band count, dtype, CRS string, affine transform, nodata
        value and the ground sample distance implied by the transform.
    """
    with rasterio.open(href_or_path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "nodata": src.nodata,
            "gsd": (abs(src.transform.a), abs(src.transform.e)),
        }
