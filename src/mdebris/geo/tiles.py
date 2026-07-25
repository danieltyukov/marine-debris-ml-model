"""Slippy-map tile addressing and raster windowing.

The legacy pipeline addressed imagery exclusively as XYZ tiles named
``{x}-{y}-{z}.jpg`` and rebuilt the tile-to-world transform inline in the
inference loop. That math is correct and worth keeping, so it lives here as a
named function instead. Everything that was hardcoded in the original (the 256
pixel tile size, the naive ``split('-')`` parse) is a parameter or a guarded
parse now.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import affine
import mercantile

from mdebris.config import settings
from mdebris.types import GeoBBox, TileRef

__all__ = [
    "WEB_MERCATOR_MAX_LAT",
    "deg2num",
    "num2deg",
    "parse_tile_name",
    "tile_affine",
    "tile_bounds",
    "tiles_for_bbox",
    "windows_for_raster",
]

# Web Mercator is only defined up to atan(sinh(pi)) in latitude, which is where the
# projection would run to infinity. Tile schemes clamp to this value.
WEB_MERCATOR_MAX_LAT = 85.0511287798066


def parse_tile_name(name: str | os.PathLike[str]) -> TileRef:
    """Parse a tile reference out of a tile filename or path.

    Accepts ``"12-34-5"``, ``"12-34-5.jpg"``, ``"/data/tiles/12-34-5.tif"`` and
    names carrying a prefix such as ``"scene-abc-12-34-5.png"``.

    The legacy parse was ``[int(x) for x in basename.split('-')]``, which raised
    an unhelpful "too many values to unpack" on any filename containing an extra
    hyphen (scene ids routinely do). The last three hyphen-separated components
    are the coordinates, so that is what is read.

    Args:
        name: Tile filename, basename or full path.

    Returns:
        The parsed tile reference.

    Raises:
        ValueError: If fewer than three components are present, if any of the
            last three is not an integer, or if the coordinates are out of range
            for the zoom level.
    """
    text = os.fspath(name).replace("\\", "/")
    stem = os.path.splitext(text.rsplit("/", 1)[-1])[0]
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"cannot parse tile name {os.fspath(name)!r}: expected '{{x}}-{{y}}-{{z}}' "
            f"(optionally prefixed and with an extension), got {len(parts)} component(s)"
        )
    try:
        x, y, z = (int(part) for part in parts[-3:])
    except ValueError as exc:
        raise ValueError(
            f"cannot parse tile name {os.fspath(name)!r}: the last three components "
            f"{parts[-3:]!r} are not all integers"
        ) from exc
    return TileRef(x=x, y=y, z=z)


def tile_bounds(tile: TileRef) -> GeoBBox:
    """Geographic bounds of a tile in EPSG:4326 degrees.

    Args:
        tile: Tile to bound.

    Returns:
        The tile envelope as ``west, south, east, north``.
    """
    b = mercantile.bounds(tile.x, tile.y, tile.z)
    return GeoBBox(west=b.west, south=b.south, east=b.east, north=b.north)


def tile_affine(tile: TileRef, tile_size: int | None = None) -> affine.Affine:
    """Build the pixel-to-lon/lat transform for a tile image.

    This is the legacy transform, generalized off the hardcoded 256. Column
    scale is positive because longitude grows eastward with the column index;
    row scale is negative because latitude *decreases* as the row index grows,
    and the origin sits at the tile's north-west corner.

    The result is a plate-carree style affine over the tile extent, not a true
    Web Mercator transform. Within a single tile the latitude error from
    treating the tile as linear in degrees is well below a pixel at the zoom
    levels used here, and it is what the legacy GeoJSON was built with, so
    parity is preserved.

    Args:
        tile: Tile the image belongs to.
        tile_size: Side length of the tile image in pixels. Defaults to
            ``settings.tile_size``.

    Returns:
        Affine mapping ``(column, row)`` to ``(longitude, latitude)``.

    Raises:
        ValueError: If ``tile_size`` is not positive.
    """
    size = settings.tile_size if tile_size is None else tile_size
    if size <= 0:
        raise ValueError(f"tile_size must be positive, got {size}")
    b = tile_bounds(tile)
    width = b.east - b.west
    height = b.north - b.south
    return affine.Affine(width / size, 0.0, b.west, 0.0, -height / size, b.north)


def num2deg(x: float, y: float, z: int) -> tuple[float, float]:
    """Convert fractional tile coordinates to longitude and latitude.

    Implemented directly rather than delegating to mercantile so the projection
    math is visible and independently testable. ``test_geo_tiles`` asserts it
    agrees with mercantile.

    Args:
        x: Tile column, fractional values addressing points inside a tile.
        y: Tile row, fractional values addressing points inside a tile.
        z: Zoom level.

    Returns:
        ``(longitude, latitude)`` in degrees.
    """
    n = 2.0**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def deg2num(lon: float, lat: float, z: int) -> tuple[int, int]:
    """Convert longitude and latitude to the tile containing that point.

    Latitude is clamped to the Web Mercator limit and the result is clamped to
    the valid tile range, so a point on the antimeridian or at a pole returns
    the edge tile instead of an out-of-range index.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        z: Zoom level.

    Returns:
        ``(x, y)`` integer tile coordinates.
    """
    lat = min(max(lat, -WEB_MERCATOR_MAX_LAT), WEB_MERCATOR_MAX_LAT)
    n = 2.0**z
    x = math.floor((lon + 180.0) / 360.0 * n)
    y = math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    limit = (1 << z) - 1
    return min(max(x, 0), limit), min(max(y, 0), limit)


def tiles_for_bbox(bbox: GeoBBox, zoom: int) -> Iterator[TileRef]:
    """Enumerate the tiles covering a geographic box at one zoom level.

    Args:
        bbox: Area of interest in EPSG:4326 degrees.
        zoom: Zoom level to tile at.

    Yields:
        Every tile intersecting ``bbox``, in row-major order.

    Raises:
        ValueError: If ``zoom`` is outside 0..30.
    """
    if not 0 <= zoom <= 30:
        raise ValueError(f"zoom {zoom} outside supported range 0..30")
    for t in mercantile.tiles(bbox.west, bbox.south, bbox.east, bbox.north, [zoom]):
        yield TileRef(x=t.x, y=t.y, z=t.z)


def windows_for_raster(
    width: int,
    height: int,
    tile_size: int | None = None,
    overlap: int | None = None,
) -> Iterator[tuple[int, int, int, int]]:
    """Slide overlapping windows across a raster, covering every pixel.

    Windows overlap so an object lying on a seam is seen whole by at least one
    window. Right and bottom edges emit a *partial* window rather than a window
    clamped back into the raster: a clamped window would re-read pixels the
    previous window already covered and, more importantly, would report offsets
    that do not match the caller's grid arithmetic. No window ever extends past
    the raster, so every offset can be passed straight to rasterio without a
    boundless read.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        tile_size: Window side length. Defaults to ``settings.tile_size``.
        overlap: Pixels shared between adjacent windows. Defaults to
            ``settings.tile_overlap``.

    Yields:
        ``(col_off, row_off, win_width, win_height)`` tuples in row-major order.

    Raises:
        ValueError: If any dimension is non-positive or if ``overlap`` is
            negative or not smaller than ``tile_size``.
    """
    size = settings.tile_size if tile_size is None else tile_size
    step_overlap = settings.tile_overlap if overlap is None else overlap
    if width <= 0 or height <= 0:
        raise ValueError(f"raster dimensions must be positive, got {width}x{height}")
    if size <= 0:
        raise ValueError(f"tile_size must be positive, got {size}")
    if not 0 <= step_overlap < size:
        raise ValueError(f"overlap {step_overlap} must satisfy 0 <= overlap < tile_size {size}")

    step = size - step_overlap
    col_offsets = _axis_offsets(width, size, step)
    row_offsets = _axis_offsets(height, size, step)
    for row_off in row_offsets:
        win_h = min(size, height - row_off)
        for col_off in col_offsets:
            yield (col_off, row_off, min(size, width - col_off), win_h)


def _axis_offsets(extent: int, size: int, step: int) -> list[int]:
    """Window start offsets along one axis, including the partial tail window.

    ``range`` only yields offsets where a full window fits. When the raster does
    not divide evenly, one more offset is appended so the remaining strip is
    covered; because ``step <= size`` that strip is contiguous with the window
    before it.
    """
    if extent <= size:
        return [0]
    offsets = list(range(0, extent - size + 1, step))
    if offsets[-1] + size < extent:
        offsets.append(offsets[-1] + step)
    return offsets
