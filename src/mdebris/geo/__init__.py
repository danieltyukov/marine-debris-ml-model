"""Geospatial support: tile math, georeferencing and windowed raster access.

Nothing here imports torch or any model code, so the geometry can be exercised
(and tested) without model weights on disk.
"""

from mdebris.geo.georef import (
    WGS84,
    boxes_to_detections,
    default_output_path,
    detections_to_geodataframe,
    georeference_detections,
    pixel_bbox_to_polygon,
    read_geojson,
    write_geojson,
)
from mdebris.geo.raster import (
    SENTINEL2_RESOLUTION_M,
    WindowLike,
    raster_profile,
    read_bands,
    read_window,
    to_rgb,
    window_transform,
)
from mdebris.geo.tiles import (
    WEB_MERCATOR_MAX_LAT,
    deg2num,
    num2deg,
    parse_tile_name,
    tile_affine,
    tile_bounds,
    tiles_for_bbox,
    windows_for_raster,
)

__all__ = [
    "SENTINEL2_RESOLUTION_M",
    "WEB_MERCATOR_MAX_LAT",
    "WGS84",
    "WindowLike",
    "boxes_to_detections",
    "default_output_path",
    "deg2num",
    "detections_to_geodataframe",
    "georeference_detections",
    "num2deg",
    "parse_tile_name",
    "pixel_bbox_to_polygon",
    "raster_profile",
    "read_bands",
    "read_geojson",
    "read_window",
    "tile_affine",
    "tile_bounds",
    "tiles_for_bbox",
    "to_rgb",
    "window_transform",
    "windows_for_raster",
    "write_geojson",
]
