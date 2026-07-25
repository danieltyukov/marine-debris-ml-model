"""Data connectors: STAC imagery search, the MARIDA benchmark, and bundled sample chips.

Three sources, three roles:

- :mod:`~mdebris.data.stac` finds and resolves free Sentinel-2 scenes, normalizing band
  names so the rest of the codebase never learns which provider it is talking to.
- :mod:`~mdebris.data.marida` downloads the public marine-debris benchmark, which is what
  makes the confuser classes in :class:`~mdebris.types.SurfaceClass` measurable rather
  than aspirational.
- :mod:`~mdebris.data.samples` ships small real chips inside the package so tests, the
  README and the demo all run with no network and no credentials.

:mod:`~mdebris.data.planet` is optional and commercial. It is deliberately not imported
here, so nothing on the default path can fail over a missing API key.
"""

from __future__ import annotations

from mdebris.data.marida import (
    MARIDA_BANDS,
    MARIDA_CLASSES,
    MARIDA_TO_SURFACE,
    MaridaError,
    MaridaPatch,
    download_marida,
    load_marida_split,
)
from mdebris.data.samples import (
    HOTSPOTS,
    SAMPLE_BANDS,
    SampleError,
    fetch_sample_chips,
    list_samples,
    load_sample,
    reflectance_from_meta,
    sample_bbox,
    sample_reflectance,
    sample_scene,
)
from mdebris.data.scaling import (
    BOA_OFFSET,
    NON_REFLECTANCE_BANDS,
    REFLECTANCE_SCALE,
    reflectance_params_for_baseline,
    reflectance_params_for_item,
    scale_bands,
    to_reflectance,
)
from mdebris.data.stac import (
    Band,
    SceneNotFoundError,
    StacClient,
    StacError,
    canonical_band,
    get_scene_assets,
    normalize_assets,
    search_scenes,
)

__all__ = [
    "BOA_OFFSET",
    "HOTSPOTS",
    "MARIDA_BANDS",
    "MARIDA_CLASSES",
    "MARIDA_TO_SURFACE",
    "NON_REFLECTANCE_BANDS",
    "REFLECTANCE_SCALE",
    "SAMPLE_BANDS",
    "Band",
    "MaridaError",
    "MaridaPatch",
    "SampleError",
    "SceneNotFoundError",
    "StacClient",
    "StacError",
    "canonical_band",
    "download_marida",
    "fetch_sample_chips",
    "get_scene_assets",
    "list_samples",
    "load_marida_split",
    "load_sample",
    "normalize_assets",
    "reflectance_from_meta",
    "reflectance_params_for_baseline",
    "reflectance_params_for_item",
    "sample_bbox",
    "sample_reflectance",
    "sample_scene",
    "scale_bands",
    "search_scenes",
    "to_reflectance",
]
