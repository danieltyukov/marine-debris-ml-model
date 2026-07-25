"""Spectral indices and prescreen masks for floating-material detection.

Two layers:

* ``spectral`` holds the indices themselves (FDI, FAI, NDVI, NDWI, MNDWI, NDMI, RNDVI,
  PI, kNDVI) as pure numpy functions over float32 reflectance, plus ``INDEX_REGISTRY``
  so other modules can enumerate them without knowing their names in advance.
* ``masks`` turns those indices into the boolean decisions the cascade needs: what is
  water, what is cloud, which pixels are debris candidates, and which bounding boxes are
  worth handing to a detector.

Nothing here imports torch or rasterio, so the prescreen runs wherever the arrays do.
"""

from __future__ import annotations

from mdebris.indices.masks import (
    CLOUD_SCL_CLASSES,
    DEFAULT_FDI_THRESHOLD,
    DEFAULT_NDWI_THRESHOLD,
    SCL_CLASSES,
    BoolArray,
    candidate_regions,
    cloud_mask_from_scl,
    debris_candidate_mask,
    water_mask,
)
from mdebris.indices.spectral import (
    BAND_ALIASES,
    FDI_WAVELENGTHS_NM,
    FDI_WAVELENGTHS_NM_USGS,
    INDEX_REGISTRY,
    S2A_CENTRAL_WAVELENGTHS_NM,
    FloatArray,
    IndexSpec,
    available_indices,
    compute_indices,
    fai,
    fdi,
    fdi_baseline_slope,
    kndvi,
    mndwi,
    ndmi,
    ndvi,
    ndwi,
    normalize_bands,
    plastic_index,
    rndvi,
)

__all__ = [
    "BAND_ALIASES",
    "CLOUD_SCL_CLASSES",
    "DEFAULT_FDI_THRESHOLD",
    "DEFAULT_NDWI_THRESHOLD",
    "FDI_WAVELENGTHS_NM",
    "FDI_WAVELENGTHS_NM_USGS",
    "INDEX_REGISTRY",
    "S2A_CENTRAL_WAVELENGTHS_NM",
    "SCL_CLASSES",
    "BoolArray",
    "FloatArray",
    "IndexSpec",
    "available_indices",
    "candidate_regions",
    "cloud_mask_from_scl",
    "compute_indices",
    "debris_candidate_mask",
    "fai",
    "fdi",
    "fdi_baseline_slope",
    "kndvi",
    "mndwi",
    "ndmi",
    "ndvi",
    "ndwi",
    "normalize_bands",
    "plastic_index",
    "rndvi",
    "water_mask",
]
