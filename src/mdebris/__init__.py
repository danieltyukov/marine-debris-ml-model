"""Marine debris detection on satellite imagery.

Open-vocabulary detection, promptable segmentation and spectral indices over free
Sentinel-2 imagery. No training and no API credentials are required for the default
zero-shot pipeline.
"""

from mdebris.config import Settings, settings
from mdebris.types import (
    BBox,
    Detection,
    DetectionSet,
    Detector,
    GeoBBox,
    SceneRef,
    SurfaceClass,
    TileRef,
)

__version__ = "2.0.0"

__all__ = [
    "BBox",
    "Detection",
    "DetectionSet",
    "Detector",
    "GeoBBox",
    "SceneRef",
    "Settings",
    "SurfaceClass",
    "TileRef",
    "__version__",
    "settings",
]
