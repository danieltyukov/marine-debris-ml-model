"""Core data contracts shared by every module.

This module is deliberately dependency-light: it imports only from the standard
library and numpy so that geo, indices, models, eval and api can all depend on it
without creating import cycles or dragging torch into pure-geometry code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BBox",
    "Detection",
    "DetectionSet",
    "Detector",
    "GeoBBox",
    "SceneRef",
    "SurfaceClass",
    "TileRef",
]


class SurfaceClass(StrEnum):
    """Sea-surface classes.

    The legacy model had exactly one class, ``marine_debris``, which made it
    structurally unable to distinguish plastic from the things that look like
    plastic from orbit. Sargassum mats, ship wakes and sea foam are the dominant
    false positives reported in the marine-litter literature, so they are
    first-class labels here rather than unmodelled noise.
    """

    DEBRIS = "marine_debris"
    SARGASSUM = "sargassum"
    SHIP = "ship"
    WAKE = "ship_wake"
    FOAM = "sea_foam"
    CLOUD = "cloud"
    SEDIMENT = "sediment_plume"
    WATER = "open_water"
    UNKNOWN = "unknown"

    @property
    def is_target(self) -> bool:
        """True for the class we are actually hunting."""
        return self is SurfaceClass.DEBRIS

    @property
    def is_confuser(self) -> bool:
        """True for classes that are commonly mistaken for debris."""
        return self in {
            SurfaceClass.SARGASSUM,
            SurfaceClass.WAKE,
            SurfaceClass.FOAM,
            SurfaceClass.SEDIMENT,
        }


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in **pixel** coordinates, ``xmin, ymin, xmax, ymax``.

    Pixel-space and geographic-space boxes are separate types on purpose. The
    legacy code kept both as bare lists and relied on a manual index shuffle
    (``[b[1], b[0], b[3], b[2]]``) to reconcile TensorFlow's ``[ymin, xmin,
    ymax, xmax]`` ordering with shapely's ``[xmin, ymin, xmax, ymax]``. Getting
    that backwards silently produces transposed detections, so the conversion
    now lives in one named constructor instead of being open-coded per call site.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError(
                f"degenerate bbox: ({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"
            )

    @classmethod
    def from_yxyx(cls, ymin: float, xmin: float, ymax: float, xmax: float) -> BBox:
        """Build from the ``[ymin, xmin, ymax, xmax]`` order used by TF/most detectors."""
        return cls(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @classmethod
    def from_normalized(
        cls, ymin: float, xmin: float, ymax: float, xmax: float, *, width: int, height: int
    ) -> BBox:
        """Scale normalized ``[0, 1]`` ``yxyx`` coordinates up to pixel space."""
        return cls(xmin=xmin * width, ymin=ymin * height, xmax=xmax * width, ymax=ymax * height)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def iou(self, other: BBox) -> float:
        """Intersection over union. Returns 0.0 for disjoint boxes."""
        ix0, iy0 = max(self.xmin, other.xmin), max(self.ymin, other.ymin)
        ix1, iy1 = min(self.xmax, other.xmax), min(self.ymax, other.ymax)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class GeoBBox:
    """Geographic bounds in EPSG:4326 degrees, ``west, south, east, north``."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError(f"longitude out of range: {self.west}, {self.east}")
        if not (-90.0 <= self.south <= 90.0 and -90.0 <= self.north <= 90.0):
            raise ValueError(f"latitude out of range: {self.south}, {self.north}")
        if self.north < self.south:
            raise ValueError(f"north {self.north} is below south {self.south}")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def centroid(self) -> tuple[float, float]:
        """(lon, lat) of the box centre."""
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)


@dataclass(frozen=True, slots=True)
class TileRef:
    """A slippy-map XYZ tile, the addressing scheme the legacy pipeline used.

    Legacy tiles were named ``{x}-{y}-{z}.jpg`` and parsed with a bare
    ``basename.split('-')``, which breaks on negative coordinates and on any
    filename containing another hyphen. Parsing now lives in ``geo.tiles``.
    """

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if not 0 <= self.z <= 30:
            raise ValueError(f"zoom {self.z} outside supported range 0..30")
        limit = 1 << self.z
        if not (0 <= self.x < limit and 0 <= self.y < limit):
            raise ValueError(f"tile ({self.x}, {self.y}) out of range for zoom {self.z}")

    def __str__(self) -> str:
        return f"{self.x}-{self.y}-{self.z}"


@dataclass(frozen=True, slots=True)
class SceneRef:
    """Identifies a source satellite scene so detections stay traceable to provenance."""

    scene_id: str
    collection: str = "sentinel-2-l2a"
    datetime: str | None = None
    platform: str | None = None
    cloud_cover: float | None = None
    source: str = "planetary-computer"


@dataclass(slots=True)
class Detection:
    """One detected object, carrying pixel geometry, geography and provenance together.

    Keeping the geographic polygon on the same object as the pixel box is what lets
    the pipeline emit GeoJSON without a second lookup, and what makes a detection
    self-describing when it is serialized into an API response.
    """

    bbox: BBox
    score: float
    label: SurfaceClass = SurfaceClass.DEBRIS
    geometry: Any | None = None  # shapely Polygon in EPSG:4326, set by geo.georef
    tile: TileRef | None = None
    scene: SceneRef | None = None
    mask: np.ndarray | None = None  # optional boolean mask from SAM2
    indices: dict[str, float] = field(default_factory=dict)  # e.g. {"FDI": 0.012}
    source_model: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score {self.score} outside [0, 1]")
        if isinstance(self.label, str) and not isinstance(self.label, SurfaceClass):
            self.label = SurfaceClass(self.label)

    @property
    def area_m2(self) -> float | None:
        """Footprint in square metres, if a ground sample distance is known.

        Populated by the pipeline via ``indices['gsd_m']``; ``None`` when the
        detection came from an image with no known scale.
        """
        gsd = self.indices.get("gsd_m")
        return None if gsd is None else self.bbox.area * gsd * gsd

    def to_feature(self) -> dict[str, Any]:
        """GeoJSON Feature. Requires ``geometry`` to have been set."""
        if self.geometry is None:
            raise ValueError("detection has no geographic geometry; run geo.georef first")
        from shapely.geometry import mapping

        props: dict[str, Any] = {
            "score": round(float(self.score), 6),
            "label": str(self.label),
            "model": self.source_model,
        }
        if self.tile is not None:
            props["tile"] = str(self.tile)
        if self.scene is not None:
            props["scene_id"] = self.scene.scene_id
            props["collection"] = self.scene.collection
            if self.scene.datetime:
                props["datetime"] = self.scene.datetime
        if self.indices:
            props |= {k: round(float(v), 6) for k, v in self.indices.items()}
        if (area := self.area_m2) is not None:
            props["area_m2"] = round(area, 2)
        return {"type": "Feature", "geometry": mapping(self.geometry), "properties": props}


@dataclass(slots=True)
class DetectionSet:
    """Detections for one scene or AOI, plus the metadata needed to reproduce them."""

    detections: list[Detection] = field(default_factory=list)
    scene: SceneRef | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def filter(
        self, *, min_score: float = 0.0, labels: set[SurfaceClass] | None = None
    ) -> DetectionSet:
        keep = [
            d
            for d in self.detections
            if d.score >= min_score and (labels is None or d.label in labels)
        ]
        return DetectionSet(detections=keep, scene=self.scene, meta=dict(self.meta))

    def targets_only(self) -> DetectionSet:
        """Drop confusers, keep actual debris."""
        return self.filter(labels={SurfaceClass.DEBRIS})

    def to_geojson(self) -> dict[str, Any]:
        """A GeoJSON FeatureCollection.

        An explicit CRS member is included. The legacy writer emitted no CRS at
        all; EPSG:4326 is the GeoJSON default so the output was still readable,
        but being explicit makes the files self-describing to GIS tools.
        """
        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [d.to_feature() for d in self.detections if d.geometry is not None],
            "properties": {
                "count": len(self.detections),
                "scene_id": self.scene.scene_id if self.scene else None,
                **self.meta,
            },
        }


@runtime_checkable
class Detector(Protocol):
    """Structural interface every detector satisfies.

    Zero-shot (OWLv2), supervised (RT-DETRv2) and the cascade all implement this,
    so the pipeline, CLI and API never branch on which model is in use.
    """

    name: str

    def detect(self, image: np.ndarray, *, threshold: float = 0.1) -> list[Detection]:
        """Detect objects in an HxWx3 uint8 RGB array."""
        ...
