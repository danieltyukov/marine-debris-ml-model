"""Request and response models for the HTTP API.

Kept free of torch and rasterio imports so that OpenAPI schema generation, and any
client that only wants the types, stays cheap.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from mdebris.types import SurfaceClass

__all__ = [
    "DetectRequest",
    "DetectResponse",
    "DetectionModel",
    "HealthResponse",
    "IndexStatsResponse",
    "SampleInfo",
    "SceneModel",
    "SearchRequest",
]


class SceneModel(BaseModel):
    scene_id: str
    collection: str = "sentinel-2-l2a"
    datetime: str | None = None
    platform: str | None = None
    cloud_cover: float | None = None
    source: str = "planetary-computer"


class DetectionModel(BaseModel):
    label: SurfaceClass
    score: float = Field(ge=0.0, le=1.0)
    bbox_pixel: tuple[float, float, float, float] = Field(
        description="xmin, ymin, xmax, ymax in pixel coordinates of the analysed chip."
    )
    lon: float | None = None
    lat: float | None = None
    area_m2: float | None = None
    indices: dict[str, float] = Field(default_factory=dict)
    source_model: str = ""


class DetectRequest(BaseModel):
    """Run detection on a bundled sample or on live imagery for a bounding box."""

    sample: str | None = Field(
        default=None, description="Bundled sample name. Mutually exclusive with bbox."
    )
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="west, south, east, north in EPSG:4326 degrees."
    )
    start: str | None = Field(default=None, description="Start date, YYYY-MM-DD.")
    end: str | None = Field(default=None, description="End date, YYYY-MM-DD.")
    model: Literal["owlv2", "grounding-dino", "rtdetr"] = "owlv2"
    threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    cascade: bool = True
    targets_only: bool = Field(
        default=False, description="Drop confuser classes such as ship and sargassum."
    )
    max_cloud: float = Field(default=20.0, ge=0.0, le=100.0)

    @field_validator("bbox")
    @classmethod
    def _bbox_is_sane(cls, v):
        if v is None:
            return v
        west, south, east, north = v
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("longitude must be within [-180, 180]")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("latitude must be within [-90, 90]")
        if north <= south:
            raise ValueError("north must be greater than south")
        # An unbounded request would tile an entire hemisphere at 18 seconds a tile.
        if (east - west) * (north - south) > 4.0:
            raise ValueError("bbox area is too large, keep it under 4 square degrees")
        return v

    def validate_selection(self) -> None:
        """Enforce that exactly one of sample or bbox is supplied."""
        if (self.sample is None) == (self.bbox is None):
            raise ValueError("supply exactly one of 'sample' or 'bbox'")
        if self.bbox is not None and (self.start is None or self.end is None):
            raise ValueError("'bbox' requires both 'start' and 'end'")


class DetectResponse(BaseModel):
    detections: list[DetectionModel]
    geojson: dict[str, Any]
    scene: SceneModel | None = None
    screening: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    count: int = 0


class SearchRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    start: str
    end: str
    max_cloud: float = Field(default=20.0, ge=0.0, le=100.0)
    limit: int = Field(default=10, ge=1, le=100)


class SampleInfo(BaseModel):
    name: str
    scene_id: str
    datetime: str | None = None
    width: int
    height: int
    cloud_cover: float | None = None
    description: str = ""
    bands: list[str] = Field(default_factory=list)


class IndexStatsResponse(BaseModel):
    sample: str
    water_fraction: float
    indices: dict[str, dict[str, float]]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    device: str
    samples: list[str] = Field(default_factory=list)
