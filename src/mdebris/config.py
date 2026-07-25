"""Runtime configuration.

Everything is environment-driven with working defaults. The legacy scripts hardcoded
output paths (``./marine_litter/data_geo/``) and required a paid Planet key before
anything would run at all; here the defaults produce a working zero-shot pipeline
against free imagery with no credentials of any kind.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "settings"]

# OWLv2 resizes every input to 960x960 internally, so tiling the source at 960 covers
# the most ground per forward pass. Measured: a 512x512 tile and a 960x960 tile cost
# the same ~18s on CPU, meaning smaller tiles simply waste compute.
OWLV2_NATIVE_SIZE = 960


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MDEBRIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- paths -------------------------------------------------------------
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "mdebris",
        description="Downloaded scenes, datasets and derived chips.",
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Where GeoJSON, figures and reports are written.",
    )

    # ---- data sources ------------------------------------------------------
    stac_endpoint: str = Field(
        default="https://planetarycomputer.microsoft.com/api/stac/v1",
        description="Default STAC API. Anonymous search verified working.",
    )
    stac_fallback_endpoint: str = Field(
        default="https://earth-search.aws.element84.com/v1",
        description="Fallback STAC API (AWS Open Data, no auth).",
    )
    stac_collection: str = "sentinel-2-l2a"
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)

    planet_api_key: str | None = Field(
        default=None,
        description="Optional. Only needed for the commercial Planet connector.",
        alias="PL_API_KEY",
    )

    # ---- models ------------------------------------------------------------
    zeroshot_model: str = "google/owlv2-base-patch16-ensemble"
    segment_model: str = "facebook/sam2.1-hiera-tiny"
    supervised_model: str = "PekingU/rtdetr_v2_r18vd"
    device: str = Field(default="auto", description="'auto', 'cpu', 'cuda' or 'mps'.")
    torch_threads: int | None = Field(
        default=None, description="CPU threads for torch. None uses the torch default."
    )

    # ---- tiling and thresholds --------------------------------------------
    tile_size: int = Field(default=OWLV2_NATIVE_SIZE, ge=64, le=4096)
    tile_overlap: int = Field(
        default=96,
        ge=0,
        description="Pixel overlap between tiles so debris on a tile seam is not cut in half.",
    )
    score_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # ---- cascade -----------------------------------------------------------
    use_cascade: bool = Field(
        default=True,
        description="Screen tiles with cheap spectral indices before running the detector.",
    )
    fdi_threshold: float = Field(
        default=0.006,
        description="Floating Debris Index above which a water pixel is a candidate.",
    )
    ndwi_water_threshold: float = Field(default=0.0, description="NDWI above this counts as water.")
    min_candidate_pixels: int = Field(
        default=4,
        ge=1,
        description="A tile needs this many candidate pixels before the detector runs on it.",
    )

    @field_validator("tile_overlap")
    @classmethod
    def _overlap_fits(cls, v: int, info) -> int:
        size = info.data.get("tile_size", OWLV2_NATIVE_SIZE)
        if v >= size:
            raise ValueError(f"tile_overlap {v} must be smaller than tile_size {size}")
        return v

    def resolve_device(self) -> str:
        """Pick a torch device, honouring an explicit override."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
