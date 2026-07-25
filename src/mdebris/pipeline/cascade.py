"""The cost cascade: cheap spectral screening in front of an expensive detector.

A vision transformer costs roughly 18 seconds per 960x960 tile on CPU. A Sentinel-2
scene is 10,980 x 10,980 pixels, about 130 such tiles, so detecting everywhere would
take around 40 minutes per scene. Nearly all of that area is land, cloud or empty
water which cannot contain floating debris.

Spectral indices are array arithmetic costing microseconds per tile. Screening first
and running the detector only on tiles that are simultaneously water, cloud-free and
high-FDI is what makes the project usable without a GPU.

The screen is deliberately permissive. A false positive here costs one detector call,
which is recoverable. A false negative discards a tile the detector will never see,
which is not. Thresholds are tuned toward recall for that reason.

Why the threshold is adaptive by default
----------------------------------------
A fixed FDI cutoff does not transfer between scenes. Measured on a real Sentinel-2
chip of open ocean off Accra (S2A_MSIL2A_20240527T100601_R022_T30NZM, 100% water,
no known debris), FDI over water was:

    mean +0.00178, p50 +0.00157, p99 +0.00917, p99.9 +0.01316

so the naive fixed cutoff of 0.006 flagged 6.35% of *pure water* as candidate and
accepted every tile, delivering no speedup at all. FDI magnitude moves with
atmospheric correction, sun angle, sea state and water type, so a constant tuned on
one scene is simply wrong on the next.

Taking a high percentile of each scene's own water-FDI distribution instead makes the
screen self-calibrating: it looks for the extreme upper tail relative to the water
actually present. An absolute floor is still applied so that a scene containing
genuinely nothing does not have its own noise promoted into candidates.

Acceptance is also based on connected regions rather than raw pixel counts. Scattered
single pixels above the threshold are sensor and sun-glint noise; a real debris patch
is spatially coherent.

Known limitation of the adaptive threshold
------------------------------------------
A percentile cutoff assumes the target is rare within the tile. If debris covered more
than (100 - percentile)% of the water in a tile, the threshold would climb into the
debris distribution itself and the patch would mask its own detection. At the p99.9
default that means debris would have to fill over 0.1% of the tile's water, which for
a 960 px tile is roughly 920 pixels, or about 9 hectares of continuous material.
Patches that large are rare, and the absolute floor plus the detector pass downstream
both still apply. Callers working a known mass-stranding event should pass an explicit
``fdi_threshold`` rather than relying on the adaptive path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mdebris.config import settings
from mdebris.indices.masks import (
    candidate_regions,
    cloud_mask_from_scl,
    debris_candidate_mask,
    water_mask,
)
from mdebris.indices.spectral import compute_indices
from mdebris.types import BBox

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = ["ScreenResult", "adaptive_fdi_threshold", "screen_tile", "summarize_screening"]


@dataclass(slots=True)
class ScreenResult:
    """Outcome of screening a single tile.

    Attributes:
        accepted: Whether the tile is worth running the detector on.
        candidate_pixels: Count of pixels passing the combined debris mask.
        regions: Connected-component boxes of candidate pixels, in tile pixel
            coordinates. These seed the detector and can prompt SAM 2 directly.
        indices: Computed index arrays, reused downstream so nothing is recomputed.
        mask: The combined boolean candidate mask.
        reason: Why the tile was rejected, empty when accepted. Kept for reporting,
            since "why did the pipeline skip this scene" is the first question
            anyone asks when a known debris event is missed.
    """

    accepted: bool
    candidate_pixels: int = 0
    regions: list[BBox] = field(default_factory=list)
    indices: dict[str, np.ndarray] = field(default_factory=dict)
    mask: np.ndarray | None = None
    threshold: float = 0.0
    reason: str = ""

    @property
    def candidate_fraction(self) -> float:
        """Fraction of the tile flagged as candidate, 0.0 when the mask is absent."""
        if self.mask is None or self.mask.size == 0:
            return 0.0
        return float(self.candidate_pixels) / float(self.mask.size)


def adaptive_fdi_threshold(
    fdi: np.ndarray,
    water: np.ndarray | None = None,
    *,
    percentile: float = 99.9,
    floor: float | None = None,
) -> float:
    """Derive an FDI cutoff from the scene's own water pixels.

    Args:
        fdi: Floating Debris Index array.
        water: Boolean water mask. When omitted every finite pixel is used.
        percentile: Percentile of the water FDI distribution to cut at.
        floor: Absolute lower bound. Defaults to ``settings.fdi_threshold``. Without
            it, a scene containing nothing unusual would have its own noise floor
            promoted into candidates, since a percentile always returns some value.

    Returns:
        The threshold, never below the floor.
    """
    floor = settings.fdi_threshold if floor is None else floor
    values = np.asarray(fdi, dtype=np.float32)
    sample = values[water.astype(bool)] if water is not None else values
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float(floor)
    return float(max(floor, np.percentile(sample, percentile)))


def screen_tile(
    bands: Mapping[str, np.ndarray],
    *,
    fdi_threshold: float | None = None,
    fdi_percentile: float | None = 99.9,
    ndwi_threshold: float | None = None,
    min_candidate_pixels: int | None = None,
    min_region_pixels: int = 8,
    scl: np.ndarray | None = None,
    compute_all_indices: bool = False,
) -> ScreenResult:
    """Decide whether a tile deserves a detector pass.

    Args:
        bands: Reflectance arrays keyed by Sentinel-2 band name (B02, B03, ...).
            Values are expected in the 0 to 1 reflectance range.
        fdi_threshold: Absolute FDI cutoff. When given, it overrides the adaptive
            threshold entirely. Defaults to settings when ``fdi_percentile`` is None.
        fdi_percentile: Percentile of this tile's water FDI used as the cutoff.
            Set to None to use a fixed absolute threshold instead. Adaptive is the
            default because a fixed cutoff does not transfer between scenes, see the
            module docstring for the measurements behind that.
        ndwi_threshold: NDWI cutoff separating water from land. Defaults to settings.
        min_candidate_pixels: Minimum total candidate pixels. Defaults to settings.
        min_region_pixels: Minimum size of a single connected candidate region for
            the tile to be accepted. Spatial coherence is what separates a debris
            patch from scattered glint noise.
        scl: Optional Sentinel-2 Scene Classification Layer for cloud masking.
        compute_all_indices: Compute every registered index rather than only those
            the screen needs. Useful when the caller wants them for figures.

    Returns:
        A ScreenResult. When rejected, ``regions`` is empty and ``reason`` explains why.

    Raises:
        ValueError: If no bands are supplied.
    """
    if not bands:
        raise ValueError("screen_tile requires at least one band array")

    ndwi_threshold = settings.ndwi_water_threshold if ndwi_threshold is None else ndwi_threshold
    min_pixels = (
        settings.min_candidate_pixels if min_candidate_pixels is None else min_candidate_pixels
    )

    # SCL travels alongside the reflectance bands but is a classification raster,
    # not reflectance, so it must not be fed to the index arithmetic.
    reflectance = {k: v for k, v in bands.items() if k != "SCL"}
    if scl is None:
        scl = bands.get("SCL")

    indices = compute_indices(reflectance) if compute_all_indices else {}

    threshold = fdi_threshold
    if threshold is None:
        if fdi_percentile is None:
            threshold = settings.fdi_threshold
        else:
            try:
                fdi_arr = (indices or compute_indices(reflectance, ["FDI", "FDI_B04", "NDWI"])).get(
                    "FDI"
                )
                if fdi_arr is None:
                    fdi_arr = compute_indices(reflectance, ["FDI_B04"]).get("FDI_B04")

                # Clouds must be excluded BEFORE the percentile is taken, not just
                # from the final mask. Cloud tops have very large FDI, so leaving
                # them in the sample drags the cutoff upward and desensitises the
                # screen on exactly the cloudy scenes that need it most. Measured on
                # the Accra chip: including cloud gave a threshold of 0.3206 and only
                # 83 candidate pixels, against 0.0131 over clean water.
                sample_mask = water_mask(reflectance, ndwi_threshold=ndwi_threshold)
                if scl is not None:
                    sample_mask = sample_mask & ~cloud_mask_from_scl(scl)

                threshold = (
                    adaptive_fdi_threshold(fdi_arr, sample_mask, percentile=fdi_percentile)
                    if fdi_arr is not None
                    else settings.fdi_threshold
                )
            except (KeyError, ValueError):
                threshold = settings.fdi_threshold

    try:
        mask = debris_candidate_mask(
            reflectance,
            fdi_threshold=threshold,
            ndwi_threshold=ndwi_threshold,
            scl=scl,
        )
    except (KeyError, ValueError) as exc:
        # Missing bands mean the screen cannot run. Accepting the tile is the safe
        # failure mode: the detector still gets a look rather than the tile being
        # silently dropped because of an incomplete band set.
        return ScreenResult(
            accepted=True,
            indices=indices,
            threshold=float(threshold),
            reason=f"screen unavailable ({exc}), tile passed through",
        )

    count = int(np.count_nonzero(mask))
    if count < min_pixels:
        return ScreenResult(
            accepted=False,
            candidate_pixels=count,
            indices=indices,
            mask=mask,
            threshold=float(threshold),
            reason=f"{count} candidate pixels below minimum {min_pixels}",
        )

    regions = candidate_regions(mask, min_pixels=min_region_pixels)
    if not regions:
        return ScreenResult(
            accepted=False,
            candidate_pixels=count,
            indices=indices,
            mask=mask,
            threshold=float(threshold),
            reason=(
                f"{count} candidate pixels but no connected region of at least "
                f"{min_region_pixels} px, consistent with glint or sensor noise"
            ),
        )

    return ScreenResult(
        accepted=True,
        candidate_pixels=count,
        regions=regions,
        indices=indices,
        mask=mask,
        threshold=float(threshold),
    )


def summarize_screening(results: Sequence[ScreenResult]) -> dict[str, float | int]:
    """Aggregate screening outcomes into reportable statistics.

    The saved-work figure is the number worth surfacing: it is the fraction of
    detector calls the cascade avoided, which is the entire justification for the
    cascade existing.
    """
    total = len(results)
    if total == 0:
        return {
            "tiles_total": 0,
            "tiles_accepted": 0,
            "tiles_skipped": 0,
            "accept_rate": 0.0,
            "work_avoided": 0.0,
            "candidate_pixels": 0,
        }
    accepted = sum(1 for r in results if r.accepted)
    return {
        "tiles_total": total,
        "tiles_accepted": accepted,
        "tiles_skipped": total - accepted,
        "accept_rate": round(accepted / total, 4),
        "work_avoided": round(1.0 - accepted / total, 4),
        "candidate_pixels": int(sum(r.candidate_pixels for r in results)),
    }
