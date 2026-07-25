"""HTTP API.

Detector instances are cached per model name because loading OWLv2 costs several
seconds and roughly 600 MB, which is not something to repeat per request.

Detection is CPU-bound and slow (tens of seconds per tile), so the handlers run it in
a worker thread. Doing that work inline would block the event loop and stall every
other request, including the health check.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from mdebris import __version__
from mdebris.api.schemas import (
    DetectionModel,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    IndexStatsResponse,
    SampleInfo,
    SceneModel,
    SearchRequest,
)
from mdebris.config import settings

log = logging.getLogger(__name__)

app = FastAPI(
    title="mdebris",
    version=__version__,
    summary="Marine debris detection on free Sentinel-2 imagery.",
    description=(
        "Open-vocabulary detection with OWLv2, spectral index screening and "
        "geo-registered GeoJSON output. No credentials required."
    ),
)


@lru_cache(maxsize=4)
def _detector(name: str):
    """Load and cache a detector. Cached because weights cost seconds and megabytes."""
    from mdebris.models import get_detector

    return get_detector(name)


def _to_detection_models(detections) -> list[DetectionModel]:
    out: list[DetectionModel] = []
    for d in detections:
        lon = lat = None
        if d.geometry is not None:
            lon, lat = d.geometry.centroid.coords[0]
        out.append(
            DetectionModel(
                label=d.label,
                score=round(float(d.score), 6),
                bbox_pixel=tuple(round(v, 2) for v in d.bbox.as_xyxy()),  # type: ignore[arg-type]
                lon=lon,
                lat=lat,
                area_m2=d.area_m2,
                indices={k: round(float(v), 6) for k, v in d.indices.items()},
                source_model=d.source_model,
            )
        )
    return out


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness check plus the resolved compute device."""
    from mdebris.data import list_samples

    return HealthResponse(
        version=__version__, device=settings.resolve_device(), samples=list_samples()
    )


@app.get("/samples", response_model=list[SampleInfo], tags=["data"])
def samples() -> list[SampleInfo]:
    """The Sentinel-2 chips bundled with the package, usable with no network."""
    from mdebris.data import list_samples, load_sample

    infos: list[SampleInfo] = []
    for name in list_samples():
        _, meta = load_sample(name)
        infos.append(
            SampleInfo(
                name=name,
                scene_id=meta.get("scene_id", ""),
                datetime=meta.get("datetime"),
                width=int(meta.get("width", 0)),
                height=int(meta.get("height", 0)),
                cloud_cover=meta.get("cloud_cover"),
                description=meta.get("description", ""),
                bands=list(meta.get("bands", [])),
            )
        )
    return infos


@app.get("/indices/{sample}", response_model=IndexStatsResponse, tags=["data"])
def indices(sample: str) -> IndexStatsResponse:
    """Spectral index statistics over a bundled sample."""
    import numpy as np

    from mdebris.data import SampleError, sample_reflectance
    from mdebris.indices.masks import water_mask
    from mdebris.indices.spectral import compute_indices

    try:
        bands, _ = sample_reflectance(sample)
    except (SampleError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=f"unknown sample {sample!r}") from exc

    refl = {k: v for k, v in bands.items() if k != "SCL"}
    values = compute_indices(refl)
    stats: dict[str, dict[str, float]] = {}
    for name, arr in values.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        stats[name] = {
            "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)),
            "p99": float(np.percentile(finite, 99)),
            "p99_9": float(np.percentile(finite, 99.9)),
        }
    return IndexStatsResponse(
        sample=sample, water_fraction=float(water_mask(refl).mean()), indices=stats
    )


@app.post("/search", tags=["data"])
def search(req: SearchRequest) -> list[SceneModel]:
    """Search free Sentinel-2 imagery."""
    from mdebris.data import StacError, search_scenes
    from mdebris.types import GeoBBox

    west, south, east, north = req.bbox
    try:
        scenes = search_scenes(
            GeoBBox(west=west, south=south, east=east, north=north),
            req.start,
            req.end,
            max_cloud=req.max_cloud,
            limit=req.limit,
        )
    except StacError as exc:
        # A provider outage is upstream's fault, not the caller's.
        raise HTTPException(status_code=502, detail=f"imagery search failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return [SceneModel(**vars(s)) for s in scenes]


def _run_detection(req: DetectRequest) -> dict[str, Any]:
    """Synchronous detection body, executed in a worker thread."""
    import rasterio.transform as rt

    from mdebris.data import sample_reflectance
    from mdebris.pipeline import detect_in_arrays
    from mdebris.types import SceneRef

    detector = _detector(req.model)

    if req.sample is not None:
        bands, meta = sample_reflectance(req.sample)
        transform = rt.Affine(*meta["transform"])
        crs = meta.get("crs")
        scene = SceneRef(
            scene_id=meta.get("scene_id", req.sample),
            datetime=meta.get("datetime"),
            cloud_cover=meta.get("cloud_cover"),
        )
    else:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        from mdebris.data import StacClient, reflectance_params_for_item, scale_bands
        from mdebris.geo.raster import read_bands
        from mdebris.types import GeoBBox

        west, south, east, north = req.bbox  # type: ignore[misc]
        client = StacClient()
        items = client.search_items(
            GeoBBox(west=west, south=south, east=east, north=north),
            req.start,  # type: ignore[arg-type]
            req.end,  # type: ignore[arg-type]
            max_cloud=req.max_cloud,
            limit=1,
        )
        if not items:
            raise LookupError("no scene matched that area, date range and cloud limit")
        item = items[0]
        scene = SceneRef(
            scene_id=item.id,
            datetime=str(item.datetime),
            cloud_cover=item.properties.get("eo:cloud_cover"),
        )
        hrefs = client.asset_hrefs(item, ["B02", "B03", "B04", "B06", "B08", "B11", "SCL"])
        with rasterio.open(hrefs["B04"]) as src:
            crs = src.crs
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, west, south, east, north
            )
            window = from_bounds(left, bottom, right, top, src.transform)
            transform = src.window_transform(window)
        scale, offset = reflectance_params_for_item(item)
        bands = scale_bands(read_bands(hrefs, window, reference="B04"), scale=scale, offset=offset)

    result = detect_in_arrays(
        bands,
        detector,
        transform=transform,
        crs=crs,
        scene=scene,
        threshold=req.threshold,
        use_cascade=req.cascade,
    )
    detections = result.detections.targets_only() if req.targets_only else result.detections
    return {
        "detections": _to_detection_models(detections.detections),
        "geojson": detections.to_geojson(),
        "scene": SceneModel(**vars(scene)),
        "screening": {k: float(v) for k, v in result.screening.items()},
        "timings": result.timings,
        "count": len(detections),
    }


@app.post("/detect", response_model=DetectResponse, tags=["detection"])
async def detect(req: DetectRequest) -> DetectResponse:
    """Detect debris in a bundled sample or in live Sentinel-2 imagery.

    This is slow on CPU, tens of seconds per tile. The work runs in a thread so it
    does not block the event loop.
    """
    try:
        req.validate_selection()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        payload = await anyio.to_thread.run_sync(_run_detection, req)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=f"sample not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("detection failed")
        raise HTTPException(status_code=500, detail=f"detection failed: {exc}") from exc

    return DetectResponse(**payload)


@app.get("/detect/{sample}", response_model=DetectResponse, tags=["detection"])
async def detect_sample(
    sample: str,
    threshold: float = Query(default=0.10, ge=0.0, le=1.0),
    targets_only: bool = False,
) -> DetectResponse:
    """Convenience GET for running the default model over a bundled sample."""
    return await detect(
        DetectRequest(sample=sample, threshold=threshold, targets_only=targets_only)
    )


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "mdebris",
            "version": __version__,
            "docs": "/docs",
            "endpoints": ["/health", "/samples", "/indices/{sample}", "/search", "/detect"],
        }
    )
