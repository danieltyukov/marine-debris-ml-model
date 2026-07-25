# Marine Debris Detection: Modernization Design

Date: 2026-07-25
Status: approved scope, implementation in progress

## 1. Why this rewrite

The repository as archived is not merely dated, it is unrunnable. Every claim below
was verified on the target machine, not assumed.

| Problem | Evidence |
|---|---|
| TensorFlow 1.14 has no wheel for Python 3.7+ | Target machine runs Python 3.12; install is impossible |
| 19 MB of vendored TF Object Detection API | `du -sh object_detection_api` = 19M; only 2 of its files are local additions |
| Deprecated NumPy aliases | `np.int` at `inference_utils/tf_od_predict_image_aug_to_geo_corrected.py:107`, removed in NumPy 1.24 |
| Commercial-only data path | Planet API requires a paid key; no free fallback |
| Single-class taxonomy | `configs/marine_debris.pbtxt` defines exactly one class, `marine_debris` |
| No tests, no CI, no package | Repo is a set of loose scripts to be copied into a vendored tree |
| Training required before any use | 500k steps on GPU before a single prediction |

### Silent correctness bugs found in the legacy inference script

These are worth recording because they affect published results, not just style.

1. **Single-detection tiles are silently dropped.** `np.squeeze` on a one-row box array
   yields a 1-D array; `bboxes_256.tolist()` then yields floats, `bbox[1]` raises
   `TypeError`, and the bare `except TypeError: continue` discards the whole tile.
   Any tile with exactly one detection contributed nothing to the output.
2. **Hardcoded output path.** Results are written to `./marine_litter/data_geo/{scene}.geojson`,
   a directory the repo never creates. The script crashes at the last step unless that
   path happens to exist.
3. **The filename promises augmentation that does not happen.** The script is named
   `..._image_aug_...` and defines `darken_img()`, but the call site is commented out
   (line 78). No test-time augmentation is performed.

## 2. Verified platform facts

Measured on this machine, 22 cores, 62 GB RAM, **no GPU**.

| Fact | Value |
|---|---|
| PyTorch | 2.13.0+cpu |
| transformers | 5.14.1 (major version 5) |
| OWLv2 `google/owlv2-base-patch16-ensemble` | loads, 155.0M params |
| OWLv2 CPU forward | 18.25 s fp32, 14.36 s int8-dynamic (only 1.27x, not worth it) |
| OWLv2 internal input size | always `(1, 3, 960, 960)` regardless of input |
| `Sam2Model` / `Sam2Processor` | native in transformers 5.14.1, no extra package |
| `RTDetrV2ForObjectDetection` | available, Apache-2.0 |
| `GroundingDinoForObjectDetection` | available |
| `Dinov3Model` | NOT in transformers 5.14.1 |
| Planetary Computer STAC | anonymous search works, 135 collections, full B01-B12 + SCL |
| Element84 earth-search | reachable, returned 0 items for the Accra test bbox |
| CDSE STAC | reachable but serves `s3://eodata/...` hrefs needing CDSE credentials |

End-to-end data path proven with no credentials:
`S2A_MSIL2A_20240527T100601_R022_T30NZM`, 12.2% cloud, windowed COG read,
FDI/NDVI/NDWI computed over real pixels.

## 3. Architecture

### 3.1 The core idea: a cost cascade

OWLv2 costs ~18 s per 960x960 tile on CPU. A full Sentinel-2 scene is 10,980x10,980 px,
which is 130 tiles at 960 px, or roughly 40 minutes of pure transformer time. Most of
those tiles are land, cloud, or empty deep water and cannot contain floating debris.

Spectral indices are pure NumPy arithmetic costing microseconds per tile. So:

```
scene
  -> windowed COG read (bytes only for the AOI)
  -> water mask (NDWI) + cloud mask (SCL band)
  -> spectral indices (FDI, FAI, NDVI, PI, kNDVI)   [microseconds/tile]
  -> candidate tiles only
       -> OWLv2 open-vocabulary detection            [~18 s/tile, few tiles]
       -> SAM2 mask refinement on accepted boxes
  -> geo-registration -> GeoJSON
```

This is how operational remote-sensing pipelines are actually built, and it is what makes
the project usable on a laptop with no GPU.

### 3.2 Tiling at 960 px

Because OWLv2 resizes everything to 960x960, tile size is a model-derived constant rather
than an arbitrary choice. Measured on this machine, in megapixels of *source imagery*
processed per second:

| tile | batch | s/tile | MP/s of source |
|---|---|---|---|
| 512 | 1 | 19.44 | 0.013 |
| 960 | 1 | 18.46 | **0.050** |
| 960 | 2 | 16.96 | 0.054 |
| 960 | 4 | 18.45 | 0.050 |

Two conclusions. Tiling at 960 instead of 512 gives a **3.8x throughput gain for free**,
because a 512 px tile pays the full 960x960 compute anyway. And **batching is worthless
here**: 22 CPU cores are already saturated by a single forward pass, so batch 2 gains 8%
and batch 4 gains nothing. The pipeline therefore streams tiles one at a time.

### 3.2.1 Why the cascade is not optional

A full Sentinel-2 tile is 10,980 x 10,980 px, about 120 megapixels. At the measured
0.050 MP/s that is roughly **40 minutes of transformer time per scene** on CPU. Screening
with spectral indices first, where typically only a small percentage of tiles are
simultaneously water, cloud-free and high-FDI, cuts that to minutes. The cascade is what
makes the project usable without a GPU.

### 3.3 Package layout

```
src/mdebris/
├── config.py            pydantic-settings, env-driven, no hardcoded paths
├── geo/
│   ├── tiles.py         slippy XYZ <-> lat/lon (ported from legacy), 960px windowing
│   ├── georef.py        pixel bbox -> geographic polygon, GeoJSON FeatureCollection
│   └── raster.py        windowed COG reads, band stacking, 20m -> 10m resampling
├── data/
│   ├── stac.py          Planetary Computer (default) + Element84 fallback
│   ├── marida.py        MARIDA benchmark loader
│   ├── planet.py        optional Planet connector, legacy parity
│   └── sample.py        small real chips shipped in-repo for offline tests
├── indices/spectral.py  FDI, FAI, NDVI, NDWI, PI, kNDVI + water/cloud masking
├── models/
│   ├── base.py          Detector protocol, Detection dataclass
│   ├── zeroshot.py      OWLv2 open-vocabulary (default, no training)
│   ├── segment.py       SAM2 box-prompted mask refinement
│   ├── supervised.py    RT-DETRv2 fine-tune + inference (Apache-2.0)
│   └── cascade.py       index prescreen -> model routing -> fusion
├── pipeline/scene.py    scene -> detections orchestration
├── eval/metrics.py      mAP, precision/recall/F1, confusion matrix @IoU
├── viz/figures.py       every README figure, reproducibly generated
├── api/app.py           FastAPI service
└── cli.py               typer CLI
```

Each module is independently testable and depends only on layers below it.

### 3.4 Spectral indices

FDI is the load-bearing one (Biermann et al. 2020, *Nature Scientific Reports*). It
detects floating material by measuring how far the NIR reflectance sits above a baseline
interpolated between RED and SWIR1:

```
NIR' = R + (SWIR1 - R) * ((833 - 665) / (1610 - 665)) * 10
FDI  = NIR - NIR'
```

Sentinel-2 band mapping: RED = B04 (665 nm), NIR = B08 (833 nm), SWIR1 = B11 (1610 nm).
B11 is 20 m and must be resampled to the 10 m grid before the arithmetic.

Companion indices: NDWI for water masking, NDVI to separate vegetation, FAI for algae,
PI for plastic, kNDVI as a nonlinear vegetation contrast. Distinguishing debris from
*Sargassum* and foam is the central scientific difficulty and needs several indices, not one.

### 3.5 Taxonomy

The single `marine_debris` class is replaced by a prompt set covering the confusers that
actually cause false positives: floating plastic debris, sargassum/algae mat, ship/vessel,
ship wake, sea foam, cloud, sediment plume, open water. Open-vocabulary detection makes
this a configuration change rather than a retraining job.

### 3.6 What is preserved from the original

The geo-referencing math is the genuinely valuable part and is ported forward, cleaned:

- tile filename `{x}-{y}-{z}` parsing
- `mercantile.bounds(x, y, z)` -> (west, south, east, north)
- affine transform `(width/N, 0, west, 0, -height/N, north)`
- normalized `[ymin, xmin, ymax, xmax]` -> pixel -> `[xmin, ymin, xmax, ymax]` -> polygon
- GeoJSON FeatureCollection output with tile, class, score properties

with the single-detection bug fixed, the hardcoded path removed, tile size no longer
hardcoded to 256, and an explicit CRS.

Evaluation keeps parity with the legacy metrics (confusion matrix, precision, recall, F1,
mAP at 0.5 IoU) so old and new numbers remain comparable.

## 4. Data strategy

| Purpose | Source | Auth |
|---|---|---|
| Default live imagery | Planetary Computer STAC, Sentinel-2 L2A | none (verified) |
| Fallback | Element84 earth-search v1 | none |
| Benchmark/eval | MARIDA (Sentinel-2 marine debris, 11+ classes) | none |
| Offline tests + demo | small real chips committed to repo | none |
| Optional commercial | Planet, legacy parity | `PL_API_KEY` env |

No credentials are ever committed. The legacy code was already clean here: all keys come
from environment variables.

## 5. Deliverables

1. Modern PyTorch package, `pip install -e .`, Python 3.11+
2. Zero-shot detection working with no training and no API keys
3. Supervised RT-DETRv2 path for when labels and a GPU exist
4. Free STAC data connectors, verified
5. Spectral index engine with cited formulas
6. Evaluation suite with legacy metric parity
7. FastAPI service + typer CLI
8. Test suite + GitHub Actions CI
9. All README figures regenerated from real data by a committed script
10. Architecture diagram and a published interactive demo page

## 6. Explicitly out of scope

- Claude/LLM analysis layer (user decision: vision models only)
- Retraining a detector from scratch
- Any paid data dependency in the default path
