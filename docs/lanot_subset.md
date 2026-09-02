# MARIDA pixels for testing a cloud, shadow or shallow-water mask inside LANOT's footprint

Produced by `python scripts/package_lanot_subset.py`. The data is `lanot_subset.csv.gz`
next to this file, one row per pixel, gzip-compressed CSV.

## What it is

The LANOT sargassum platform (Instituto de Geografia, UNAM) reports three sources
of false detections in operation: isolated pixels along the edges of thin cloud,
cloud shadows, and shallow water near the coast where the bottom shows through.
MARIDA (Kikaki et al. 2022) annotates all three as classes, and four of its
seventeen sites are on the 18 Sentinel-2 tiles the platform processes. This file is
every MARIDA pixel on those four tiles annotated as one of those three confusers or
as sargassum, so a candidate mask can be scored on both halves of its job: reject
the confusers, keep the sargassum.

| | |
|---|---|
| pixels | 120,239 |
| patches | 280 |
| acquisition dates | 24 (2016-09-04 to 2020-11-15) |
| tiles | 16PCC, 16PDC, 16PEC, 16QED |
| splits | train, val, test |

## Baseline: what expression (1) does on each class

Expression (1) of Arellano-Verdejo et al. 2025 is the platform's per-pixel detection
rule, `(b8A < 0.07) and (b04 < 0.1) and (b11 < 0.05) and (b04 < b8A) and (b04 < b08)`.
For the sargassum rows the `fires` column is recall, and higher is better. For the
confuser rows it is the leak a mask would have to close, and lower is better. The
`b11 < 0.05` column is that one term alone.

| class | pixels | patches | expression (1) fires | passes b11 < 0.05 | median B11 |
|---|---|---|---|---|---|
| Dense Sargassum | 2,793 | 48 | 152 (5.4%) | 2,791 (99.9%) | 0.0150 |
| Sparse Sargassum | 2,136 | 99 | 1,725 (80.8%) | 2,136 (100.0%) | 0.0060 |
| Clouds | 91,365 | 138 | 2,308 (2.5%) | 29,081 (31.8%) | 0.0716 |
| Cloud Shadows | 8,042 | 58 | 111 (1.4%) | 8,042 (100.0%) | 0.0090 |
| Shallow Water | 15,903 | 44 | 386 (2.4%) | 15,856 (99.7%) | 0.0156 |

The cloud the platform actually sees as noise is the thin kind, and the SWIR term
does not remove it because it is not bright. On these rows:

| thin cloud (Clouds with B11 < 0.05) | |
|---|---|
| pixels | 29,081 (31.8% of all cloud) |
| expression (1) fires on | 2,308 (7.9%) |
| median B11 | 0.0308 |

Those are the rows a thin-cloud mask is for. `df[(df.label == 'Clouds') & df.b11_gate]`
selects them.

## Which term does the work

Fraction of each class that passes each term of expression (1) on its own. A
confuser is rejected by whichever term has the lowest number in its row; a
sargassum class is lost to it.

| class | b8A < 0.07 | b04 < 0.1 | b11 < 0.05 | b04 < b8A | b04 < b08 | median B8A |
|---|---|---|---|---|---|---|
| Dense Sargassum | 5.4% | 100.0% | 99.9% | 100.0% | 100.0% | 0.135 |
| Sparse Sargassum | 85.7% | 100.0% | 100.0% | 98.0% | 96.9% | 0.043 |
| Clouds | 39.7% | 59.3% | 31.8% | 69.9% | 15.6% | 0.081 |
| Cloud Shadows | 100.0% | 100.0% | 100.0% | 26.6% | 1.7% | 0.015 |
| Shallow Water | 99.8% | 100.0% | 99.7% | 29.8% | 2.5% | 0.023 |

Two things follow. Cloud shadow and shallow water are rejected by the red-edge
terms, `b04 < b08` above all, not by the SWIR gate, which nearly all of them pass.
And what MARIDA annotates as Dense Sargassum is lost almost entirely to the
`b8A < 0.07` ceiling: those mats are bright in the near-infrared, well above the
cap, while passing the other four terms. Some of that gap is the processor, since
Rayleigh-corrected reflectance sits above Sen2Cor's in the NIR, but not a doubling.
Whether the ceiling is a deliberate choice against land and foam, or whether dense
mats in Sen2Cor L2A really do stay under it, is a question for the people who
calibrated the rule.

## Per split

All three of MARIDA's published splits are included and tagged. A mask has no
training split to keep clean, so use everything. A learned model scored on these
rows should keep to `test`; the classifier in this repository was trained on `train`.

| split | Dense Sargassum | Sparse Sargassum | Clouds | Cloud Shadows | Shallow Water |
|---|---|---|---|---|---|
| train | 866 | 870 | 45,440 | 3,143 | 12,809 |
| val | 1,167 | 385 | 14,127 | 1,511 | 606 |
| test | 760 | 881 | 31,798 | 3,388 | 2,488 |

## Per tile

| tile | site | patches | Dense Sargassum | Sparse Sargassum | Clouds | Cloud Shadows | Shallow Water |
|---|---|---|---|---|---|---|---|
| 16PCC | Motagua, Guatemala | 128 | 2,048 | 574 | 62,082 | 3,585 | 3,960 |
| 16PDC | Ulua, Honduras | 50 | 49 | 226 | 13,507 | 883 | 2,251 |
| 16PEC | La Ceiba, Honduras | 75 | 222 | 645 | 11,678 | 1,733 | 3,782 |
| 16QED | Roatan, Honduras | 27 | 474 | 691 | 4,098 | 1,841 | 5,910 |

## Columns

| column | meaning |
|---|---|
| `roi` | MARIDA patch id, `d-m-yy_TILE_index`, as in the split files |
| `split` | train, val or test, MARIDA's published split |
| `tile` | Sentinel-2 MGRS tile |
| `date` | acquisition date, ISO |
| `row` | pixel row inside the 256x256 patch, 0 at the top |
| `col` | pixel column inside the patch, 0 at the left |
| `epsg` | EPSG code of the patch CRS (UTM zone 16N here) |
| `easting` | pixel-centre easting in that CRS, metres |
| `northing` | pixel-centre northing in that CRS, metres |
| `lon` | pixel-centre longitude, WGS84 |
| `lat` | pixel-centre latitude, WGS84 |
| `label` | MARIDA class name |
| `confidence` | annotator confidence, High / Moderate / Low |
| `B01 .. B12` | ACOLITE Rayleigh-corrected reflectance, 0-1, the 11 MARIDA bands |
| `lanot_expr1` | True where expression (1) of Arellano-Verdejo et al. 2025 fires |
| `b11_gate` | True where B11 < 0.05, the SWIR term of expression (1) on its own |

Coordinates are pixel centres. `row` and `col` index the 256x256 MARIDA patch named
in `roi`, so any row can be traced back to the original GeoTIFF.

## What the reflectance is, and why it matters here

MARIDA is ACOLITE output on Level-1C: Rayleigh-corrected reflectance from the dark
spectrum fitting processor, with the 20 m and 60 m bands replicated to 10 m. It is
not Sen2Cor Level-2A, which is what the LANOT pipeline runs and what expression (1)
was calibrated on. Both are unitless 0-1 reflectance, but the Rayleigh-corrected
product still carries the aerosol term, which is largest in the blue and smallest in
the SWIR. So the `lanot_expr1` column is the published rule applied to reflectance it
was not tuned for. It is a baseline for these classes, not a measurement of the
platform, which segments and filters the pixel mask before anything is published.

The same fact cuts the other way: LANOT are evaluating ACOLITE for the platform, and
these rows are already ACOLITE reflectance with labels attached.

## Scoring a candidate mask

```python
import pandas as pd

df = pd.read_csv("lanot_subset.csv.gz")

# A mask is a boolean per row, True where the pixel is kept for detection.
# The SWIR term of expression (1) is the trivial baseline:
keep = df["B11"] < 0.05

positive = df["label"].isin(["Dense Sargassum", "Sparse Sargassum"])
print("sargassum kept:", f"{keep[positive].mean():.3f}")
for name in ["Clouds", "Cloud Shadows", "Shallow Water"]:
    rejected = 1 - keep[df["label"] == name].mean()
    print(f"{name} rejected: {rejected:.3f}")
```

A mask that uses more than the 11 bands, for instance the spatial context around a
cloud edge, needs the patches themselves. `roi`, `row` and `col` locate every pixel
in the MARIDA GeoTIFFs, which are on Zenodo.

## Licence and citation

The pixels are MARIDA, CC-BY-4.0. Cite the dataset, not this repository:

> Kikaki K, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K (2022) MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. PLoS ONE 17(1): e0262247. https://doi.org/10.1371/journal.pone.0262247

The detection rule is expression (1) of Arellano-Verdejo, J., Lazcano-Hernandez, H.E.,
Prado Molina, J. et al. *Towards enhanced Sargassum monitoring in the Caribbean Sea.*
Sci Rep 15, 8965 (2025). <https://doi.org/10.1038/s41598-025-93001-9>

MARIDA's labels are annotations by remote-sensing researchers, not field
observations, and MARIDA annotates a confuser-rich selection of each scene rather
than whole scenes. Class proportions here say nothing about how common each
confuser is in an operational scene.
