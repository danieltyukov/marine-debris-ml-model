# Which bands carry the signal: a sensor ablation

Produced by `python scripts/eval_band_ablation.py`. The same gradient-boosting
classifier is fitted on MARIDA with only the bands a given sensor has, the
rest set to missing, and an index that needs a missing band is dropped with
it. Thresholds are chosen on the validation split and applied to the test
split, so every row is scored on pixels its threshold never saw.

Train 429,412 pixels, validation 213,102, test 194,863 in 359 patches. Test positives: 1,641 sargassum, 381 debris. Intervals are 95% percentile
intervals from a cluster bootstrap over test patches, because pixels in one
patch share a scene and an annotator and are not independent.

## Learned model per band set

| scenario | stands for | features | sargassum P | R | F1 (95% CI) | debris P | R | F1 (95% CI) | fit s |
|---|---|---|---|---|---|---|---|---|---|
| `sentinel2` | all 11 Sentinel-2 bands | 18 | 0.952 | 0.928 | **0.940** (0.87 to 0.98) | 0.414 | 0.667 | **0.511** (0.27 to 0.78) | 8 |
| `no_swir` | no shortwave infrared | 13 | 0.853 | 0.956 | **0.901** (0.81 to 0.95) | 0.280 | 0.640 | **0.389** (0.29 to 0.50) | 8 |
| `no_red_edge` | Landsat-like, no red edge | 13 | 0.738 | 0.948 | **0.830** (0.61 to 0.96) | 0.683 | 0.538 | **0.602** (0.52 to 0.67) | 8 |
| `superdove` | PlanetScope 8-band, nearest S2 bands | 10 | 0.880 | 0.951 | **0.914** (0.83 to 0.96) | 0.296 | 0.588 | **0.394** (0.30 to 0.49) | 10 |
| `dove4` | PlanetScope 4-band, the NASA IMPACT imagery | 8 | 0.922 | 0.934 | **0.928** (0.86 to 0.96) | 0.054 | 0.152 | **0.080** (0.04 to 0.15) | 18 |
| `rgb` | visible only | 3 | 0.395 | 0.528 | **0.452** (0.25 to 0.66) | 0.012 | 0.207 | **0.023** (0.01 to 0.05) | 8 |
| `indices_only` | the seven physical indices, no raw band | 7 | 0.933 | 0.955 | **0.944** (0.89 to 0.97) | 0.105 | 0.690 | **0.182** (0.06 to 0.77) | 8 |

## One physical index, one threshold, no learning

| index | sargassum P | R | F1 (95% CI) | cut | debris P | R | F1 (95% CI) | cut |
|---|---|---|---|---|---|---|---|---|
| `FDI >` | 0.015 | 0.981 | **0.029** (0.01 to 0.08) | 0.0240 | 0.003 | 0.803 | **0.006** (0.00 to 0.01) | 0.0222 |
| `FAI >` | 0.864 | 0.562 | **0.681** (0.49 to 0.79) | 0.0443 | 0.009 | 0.630 | **0.017** (0.01 to 0.05) | 0.0005 |

## Protocol note

The full-band sargassum F1 here is 0.940. The 0.948 in
`sargassum_report.md` is the same model at the threshold that maximises F1 on
the test split itself, which this protocol reproduces as 0.948 and does not report as a
result. Choosing a threshold on the pixels you then score is optimistic by
construction; the gap between the two numbers is the size of that optimism.

## Reading it

A scenario close to the full band set means the score does not depend on the
bands it lacks, and a resolution comparison against a sensor with that band set
is a resolution comparison. A scenario far below it means the comparison would
be confounded and has to be reported as bands plus resolution.

The physical rows are the other end of the axis this repository keeps returning
to: a single index with a single cut is what most of the operational sargassum
literature uses, and the learned rows show what the extra bands buy over it at
10 m, on this benchmark, for each target.
