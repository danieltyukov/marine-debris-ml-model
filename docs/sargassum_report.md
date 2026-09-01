# Sargassum detection on MARIDA

Produced by `python scripts/eval_sargassum.py`. No retraining happens here:
the classifier fitted by `scripts/train_marida.py` is loaded and re-scored on
MARIDA's own scene-grouped held-out test split, so these numbers and the ones
in `marida_report.md` come from the same model.

## Why this is reported separately

The headline result in this repository is a negative one about *marine debris*:
at 10 m ground sampling a debris filament is a few low-contrast pixels, and the
best F1 is 0.515. Sargassum is a different physical target. It floats in mats
tens of metres across and carries a chlorophyll red edge, so it fills pixels and
the spectral indices key on it directly. Same model, same 18 features, same split.

## Per class

| class | precision | recall | F1 | test pixels |
|---|---|---|---|---|
| Dense Sargassum | 0.946 | 0.920 | **0.933** | 760 |
| Sparse Sargassum | 0.586 | 0.904 | **0.711** | 881 |

## Any sargassum vs everything else

The operational question is not which density it is, it is whether there is any. 1,641 of 194,863 held-out pixels are sargassum of either class.

| operating point | precision | recall | F1 |
|---|---|---|---|
| `argmax` default | 0.757 | 0.967 | 0.849 |
| best F1, threshold 0.951 | 0.987 | 0.913 | **0.948** |
| 90% precision, threshold 0.736 | 0.900 | 0.947 | 0.923 |

The last row is the one that matters for dispatch. Sending a crew to a clean
beach costs a shift, so precision is the constraint and recall is whatever can
be had under it.

## What it gets wrong

Sargassum pixels the model missed, by what it called them instead:

| called | pixels |
|---|---|
| Natural Organic Material | 40 |
| Mixed Water | 11 |
| Marine Debris | 3 |

False alarms, by what they actually were:

| actually | pixels |
|---|---|
| Clouds | 390 |
| Turbid Water | 47 |
| Marine Water | 45 |
| Shallow Water | 13 |
| Cloud Shadows | 6 |
| Natural Organic Material | 4 |
| Mixed Water | 2 |
| Wakes | 2 |

## What this does not say

MARIDA's sargassum labels are Sentinel-2 pixels annotated by remote-sensing
researchers, not field observations. A per-pixel score on a benchmark is not a
validated landfall forecast, and nothing here measures whether material detected
offshore reaches a particular beach. Those are separate claims and this file
supports neither of them.

What it does support, which an earlier version of this file got wrong: these
pixels are in the right ocean. Every sargassum pixel in the held-out split falls
on 16PCC, 16PDC, 16PEC or 16QED, all four of which are among the 18 Sentinel-2
tiles the LANOT platform processes operationally for the Mexican Caribbean, Belize,
Guatemala and Honduras. None are on the Mexican stretch of that footprint. See
[`lanot_comparison.md`](lanot_comparison.md).
