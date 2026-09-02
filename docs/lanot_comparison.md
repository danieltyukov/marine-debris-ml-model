# Where this classifier and LANOT's published operator disagree

Produced by `python scripts/eval_lanot_operator.py`.

## Read this first

**This is not a score of the LANOT platform, and the table below should not be
quoted as one.** Expression (1) is the per-pixel gate at the front of their
pipeline. Everything that reaches a published LANOT polygon has been through
segmentation, entropy filtering and threshold denoising afterwards, none of which
is run here, and all of which exists precisely to clear up isolated per-pixel
false positives. Scoring the gate alone measures the gate alone.

The comparison is here for one reason: the two detectors fail on different pixels,
and the union is better than either. That is the result, not the ranking.

## The systems

The LANOT platform (Instituto de Geografia UNAM) is the operational Sargassum
monitoring system for the Mexican Caribbean, Belize, Guatemala and Honduras. Its
detection rule is published in full and is not a learned model:

> (b8A < 0.07) and (b04 < 0.1) and (b11 < 0.05) and (b04 < b8A) and (b04 < b08)

Arellano-Verdejo, J., Lazcano-Hernandez, H.E., Prado Molina, J. et al. *Towards
enhanced Sargassum monitoring in the Caribbean Sea.* Sci Rep 15, 8965 (2025).
<https://doi.org/10.1038/s41598-025-93001-9>

The first three terms reject bright surfaces: cloud, land and sun glint are bright
somewhere in 665-1610 nm and open water is nearly black at 1610 nm. The last two
are the detection, requiring near-infrared above red, the red-edge signature of
floating vegetation.

## Why these pixels

The paper names the 18 Sentinel-2 tiles the platform processes. Four of MARIDA's
seventeen sites are on that list: 16PCC (Motagua, Guatemala), 16PDC (Ulua,
Honduras), 16PEC (La Ceiba, Honduras) and 16QED (Roatan, Honduras).

Restricted to those tiles, MARIDA's held-out test split has 190,431 labelled
pixels, 1,641 of them annotated Dense or Sparse Sargassum. That is
every sargassum pixel in the test split: on this benchmark, sargassum occurs only
inside LANOT's footprint. The classifier is scored on scenes it never trained on,
and the operator, which has no training, is scored on the same pixels.

## The result: they miss different things

| | sargassum pixels |
|---|---|
| found by both | 715 |
| found only by this classifier | 872 |
| found only by LANOT expression (1) | 41 |
| **found by either** | **1,628** |
| missed by both | 13 |

Union recall is 0.992. Only 13 of
1,641 annotated sargassum pixels are invisible to both methods, so
the floor on this benchmark is far lower than either detector reaches alone. A
hand-calibrated physical rule and a gradient-boosted classifier trained on different
evidence keep some genuinely independent signal from each other.

They make identical calls on 98.6% of all labelled pixels, and
291 false positives are shared. What those shared errors are:

| actually | pixels |
|---|---|
| Clouds | 260 |
| Shallow Water | 13 |
| Marine Water | 12 |
| Cloud Shadows | 6 |

## The cloud that neither one rejects

Cloud near-edge error is the failure mode both systems have, and the interesting
part is that it is a specific physical population rather than a tuning problem.

The `b11 < 0.05` term is a good gate. Of the 31,798 cloud pixels on these
tiles it rejects 61%, while keeping
99.9% of the sargassum. Thick cloud is bright at
1610 nm and sargassum mats are not, so the separation is real.

It does not help with the cloud that actually causes false positives. All
390 of the 390 cloud pixels this
classifier calls sargassum already pass the gate, which is why adding the term to
the learned model changes nothing at all: the two rows are identical in the table
below. Those pixels are optically thin cloud, with a median B11 of
0.0406 against 0.0612 for cloud
generally. LANOT's own cloud false positives sit in the same population, median B11
0.0392.

Thin cloud over dark water is dim in SWIR and lifts red-edge reflectance, so it
looks like floating vegetation to a physical rule and to a learned classifier alike.
No threshold on B11 separates it, because it is not bright. That makes it a distinct
target for a mask rather than a parameter to retune, and MARIDA labels those exact
pixels, so any candidate cloud mask can be tested against them directly.

## Operating points

Per-pixel, on the tiles and split described above. The first row is a front gate
with no post-processing behind it and the others are a trained classifier, so the
rows are not like for like and the F1 column is not a ranking.

| detector | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| LANOT expression (1) | 0.294 | 0.461 | 0.359 | 756 | 1,816 | 885 |
| this classifier, argmax | 0.759 | 0.967 | 0.850 | 1,587 | 505 | 54 |
| this classifier, best F1 | 0.987 | 0.913 | 0.948 | 1,498 | 20 | 143 |
| this classifier + b11 gate, argmax | 0.759 | 0.967 | 0.850 | 1,587 | 505 | 54 |
| this classifier + b11 gate, best F1 | 0.987 | 0.913 | 0.948 | 1,498 | 20 | 143 |

## What each one gets wrong

### LANOT expression (1)

False alarms, by what the pixel actually was:

| actually | pixels |
|---|---|
| Clouds | 1,348 |
| Shallow Water | 202 |
| Marine Water | 110 |
| Cloud Shadows | 76 |
| Waves | 45 |
| Ship | 23 |
| Mixed Water | 5 |
| Natural Organic Material | 4 |

Sargassum pixels missed, by annotated class:

| annotated | pixels |
|---|---|
| Dense Sargassum | 733 |
| Sparse Sargassum | 152 |

### this classifier, argmax

False alarms, by what the pixel actually was:

| actually | pixels |
|---|---|
| Clouds | 390 |
| Turbid Water | 47 |
| Marine Water | 45 |
| Shallow Water | 13 |
| Cloud Shadows | 6 |
| Mixed Water | 2 |
| Wakes | 2 |

Sargassum pixels missed, by annotated class:

| annotated | pixels |
|---|---|
| Sparse Sargassum | 48 |
| Dense Sargassum | 6 |

### this classifier + b11 gate, argmax

False alarms, by what the pixel actually was:

| actually | pixels |
|---|---|
| Clouds | 390 |
| Turbid Water | 47 |
| Marine Water | 45 |
| Shallow Water | 13 |
| Cloud Shadows | 6 |
| Mixed Water | 2 |
| Wakes | 2 |

Sargassum pixels missed, by annotated class:

| annotated | pixels |
|---|---|
| Sparse Sargassum | 48 |
| Dense Sargassum | 6 |

## Per tile

| tile | pixels | sargassum | LANOT F1 | classifier F1 |
|---|---|---|---|---|
| 16PCC | 112,549 | 821 | 0.155 | 0.762 |
| 16PDC | 44,430 | 67 | 0.195 | 0.591 |
| 16PEC | 25,838 | 0 | 0.000 | 0.000 |
| 16QED | 7,614 | 753 | 0.767 | 0.993 |

## What this does not say

Repeating the first section because it is the part that is easiest to misread: the
LANOT row is their per-pixel gate without the segmentation, entropy filtering and
denoising that follow it in their pipeline. It is not their platform and it is not
their published product.

Expression (1) was calibrated by photointerpretation for the optical properties of
the Mexican Caribbean. These four tiles are Guatemala and Honduras: the same
platform footprint, adjacent water, not the water it was tuned on. The classifier,
by contrast, was trained on MARIDA's own training split, so it has seen this
dataset's annotation conventions and the operator has not.

MARIDA annotates a deliberately confuser-rich subset of each scene rather than whole
scenes. Precision on these pixels is pessimistic for both detectors compared with an
operational scene that is mostly plain water.

The reflectance is not the reflectance the rule was tuned on. MARIDA is ACOLITE
output on Level-1C, Rayleigh-corrected reflectance from dark spectrum fitting (Kikaki
et al. 2022). The LANOT pipeline runs Sen2Cor to Level-2A, and expression (1) was
calibrated on that. Both are unitless 0-1 reflectance, so the thresholds apply
without rescaling, but the Rayleigh-corrected product still carries the aerosol term,
largest in the blue and smallest in the SWIR. The `b11 < 0.05` term is the least
affected of the five and the `b04 < 0.1` term the most. A fairer test of the rule
would run it on Sen2Cor reflectance for the same pixels, which MARIDA does not ship.

MARIDA's labels are annotations by remote-sensing researchers, not field
observations. Nothing here is validated against sargassum that anyone touched.

