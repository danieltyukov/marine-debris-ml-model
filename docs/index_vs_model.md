# Physical indices against the classifier, Bonaire 2025

How does a single physical index with a single cut compare with the learned classifier,
pixel by pixel, on real Caribbean passes? On every Sentinel-2 pass of Bonaire's January
to August 2025 season (the same 64 passes, surf zones, land mask and cloud handling as the
2.0 season run), this compares what the classifier flags as sargassum with what the
Floating Algae Index (FAI), the Floating Debris Index (FDI) and NDVI flag at a cut chosen
on MARIDA.

These numbers come from the 2.0 run, before the mangrove mask, so the Lac Bay mangrove
canopy is on the water side here. That is what makes the comparison informative about
canopy, and it is one of the analyses that showed the Lac Bay flags were mangrove
([lac_bay_mangroves.md](lac_bay_mangroves.md)). `scripts/eval_index_vs_model.py` reruns
it, with the mask (the default) or without it (`--no-mangrove-mask`, which reproduces
these tables).

## In short

1. **At 10 m the two are nested, not rivals.** In Lac Bay 99% of the classifier's
   sargassum flags are also over the FAI cut, but only 1.2% of FAI's flags are classifier
   flags: FAI flags 85 times as much (297,424 against 3,508 pixel looks), and the
   pixel-level Jaccard is 0.012. On the other six windward segments it is 93% and 1.1%.
   NDVI gives the same picture; FDI at its MARIDA cut is worse.
2. **What the indices add in Lac Bay is rooted vegetation.** 98% of FAI's Lac Bay flags
   fall on pixels FAI flags on at least half of the passes that see them: 10,003 pixels,
   about 1 km2, the mangrove canopy the land mask left on the water side. The classifier
   calls 62% of FAI-only flags clear (p below 0.056) and puts the rest in its uncertain
   band.
3. **Where there is no canopy the indices false-alarm and the classifier does not.** On
   the Kralendijk leeward control FAI flags 4.9, NDVI 14 and FDI 19 per 1,000 visible
   water pixels; the classifier flags none, and calls 99% of FAI's flags there clear.
4. **The classifier's own Lac Bay flags sit on the inner edge of that canopy.** 96% of
   them are on pixels FAI flags on most passes and 99% within one pixel (10 m) of them,
   although those pixels are 24% of Lac Bay's visible water (27% with the one-pixel ring).
5. **The January peak is the classifier's alone.** Per 1,000 visible Lac Bay pixels the
   classifier flags 8.4 in January and 0.65 from April to June. FAI flags 226 and 258,
   NDVI 233 and 255, FDI 285 and 372: flat or rising. That fits the mangrove-edge reading:
   the canopy is always there, and the classifier's edge flags depend on the atmosphere.
6. **Scale.** Compared on coarser blocks, agreement in Lac Bay rises from a Jaccard of
   0.01 at 10 m to 0.38 at 1 km for FAI (0.43 for NDVI); on the rest of the windward coast
   it stays at or below 0.04. At 1 km the index and the classifier agree that something is
   going on in Lac Bay; at 10 m they disagree about which pixel it is.

Read as a pipeline, the index is a near-lossless prescreen (it already contains 93% to
100% of the classifier's flags), repeat passes say which of its flags never move, and the
classifier keeps about 1% of what the index flags. Where the chain runs out is the
mangrove edge: a 10 m pixel there is part canopy and part water, and nothing at this
resolution separates it from sargassum caught in the roots.

## The rules

All as `>=`, the sign `mdebris.eval.ablation.binary_metrics` scores with.

| rule | cut | source | MARIDA test precision | recall | F1 (95% interval) |
|---|---|---|---|---|---|
| classifier | calibrated p >= 0.90 | season run edge, `calibration.json` | 0.989 | 0.907 | see `calibration.md` |
| FAI | >= 0.0443 | `band_ablation.json`, sargassum row | 0.864 | 0.562 | 0.681 (0.49 to 0.79) |
| FDI | >= 0.0240 | `band_ablation.json`, sargassum row | 0.015 | 0.981 | 0.029 (0.01 to 0.08) |
| NDVI | >= 0.2375 | chosen here on MARIDA validation, same protocol | 0.928 | 0.915 | 0.921 (0.85 to 0.96) |

The FAI and FDI cuts are the sargassum rows of [`band_ablation.json`](band_ablation.json),
chosen for best F1 on MARIDA validation and scored on test. The repository has no NDVI
rule, so one was chosen with the same protocol. In [`band_ablation.md`](band_ablation.md)
the FDI row's 0.015 is that rule's precision; its cut is 0.0240.

## What each rule flags, whole season

Flagged pixel looks over all 64 passes, with the rate per 1,000 visible water pixel looks
in brackets. A pixel look is one visible, non-cloud water pixel on one pass; group totals
add the segments' rows, so pixels where two surf zones overlap count in both.

| group | visible water pixel looks | classifier | FAI | NDVI | FDI |
|---|---|---|---|---|---|
| Lac Bay | 1,207,148 | 3,508 (2.9) | 297,424 (246) | 298,925 (248) | 409,482 (339) |
| other six windward | 9,969,003 | 649 (0.065) | 55,805 (5.6) | 86,680 (8.7) | 394,778 (40) |
| Kralendijk (leeward control) | 487,038 | 0 | 2,374 (4.9) | 6,770 (14) | 9,310 (19) |

Overlap with the classifier on the same passes:

| group | index | classifier flags also flagged by the index | index flags also flagged by the classifier | Jaccard | index flags on pixels stationary for that index | index-only flags the classifier called clear |
|---|---|---|---|---|---|---|
| Lac Bay | FAI | 99% | 1.2% | 0.012 | 98% | 62% |
| Lac Bay | NDVI | 98% | 1.1% | 0.011 | 98% | 62% |
| Lac Bay | FDI | 100% | 0.9% | 0.009 | 89% | 72% |
| other six windward | FAI | 93% | 1.1% | 0.011 | 60% | 80% |
| other six windward | NDVI | 93% | 0.7% | 0.007 | 30% | 86% |
| other six windward | FDI | 100% | 0.2% | 0.002 | 52% | 96% |
| Kralendijk (leeward control) | FAI | - | 0% | 0 | 0% | 99% |
| Kralendijk (leeward control) | NDVI | - | 0% | 0 | 0% | 98% |
| Kralendijk (leeward control) | FDI | - | 0% | 0 | 5% | 100% |

Distinct pixels ever flagged, with the number stationary for that rule (flagged on at
least half of the at least five passes that saw them) in brackets:

| group | classifier | FAI | NDVI | FDI |
|---|---|---|---|---|
| Lac Bay | 1,167 (25) | 13,468 (10,003) | 14,006 (10,077) | 27,608 (13,211) |
| other six windward | 375 (5) | 15,947 (1,502) | 47,780 (1,269) | 81,137 (9,284) |
| Kralendijk (leeward control) | 0 | 2,092 (1) | 5,424 (0) | 6,535 (55) |

The classifier column reproduces the 2.0 season run exactly. On the other six windward
segments, 1,253 of FAI's 1,502 stationary pixels are in Cai to Boka Washikemba, next to
Lac Bay, where the mangrove mask also applies.

## Lac Bay month by month

Flags per 1,000 visible water pixel looks, on segment-passes that were not blind.
"Transient" counts only flags on pixels that are not stationary for that rule.

| month | passes | visible pixel looks | classifier | FAI | NDVI | FDI | classifier, transient | FAI, transient | NDVI, transient | FDI, transient |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-01 | 6 | 195,715 | 8.4 | 226 | 233 | 285 | 7.9 | 4.7 | 9.5 | 17 |
| 2025-02 | 4 | 135,716 | 4.2 | 214 | 214 | 299 | 3.7 | 2.2 | 2.3 | 28 |
| 2025-03 | 6 | 167,098 | 2.5 | 234 | 234 | 318 | 2.2 | 1.7 | 1.3 | 32 |
| 2025-04 | 2 | 43,409 | 0.60 | 240 | 239 | 314 | 0.58 | 3.3 | 1.9 | 24 |
| 2025-05 | 5 | 123,066 | 0.67 | 255 | 249 | 363 | 0.54 | 9.3 | 3.3 | 56 |
| 2025-06 | 6 | 140,434 | 0.66 | 265 | 266 | 398 | 0.52 | 3.9 | 2.5 | 72 |
| 2025-07 | 5 | 157,487 | 1.2 | 287 | 287 | 378 | 0.76 | 3.5 | 2.1 | 27 |
| 2025-08 | 7 | 210,664 | 1.9 | 266 | 278 | 342 | 1.7 | 1.5 | 12 | 12 |

No index rule shows an April to June rise in Lac Bay that stands clear of its own noise.
FAI and NDVI flag the canopy at the same rate all season, and their transient flags stay
at a few per thousand. FAI's highest transient month is May (9.3), and nothing here can
tell that apart from canopy pixels that fall just short of the stationary rule.

On the clear 17 June pass the classifier flags 56 Lac Bay pixels, all also FAI flags; FAI
flags 9,837, NDVI 9,950 and FDI 12,332 there, nearly all canopy. 11 of the 56 are more
than a pixel from the canopy, among them the small open-lagoon patch described in
[lac_bay_mangroves.md](lac_bay_mangroves.md#two-candidates-not-confirmed). Over April to
June, 35 of the classifier's 201 Lac Bay flags are away from the canopy, against 2 of
1,730 in January.

## What the flagged pixels look like

| pixels | n | NDVI median (10th to 90th percentile) | FAI median (10th to 90th percentile) |
|---|---|---|---|
| classifier flags, Lac Bay | 3,508 | 0.68 (0.33 to 0.88) | 0.160 (0.067 to 0.274) |
| classifier flags, other six windward | 643 | 0.64 (0.31 to 0.87) | 0.144 (0.054 to 0.248) |
| MARIDA, Dense Sargassum (test) | 760 | 0.49 (0.41 to 0.57) | 0.092 (0.058 to 0.122) |
| MARIDA, Sparse Sargassum (test) | 881 | 0.38 (0.20 to 0.47) | 0.030 (0.010 to 0.053) |

More than half of the classifier's Bonaire flags are greener than 90% of MARIDA's dense
sargassum on each index. On 16 January, the 9,107 Lac Bay pixels only FAI flags have
median B08 0.348 and NDVI 0.88, a canopy spectrum. MARIDA has no mangrove or land
vegetation class, so the classifier is outside its training data there; its sargassum
probability on those pixels stays below 0.90.

## Agreement on coarser blocks

Jaccard between the classifier's flags and each index's flags when both 10 m flag maps are
compared on k by k pixel blocks, pass by pass (a block counts as flagged if any pixel in
it is):

| group | index | 10 m | 30 m | 100 m | 300 m | 1 km |
|---|---|---|---|---|---|---|
| Lac Bay | FAI | 0.01 | 0.04 | 0.12 | 0.21 | 0.38 |
| Lac Bay | NDVI | 0.01 | 0.04 | 0.13 | 0.22 | 0.43 |
| Lac Bay | FDI | 0.01 | 0.03 | 0.08 | 0.14 | 0.30 |
| other six windward | FAI | 0.01 | 0.02 | 0.03 | 0.03 | 0.04 |
| other six windward | NDVI | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 |
| other six windward | FDI | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 |

This coarsens the comparison, not the sensor. A 300 m or 1 km sensor records the block's
mixed reflectance, so the cut would change and the classifier would need retraining; this
is not a simulation of OLCI or MODIS. It shows that the disagreement in Lac Bay is about
which pixel, not which place, and that on the open coast the index flags are somewhere
else altogether (ships, cloud edges, dark-water noise).

## Caveats

- No pixel has been checked on the ground. Agreement between two rules is not accuracy.
- The nesting is partly built in: FAI, FDI and NDVI are 3 of the classifier's 18 input
  features, so a classifier flag being index-positive is expected. The informative
  direction is the other one, the 99% of index flags the classifier rejects.
- The cuts were chosen on MARIDA, open water off Central America and Hispaniola with no
  scene from the ABC islands. There are no Bonaire labels to tune a cut on, which is the
  same gap the classifier has.
- NDVI is a ratio, and over Bonaire's dark open water red and near-infrared are both close
  to zero: on 16 January, 1,493 of the 1,833 pixels only NDVI flagged on Spelonk to Boka
  Onima, and 450 of 514 on the Kralendijk control, had B08 below 0.01. The NDVI cut scores
  F1 0.921 on MARIDA, so this is a transfer failure, not a bad cut on its home data.
- FDI at its MARIDA cut is a poor sargassum rule to begin with (F1 0.029 on MARIDA). It is
  here because the band ablation reports it.
- An Aruba run of the same comparison was stopped after 31 of 63 passes when the machine
  was overloaded; nothing about Aruba is reported here.
