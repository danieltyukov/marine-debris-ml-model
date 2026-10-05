# Sint Maarten: one season of Sentinel-2 passes

Produced by `python scripts/run_island_season.py --island sint_maarten`. 64 passes over
tile 20QMF (relative orbit R039) from 2025-01-01 to 2025-08-31, 19 from Sentinel-2A, 23
from Sentinel-2B, 22 from Sentinel-2C. The search leaves out granules reported as 100%
cloud, and keeps one granule per datatake whose footprint covers at least 99% of every
surf zone. Segments are cut from the OpenStreetMap coastline at named landmarks
(`assets/islands/sint_maarten/segments.geojson`); the surf zone is the water within 500
m of each. Land is removed with the OpenStreetMap coastline polygons plus a 15 m seaward
strip, and mapped mangrove and other vegetated wetland is removed with it, plus a 20 m
strip (`assets/islands/sint_maarten/mangroves.geojson`). A pixel is called sargassum at
calibrated probability >= 0.90, which after calibration means the model's own estimate
is at least 90%; it is uncertain in [0.056, 0.90) and clear below. The lower edge keeps
98% of true sargassum on MARIDA's validation split, and `docs/calibration.md` records
what the upper edge achieves on the test split.

## How often each stretch of coast could be seen

| segment | exposure | passes | observed | partial | blind | usable | longest gap, days | median gap | longest wait for a fully clear pass | passes with sargassum | most front affected, m | pixels ever flagged | of which stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oyster Pond to Guana Bay Point | windward | 64 | 22 | 18 | 24 | 62% | 28 | 5.0 | 38 days (15 Apr to 23 May) | 0 | 0 | 1 | 0 |
| Guana Bay Point to Point Blanche Bay | windward | 64 | 32 | 10 | 22 | 66% | 28 | 5.0 | 28 days (15 Apr to 13 May) | 0 | 0 | 5 | 0 |
| Coralita to Orient Bay (French Saint-Martin) | windward | 64 | 13 | 24 | 27 | 58% | 30 | 5.0 | 68 days (26 Mar to 2 Jun) | 1 | 600 | 10 | 0 |
| Fort Amsterdam to Cole Bay | leeward | 64 | 19 | 23 | 22 | 66% | 28 | 5.0 | 48 days (26 Mar to 13 May) | 0 | 0 | 1 | 0 |

`usable` is observed plus partial: passes on which a detection means something.
A segment is blind on a pass when more than 70% of the water side of its surf
zone is cloud by the scene classification layer, partial above 20%. Every
non-cloud pixel on the water side is classified, surf included: there is no NDWI
water gate, because on MARIDA no dense sargassum pixel passes one. The longest
gap is the most days between two consecutive usable looks; the longest wait for a
fully clear pass is the same between two passes on which the segment was observed
in full.

## Checks on the detection columns

Flagged pixels are not confirmed sargassum. Two per-pixel counts kept across the
season name the flags that are probably something else.

- Stationary: flagged on at least 50% of the passes that could see it, with
  at least 5 such passes. Floating material moves; a reef, a wreck or a
  structure does not.
- Near persistent vegetation: on or within 20 m of a pixel that looked like leaf canopy
  (NDVI above 0.5, B08 above 0.08) on at least 90% of at least
  10 clear looks. That is canopy the mangrove map missed, or its edge. The
  check removes nothing; it only counts.

| segment | water-side pixels | removed by the mangrove mask | persistent vegetation pixels | flagged pixels | of which near persistent vegetation |
|---|---|---|---|---|---|
| Oyster Pond to Guana Bay Point | 19,976 | 0 | 0 | 1 | 0 |
| Guana Bay Point to Point Blanche Bay | 21,327 | 0 | 0 | 5 | 0 |
| Coralita to Orient Bay (French Saint-Martin) | 27,625 | 0 | 0 | 10 | 0 |
| Fort Amsterdam to Cole Bay | 24,592 | 0 | 0 | 1 | 0 |

Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones
overlap counts in both. `assets/sint_maarten_persistence.png` shows where the
flags sit.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-03 | Sentinel-2A | 30 | 99 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-01-08 | Sentinel-2B | 0 | 0 | 0 | 4 | 0 | 0 | 0 | - |
| 2025-01-13 | Sentinel-2A | 4 | 0 | 1 | 4 | 0 | 0 | 1 | - |
| 2025-01-18 | Sentinel-2B | 51 | 45 | 2 | 2 | 1 | 1 | 0 | - |
| 2025-01-28 | Sentinel-2B | 7 | 23 | 1 | 2 | 2 | 0 | 0 | - |
| 2025-02-02 | Sentinel-2C | 37 | 35 | 8 | 3 | 0 | 1 | 0 | - |
| 2025-02-07 | Sentinel-2B | 23 | 10 | 1 | 3 | 1 | 0 | 1 | - |
| 2025-02-12 | Sentinel-2C | 9 | 9 | 2 | 3 | 1 | 0 | 0 | - |
| 2025-02-17 | Sentinel-2B | 40 | 2 | 0 | 4 | 0 | 0 | 0 | - |
| 2025-02-22 | Sentinel-2C | 20 | 3 | 0 | 4 | 0 | 0 | 0 | - |
| 2025-02-27 | Sentinel-2B | 21 | 34 | 0 | 1 | 2 | 1 | 0 | - |
| 2025-03-04 | Sentinel-2C | 15 | 85 | 20 | 0 | 1 | 3 | 0 | - |
| 2025-03-09 | Sentinel-2B | 9 | 2 | 0 | 4 | 0 | 0 | 0 | - |
| 2025-03-14 | Sentinel-2C | 89 | 98 | 76 | 0 | 0 | 4 | 0 | - |
| 2025-03-16 | Sentinel-2A | 0 | 19 | 3 | 3 | 1 | 0 | 0 | - |
| 2025-03-19 | Sentinel-2B | 46 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-03-24 | Sentinel-2C | 16 | 9 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-03-26 | Sentinel-2A | 8 | 5 | 0 | 4 | 0 | 0 | 0 | - |
| 2025-03-29 | Sentinel-2B | 13 | 52 | 2 | 0 | 3 | 1 | 0 | - |
| 2025-04-03 | Sentinel-2C | 81 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-05 | Sentinel-2A | 8 | 73 | 3 | 0 | 2 | 2 | 0 | - |
| 2025-04-08 | Sentinel-2B | 53 | 100 | 3 | 0 | 0 | 4 | 0 | - |
| 2025-04-13 | Sentinel-2C | 7 | 32 | 1 | 2 | 2 | 0 | 0 | - |
| 2025-04-15 | Sentinel-2A | 8 | 29 | 0 | 2 | 2 | 0 | 0 | - |
| 2025-04-18 | Sentinel-2B | 99 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-23 | Sentinel-2C | 97 | 91 | 16 | 0 | 0 | 4 | 0 | - |
| 2025-04-25 | Sentinel-2A | 57 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-28 | Sentinel-2B | 92 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-05-03 | Sentinel-2C | 56 | 100 | 11 | 0 | 0 | 4 | 0 | - |
| 2025-05-05 | Sentinel-2A | 18 | 83 | 5 | 0 | 0 | 4 | 0 | - |
| 2025-05-08 | Sentinel-2B | 34 | 90 | 3 | 0 | 0 | 4 | 0 | - |
| 2025-05-13 | Sentinel-2C | 28 | 42 | 3 | 2 | 1 | 1 | 0 | - |
| 2025-05-15 | Sentinel-2A | 44 | 34 | 2 | 1 | 3 | 0 | 0 | - |
| 2025-05-18 | Sentinel-2B | 94 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-05-23 | Sentinel-2C | 4 | 21 | 1 | 2 | 2 | 0 | 0 | - |
| 2025-05-25 | Sentinel-2A | 8 | 46 | 10 | 1 | 3 | 0 | 0 | - |
| 2025-05-28 | Sentinel-2B | 99 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-02 | Sentinel-2C | 9 | 13 | 70 | 4 | 0 | 0 | 0 | - |
| 2025-06-04 | Sentinel-2A | 5 | 22 | 5 | 2 | 2 | 0 | 0 | - |
| 2025-06-07 | Sentinel-2B | 12 | 28 | 25 | 2 | 2 | 0 | 0 | - |
| 2025-06-12 | Sentinel-2C | 4 | 20 | 7 | 2 | 2 | 0 | 0 | - |
| 2025-06-14 | Sentinel-2A | 100 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-17 | Sentinel-2B | 78 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-22 | Sentinel-2C | 62 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-24 | Sentinel-2A | 95 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-27 | Sentinel-2B | 15 | 34 | 5 | 0 | 4 | 0 | 0 | - |
| 2025-07-02 | Sentinel-2C | 7 | 28 | 2 | 2 | 2 | 0 | 0 | - |
| 2025-07-04 | Sentinel-2A | 7 | 10 | 21 | 4 | 0 | 0 | 0 | - |
| 2025-07-07 | Sentinel-2B | 31 | 27 | 3 | 3 | 1 | 0 | 15 | Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-12 | Sentinel-2C | 38 | 73 | 57 | 0 | 2 | 2 | 0 | - |
| 2025-07-14 | Sentinel-2A | 21 | 22 | 7 | 3 | 1 | 0 | 0 | - |
| 2025-07-17 | Sentinel-2B | 17 | 76 | 21 | 0 | 1 | 3 | 0 | - |
| 2025-07-22 | Sentinel-2C | 25 | 39 | 8 | 0 | 4 | 0 | 0 | - |
| 2025-07-24 | Sentinel-2A | 35 | 28 | 4 | 0 | 4 | 0 | 0 | - |
| 2025-07-27 | Sentinel-2B | 51 | 39 | 1 | 1 | 2 | 1 | 0 | - |
| 2025-08-01 | Sentinel-2C | 36 | 37 | 3 | 1 | 3 | 0 | 0 | - |
| 2025-08-03 | Sentinel-2A | 6 | 42 | 10 | 1 | 2 | 1 | 0 | - |
| 2025-08-06 | Sentinel-2B | 9 | 28 | 2 | 2 | 2 | 0 | 0 | - |
| 2025-08-11 | Sentinel-2C | 7 | 38 | 1 | 1 | 3 | 0 | 0 | - |
| 2025-08-13 | Sentinel-2A | 12 | 30 | 11 | 2 | 2 | 0 | 0 | - |
| 2025-08-21 | Sentinel-2C | 16 | 59 | 13 | 0 | 4 | 0 | 0 | - |
| 2025-08-23 | Sentinel-2A | 34 | 93 | 79 | 0 | 0 | 4 | 0 | - |
| 2025-08-26 | Sentinel-2B | 41 | 56 | 11 | 0 | 3 | 1 | 0 | - |
| 2025-08-31 | Sentinel-2C | 6 | 52 | 5 | 1 | 2 | 1 | 0 | - |

## Granules not counted as passes

- `S2C_MSIL2A_20250622T145741_R039_T20QMF_20250622T171857`: second granule of the same datatake as `S2C_MSIL2A_20250622T145741_R039_T20QMF_20250707T090915`; its footprint covers 100% of the least-covered surf zone, the kept one 100%

## What this does not say

No pixel here has been checked against the beach. The classifier was trained and
calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and
around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is
from Sint Maarten. Reef flats, sand bottoms, lagoons and wrecks here are surfaces
the model has never been graded on. The observability columns do not depend on
the classifier at all and are the part of this report to trust first: they say
how often an optical satellite could look at each stretch of coast, which is a
property of the weather and the orbit, not of the model.

Fort Amsterdam to Cole Bay is a leeward control. Sargassum arrives on the windward
side; if that row fills with detections, the model is finding something other
than sargassum.

Saint Martin is the only one of these islands inside a swath overlap. Relative orbit R039 covers the whole island and is the run compared with the other islands. Relative orbit R139 sees the windward east coast on other days and never the leeward control; it is run on the windward segments and reported on its own. Reading the R039 numbers alone as how often this coast can be seen undercounts it.

The obvious landmark at the Oyster Pond inlet, Babit Point, is on the French headland. The Dutch windward coast starts at the south end of the line OpenStreetMap closes Oyster Pond with across the inlet (relation 9589207).

The Dutch windward coast stops at Point Blanche Bay, about 950 m from the cruise piers, so the piers stay out of a 500 m zone. The Dutch windward coast is only about 7 km long, so its two segments are shorter than any windward segment on the other islands, and their partial and blind verdicts are a little noisier.

Simpson Bay Lagoon is open to the sea at both ends, so the coastline does not close into one ring. Segments are cut on the two largest rings, and the land mask is every closed ring, islets included, so islet rock inside a windward zone is not classified as water.

Orbit R039 looks into the sun's glint over the windward sea in summer (median glint angle 10.7 degrees at the coast, under 20 degrees on 49 of 64 passes); R139 never does. On 16 mornings when both passed about 10 minutes apart, R039 saw about twice the cloud over the windward zones and flagged no pixel where R139 flagged 437 to 846 per segment. docs/islands.md has the measurements.
