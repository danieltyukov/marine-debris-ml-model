# Sint Maarten, windward, second orbit R139: one season of Sentinel-2 passes

Produced by `python scripts/run_island_season.py --island sint_maarten`. 64 passes over
tile 20QMF (relative orbit R139) from 2025-01-01 to 2025-08-31, 20 from Sentinel-2A, 22
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
| Oyster Pond to Guana Bay Point | windward | 64 | 37 | 12 | 15 | 77% | 15 | 5.0 | 20 days (30 Jan to 19 Feb) | 14 | 3408 | 579 | 0 |
| Guana Bay Point to Point Blanche Bay | windward | 64 | 35 | 16 | 13 | 80% | 13 | 5.0 | 25 days (30 Jan to 24 Feb) | 20 | 3640 | 548 | 0 |
| Coralita to Orient Bay (French Saint-Martin) | windward | 64 | 37 | 12 | 15 | 77% | 15 | 5.0 | 15 days (20 May to 4 Jun) | 23 | 4709 | 1,353 | 0 |

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
| Oyster Pond to Guana Bay Point | 19,976 | 0 | 0 | 579 | 0 |
| Guana Bay Point to Point Blanche Bay | 21,327 | 0 | 0 | 548 | 0 |
| Coralita to Orient Bay (French Saint-Martin) | 27,625 | 0 | 0 | 1353 | 0 |

Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones
overlap counts in both. `assets/sint_maarten_r139_persistence.png` shows where the
flags sit.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-05 | Sentinel-2B | 25 | 25 | 0 | 1 | 2 | 0 | 0 | - |
| 2025-01-10 | Sentinel-2A | 27 | 28 | 0 | 2 | 0 | 1 | 1 | - |
| 2025-01-15 | Sentinel-2B | 7 | 12 | 1 | 2 | 1 | 0 | 0 | - |
| 2025-01-20 | Sentinel-2A | 3 | 6 | 0 | 3 | 0 | 0 | 0 | - |
| 2025-01-25 | Sentinel-2B | 10 | 8 | 2 | 2 | 1 | 0 | 0 | - |
| 2025-01-30 | Sentinel-2C | 24 | 1 | 0 | 3 | 0 | 0 | 1 | - |
| 2025-02-09 | Sentinel-2C | 19 | 81 | 2 | 0 | 1 | 2 | 0 | - |
| 2025-02-14 | Sentinel-2B | 19 | 38 | 0 | 1 | 1 | 1 | 0 | - |
| 2025-02-19 | Sentinel-2C | 29 | 21 | 0 | 2 | 1 | 0 | 1 | - |
| 2025-02-24 | Sentinel-2B | 2 | 2 | 0 | 3 | 0 | 0 | 0 | - |
| 2025-03-01 | Sentinel-2C | 2 | 2 | 0 | 3 | 0 | 0 | 0 | - |
| 2025-03-06 | Sentinel-2B | 91 | 98 | 0 | 0 | 0 | 3 | 0 | - |
| 2025-03-11 | Sentinel-2C | 12 | 0 | 0 | 3 | 0 | 0 | 1 | - |
| 2025-03-13 | Sentinel-2A | 0 | 1 | 0 | 3 | 0 | 0 | 3 | - |
| 2025-03-16 | Sentinel-2B | 4 | 16 | 2 | 2 | 1 | 0 | 0 | - |
| 2025-03-21 | Sentinel-2C | 23 | 41 | 2 | 0 | 3 | 0 | 0 | - |
| 2025-03-23 | Sentinel-2A | 4 | 1 | 0 | 3 | 0 | 0 | 4 | - |
| 2025-03-26 | Sentinel-2B | 18 | 1 | 0 | 3 | 0 | 0 | 1 | - |
| 2025-03-31 | Sentinel-2C | 40 | 92 | 0 | 0 | 0 | 3 | 0 | - |
| 2025-04-02 | Sentinel-2A | 9 | 3 | 0 | 3 | 0 | 0 | 0 | - |
| 2025-04-05 | Sentinel-2B | 23 | 69 | 0 | 0 | 1 | 2 | 0 | - |
| 2025-04-10 | Sentinel-2C | 24 | 52 | 0 | 0 | 2 | 1 | 0 | - |
| 2025-04-12 | Sentinel-2A | 11 | 95 | 29 | 0 | 0 | 3 | 0 | - |
| 2025-04-15 | Sentinel-2B | 3 | 3 | 1 | 3 | 0 | 0 | 65 | Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-04-20 | Sentinel-2C | 45 | 88 | 3 | 0 | 0 | 3 | 0 | - |
| 2025-04-22 | Sentinel-2A | 5 | 0 | 0 | 3 | 0 | 0 | 0 | - |
| 2025-04-25 | Sentinel-2B | 32 | 1 | 0 | 3 | 0 | 0 | 82 | Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-04-30 | Sentinel-2C | 38 | 100 | 5 | 0 | 0 | 3 | 7 | - |
| 2025-05-02 | Sentinel-2A | 36 | 74 | 68 | 0 | 2 | 1 | 4 | Guana Bay Point to Point Blanche Bay |
| 2025-05-05 | Sentinel-2B | 16 | 1 | 3 | 3 | 0 | 0 | 403 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-05-10 | Sentinel-2C | 5 | 1 | 1 | 3 | 0 | 0 | 148 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-05-12 | Sentinel-2A | 37 | 54 | 3 | 0 | 3 | 0 | 0 | - |
| 2025-05-15 | Sentinel-2B | 37 | 23 | 2 | 2 | 1 | 0 | 284 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-05-20 | Sentinel-2C | 4 | 0 | 2 | 3 | 0 | 0 | 303 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-05-22 | Sentinel-2A | 20 | 62 | 8 | 0 | 2 | 1 | 37 | Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-05-25 | Sentinel-2B | 11 | 30 | 9 | 1 | 2 | 0 | 212 | Oyster Pond to Guana Bay Point, Coralita to Orient Bay (French Saint-Martin) |
| 2025-05-30 | Sentinel-2C | 93 | 41 | 8 | 1 | 2 | 0 | 0 | - |
| 2025-06-01 | Sentinel-2A | 73 | 100 | 0 | 0 | 0 | 3 | 0 | - |
| 2025-06-04 | Sentinel-2B | 8 | 17 | 7 | 2 | 1 | 0 | 265 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-06-09 | Sentinel-2C | 48 | 85 | 6 | 0 | 1 | 2 | 0 | - |
| 2025-06-11 | Sentinel-2A | 13 | 5 | 5 | 3 | 0 | 0 | 3 | - |
| 2025-06-19 | Sentinel-2C | 5 | 2 | 3 | 3 | 0 | 0 | 549 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-06-21 | Sentinel-2A | 24 | 28 | 4 | 0 | 3 | 0 | 142 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-06-24 | Sentinel-2B | 98 | 100 | 0 | 0 | 0 | 3 | 0 | - |
| 2025-06-29 | Sentinel-2C | 2 | 2 | 7 | 3 | 0 | 0 | 146 | Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-01 | Sentinel-2A | 46 | 96 | 30 | 0 | 0 | 3 | 0 | - |
| 2025-07-04 | Sentinel-2B | 3 | 4 | 46 | 3 | 0 | 0 | 0 | - |
| 2025-07-09 | Sentinel-2C | 34 | 27 | 8 | 1 | 2 | 0 | 23 | Guana Bay Point to Point Blanche Bay |
| 2025-07-11 | Sentinel-2A | 38 | 53 | 32 | 0 | 2 | 1 | 0 | - |
| 2025-07-14 | Sentinel-2B | 4 | 1 | 6 | 3 | 0 | 0 | 354 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-19 | Sentinel-2C | 15 | 11 | 4 | 3 | 0 | 0 | 187 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-21 | Sentinel-2A | 19 | 0 | 2 | 3 | 0 | 0 | 111 | Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-24 | Sentinel-2B | 27 | 14 | 5 | 2 | 1 | 0 | 26 | Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-29 | Sentinel-2C | 7 | 2 | 4 | 3 | 0 | 0 | 15 | Coralita to Orient Bay (French Saint-Martin) |
| 2025-07-31 | Sentinel-2A | 9 | 27 | 4 | 1 | 2 | 0 | 186 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-08-03 | Sentinel-2B | 7 | 17 | 5 | 2 | 1 | 0 | 32 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay |
| 2025-08-08 | Sentinel-2C | 8 | 6 | 4 | 3 | 0 | 0 | 55 | Coralita to Orient Bay (French Saint-Martin) |
| 2025-08-10 | Sentinel-2A | 5 | 11 | 7 | 3 | 0 | 0 | 254 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |
| 2025-08-13 | Sentinel-2B | 2 | 0 | 29 | 3 | 0 | 0 | 24 | Coralita to Orient Bay (French Saint-Martin) |
| 2025-08-18 | Sentinel-2C | 58 | 58 | 2 | 1 | 0 | 2 | 1 | - |
| 2025-08-20 | Sentinel-2A | 3 | 0 | 3 | 3 | 0 | 0 | 46 | Coralita to Orient Bay (French Saint-Martin) |
| 2025-08-23 | Sentinel-2B | 60 | 70 | 99 | 0 | 1 | 2 | 0 | - |
| 2025-08-28 | Sentinel-2C | 87 | 100 | 0 | 0 | 0 | 3 | 0 | - |
| 2025-08-30 | Sentinel-2A | 5 | 3 | 2 | 3 | 0 | 0 | 216 | Oyster Pond to Guana Bay Point, Guana Bay Point to Point Blanche Bay, Coralita to Orient Bay (French Saint-Martin) |

## Granules not counted as passes

- `S2A_MSIL2A_20250512T144741_R139_T20QMF_20250512T184731`: second granule of the same datatake as `S2A_MSIL2A_20250512T144741_R139_T20QMF_20250512T214317`; its footprint covers 0% of the least-covered surf zone, the kept one 100%
- `S2A_MSIL2A_20250711T144751_R139_T20QMF_20250711T184613`: second granule of the same datatake as `S2A_MSIL2A_20250711T144751_R139_T20QMF_20250711T214314`; its footprint covers 0% of the least-covered surf zone, the kept one 100%

## What this does not say

No pixel here has been checked against the beach. The classifier was trained and
calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and
around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is
from Sint Maarten. Reef flats, sand bottoms, lagoons and wrecks here are surfaces
the model has never been graded on. The observability columns do not depend on
the classifier at all and are the part of this report to trust first: they say
how often an optical satellite could look at each stretch of coast, which is a
property of the weather and the orbit, not of the model.

This run is a second look at part of the coast from another orbit. It is reported
on its own and is not mixed into tables that compare islands on one orbit each.

Saint Martin is the only one of these islands inside a swath overlap. Relative orbit R039 covers the whole island and is the run compared with the other islands. Relative orbit R139 sees the windward east coast on other days and never the leeward control; it is run on the windward segments and reported on its own. Reading the R039 numbers alone as how often this coast can be seen undercounts it.

The obvious landmark at the Oyster Pond inlet, Babit Point, is on the French headland. The Dutch windward coast starts at the south end of the line OpenStreetMap closes Oyster Pond with across the inlet (relation 9589207).

The Dutch windward coast stops at Point Blanche Bay, about 950 m from the cruise piers, so the piers stay out of a 500 m zone. The Dutch windward coast is only about 7 km long, so its two segments are shorter than any windward segment on the other islands, and their partial and blind verdicts are a little noisier.

Simpson Bay Lagoon is open to the sea at both ends, so the coastline does not close into one ring. Segments are cut on the two largest rings, and the land mask is every closed ring, islets included, so islet rock inside a windward zone is not classified as water.

Orbit R039 looks into the sun's glint over the windward sea in summer (median glint angle 10.7 degrees at the coast, under 20 degrees on 49 of 64 passes); R139 never does. On 16 mornings when both passed about 10 minutes apart, R039 saw about twice the cloud over the windward zones and flagged no pixel where R139 flagged 437 to 846 per segment. docs/islands.md has the measurements.
