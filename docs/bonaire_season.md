# Bonaire: one season of Sentinel-2 passes

Produced by `python scripts/run_bonaire_season.py`. 64 passes over tile 19PEP (relative
orbit R082) from 2025-01-01 to 2025-08-31, 18 from Sentinel-2A, 24 from Sentinel-2B, 22
from Sentinel-2C. The search leaves out granules reported as 100% cloud, and keeps one
granule per datatake whose footprint covers at least 99% of every surf zone. Segments
are cut from the OpenStreetMap coastline at named landmarks
(`assets/islands/bonaire/segments.geojson`); the surf zone is the water within 500 m of
each. Land is removed with the OpenStreetMap coastline polygons plus a 15 m seaward
strip, and mapped mangrove and other vegetated wetland is removed with it, plus a 20 m
strip (`assets/islands/bonaire/mangroves.geojson`). A pixel is called sargassum at
calibrated probability >= 0.90, which after calibration means the model's own estimate
is at least 90%; it is uncertain in [0.056, 0.90) and clear below. The lower edge keeps
98% of true sargassum on MARIDA's validation split, and `docs/calibration.md` records
what the upper edge achieves on the test split.

## How often each stretch of coast could be seen

| segment | exposure | passes | observed | partial | blind | usable | longest gap, days | median gap | longest wait for a fully clear pass | passes with sargassum | most front affected, m | pixels ever flagged | of which stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Willemstoren to Sorobon | windward | 64 | 31 | 15 | 18 | 72% | 15 | 5.0 | 25 days (27 Mar to 21 Apr) | 1 | 1605 | 12 | 0 |
| Lac Bay | windward | 64 | 15 | 26 | 23 | 64% | 20 | 5.0 | 97 days (12 Mar to 17 Jun) | 8 | 1511 | 204 | 1 |
| Cai to Boka Washikemba | windward | 64 | 33 | 13 | 18 | 72% | 15 | 5.0 | 20 days (21 Apr to 11 May) | 4 | 1192 | 69 | 0 |
| Boka Washikemba to Spelonk | windward | 64 | 34 | 11 | 19 | 70% | 15 | 5.0 | 35 days (17 Mar to 21 Apr) | 3 | 1178 | 41 | 0 |
| Spelonk to Boka Onima | windward | 64 | 32 | 15 | 17 | 73% | 15 | 5.0 | 35 days (17 Mar to 21 Apr) | 2 | 1121 | 24 | 0 |
| Boka Onima to Playa Chikitu | windward | 64 | 34 | 13 | 17 | 73% | 15 | 5.0 | 30 days (7 Jun to 7 Jul) | 0 | 0 | 5 | 0 |
| Playa Chikitu to Boka Kokolishi | windward | 64 | 38 | 9 | 17 | 73% | 15 | 5.0 | 18 days (29 Mar to 16 Apr) | 0 | 0 | 13 | 0 |
| Kralendijk waterfront | leeward | 64 | 14 | 22 | 28 | 56% | 20 | 5.0 | 75 days (12 Mar to 26 May) | 0 | 0 | 0 | 0 |

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
| Willemstoren to Sorobon | 45,396 | 47 | 0 | 12 | 0 |
| Lac Bay | 30,584 | 8,752 | 2849 | 204 | 151 |
| Cai to Boka Washikemba | 54,766 | 1,256 | 147 | 69 | 18 |
| Boka Washikemba to Spelonk | 27,581 | 7 | 0 | 41 | 0 |
| Spelonk to Boka Onima | 75,758 | 0 | 0 | 24 | 0 |
| Boka Onima to Playa Chikitu | 27,137 | 0 | 0 | 5 | 0 |
| Playa Chikitu to Boka Kokolishi | 22,273 | 0 | 0 | 13 | 0 |
| Kralendijk waterfront | 17,593 | 0 | 0 | 0 | 0 |

Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones
overlap counts in both. `assets/bonaire_persistence.png` shows where the
flags sit.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-01 | Sentinel-2B | 14 | 20 | 6 | 7 | 1 | 0 | 19 | Boka Washikemba to Spelonk |
| 2025-01-06 | Sentinel-2A | 28 | 20 | 9 | 3 | 5 | 0 | 2 | - |
| 2025-01-11 | Sentinel-2B | 9 | 8 | 3 | 7 | 1 | 0 | 69 | Lac Bay |
| 2025-01-16 | Sentinel-2A | 13 | 4 | 2 | 8 | 0 | 0 | 73 | Lac Bay, Cai to Boka Washikemba, Boka Washikemba to Spelonk |
| 2025-01-21 | Sentinel-2B | 32 | 27 | 2 | 4 | 3 | 1 | 21 | - |
| 2025-01-26 | Sentinel-2C | 8 | 4 | 4 | 8 | 0 | 0 | 40 | - |
| 2025-01-31 | Sentinel-2B | 21 | 19 | 13 | 6 | 1 | 1 | 4 | - |
| 2025-02-05 | Sentinel-2C | 30 | 9 | 11 | 7 | 1 | 0 | 11 | - |
| 2025-02-15 | Sentinel-2C | 43 | 11 | 2 | 7 | 0 | 1 | 16 | Lac Bay |
| 2025-02-20 | Sentinel-2B | 1 | 3 | 2 | 8 | 0 | 0 | 18 | - |
| 2025-02-25 | Sentinel-2C | 14 | 8 | 10 | 7 | 1 | 0 | 23 | Lac Bay, Cai to Boka Washikemba |
| 2025-03-02 | Sentinel-2B | 5 | 5 | 7 | 7 | 1 | 0 | 12 | - |
| 2025-03-07 | Sentinel-2C | 22 | 29 | 17 | 2 | 5 | 1 | 7 | - |
| 2025-03-12 | Sentinel-2B | 2 | 3 | 7 | 8 | 0 | 0 | 4 | - |
| 2025-03-17 | Sentinel-2C | 20 | 21 | 4 | 5 | 2 | 1 | 14 | - |
| 2025-03-19 | Sentinel-2A | 69 | 44 | 13 | 1 | 6 | 1 | 20 | Spelonk to Boka Onima |
| 2025-03-22 | Sentinel-2B | 41 | 88 | 4 | 0 | 2 | 6 | 0 | - |
| 2025-03-27 | Sentinel-2C | 34 | 52 | 10 | 3 | 2 | 3 | 0 | - |
| 2025-03-29 | Sentinel-2A | 72 | 91 | 9 | 1 | 0 | 7 | 0 | - |
| 2025-04-01 | Sentinel-2B | 34 | 30 | 8 | 1 | 7 | 0 | 0 | - |
| 2025-04-06 | Sentinel-2C | 55 | 85 | 30 | 1 | 1 | 6 | 0 | - |
| 2025-04-08 | Sentinel-2A | 97 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-04-11 | Sentinel-2B | 75 | 85 | 2 | 0 | 1 | 7 | 0 | - |
| 2025-04-16 | Sentinel-2C | 47 | 37 | 7 | 3 | 3 | 2 | 0 | - |
| 2025-04-21 | Sentinel-2B | 19 | 6 | 7 | 6 | 2 | 0 | 18 | - |
| 2025-04-26 | Sentinel-2C | 100 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-04-28 | Sentinel-2A | 99 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-05-01 | Sentinel-2B | 60 | 100 | 45 | 0 | 0 | 8 | 0 | - |
| 2025-05-06 | Sentinel-2C | 27 | 26 | 49 | 3 | 4 | 1 | 0 | - |
| 2025-05-08 | Sentinel-2A | 57 | 68 | 21 | 1 | 2 | 5 | 2 | - |
| 2025-05-11 | Sentinel-2B | 20 | 7 | 7 | 6 | 2 | 0 | 39 | Lac Bay, Cai to Boka Washikemba |
| 2025-05-16 | Sentinel-2C | 72 | 27 | 7 | 4 | 2 | 2 | 0 | - |
| 2025-05-18 | Sentinel-2A | 97 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-05-21 | Sentinel-2B | 25 | 23 | 5 | 4 | 3 | 1 | 0 | - |
| 2025-05-26 | Sentinel-2C | 16 | 6 | 18 | 7 | 1 | 0 | 0 | - |
| 2025-05-28 | Sentinel-2A | 92 | 82 | 51 | 0 | 3 | 5 | 0 | - |
| 2025-05-31 | Sentinel-2B | 95 | 97 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-06-05 | Sentinel-2C | 14 | 20 | 10 | 5 | 2 | 1 | 6 | - |
| 2025-06-07 | Sentinel-2A | 22 | 16 | 7 | 5 | 3 | 0 | 8 | - |
| 2025-06-10 | Sentinel-2B | 76 | 86 | 4 | 0 | 1 | 7 | 0 | - |
| 2025-06-15 | Sentinel-2C | 40 | 66 | 21 | 0 | 4 | 4 | 0 | - |
| 2025-06-17 | Sentinel-2A | 10 | 12 | 8 | 7 | 1 | 0 | 27 | Lac Bay, Cai to Boka Washikemba |
| 2025-06-20 | Sentinel-2B | 32 | 38 | 5 | 2 | 6 | 0 | 0 | - |
| 2025-06-25 | Sentinel-2C | 99 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-06-27 | Sentinel-2A | 13 | 47 | 11 | 0 | 8 | 0 | 0 | - |
| 2025-06-30 | Sentinel-2B | 55 | 51 | 7 | 0 | 7 | 1 | 0 | - |
| 2025-07-05 | Sentinel-2C | 31 | 19 | 10 | 5 | 3 | 0 | 5 | - |
| 2025-07-07 | Sentinel-2A | 16 | 34 | 3 | 4 | 2 | 2 | 14 | Spelonk to Boka Onima |
| 2025-07-10 | Sentinel-2B | 14 | 9 | 5 | 6 | 2 | 0 | 16 | Lac Bay |
| 2025-07-15 | Sentinel-2C | 11 | 17 | 8 | 6 | 1 | 1 | 2 | - |
| 2025-07-17 | Sentinel-2A | 7 | 7 | 23 | 6 | 2 | 0 | 11 | - |
| 2025-07-20 | Sentinel-2B | 97 | 100 | 2 | 0 | 0 | 8 | 0 | - |
| 2025-07-25 | Sentinel-2C | 19 | 59 | 6 | 2 | 2 | 4 | 0 | - |
| 2025-07-27 | Sentinel-2A | 46 | 90 | 4 | 0 | 1 | 7 | 0 | - |
| 2025-07-30 | Sentinel-2B | 6 | 10 | 6 | 6 | 2 | 0 | 8 | - |
| 2025-08-04 | Sentinel-2C | 10 | 6 | 14 | 7 | 1 | 0 | 0 | - |
| 2025-08-06 | Sentinel-2A | 14 | 5 | 15 | 7 | 1 | 0 | 4 | - |
| 2025-08-09 | Sentinel-2B | 11 | 8 | 8 | 6 | 2 | 0 | 1 | - |
| 2025-08-14 | Sentinel-2C | 15 | 36 | 9 | 3 | 4 | 1 | 2 | - |
| 2025-08-16 | Sentinel-2A | 17 | 26 | 11 | 4 | 3 | 1 | 95 | Willemstoren to Sorobon, Lac Bay, Boka Washikemba to Spelonk |
| 2025-08-19 | Sentinel-2B | 89 | 100 | 11 | 0 | 0 | 8 | 0 | - |
| 2025-08-24 | Sentinel-2C | 21 | 4 | 14 | 7 | 1 | 0 | 0 | - |
| 2025-08-26 | Sentinel-2A | 54 | 73 | 3 | 2 | 1 | 5 | 0 | - |
| 2025-08-29 | Sentinel-2B | 22 | 9 | 11 | 6 | 2 | 0 | 1 | - |

## What this does not say

No pixel here has been checked against the beach. The classifier was trained and
calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and
around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is
from Bonaire. Reef flats, sand bottoms, lagoons and wrecks here are surfaces
the model has never been graded on. The observability columns do not depend on
the classifier at all and are the part of this report to trust first: they say
how often an optical satellite could look at each stretch of coast, which is a
property of the weather and the orbit, not of the model.

Kralendijk waterfront is a leeward control. Sargassum arrives on the windward
side; if that row fills with detections, the model is finding something other
than sargassum.
