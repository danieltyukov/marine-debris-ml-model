# Curaçao, east, Hato to Oostpunt: one season of Sentinel-2 passes

Produced by `python scripts/run_island_season.py --island curacao --part east`. 64
passes over tile 19PEP (relative orbit R082) from 2025-01-01 to 2025-08-31, 18 from
Sentinel-2A, 24 from Sentinel-2B, 22 from Sentinel-2C. The search leaves out granules
reported as 100% cloud, and keeps one granule per datatake whose footprint covers at
least 99% of every surf zone. Segments are cut from the OpenStreetMap coastline at named
landmarks (`assets/islands/curacao/segments.geojson`); the surf zone is the water within
500 m of each. Land is removed with the OpenStreetMap coastline polygons plus a 15 m
seaward strip, and mapped mangrove and other vegetated wetland is removed with it, plus
a 20 m strip (`assets/islands/curacao/mangroves.geojson`). A pixel is called sargassum
at calibrated probability >= 0.90, which after calibration means the model's own
estimate is at least 90%; it is uncertain in [0.056, 0.90) and clear below. The lower
edge keeps 98% of true sargassum on MARIDA's validation split, and `docs/calibration.md`
records what the upper edge achieves on the test split.

## How often each stretch of coast could be seen

| segment | exposure | passes | observed | partial | blind | usable | longest gap, days | median gap | longest wait for a fully clear pass | passes with sargassum | most front affected, m | pixels ever flagged | of which stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hato to Playa Kanoa | windward | 64 | 35 | 12 | 17 | 73% | 10 | 5.0 | 15 days (11 May to 26 May) | 1 | 559 | 28 | 0 |
| Playa Kanoa to Sint Joris Baai | windward | 64 | 35 | 12 | 17 | 73% | 15 | 5.0 | 15 days (11 May to 26 May) | 1 | 1044 | 15 | 0 |
| Sint Joris Baai | windward | 64 | 34 | 10 | 20 | 69% | 15 | 5.0 | 15 days (11 May to 26 May) | 1 | 2207 | 6 | 0 |
| Sint Joris Baai to Oostpunt | windward | 64 | 31 | 12 | 21 | 67% | 15 | 5.0 | 17 days (21 May to 7 Jun) | 0 | 0 | 6 | 0 |

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
| Hato to Playa Kanoa | 48,045 | 0 | 0 | 28 | 0 |
| Playa Kanoa to Sint Joris Baai | 48,261 | 0 | 0 | 15 | 0 |
| Sint Joris Baai | 27,225 | 2,455 | 0 | 6 | 0 |
| Sint Joris Baai to Oostpunt | 65,704 | 0 | 0 | 6 | 0 |

Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones
overlap counts in both. `assets/curacao_east_persistence.png` shows where the
flags sit.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-01 | Sentinel-2B | 14 | 27 | 7 | 1 | 3 | 0 | 0 | - |
| 2025-01-06 | Sentinel-2A | 28 | 43 | 2 | 0 | 4 | 0 | 0 | - |
| 2025-01-11 | Sentinel-2B | 9 | 7 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-01-16 | Sentinel-2A | 13 | 4 | 1 | 4 | 0 | 0 | 0 | - |
| 2025-01-21 | Sentinel-2B | 32 | 66 | 61 | 1 | 1 | 2 | 0 | - |
| 2025-01-26 | Sentinel-2C | 8 | 7 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-01-31 | Sentinel-2B | 21 | 27 | 1 | 2 | 2 | 0 | 0 | - |
| 2025-02-05 | Sentinel-2C | 30 | 5 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-02-15 | Sentinel-2C | 43 | 75 | 41 | 1 | 0 | 3 | 0 | - |
| 2025-02-20 | Sentinel-2B | 1 | 0 | 1 | 4 | 0 | 0 | 2 | - |
| 2025-02-25 | Sentinel-2C | 14 | 27 | 4 | 2 | 2 | 0 | 0 | - |
| 2025-03-02 | Sentinel-2B | 5 | 1 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-03-07 | Sentinel-2C | 22 | 5 | 1 | 4 | 0 | 0 | 0 | - |
| 2025-03-12 | Sentinel-2B | 2 | 5 | 3 | 4 | 0 | 0 | 0 | - |
| 2025-03-17 | Sentinel-2C | 20 | 32 | 1 | 3 | 0 | 1 | 4 | - |
| 2025-03-19 | Sentinel-2A | 69 | 91 | 51 | 0 | 0 | 4 | 0 | - |
| 2025-03-22 | Sentinel-2B | 41 | 56 | 2 | 1 | 2 | 1 | 0 | - |
| 2025-03-27 | Sentinel-2C | 34 | 8 | 4 | 4 | 0 | 0 | 0 | - |
| 2025-03-29 | Sentinel-2A | 72 | 9 | 10 | 3 | 1 | 0 | 0 | - |
| 2025-04-01 | Sentinel-2B | 34 | 21 | 2 | 2 | 2 | 0 | 0 | - |
| 2025-04-06 | Sentinel-2C | 55 | 9 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-04-08 | Sentinel-2A | 97 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-11 | Sentinel-2B | 75 | 100 | 12 | 0 | 0 | 4 | 0 | - |
| 2025-04-16 | Sentinel-2C | 47 | 13 | 6 | 3 | 1 | 0 | 0 | - |
| 2025-04-21 | Sentinel-2B | 19 | 6 | 3 | 4 | 0 | 0 | 0 | - |
| 2025-04-26 | Sentinel-2C | 100 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-28 | Sentinel-2A | 99 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-05-01 | Sentinel-2B | 60 | 6 | 10 | 4 | 0 | 0 | 0 | - |
| 2025-05-06 | Sentinel-2C | 27 | 36 | 47 | 0 | 4 | 0 | 0 | - |
| 2025-05-08 | Sentinel-2A | 57 | 53 | 13 | 0 | 3 | 1 | 0 | - |
| 2025-05-11 | Sentinel-2B | 20 | 3 | 2 | 4 | 0 | 0 | 5 | Sint Joris Baai |
| 2025-05-16 | Sentinel-2C | 72 | 70 | 8 | 0 | 1 | 3 | 0 | - |
| 2025-05-18 | Sentinel-2A | 97 | 96 | 1 | 0 | 0 | 4 | 0 | - |
| 2025-05-21 | Sentinel-2B | 25 | 26 | 3 | 1 | 3 | 0 | 0 | - |
| 2025-05-26 | Sentinel-2C | 16 | 27 | 32 | 3 | 1 | 0 | 0 | - |
| 2025-05-28 | Sentinel-2A | 92 | 94 | 74 | 0 | 0 | 4 | 0 | - |
| 2025-05-31 | Sentinel-2B | 95 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-05 | Sentinel-2C | 14 | 48 | 3 | 1 | 1 | 2 | 0 | - |
| 2025-06-07 | Sentinel-2A | 22 | 8 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-06-10 | Sentinel-2B | 76 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-15 | Sentinel-2C | 40 | 45 | 8 | 1 | 2 | 1 | 0 | - |
| 2025-06-17 | Sentinel-2A | 10 | 7 | 3 | 4 | 0 | 0 | 20 | Hato to Playa Kanoa |
| 2025-06-20 | Sentinel-2B | 32 | 71 | 13 | 0 | 3 | 1 | 0 | - |
| 2025-06-25 | Sentinel-2C | 99 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-27 | Sentinel-2A | 13 | 5 | 5 | 4 | 0 | 0 | 0 | - |
| 2025-06-30 | Sentinel-2B | 55 | 96 | 4 | 0 | 0 | 4 | 0 | - |
| 2025-07-05 | Sentinel-2C | 31 | 62 | 7 | 0 | 1 | 3 | 1 | - |
| 2025-07-07 | Sentinel-2A | 16 | 9 | 5 | 4 | 0 | 0 | 17 | Playa Kanoa to Sint Joris Baai |
| 2025-07-10 | Sentinel-2B | 14 | 9 | 3 | 3 | 1 | 0 | 0 | - |
| 2025-07-15 | Sentinel-2C | 11 | 28 | 8 | 2 | 2 | 0 | 0 | - |
| 2025-07-17 | Sentinel-2A | 7 | 1 | 6 | 4 | 0 | 0 | 0 | - |
| 2025-07-20 | Sentinel-2B | 97 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-07-25 | Sentinel-2C | 19 | 13 | 6 | 3 | 1 | 0 | 0 | - |
| 2025-07-27 | Sentinel-2A | 46 | 6 | 4 | 4 | 0 | 0 | 0 | - |
| 2025-07-30 | Sentinel-2B | 6 | 0 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-08-04 | Sentinel-2C | 10 | 16 | 7 | 3 | 1 | 0 | 0 | - |
| 2025-08-06 | Sentinel-2A | 14 | 6 | 3 | 3 | 1 | 0 | 0 | - |
| 2025-08-09 | Sentinel-2B | 11 | 2 | 4 | 4 | 0 | 0 | 0 | - |
| 2025-08-14 | Sentinel-2C | 15 | 1 | 3 | 4 | 0 | 0 | 0 | - |
| 2025-08-16 | Sentinel-2A | 17 | 21 | 1 | 3 | 1 | 0 | 8 | - |
| 2025-08-19 | Sentinel-2B | 89 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-08-24 | Sentinel-2C | 21 | 29 | 3 | 3 | 0 | 1 | 0 | - |
| 2025-08-26 | Sentinel-2A | 54 | 31 | 2 | 2 | 2 | 0 | 0 | - |
| 2025-08-29 | Sentinel-2B | 22 | 99 | 2 | 0 | 0 | 4 | 0 | - |

## What this does not say

No pixel here has been checked against the beach. The classifier was trained and
calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and
around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is
from Curaçao. Reef flats, sand bottoms, lagoons and wrecks here are surfaces
the model has never been graded on. The observability columns do not depend on
the classifier at all and are the part of this report to trust first: they say
how often an optical satellite could look at each stretch of coast, which is a
property of the weather and the orbit, not of the model.

Curaçao does not fit on one tile. The west part is read from 19PDP and the east part from 19PEP, both on relative orbit R082 (Bonaire's), so a date means the same overpass on both and the halves join by date. The split is on the north coast below Landhuis Hato, inside the tile overlap; every surf zone is at least 3.7 km inside the tile it is read from.

Relative orbit R125 (Aruba's) also clips the western tip, but no R125 granule covers a whole segment, so it is not counted. The Watamula end of the north coast is therefore seen somewhat more often than these tables say.

Sint Joris Baai's mouth has no named node. Its two ends are the ends of the line OpenStreetMap closes the bay polygon with (way 298797383), 226 m across.

Daaibooi to Playa Largu is the leeward control rather than a stretch next to Willemstad, whose coast has the harbour mouth, Piscadera Bay, Spanish Water and the Bullenbaai oil terminal. The dive piers at Porto Marie and Cas Abao still put a few structures in it.
