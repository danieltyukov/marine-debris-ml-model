# Curaçao, west, Watamula to Hato: one season of Sentinel-2 passes

Produced by `python scripts/run_island_season.py --island curacao --part west`. 64
passes over tile 19PDP (relative orbit R082) from 2025-01-01 to 2025-08-31, 18 from
Sentinel-2A, 25 from Sentinel-2B, 21 from Sentinel-2C. The search leaves out granules
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
| Watamula to Bartolbaai | windward | 64 | 23 | 19 | 22 | 66% | 25 | 5.0 | 35 days (16 Apr to 21 May) | 3 | 2901 | 216 | 0 |
| Bartolbaai to Boka San Pedro | windward | 64 | 30 | 12 | 22 | 66% | 15 | 5.0 | 28 days (7 Jun to 5 Jul) | 2 | 1117 | 26 | 0 |
| Boka San Pedro to Hato | windward | 64 | 27 | 19 | 18 | 72% | 15 | 5.0 | 30 days (22 Mar to 21 Apr) | 4 | 2176 | 88 | 0 |
| Daaibooi to Playa Largu | leeward | 64 | 26 | 12 | 26 | 59% | 15 | 5.0 | 37 days (11 May to 17 Jun) | 0 | 0 | 0 | 0 |

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
| Watamula to Bartolbaai | 69,945 | 0 | 15 | 216 | 0 |
| Bartolbaai to Boka San Pedro | 47,873 | 16 | 39 | 26 | 0 |
| Boka San Pedro to Hato | 64,628 | 0 | 0 | 88 | 0 |
| Daaibooi to Playa Largu | 18,722 | 0 | 0 | 0 | 0 |

Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones
overlap counts in both. `assets/curacao_west_persistence.png` shows where the
flags sit.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-01 | Sentinel-2B | 40 | 67 | 52 | 0 | 1 | 3 | 0 | - |
| 2025-01-06 | Sentinel-2A | 61 | 90 | 4 | 0 | 0 | 4 | 12 | - |
| 2025-01-11 | Sentinel-2B | 24 | 9 | 3 | 4 | 0 | 0 | 3 | - |
| 2025-01-16 | Sentinel-2A | 9 | 6 | 1 | 4 | 0 | 0 | 0 | - |
| 2025-01-21 | Sentinel-2B | 62 | 48 | 52 | 2 | 1 | 1 | 0 | - |
| 2025-01-26 | Sentinel-2C | 14 | 17 | 5 | 3 | 1 | 0 | 0 | - |
| 2025-01-31 | Sentinel-2B | 29 | 82 | 7 | 0 | 1 | 3 | 0 | - |
| 2025-02-05 | Sentinel-2C | 18 | 1 | 1 | 4 | 0 | 0 | 0 | - |
| 2025-02-10 | Sentinel-2B | 22 | 9 | 1 | 4 | 0 | 0 | 1 | - |
| 2025-02-15 | Sentinel-2C | 44 | 42 | 38 | 0 | 4 | 0 | 0 | - |
| 2025-02-20 | Sentinel-2B | 3 | 5 | 3 | 4 | 0 | 0 | 0 | - |
| 2025-02-25 | Sentinel-2C | 13 | 25 | 3 | 2 | 2 | 0 | 12 | Bartolbaai to Boka San Pedro |
| 2025-03-02 | Sentinel-2B | 2 | 1 | 2 | 4 | 0 | 0 | 0 | - |
| 2025-03-07 | Sentinel-2C | 25 | 24 | 4 | 2 | 2 | 0 | 0 | - |
| 2025-03-12 | Sentinel-2B | 7 | 11 | 4 | 3 | 0 | 1 | 2 | - |
| 2025-03-17 | Sentinel-2C | 20 | 13 | 12 | 3 | 1 | 0 | 0 | - |
| 2025-03-19 | Sentinel-2A | 72 | 64 | 33 | 0 | 2 | 2 | 0 | - |
| 2025-03-22 | Sentinel-2B | 30 | 19 | 2 | 3 | 1 | 0 | 0 | - |
| 2025-03-27 | Sentinel-2C | 50 | 51 | 7 | 1 | 2 | 1 | 0 | - |
| 2025-03-29 | Sentinel-2A | 40 | 86 | 4 | 1 | 0 | 3 | 0 | - |
| 2025-04-01 | Sentinel-2B | 57 | 81 | 0 | 0 | 1 | 3 | 10 | Boka San Pedro to Hato |
| 2025-04-06 | Sentinel-2C | 35 | 68 | 20 | 1 | 1 | 2 | 0 | - |
| 2025-04-08 | Sentinel-2A | 100 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-11 | Sentinel-2B | 64 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-04-16 | Sentinel-2C | 27 | 29 | 2 | 2 | 2 | 0 | 0 | - |
| 2025-04-21 | Sentinel-2B | 23 | 36 | 8 | 1 | 3 | 0 | 187 | Watamula to Bartolbaai, Boka San Pedro to Hato |
| 2025-04-28 | Sentinel-2A | 96 | 96 | 3 | 0 | 0 | 4 | 0 | - |
| 2025-05-01 | Sentinel-2B | 59 | 85 | 34 | 0 | 1 | 3 | 0 | - |
| 2025-05-06 | Sentinel-2C | 48 | 25 | 29 | 2 | 2 | 0 | 0 | - |
| 2025-05-08 | Sentinel-2A | 76 | 74 | 4 | 0 | 2 | 2 | 0 | - |
| 2025-05-11 | Sentinel-2B | 36 | 32 | 4 | 3 | 0 | 1 | 4 | - |
| 2025-05-16 | Sentinel-2C | 86 | 93 | 27 | 0 | 0 | 4 | 0 | - |
| 2025-05-18 | Sentinel-2A | 74 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-05-21 | Sentinel-2B | 26 | 23 | 1 | 2 | 1 | 1 | 16 | Watamula to Bartolbaai |
| 2025-05-26 | Sentinel-2C | 16 | 10 | 3 | 3 | 1 | 0 | 0 | - |
| 2025-05-28 | Sentinel-2A | 72 | 97 | 53 | 0 | 0 | 4 | 0 | - |
| 2025-05-31 | Sentinel-2B | 100 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-05 | Sentinel-2C | 14 | 22 | 3 | 2 | 2 | 0 | 0 | - |
| 2025-06-07 | Sentinel-2A | 29 | 13 | 4 | 2 | 2 | 0 | 0 | - |
| 2025-06-10 | Sentinel-2B | 96 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-15 | Sentinel-2C | 47 | 87 | 42 | 0 | 1 | 3 | 0 | - |
| 2025-06-17 | Sentinel-2A | 14 | 20 | 1 | 2 | 2 | 0 | 19 | Bartolbaai to Boka San Pedro, Boka San Pedro to Hato |
| 2025-06-20 | Sentinel-2B | 51 | 40 | 8 | 1 | 2 | 1 | 0 | - |
| 2025-06-25 | Sentinel-2C | 100 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-06-27 | Sentinel-2A | 11 | 8 | 9 | 3 | 1 | 0 | 0 | - |
| 2025-06-30 | Sentinel-2B | 71 | 98 | 6 | 0 | 0 | 4 | 0 | - |
| 2025-07-05 | Sentinel-2C | 53 | 43 | 51 | 1 | 3 | 0 | 1 | - |
| 2025-07-07 | Sentinel-2A | 22 | 9 | 3 | 4 | 0 | 0 | 24 | Boka San Pedro to Hato |
| 2025-07-10 | Sentinel-2B | 11 | 16 | 7 | 3 | 1 | 0 | 2 | - |
| 2025-07-15 | Sentinel-2C | 7 | 34 | 6 | 2 | 1 | 1 | 0 | - |
| 2025-07-17 | Sentinel-2A | 8 | 24 | 11 | 2 | 2 | 0 | 0 | - |
| 2025-07-20 | Sentinel-2B | 100 | 100 | 0 | 0 | 0 | 4 | 0 | - |
| 2025-07-25 | Sentinel-2C | 23 | 32 | 10 | 2 | 2 | 0 | 0 | - |
| 2025-07-27 | Sentinel-2A | 47 | 38 | 5 | 1 | 2 | 1 | 0 | - |
| 2025-07-30 | Sentinel-2B | 7 | 0 | 1 | 4 | 0 | 0 | 0 | - |
| 2025-08-04 | Sentinel-2C | 25 | 18 | 6 | 3 | 0 | 1 | 0 | - |
| 2025-08-06 | Sentinel-2A | 14 | 36 | 3 | 1 | 2 | 1 | 0 | - |
| 2025-08-09 | Sentinel-2B | 28 | 19 | 11 | 3 | 1 | 0 | 0 | - |
| 2025-08-14 | Sentinel-2C | 19 | 11 | 3 | 4 | 0 | 0 | 0 | - |
| 2025-08-16 | Sentinel-2A | 18 | 31 | 2 | 2 | 1 | 1 | 39 | Watamula to Bartolbaai |
| 2025-08-19 | Sentinel-2B | 97 | 100 | 4 | 0 | 0 | 4 | 0 | - |
| 2025-08-24 | Sentinel-2C | 25 | 14 | 2 | 3 | 1 | 0 | 0 | - |
| 2025-08-26 | Sentinel-2A | 42 | 67 | 4 | 1 | 2 | 1 | 0 | - |
| 2025-08-29 | Sentinel-2B | 62 | 18 | 2 | 2 | 2 | 0 | 0 | - |

## What this does not say

No pixel here has been checked against the beach. The classifier was trained and
calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and
around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is
from Curaçao. Reef flats, sand bottoms, lagoons and wrecks here are surfaces
the model has never been graded on. The observability columns do not depend on
the classifier at all and are the part of this report to trust first: they say
how often an optical satellite could look at each stretch of coast, which is a
property of the weather and the orbit, not of the model.

Daaibooi to Playa Largu is a leeward control. Sargassum arrives on the windward
side; if that row fills with detections, the model is finding something other
than sargassum.

Curaçao does not fit on one tile. The west part is read from 19PDP and the east part from 19PEP, both on relative orbit R082 (Bonaire's), so a date means the same overpass on both and the halves join by date. The split is on the north coast below Landhuis Hato, inside the tile overlap; every surf zone is at least 3.7 km inside the tile it is read from.

Relative orbit R125 (Aruba's) also clips the western tip, but no R125 granule covers a whole segment, so it is not counted. The Watamula end of the north coast is therefore seen somewhat more often than these tables say.

Sint Joris Baai's mouth has no named node. Its two ends are the ends of the line OpenStreetMap closes the bay polygon with (way 298797383), 226 m across.

Daaibooi to Playa Largu is the leeward control rather than a stretch next to Willemstad, whose coast has the harbour mouth, Piscadera Bay, Spanish Water and the Bullenbaai oil terminal. The dive piers at Porto Marie and Cas Abao still put a few structures in it.
