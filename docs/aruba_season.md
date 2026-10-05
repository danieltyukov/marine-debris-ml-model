# Aruba: one season of Sentinel-2 passes

Produced by `python scripts/run_island_season.py --island aruba`. 63 passes over tile
19PCP (relative orbit R125) from 2025-01-01 to 2025-08-31, 18 from Sentinel-2A, 23 from
Sentinel-2B, 22 from Sentinel-2C. The search leaves out granules reported as 100% cloud,
and keeps one granule per datatake whose footprint covers at least 99% of every surf
zone. Segments are cut from the OpenStreetMap coastline at named landmarks
(`assets/islands/aruba/segments.geojson`); the surf zone is the water within 500 m of
each. Land is removed with the OpenStreetMap coastline polygons plus a 15 m seaward
strip, and mapped mangrove and other vegetated wetland is removed with it, plus a 20 m
strip (`assets/islands/aruba/mangroves.geojson`). A pixel is called sargassum at
calibrated probability >= 0.90, which after calibration means the model's own estimate
is at least 90%; it is uncertain in [0.056, 0.90) and clear below. The lower edge keeps
98% of true sargassum on MARIDA's validation split, and `docs/calibration.md` records
what the upper edge achieves on the test split.

## How often each stretch of coast could be seen

| segment | exposure | passes | observed | partial | blind | usable | longest gap, days | median gap | longest wait for a fully clear pass | passes with sargassum | most front affected, m | pixels ever flagged | of which stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kudareu to Boca Chiquito | windward | 63 | 27 | 10 | 26 | 59% | 15 | 5.0 | 30 days (8 Jun to 8 Jul) | 0 | 0 | 0 | 0 |
| Boca Chiquito to Bushiribana | windward | 63 | 28 | 9 | 26 | 59% | 30 | 5.0 | 30 days (8 Jun to 8 Jul) | 0 | 0 | 0 | 0 |
| Bushiribana to Andicuri | windward | 63 | 27 | 9 | 27 | 57% | 25 | 5.0 | 35 days (3 Jun to 8 Jul) | 0 | 0 | 2 | 0 |
| Andicuri to Boca Prins | windward | 63 | 25 | 20 | 18 | 71% | 15 | 5.0 | 40 days (24 May to 3 Jul) | 1 | 1600 | 31 | 0 |
| Boca Prins to Rincon | windward | 63 | 30 | 13 | 20 | 68% | 17 | 5.0 | 28 days (1 Apr to 29 Apr) | 0 | 0 | 0 | 0 |
| Rincon to Punta Basora (Boca Grandi) | windward | 63 | 28 | 12 | 23 | 63% | 15 | 5.0 | 25 days (29 Apr to 24 May) | 0 | 0 | 0 | 0 |
| Eagle Beach to Palm Beach | leeward | 63 | 14 | 22 | 27 | 57% | 20 | 5.0 | 42 days (30 Mar to 11 May) | 0 | 0 | 0 | 0 |

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
| Kudareu to Boca Chiquito | 48,677 | 0 | 0 | 0 | 0 |
| Boca Chiquito to Bushiribana | 19,876 | 0 | 0 | 0 | 0 |
| Bushiribana to Andicuri | 18,106 | 0 | 0 | 2 | 0 |
| Andicuri to Boca Prins | 42,943 | 0 | 0 | 31 | 0 |
| Boca Prins to Rincon | 28,426 | 0 | 0 | 0 | 0 |
| Rincon to Punta Basora (Boca Grandi) | 33,600 | 0 | 0 | 0 | 0 |
| Eagle Beach to Palm Beach | 20,329 | 0 | 0 | 0 | 0 |

Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones
overlap counts in both. `assets/aruba_persistence.png` shows where the
flags sit.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-09 | Sentinel-2A | 2 | 1 | 2 | 7 | 0 | 0 | 0 | - |
| 2025-01-14 | Sentinel-2B | 38 | 71 | 5 | 0 | 3 | 4 | 0 | - |
| 2025-01-19 | Sentinel-2A | 30 | 35 | 2 | 3 | 4 | 0 | 4 | - |
| 2025-01-24 | Sentinel-2B | 6 | 11 | 4 | 6 | 1 | 0 | 0 | - |
| 2025-01-29 | Sentinel-2C | 17 | 36 | 2 | 3 | 3 | 1 | 0 | - |
| 2025-02-03 | Sentinel-2B | 38 | 37 | 4 | 3 | 2 | 2 | 0 | - |
| 2025-02-08 | Sentinel-2C | 62 | 79 | 0 | 1 | 1 | 5 | 0 | - |
| 2025-02-13 | Sentinel-2B | 7 | 23 | 2 | 3 | 4 | 0 | 0 | - |
| 2025-02-18 | Sentinel-2C | 18 | 13 | 3 | 5 | 2 | 0 | 0 | - |
| 2025-02-23 | Sentinel-2B | 10 | 9 | 3 | 6 | 1 | 0 | 0 | - |
| 2025-02-28 | Sentinel-2C | 74 | 79 | 16 | 0 | 1 | 6 | 0 | - |
| 2025-03-05 | Sentinel-2B | 18 | 34 | 1 | 3 | 3 | 1 | 0 | - |
| 2025-03-10 | Sentinel-2C | 10 | 11 | 6 | 5 | 2 | 0 | 29 | Andicuri to Boca Prins |
| 2025-03-15 | Sentinel-2B | 21 | 37 | 3 | 3 | 2 | 2 | 0 | - |
| 2025-03-20 | Sentinel-2C | 80 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-03-22 | Sentinel-2A | 38 | 69 | 7 | 0 | 4 | 3 | 0 | - |
| 2025-03-25 | Sentinel-2B | 3 | 5 | 0 | 7 | 0 | 0 | 0 | - |
| 2025-03-30 | Sentinel-2C | 7 | 2 | 3 | 7 | 0 | 0 | 0 | - |
| 2025-04-01 | Sentinel-2A | 27 | 13 | 2 | 5 | 2 | 0 | 0 | - |
| 2025-04-04 | Sentinel-2B | 98 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-04-09 | Sentinel-2C | 31 | 30 | 1 | 4 | 1 | 2 | 0 | - |
| 2025-04-11 | Sentinel-2A | 52 | 57 | 0 | 1 | 3 | 3 | 0 | - |
| 2025-04-14 | Sentinel-2B | 41 | 41 | 1 | 4 | 1 | 2 | 0 | - |
| 2025-04-19 | Sentinel-2C | 69 | 64 | 8 | 1 | 2 | 4 | 0 | - |
| 2025-04-21 | Sentinel-2A | 77 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-04-24 | Sentinel-2B | 53 | 98 | 8 | 0 | 0 | 7 | 0 | - |
| 2025-04-29 | Sentinel-2C | 20 | 9 | 2 | 6 | 1 | 0 | 0 | - |
| 2025-05-01 | Sentinel-2A | 62 | 77 | 24 | 0 | 3 | 4 | 0 | - |
| 2025-05-04 | Sentinel-2B | 22 | 67 | 3 | 0 | 4 | 3 | 0 | - |
| 2025-05-09 | Sentinel-2C | 53 | 26 | 1 | 4 | 3 | 0 | 0 | - |
| 2025-05-11 | Sentinel-2A | 11 | 7 | 0 | 6 | 1 | 0 | 0 | - |
| 2025-05-14 | Sentinel-2B | 99 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-05-19 | Sentinel-2C | 79 | 98 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-05-21 | Sentinel-2A | 13 | 23 | 0 | 4 | 2 | 1 | 0 | - |
| 2025-05-24 | Sentinel-2B | 21 | 5 | 2 | 7 | 0 | 0 | 0 | - |
| 2025-05-29 | Sentinel-2C | 43 | 95 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-06-03 | Sentinel-2B | 32 | 23 | 5 | 4 | 3 | 0 | 0 | - |
| 2025-06-08 | Sentinel-2C | 27 | 41 | 1 | 2 | 3 | 2 | 0 | - |
| 2025-06-10 | Sentinel-2A | 98 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-06-13 | Sentinel-2B | 92 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-06-18 | Sentinel-2C | 62 | 81 | 0 | 0 | 2 | 5 | 0 | - |
| 2025-06-20 | Sentinel-2A | 86 | 66 | 3 | 2 | 1 | 4 | 0 | - |
| 2025-06-23 | Sentinel-2B | 45 | 89 | 0 | 0 | 1 | 6 | 0 | - |
| 2025-06-28 | Sentinel-2C | 30 | 52 | 1 | 3 | 1 | 3 | 0 | - |
| 2025-06-30 | Sentinel-2A | 53 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-07-03 | Sentinel-2B | 15 | 44 | 2 | 2 | 3 | 2 | 0 | - |
| 2025-07-08 | Sentinel-2C | 17 | 6 | 0 | 7 | 0 | 0 | 0 | - |
| 2025-07-10 | Sentinel-2A | 12 | 5 | 1 | 6 | 1 | 0 | 0 | - |
| 2025-07-13 | Sentinel-2B | 21 | 5 | 2 | 7 | 0 | 0 | 0 | - |
| 2025-07-18 | Sentinel-2C | 52 | 16 | 3 | 5 | 2 | 0 | 0 | - |
| 2025-07-20 | Sentinel-2A | 62 | 82 | 1 | 0 | 2 | 5 | 0 | - |
| 2025-07-23 | Sentinel-2B | 45 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-07-28 | Sentinel-2C | 4 | 20 | 1 | 5 | 1 | 1 | 0 | - |
| 2025-07-30 | Sentinel-2A | 19 | 13 | 4 | 5 | 1 | 1 | 0 | - |
| 2025-08-02 | Sentinel-2B | 19 | 71 | 0 | 0 | 4 | 3 | 0 | - |
| 2025-08-07 | Sentinel-2C | 5 | 3 | 3 | 7 | 0 | 0 | 0 | - |
| 2025-08-09 | Sentinel-2A | 17 | 11 | 0 | 6 | 1 | 0 | 0 | - |
| 2025-08-12 | Sentinel-2B | 2 | 1 | 4 | 7 | 0 | 0 | 0 | - |
| 2025-08-17 | Sentinel-2C | 88 | 84 | 4 | 0 | 2 | 5 | 0 | - |
| 2025-08-19 | Sentinel-2A | 93 | 100 | 0 | 0 | 0 | 7 | 0 | - |
| 2025-08-22 | Sentinel-2B | 60 | 62 | 5 | 2 | 2 | 3 | 0 | - |
| 2025-08-27 | Sentinel-2C | 19 | 40 | 0 | 1 | 6 | 0 | 0 | - |
| 2025-08-29 | Sentinel-2A | 24 | 14 | 4 | 4 | 3 | 0 | 0 | - |

## Granules not counted as passes

- `S2B_MSIL2A_20250822T151719_R125_T19PCP_20250822T185845`: second granule of the same datatake as `S2B_MSIL2A_20250822T151719_R125_T19PCP_20250822T184424`; its footprint covers 0% of the least-covered surf zone, the kept one 100%

## What this does not say

No pixel here has been checked against the beach. The classifier was trained and
calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and
around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is
from Aruba. Reef flats, sand bottoms, lagoons and wrecks here are surfaces
the model has never been graded on. The observability columns do not depend on
the classifier at all and are the part of this report to trust first: they say
how often an optical satellite could look at each stretch of coast, which is a
property of the weather and the orbit, not of the model.

Eagle Beach to Palm Beach is a leeward control. Sargassum arrives on the windward
side; if that row fills with detections, the model is finding something other
than sargassum.

Aruba sits where four Sentinel-2 tiles meet (19PCP, 19PCQ, 19PDP, 19PDQ). Tile 19PCP alone holds the island and every surf zone, and the other three carry the same datatakes, so only 19PCP is read.

OpenStreetMap has no shore node for California Lighthouse or for Colorado Point. Kudareu, the cape at the north-west point about 1.4 km from the lighthouse, and Punta Basora, below Seroe Colorado lighthouse, stand in for them.

Eagle Beach to Palm Beach is the leeward control rather than the Oranjestad waterfront, whose surf zone holds the harbour, reef islands and a lagoon. The hotel piers at Palm Beach still put boats in it.

Orbit R125 is glint-prone: in summer it looks towards the sun's reflection on the windward sea (median glint angle 18 degrees at the coast, under 20 degrees on 47 of 63 passes). Flags here fell to none from April, in the months it glints most, so a quiet season on this orbit cannot be scored as a clean coast. docs/islands.md has the measurements.
