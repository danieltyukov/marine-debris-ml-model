# Bonaire, one season of Sentinel-2 passes

Produced by `python scripts/run_bonaire_season.py --start 2025-01-01 --end 2025-08-31`.
64 passes over tile 19PEP, 18 from Sentinel-2A, 24 from Sentinel-2B, 22 from Sentinel-2C. Segments are cut from the OpenStreetMap coastline at named landmarks
(`assets/bonaire_segments.geojson`); the surf zone is the water within
500 m of each, land removed with the OpenStreetMap island polygon plus a
15 m seaward strip. A pixel is called sargassum at calibrated probability
>= 0.90, which after calibration means the model's own estimate is at least
90%; it is uncertain in [0.056, 0.90) and clear below. The
lower edge keeps 98% of true sargassum on MARIDA's validation split, and
`docs/calibration.md` records what the upper edge achieves on the test split.

## How often each stretch of coast could be seen

| segment | exposure | passes | observed | partial | blind | usable | longest gap, days | median gap | passes with sargassum | most front affected, m | pixels ever flagged | of which stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Willemstoren to Sorobon | windward | 64 | 31 | 15 | 18 | 72% | 15 | 5.0 | 1 | 1605 | 13 | 0 |
| Lac Bay | windward | 64 | 14 | 27 | 23 | 64% | 20 | 5.0 | 24 | 7552 | 1,167 | 25 |
| Cai to Boka Washikemba | windward | 64 | 32 | 14 | 18 | 72% | 15 | 5.0 | 10 | 1902 | 285 | 5 |
| Boka Washikemba to Spelonk | windward | 64 | 34 | 11 | 19 | 70% | 15 | 5.0 | 3 | 1178 | 41 | 0 |
| Spelonk to Boka Onima | windward | 64 | 32 | 15 | 17 | 73% | 15 | 5.0 | 2 | 1121 | 24 | 0 |
| Boka Onima to Playa Chikitu | windward | 64 | 34 | 13 | 17 | 73% | 15 | 5.0 | 0 | 0 | 5 | 0 |
| Playa Chikitu to Boka Kokolishi | windward | 64 | 38 | 9 | 17 | 73% | 15 | 5.0 | 0 | 0 | 13 | 0 |
| Kralendijk waterfront | leeward | 64 | 14 | 22 | 28 | 56% | 20 | 5.0 | 0 | 0 | 0 | 0 |

`usable` is observed plus partial: passes on which a detection means something.
A segment is blind on a pass when more than 70% of the water side of its surf
zone is cloud by the scene classification layer, partial above 20%. Every
non-cloud pixel on the water side is classified, surf included: there is no NDWI
water gate, because on MARIDA no dense sargassum pixel passes one. The longest
gap is the most days between two consecutive usable looks, which is the number
that decides whether a satellite product can give a beach any warning at all.

The last two columns are the honesty check on the detection columns. A pixel
flagged on at least 50% of the passes that could see it (with at least
5 such passes) is stationary, and floating material is not. Those pixels
are the shallow back-bay of Lac, the reef flat at Cai and the mangrove edge: bottom
types the MARIDA classifier was never trained on. Detection counts in segments
where most flagged pixels are stationary are that confuser, not sargassum, and the
figure `assets/bonaire_persistence.png` shows where it sits. The Wageningen report
asked for separate models for the bays and the open sea for this reason.

## Pass by pass

| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial | blind | sargassum pixels | segments with sargassum |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-01 | Sentinel-2B | 14 | 20 | 8 | 6 | 2 | 0 | 224 | Lac Bay, Cai to Boka Washikemba, Boka Washikemba to Spelonk |
| 2025-01-06 | Sentinel-2A | 28 | 22 | 9 | 2 | 6 | 0 | 54 | Lac Bay |
| 2025-01-11 | Sentinel-2B | 9 | 8 | 5 | 7 | 1 | 0 | 468 | Lac Bay, Cai to Boka Washikemba |
| 2025-01-16 | Sentinel-2A | 13 | 4 | 5 | 8 | 0 | 0 | 690 | Lac Bay, Cai to Boka Washikemba, Boka Washikemba to Spelonk |
| 2025-01-21 | Sentinel-2B | 32 | 29 | 4 | 4 | 3 | 1 | 93 | - |
| 2025-01-26 | Sentinel-2C | 8 | 4 | 6 | 8 | 0 | 0 | 359 | Lac Bay, Cai to Boka Washikemba |
| 2025-01-31 | Sentinel-2B | 21 | 19 | 15 | 5 | 2 | 1 | 152 | Lac Bay, Cai to Boka Washikemba |
| 2025-02-05 | Sentinel-2C | 30 | 10 | 13 | 7 | 1 | 0 | 163 | Lac Bay |
| 2025-02-15 | Sentinel-2C | 43 | 12 | 3 | 6 | 1 | 1 | 174 | Lac Bay |
| 2025-02-20 | Sentinel-2B | 1 | 3 | 4 | 8 | 0 | 0 | 224 | Lac Bay, Cai to Boka Washikemba |
| 2025-02-25 | Sentinel-2C | 14 | 8 | 12 | 7 | 1 | 0 | 94 | Lac Bay, Cai to Boka Washikemba |
| 2025-03-02 | Sentinel-2B | 5 | 5 | 9 | 7 | 1 | 0 | 142 | Lac Bay |
| 2025-03-07 | Sentinel-2C | 22 | 29 | 19 | 2 | 5 | 1 | 38 | Lac Bay |
| 2025-03-12 | Sentinel-2B | 2 | 4 | 9 | 8 | 0 | 0 | 170 | Lac Bay, Cai to Boka Washikemba |
| 2025-03-17 | Sentinel-2C | 20 | 20 | 6 | 5 | 2 | 1 | 129 | Lac Bay |
| 2025-03-19 | Sentinel-2A | 69 | 46 | 14 | 1 | 6 | 1 | 20 | Spelonk to Boka Onima |
| 2025-03-22 | Sentinel-2B | 41 | 89 | 4 | 0 | 2 | 6 | 0 | - |
| 2025-03-27 | Sentinel-2C | 34 | 53 | 11 | 3 | 2 | 3 | 0 | - |
| 2025-03-29 | Sentinel-2A | 72 | 91 | 9 | 1 | 0 | 7 | 0 | - |
| 2025-04-01 | Sentinel-2B | 34 | 31 | 9 | 1 | 7 | 0 | 0 | - |
| 2025-04-06 | Sentinel-2C | 55 | 85 | 30 | 1 | 1 | 6 | 0 | - |
| 2025-04-08 | Sentinel-2A | 97 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-04-11 | Sentinel-2B | 75 | 85 | 3 | 0 | 1 | 7 | 0 | - |
| 2025-04-16 | Sentinel-2C | 47 | 39 | 8 | 3 | 3 | 2 | 0 | - |
| 2025-04-21 | Sentinel-2B | 19 | 7 | 9 | 6 | 2 | 0 | 39 | Lac Bay |
| 2025-04-26 | Sentinel-2C | 100 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-04-28 | Sentinel-2A | 99 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-05-01 | Sentinel-2B | 60 | 100 | 45 | 0 | 0 | 8 | 0 | - |
| 2025-05-06 | Sentinel-2C | 27 | 28 | 49 | 3 | 4 | 1 | 0 | - |
| 2025-05-08 | Sentinel-2A | 57 | 67 | 25 | 1 | 2 | 5 | 57 | Lac Bay |
| 2025-05-11 | Sentinel-2B | 20 | 7 | 9 | 6 | 2 | 0 | 55 | Lac Bay, Cai to Boka Washikemba |
| 2025-05-16 | Sentinel-2C | 72 | 28 | 9 | 4 | 2 | 2 | 0 | - |
| 2025-05-18 | Sentinel-2A | 97 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-05-21 | Sentinel-2B | 25 | 25 | 7 | 4 | 3 | 1 | 0 | - |
| 2025-05-26 | Sentinel-2C | 16 | 7 | 21 | 7 | 1 | 0 | 0 | - |
| 2025-05-28 | Sentinel-2A | 92 | 83 | 51 | 0 | 3 | 5 | 0 | - |
| 2025-05-31 | Sentinel-2B | 95 | 97 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-06-05 | Sentinel-2C | 14 | 20 | 12 | 5 | 2 | 1 | 9 | - |
| 2025-06-07 | Sentinel-2A | 22 | 16 | 10 | 5 | 3 | 0 | 33 | Lac Bay |
| 2025-06-10 | Sentinel-2B | 76 | 86 | 4 | 0 | 1 | 7 | 0 | - |
| 2025-06-15 | Sentinel-2C | 40 | 67 | 21 | 0 | 4 | 4 | 0 | - |
| 2025-06-17 | Sentinel-2A | 10 | 12 | 11 | 7 | 1 | 0 | 88 | Lac Bay, Cai to Boka Washikemba |
| 2025-06-20 | Sentinel-2B | 32 | 38 | 7 | 2 | 6 | 0 | 1 | - |
| 2025-06-25 | Sentinel-2C | 99 | 100 | 0 | 0 | 0 | 8 | 0 | - |
| 2025-06-27 | Sentinel-2A | 13 | 47 | 14 | 0 | 8 | 0 | 0 | - |
| 2025-06-30 | Sentinel-2B | 55 | 53 | 7 | 0 | 7 | 1 | 0 | - |
| 2025-07-05 | Sentinel-2C | 31 | 19 | 13 | 5 | 3 | 0 | 50 | Lac Bay |
| 2025-07-07 | Sentinel-2A | 16 | 36 | 3 | 4 | 2 | 2 | 14 | Spelonk to Boka Onima |
| 2025-07-10 | Sentinel-2B | 14 | 9 | 7 | 7 | 1 | 0 | 67 | Lac Bay |
| 2025-07-15 | Sentinel-2C | 11 | 18 | 11 | 6 | 1 | 1 | 4 | - |
| 2025-07-17 | Sentinel-2A | 7 | 7 | 26 | 7 | 1 | 0 | 35 | Lac Bay |
| 2025-07-20 | Sentinel-2B | 97 | 100 | 2 | 0 | 0 | 8 | 0 | - |
| 2025-07-25 | Sentinel-2C | 19 | 60 | 6 | 2 | 2 | 4 | 0 | - |
| 2025-07-27 | Sentinel-2A | 46 | 90 | 4 | 0 | 1 | 7 | 0 | - |
| 2025-07-30 | Sentinel-2B | 6 | 10 | 8 | 6 | 2 | 0 | 44 | Lac Bay |
| 2025-08-04 | Sentinel-2C | 10 | 6 | 16 | 7 | 1 | 0 | 2 | - |
| 2025-08-06 | Sentinel-2A | 14 | 6 | 17 | 7 | 1 | 0 | 20 | - |
| 2025-08-09 | Sentinel-2B | 11 | 9 | 10 | 6 | 2 | 0 | 4 | - |
| 2025-08-14 | Sentinel-2C | 15 | 37 | 11 | 3 | 4 | 1 | 9 | - |
| 2025-08-16 | Sentinel-2A | 17 | 26 | 13 | 4 | 3 | 1 | 428 | Willemstoren to Sorobon, Lac Bay, Boka Washikemba to Spelonk |
| 2025-08-19 | Sentinel-2B | 89 | 100 | 11 | 0 | 0 | 8 | 0 | - |
| 2025-08-24 | Sentinel-2C | 21 | 5 | 16 | 7 | 1 | 0 | 2 | - |
| 2025-08-26 | Sentinel-2A | 54 | 74 | 7 | 2 | 1 | 5 | 0 | - |
| 2025-08-29 | Sentinel-2B | 22 | 10 | 13 | 6 | 2 | 0 | 2 | - |

## What this does not say

No pixel here has been checked against the beach. The classifier was trained
and calibrated on MARIDA scenes from Central America, not Bonaire, and the
reef flat, seagrass and the mangrove edge of Lac are surfaces it has never been
graded on. The observability columns do not depend on the classifier at all
and are the part of this table to trust first: they say how often an optical
satellite could look at each stretch of coast, which is a property of the
weather and the orbit, not of the model.

The Kralendijk waterfront is a leeward control. Sargassum arrives on the
windward side; if that row fills with detections, the model is finding
something other than sargassum.
