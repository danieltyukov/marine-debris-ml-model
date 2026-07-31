# Beach-segment brief, Cancun hotel zone

Scene `S2A_MSIL2A_20260706T161701_R140_T16QEJ_20260707T035957`, 2026-07-06, 75.5% cloud over the tile.

Produced by `python scripts/make_beach_segments.py`. Sargassum probability is
thresholded at 0.736, the 90%-precision operating point from
`sargassum_report.md`, and cloud is measured per segment from the scene's own
classification layer.

| segment | observed | cloud % | cover % | affected front m | detections |
|---|---|---|---|---|---|
| Playa Caracol | partial | 67 | 0.00 | 0 | 0 |
| Playa Tortugas | partial | 67 | 0.00 | 0 | 0 |
| Playa Langosta | partial | 68 | 0.00 | 0 | 0 |
| Punta Nizuc | blind | 83 | — | — | — |
| Playa Delfines | blind | 96 | — | — | — |
| Playa Ballenas | blind | 94 | — | — | — |
| Playa Marlin | blind | 100 | — | — | — |
| Playa Chac Mool | blind | 91 | — | — | — |
| Playa Gaviota Azul | blind | 82 | — | — | — |
| Playa Linda | blind | 92 | — | — | — |

7 of 10 segments were not observed. Those rows carry
no coverage figure on purpose: a beach under cloud has not been measured, and
reporting 0% for it is the specific mistake that sends nobody to a beach that
needed clearing.

## What this is not

This is a detection on the day of the overpass, not a landfall forecast. Nothing
here models drift between the overpass and the arrival, and no field observation
has confirmed any of these detections. The numbers say where floating material
was, at 10 m, at one instant.
