# Beach-segment brief, Cancun hotel zone

Scene `S2B_MSIL2A_20260729T160829_R140_T16QEJ_20260729T200544`, 2026-07-29, 5.9% cloud over the tile.

Produced by `python scripts/make_beach_segments.py`. Sargassum probability is
thresholded at 0.736, the 90%-precision operating point from
`sargassum_report.md`, and cloud is measured per segment from the scene's own
classification layer.

| segment | observed | cloud % | cover % | affected front m | detections |
|---|---|---|---|---|---|
| Punta Nizuc | observed | 4 | 0.00 | 0 | 0 |
| Playa Delfines | observed | 6 | 0.00 | 0 | 0 |
| Playa Ballenas | observed | 1 | 0.00 | 0 | 0 |
| Playa Marlin | observed | 1 | 0.00 | 0 | 0 |
| Playa Chac Mool | observed | 1 | 0.00 | 0 | 0 |
| Playa Gaviota Azul | observed | 1 | 0.00 | 0 | 0 |
| Playa Caracol | observed | 0 | 0.00 | 0 | 0 |
| Playa Tortugas | observed | 0 | 0.00 | 0 | 0 |
| Playa Langosta | observed | 0 | 0.00 | 0 | 0 |
| Playa Linda | observed | 0 | 0.00 | 0 | 0 |

Every segment was observed on this pass, so an absence of detections here does
mean a clear beach. That is not the usual case. LANOT, who run the nearest
comparable Sentinel-2 platform, report cloud above 90% often enough that a
fully clear day over this coast is close to non-existent.

## What this is not

This is a detection on the day of the overpass, not a landfall forecast. Nothing
here models drift between the overpass and the arrival, and no field observation
has confirmed any of these detections. The numbers say where floating material
was, at 10 m, at one instant.
