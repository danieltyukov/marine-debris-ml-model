# Sentinel-1 radar over Lac Bay, January to August 2025

Lac Bay had no fully clear Sentinel-2 look for 97 days, from 12 March to 17 June 2025.
Radar sees through cloud. Two questions: does Sentinel-1 add anything over Lac Bay while
Sentinel-2 cannot see it, and do the pixels the optical classifier flagged there behave
like floating material in radar?

The short answers: radar fills the gap in time, it found nothing in the open lagoon
water that looks floating, and the optical flags behave like a fixed surface. There is
one lead on the west shore on 6 May, which is not a detection.

The analysis was run on the 2.0 season's flags, before the mangrove mask, because those
were the flags in question ([lac_bay_mangroves.md](lac_bay_mangroves.md)).
`scripts/eval_sentinel1_lac.py` reruns the radar side of it from a clean clone; the last
section says what it covers and what it gives.

## In short

- **Radar fills the time gap.** Sentinel-1 passed over Lac 9 times in the 97 days, never
  more than 12 days apart.
- **The optical flags in Lac are mostly mangrove canopy.** Of the 1,167 pixels the 2.0
  run flagged at least once, 982 sit on canopy (median NDVI over 19 Sentinel-2 looks above
  0.3) and 144 more within 20 m of it; 23 of the 25 stationary ones are canopy. Over the
  mangroves C-band returns the canopy, the brightest steady thing in the bay.
- **Radar cannot tell flagged pixels from unflagged pixels of the same surface.** Against
  distance-matched references the separation (AUC) is 0.45 to 0.62, there is no
  brightening on the day a pixel is flagged (mean -0.24 to +0.21 dB over 6 passes), and
  the flagged pixels are as steady over 19 passes as canopy (VV temporal standard
  deviation 1.79 dB, against 1.77 for unflagged canopy edge and 2.05 to 2.34 for water).
  In radar they behave like a fixed surface. As a per-pixel feature for Lac, radar would
  add variance, not separation.
- **Over the gap the result is null.** In lagoon water more than 30 m from canopy and land
  (2.94 km2) there was no bright patch of any kind at 4 dB contrast on any of the 9 gap
  passes (3 on the 11 passes outside it), and none at 5 dB on any pass. At 3 dB the lagoon
  throws up fewer transient patches per km2 than the open sea, where they come from waves
  and speckle.
- **One lead, not a detection.** On 6 May, the calmest pass (ERA5 wind 5.2 m/s), the
  largest radar patch of the gap in the bay (5,400 m2, 5.3 dB) lies on the west, leeward
  shore at 12.1022 N, 68.2402 W. Sentinel-2 shows a red-brown band with a
  floating-vegetation signal at the same spot through cloud gaps on 8 and 11 May, and on no
  other usable look of the season.
- **The result depends on wind.** Every pass had 5.2 to 10.0 m/s, the lagoon's VV rises and
  falls with the open sea's (correlation 0.94 over 20 passes), and VH over water sits at
  the instrument noise floor on every pass.

## Passes

All 20 acquisitions over Lac from 1 January to 31 August 2025, checked against three
catalogues: Planetary Computer `sentinel-1-rtc` (19), `sentinel-1-grd` (20) and the
Copernicus Data Space catalogue (20 GRD acquisitions). Every pass is Sentinel-1A, IW mode,
VV and VH; no Sentinel-1C pass covers Lac in this window. Nineteen are ascending passes at
18:43 local time on relative orbit 33, every 12 days (10 August is missing from all three
catalogues); one is descending, at 06:25 on 24 March. The 31 March pass is in the GRD
collection only, so it was calibrated and geocoded from the GRD product for this analysis.

| date | local time | orbit | in the gap | ERA5 wind, m/s | VV open sea, dB | VV back-bay, dB | nearest Sentinel-2 over Lac |
|---|---|---|---|---|---|---|---|
| 2025-01-06 | 18:43 | asc. 33 |  | 8.2 | -11.6 | -12.5 | 01-06, partial |
| 2025-01-18 | 18:43 | asc. 33 |  | 8.6 | -10.2 | -11.2 | 01-16, observed |
| 2025-01-30 | 18:43 | asc. 33 |  | 8.7 | -11.5 | -12.4 | 01-31, partial |
| 2025-02-11 | 18:43 | asc. 33 |  | 9.2 | -11.8 | -13.0 | 02-15, partial |
| 2025-02-23 | 18:43 | asc. 33 |  | 6.3 | -13.4 | -14.2 | 02-25, observed |
| 2025-03-07 | 18:43 | asc. 33 |  | 7.1 | -13.6 | -14.3 | 03-07, partial |
| 2025-03-19 | 18:43 | asc. 33 | yes | 6.2 | -15.1 | -15.0 | 03-19, partial |
| 2025-03-24 | 06:25 | desc. 98 | yes | 10.0 | -11.4 | -12.8 | 03-22, blind |
| 2025-03-31 | 18:43 | asc. 33 (GRD) | yes | 7.3 | -12.3 | -13.3 | 04-01, partial |
| 2025-04-12 | 18:43 | asc. 33 | yes | 8.8 | -11.6 | -12.2 | 04-11, blind |
| 2025-04-24 | 18:43 | asc. 33 | yes | 6.0 | -14.4 | -15.2 | 04-26, blind |
| 2025-05-06 | 18:43 | asc. 33 | yes | 5.2 | -14.3 | -14.8 | 05-06, blind |
| 2025-05-18 | 18:43 | asc. 33 | yes | 8.0 | -14.2 | -14.7 | 05-18, blind |
| 2025-05-30 | 18:43 | asc. 33 | yes | 7.8 | -9.5 | -12.5 | 05-31, blind |
| 2025-06-11 | 18:43 | asc. 33 | yes | 9.3 | -12.2 | -13.2 | 06-10, blind |
| 2025-06-23 | 18:43 | asc. 33 |  | 9.4 | -11.7 | -12.6 | 06-25, blind |
| 2025-07-05 | 18:43 | asc. 33 |  | 7.5 | -13.3 | -13.8 | 07-05, observed |
| 2025-07-17 | 18:43 | asc. 33 |  | 8.5 | -11.2 | -12.5 | 07-17, observed |
| 2025-07-29 | 18:43 | asc. 33 |  | 8.0 | -12.1 | -13.0 | 07-30, observed |
| 2025-08-22 | 18:43 | asc. 33 |  | 6.4 | -14.3 | -15.1 | 08-24, partial |

Wind is ERA5 10 m wind at the pass hour from the Open-Meteo archive, for the grid cell at
12.0 N, 68.25 W. The Sentinel-2 verdicts are from the 2.0 season run.

## Do the optical flags behave like floating material?

Medians over the 8 radar passes within 24 hours of a Sentinel-2 pass that could see Lac
(5 with the optical pass 7.6 h before the radar, 3 with it 16.4 h after):

| class | pixel-passes | VV, dB | VH, dB | VV+3VH, dB | FAI | NDVI |
|---|---|---|---|---|---|---|
| flagged, on canopy | 325 | -9.6 | -15.7 | -7.0 | 0.157 | 0.64 |
| canopy edge, unflagged | 8,035 | -8.9 | -15.1 | -6.4 | 0.205 | 0.69 |
| canopy, unflagged | 47,399 | -7.5 | -13.4 | -4.9 | 0.307 | 0.82 |
| flagged, off canopy | 101 | -11.6 | -17.6 | -9.2 | 0.065 | 0.30 |
| fringe water, unflagged | 5,865 | -12.5 | -19.1 | -10.1 | 0.005 | 0.01 |
| back-bay water | 170,951 | -13.3 | -22.9 | -12.0 | -0.032 | -0.33 |

Flagged pixels against unflagged pixels of the same surface (AUC is the chance that a
random flagged pixel is brighter than a random reference pixel; 0.5 is no separation):

| flagged | reference | VV difference, dB (AUC) | VH difference, dB (AUC) | VV+3VH difference, dB (AUC) |
|---|---|---|---|---|
| on canopy | canopy edge | -0.66 (0.45) | -0.53 (0.46) | -0.62 (0.45) |
| off canopy | fringe water touching canopy | +0.86 (0.59) | +1.45 (0.62) | +0.94 (0.62) |
| off canopy | back-bay water | +1.70 (0.71) | +5.22 (0.96) | +2.83 (0.87) |

The last row looks like separation, but it is adjacency: every off-canopy flag on these
passes sits right against the canopy, and radar resolution is about 20 by 22 m, so those
pixels carry mangrove backscatter whatever floats on them. Matched by distance, the
separation mostly goes.

On the pass where a pixel is flagged, its radar relative to its own median over the other
passes, against unflagged pixels of the same surface on the same pass, averages +0.10 dB
(VV) on canopy and -0.24 dB off canopy over the 6 passes with flags. A brightening of about
1 dB or more would have shown. Over all 19 ascending passes the flagged canopy pixels' VV
anomaly stays within -0.20 to +0.23 dB of the canopy edge's, -0.03 dB in January when the
optical flags peak and +0.02 dB from April to June when they nearly vanish.

## Over the gap

Detector, after Biermann et al. (2024): VV+3VH in linear power, smoothed 3 by 3 over water
pixels only, compared with a 410 m local mean of the same, kept where at least 3 dB
brighter in a patch of at least 4 connected pixels (400 m2). Canopy, land and the 15 m
shore strip are left out, because radar is bright there on every pass. Each patch is
checked against every pass with a one-pixel tolerance: seen on 10 or more passes it is
stationary, on 2 or fewer transient. The same detector runs on 8.3 km2 of open sea as a
control.

Patches of any kind, 9 gap passes against 11 others:

| contrast | lagoon water more than 30 m from canopy and land (2.94 km2) | seaward of the mouth (0.70 km2) |
|---|---|---|
| 3 dB | 15 in the gap, 37 outside | 22 in the gap, 29 outside |
| 4 dB | 0 in the gap, 3 outside | 4 in the gap, 4 outside |
| 5 dB | 0 and 0 | 0 in the gap, 1 outside |

Transient patches per km2 per pass, ascending passes only:

| contrast | lagoon interior, gap / outside | seaward of the mouth, gap / outside | open sea control, gap / outside |
|---|---|---|---|
| 3 dB | 0.52 / 1.13 | 2.33 / 2.87 | 2.74 / 3.26 |
| 4 dB | 0.00 / 0.06 | 0.36 / 0.52 | 0.25 / 0.19 |
| 5 dB | 0.00 / 0.00 | 0.00 / 0.13 | 0.03 / 0.01 |

The gap passes show no excess anywhere, and the lagoon interior's rate sits below the open
sea's rate of wave and speckle false alarms. The in-gap transients in open bay water are
all 900 m2 or smaller at 4.2 dB or less; the open sea throws up patches of 2,300 m2 and
10.9 dB on the same passes.

### The west-shore lead, 6 May 2025

The largest in-gap patch in the bay is 5,400 m2 at 5.3 dB on 6 May, the calmest pass of
the season, on the west shore at 12.1022 N, 68.2402 W, about 1.2 km north-north-west of the
Sorobon end of the Lac Bay segment: an elongated return along the shoreline.

| within 100 m of it | radar detected area | optical FAI above 0.02 | classifier uncertain / flagged |
|---|---|---|---|
| 6 May (radar 18:43; the 11:07 optical pass is haze) | 5,400 m2 | 51 px, in haze | 21 / 0 |
| 8 May (optical, 55% cloud near the spot) | | 35 px, max FAI 0.24 | 17 / 0 |
| 11 May (optical, 14% cloud) | | 82 px, max FAI 0.22 | 43 / 2 |
| 18 May (radar) | 1,800 m2 | | |
| 19 Mar, 24 Apr, 23 Jun, 22 Aug (radar) | 400 to 1,600 m2 | | |
| other 14 radar passes | none | | |
| 10 other usable optical looks, January to July | | 0 to 4 px | 0 to 1 / 0 |

Open water here sits near FAI -0.03. On 8 and 11 May the true-colour image shows a
red-brown band along that shore, part of which the scene classification layer masked as
cloud or shadow; it is absent on 12 March, 5 and 17 June and in July. The classifier called
most of the band uncertain, not sargassum.

Why it is a lead and not a detection: the radar patch hugs the shoreline, within 10 m of
the land mask, where the same shore gives smaller detections on four other passes, two of
them well after the event. The optical evidence comes two and five days after the radar,
through cloud gaps. It is one event. A floating mat piled against the leeward shore fits
all of it, and so do other things. Nobody has checked it on the water.

## Caveats

- Resolution. Native resolution is about 20 by 22 m on a 10 m pixel. The Lac flags sit in
  the first 10 to 20 m of the mangrove edge, exactly where radar mixes water with canopy,
  and where material caught in the roots would be.
- Timing. The radar pass is 7.6 h after or 16.4 h before the optical one. Drifting
  material can cross the lagoon in that time; material caught in the roots stays put.
- Wind. Every pass had 5 to 10 m/s of trade wind, and the ascending pass is at 18:43 local
  every 12 days, so a calm pass cannot be had from this orbit. A floating mat shows in
  radar by damping or adding roughness against a wind-roughened background, so the null
  result holds for these conditions, not for calm water.
- Noise floor. The Planetary Computer RTC product is not noise-corrected: a check against
  the GRD product with thermal noise subtracted lowered VH over the lagoon by 4.8 dB. Over
  water, VV+3VH is therefore VV plus a constant here.
- The canopy class comes from an NDVI threshold on a 19-look median, and the mouth of the
  lagoon from a straight line; the detector thresholds (3 dB, 4 pixels, 410 m) are choices,
  which is why the sensitivity table is there.
- No ground truth was compared.

## Rerunning the radar side

```bash
python scripts/eval_sentinel1_lac.py
```

The script is a reduced version that runs from a clean clone, with no Sentinel-2
re-reads and no GRD processing. It covers the 19 passes in the Planetary Computer RTC
collection (8 of them in the gap; the 31 March pass is GRD only), takes the gap from
`docs/bonaire_season.csv`, and leaves out land plus 15 m, the mapped mangrove plus 20 m,
and every pixel that looked like leaf canopy on at least half of its clear looks in the
2.1 season run (`docs/bonaire_persistence.npz`). Its canopy class therefore differs a
little from the analysis above, which used a 19-look NDVI median over 0.3. It writes
[`sentinel1_lac.json`](sentinel1_lac.json). Its numbers, on 5 October 2026:

| contrast | lagoon interior (2.81 km2), patches in the gap / outside | transient per km2 per ascending pass, lagoon interior, gap / outside | the same, open-sea control |
|---|---|---|---|
| 3 dB | 13 / 33 | 0.46 / 1.03 | 2.57 / 3.26 |
| 4 dB | 0 / 0 | 0.00 / 0.00 | 0.21 / 0.19 |
| 5 dB | 0 / 0 | 0.00 / 0.00 | 0.00 / 0.01 |

The largest in-gap patch in the bay's water that is not there on half the passes or more
is on 6 May 2025 at 12.1022 N, 68.2402 W: 4,900 m2 at 5.5 dB, seen on 2 passes. That is the
west-shore lead above, found again with the other canopy class. ERA5 wind on the 19 passes
was 5.6 to 10.0 m/s. The conclusions do not change: nothing floating-like in the open
lagoon at 4 dB, fewer transients there than over the open sea, and one shoreline lead on
6 May.
