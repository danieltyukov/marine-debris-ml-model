# Three extensions after the Timmermans conversation

Date: 2026-09-09
Status: implemented in this branch; every number in the README sections it adds is
produced by a script in `scripts/` and can be regenerated.

## Why

On 3 September 2026 Joris Timmermans (TU Delft, Geoscience and Remote Sensing) walked
through his group's work and named the things he cares about when a classifier is put
in front of a government or a small island administration. Four of them are testable on
this repository today, with data already on disk or free over HTTP:

1. **Every satellite has a different band set**, and a method that only works on one
   sensor is a method that stops working when the sensor does. He uses whatever
   modality is available. His student's thesis made the same point from the other
   side: adding a modality that carries no new information (LiDAR height for two
   grasses of the same height) made the model worse, not better, so modalities should
   be optional and independent rather than concatenated.
2. **Data gaps from cloud are the dominant failure of optical monitoring**, and his
   group's method for studying them is ablation: remove observations by scenario,
   measure how much the answer degrades.
3. **Decision makers ask for uncertainties**, and a probability is only an uncertainty
   if it is calibrated. Data assimilation needs to know how far to trust each
   observation before it can nudge a model state.
4. **The thesis lab requires a direct connection to the Dutch Caribbean**, and Bonaire
   is the island where the data is free and the sargassum problem is acute.

Point 2 cannot be done on MARIDA the way his group does it, because MARIDA patches are
single-date. What can be done is the band-set version of the same ablation, and the
Bonaire time series measures the cloud gap directly on the site that matters.

## What is built

### A. Sensor band ablation (`mdebris.eval.ablation`, `scripts/eval_band_ablation.py`)

The same gradient-boosting classifier is trained on MARIDA with the feature columns of
a given sensor configuration, and the rest set to NaN, which the model treats as
missing. Indices that need a dropped band are dropped with it, so "no SWIR" also means
no FDI, no FAI and no MNDWI. Scenarios:

| scenario | bands kept | stands for |
|---|---|---|
| `sentinel2` | all 11 | the baseline |
| `no_swir` | drop B11, B12 | a sensor without shortwave infrared |
| `no_red_edge` | drop B05, B06, B07, B8A | Landsat-like band set |
| `superdove` | B01, B02, B03, B04, B05, B8A | PlanetScope 8-band SuperDove, nearest S2 bands |
| `dove4` | B02, B03, B04, B08 | PlanetScope 4-band, the NASA IMPACT imagery |
| `rgb` | B02, B03, B04 | what a photographic prior sees |
| `indices_only` | none, 7 indices | the physical features alone |

Two physical baselines sit beside them: a single index (FDI, then FAI) thresholded
with no learning at all. Every threshold, for the learned models and the index
baselines alike, is chosen on MARIDA's validation split and evaluated on the test
split. The README's existing 0.948 was a best-F1 threshold chosen on test, which is
optimistic; the ablation table reports the honest version and says so.

Reported per scenario, for any-sargassum and for marine debris: precision, recall, F1
at the validation-selected threshold, and the threshold. Output `docs/band_ablation.md`,
`docs/band_ablation.json`, `assets/band_ablation.png`.

What this answers before the NASA IMPACT 3 m polygons arrive: how much of the
sargassum and debris score is carried by bands a 3 m sensor does not have. If `dove4`
is close to `sentinel2`, the resolution comparison will be a resolution comparison. If
it is not, the comparison is confounded and has to be reported as bands plus
resolution.

### B. Calibration and abstention (`mdebris.eval.calibration`, `scripts/eval_calibration.py`)

For the any-sargassum probability on the held-out split:

- a reliability table (10 equal-width bins), expected calibration error and Brier score,
  before and after isotonic calibration fitted on the validation split;
- an abstention band: the probability interval in which the model should say
  "uncertain" rather than yes or no. Its upper edge is the 90% precision operating
  point already used for dispatch. Its lower edge is the largest threshold that still
  keeps 98% of true sargassum pixels above it. Pixels between the two are reported as
  a third state, with the fraction of test pixels that land there and what they are.

The calibrator is stored as `models/sargassum_calibration.json` (breakpoints only, no
pickle) and applied by the Bonaire runner so its probabilities are the calibrated ones.
Output `docs/calibration.md`, `docs/calibration.json`, `assets/calibration.png`.

### C. Bonaire windward coast, one season (`mdebris.coastal.season`, `assets/bonaire_segments.geojson`, `scripts/make_bonaire_segments.py`, `scripts/run_bonaire_season.py`)

Segments are cut from the OpenStreetMap coastline of Bonaire at named landmarks, south
tip to north coast: Willemstoren, Sorobon, Cai, Boka Washikemba, Spelonk, Boka Onima,
Playa Chikitu, Boka Kokolishi, plus one leeward control segment on the Kralendijk
waterfront that should stay clean. The geometry is ODbL, attributed in the file.

The runner takes every Sentinel-2 L2A pass over tile 19PEP between two dates, reads the
bands over the island window with HTTP range requests, classifies the water pixels
inside the surf zones, applies the calibrator, and rolls each pass onto the segments
with the existing `aggregate_segments`. It appends one row per segment per pass to
`docs/bonaire_season.csv` and then summarises:

- per segment: passes, how many were observed, partial and blind, the longest run of
  days without a usable look, and the passes on which material was detected;
- per pass: island-wide cloud over the surf zones.

`mdebris.coastal.season.summarize_history` is the pure function that turns the CSV into
those statistics, so it is tested offline. The runner itself needs the network and is a
script, not a test.

Output `docs/bonaire_season.md`, `docs/bonaire_season.csv`, `assets/bonaire_season.png`.

## What is deliberately not built

**SAR-conditioned temporal imputation.** Proposed in the meeting: treat a pixel's
Sentinel-2 time series as a masked sequence, and use Sentinel-1 backscatter as the
conditioning input to reconstruct cloud-gapped steps. Joris's caution was that this
assumes a direct link between the two sensors that may not hold for the target. For
sargassum the doubt is sharper than for vegetation: a raft is a few centimetres of
biomass on a moving surface, and whether C-band backscatter responds to it at all is
an open question in the literature, not a modelling detail. It would be built as an
experiment with a held-out clear-sky set, not as a feature, and not before someone who
knows the SAR side has said whether the signal exists.

**Drift and landfall forecasting.** Unchanged from `TASKS.md`: not before there is a
customer who wants it and field data to check it against.

## Testing

Offline, in `tests/`: feature masking by scenario and index dependency; threshold
selection on a validation set; reliability table, ECE and Brier score on synthetic
data with known calibration; isotonic round trip through JSON; abstention band
ordering; season summary statistics on hand-built histories; the Bonaire segment
file loads, every segment is named, and every geometry lies within the island's
bounding box.

Network runs are scripts and are recorded in `docs/` with the scene ids they used.

## What changed while building it

Two things the first Bonaire run exposed, both kept in the record because they change
earlier claims.

**The NDWI water gate removes dense sargassum.** The Cancun brief gates every pixel on
NDWI above zero before classifying it, to keep bright sand out of a model that has no
land class. On MARIDA's held-out split, no Dense Sargassum pixel and 3.6% of Sparse
Sargassum pixels have NDWI above zero, because a raft has the near-infrared of
vegetation. The gate therefore keeps the confusers out by discarding the target. The
first Bonaire run, gated, found at most eight isolated pixels on any pass of a season
the island's press called its worst; the same pass on 21 April 2025 without the gate
finds 26 pixels and 443 m of affected front in Lac Bay. The Bonaire runner now removes
land with the OpenStreetMap island polygon buffered 15 m seaward, which is what the
Wageningen report did with a digitised coastline and a 10 m strip, and classifies every
non-cloud pixel on the water side. The Cancun script still carries the gate; its
numbers should be read as sparse-sargassum numbers until it gets a coastline of its own.

**Observability was diluted by land.** A surf zone is a buffer on both sides of a
shoreline, so half of every zone was land that could never be cloud. The Kralendijk
control came out blind on 33 of 64 passes for a second reason: the scene classification
layer calls bright roofs cloud. `segment_cloud_fractions` gained a `valid_mask` so the
fraction is taken over the water side only.

**A stationary detection is a confuser.** Without the gate, the shallow back-bay of Lac
and the reef flat at Cai are called sargassum on every clear pass, January included,
by the hundreds of pixels; the leeward control and the open coast stay at zero. Floating
material does not hold position for months, so the runner now accumulates per pixel how
many passes flagged it and how many could see it, calls a pixel stationary when the
ratio is at least one half over at least five passes, reports per segment how many
flagged pixels are stationary, and draws the map. The reading is the one the
Wageningen report already gave: the open coast and the bays need separate models, and
MARIDA has no lagoon in it. That is the Bonaire-specific training data a thesis-lab
project would collect.

> Correction, 2026-10-05: most of these Lac Bay and Cai flags were not the bottom or the
> reef flat but the mangrove canopy edge, which the OpenStreetMap land mask left on the
> water side. Version 2.1 masks mapped mangrove; see `docs/lac_bay_mangroves.md`.
