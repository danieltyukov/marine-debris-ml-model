# Is the probability an uncertainty? Calibration of the spectral classifier

Produced by `python scripts/eval_calibration.py`. The classifier from
`train_marida.py` is loaded, its any-sargassum probability is scored on the
held-out test split, and an isotonic remap fitted on the validation split is
applied to see how much of the miscalibration is fixable without retraining.

## Summary

| task | test positives | ECE raw | ECE calibrated | Brier raw | Brier calibrated |
|---|---|---|---|---|---|
| any sargassum | 1,641 | 0.0030 | **0.0008** | 0.0016 | 0.0008 |
| marine debris | 381 | 0.0089 | **0.0019** | 0.0065 | 0.0018 |

Expected calibration error is the count-weighted gap between what the model
says and what happened, over ten equal-width probability bins; zero is a model
whose 0.9 means nine in ten. The remap is monotone, so it does not change which
pixels rank above which, only what number is attached to them.

## Reliability, any sargassum, raw

| bin | pixels | mean probability | observed fraction |
|---|---|---|---|
| 0.0 to 0.1 | 191,979 | 0.001 | 0.000 |
| 0.1 to 0.2 | 457 | 0.147 | 0.020 |
| 0.2 to 0.3 | 219 | 0.248 | 0.037 |
| 0.3 to 0.4 | 142 | 0.342 | 0.035 |
| 0.4 to 0.5 | 114 | 0.447 | 0.088 |
| 0.5 to 0.6 | 87 | 0.546 | 0.115 |
| 0.6 to 0.7 | 107 | 0.647 | 0.121 |
| 0.7 to 0.8 | 76 | 0.749 | 0.171 |
| 0.8 to 0.9 | 87 | 0.850 | 0.299 |
| 0.9 to 1.0 | 1,595 | 0.993 | 0.953 |

## Reliability, any sargassum, after isotonic calibration on validation

| bin | pixels | mean probability | observed fraction |
|---|---|---|---|
| 0.0 to 0.1 | 192,321 | 0.000 | 0.000 |
| 0.1 to 0.2 | 716 | 0.133 | 0.056 |
| 0.2 to 0.3 | 10 | 0.242 | 0.200 |
| 0.3 to 0.4 | 25 | 0.316 | 0.080 |
| 0.4 to 0.5 | 230 | 0.447 | 0.226 |
| 0.5 to 0.6 | 0 | - | - |
| 0.6 to 0.7 | 23 | 0.621 | 0.348 |
| 0.7 to 0.8 | 33 | 0.700 | 0.515 |
| 0.8 to 0.9 | 0 | - | - |
| 0.9 to 1.0 | 1,505 | 0.999 | 0.989 |

## The abstention band

Two edges, chosen on the validation split. Above the upper edge, precision is at
least 90%, so a pixel there is called sargassum. Below the lower edge,
at most 2% of true sargassum pixels are lost, so a pixel there is called
clear. In between, the model says so.

| probabilities | lower edge | upper edge | test pixels in band | of which sargassum | sargassum called clear | non-sargassum called sargassum |
|---|---|---|---|---|---|---|
| raw | 0.052 | 0.501 | 1,361 (0.70%) | 40 | 19 of 1,641 | 369 |
| calibrated | 0.056 | 0.250 | 1,542 (0.79%) | 56 | 16 of 1,641 | 253 |

## The operating point the Bonaire runner uses

Once the probability is calibrated, the number is its own statement, and a cut
at 0.9 means the model's estimate for that pixel is at least nine in ten. On the
test split that cut gives:

| task | called | precision | recall | uncertain, low edge to 0.9 | of which positive |
|---|---|---|---|---|---|
| any sargassum | 1,505 | **0.989** | 0.907 | 1,859 | 137 |
| marine debris | 230 | **0.822** | 0.496 | 24,617 | 192 |

The band is what the beach brief should carry as a third state next to
"detected" and "clear": not as a smaller number, but as a different kind of
answer, the same way `observability` already separates "not seen" from "seen
and clean".

## What this does not say

Calibration on MARIDA pixels is calibration against MARIDA's annotators over
MARIDA's scenes. A calibrated probability over Bonaire in 2025 is a hope, not a
measurement, until pixels from that coast have been checked against the ground.
The remap is stored as breakpoints in `models/sargassum_calibration.json` so it
can be refitted the moment such pixels exist.
