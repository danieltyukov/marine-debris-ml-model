# Contributing

Thanks for helping with mdebris. This page covers the environment, the tests and
linting, the data each part needs, how to add an island or a spectral index, and
what to check before opening a pull request.

## Setup

You need git and Python 3.11 or 3.12.

```sh
git clone https://github.com/danieltyukov/marine-debris-ml-model.git
cd marine-debris-ml-model
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[all,dev]"
```

Install torch from the CPU index first. The default PyPI wheel pulls about 2.5 GB of
CUDA libraries that a CPU machine never uses, and once the CPU build is present the
editable install treats torch as already satisfied. CI does the same.

The extras are split so that a partial install still works: `models` (torch,
transformers), `data` (STAC clients), `viz`, `api`, `eval` (scikit-learn, pandas) and
`dev` (pytest, ruff). `all` is everything except `dev`. Settings are read from
`MDEBRIS_*` environment variables or a `.env` file; `mdebris config` prints what
was resolved.

## Tests and linting

```sh
pytest -m "not network and not slow"
ruff check src tests scripts
ruff format --check src tests scripts
```

These are the commands CI runs, on Python 3.11 and 3.12. Run `ruff format src tests
scripts` before you push, since CI only checks the formatting.

Tests live in `tests/` and run offline with no credentials. Two markers mark the
exceptions: `network` for tests that need a live STAC endpoint, and `slow` for tests
that download model weights or large data. CI never runs `network` tests, and runs
`slow` ones only on pushes to `main` and manual runs. A new test that touches the network must carry the
marker; anything else must work offline, using the sample chips in
`src/mdebris/data/samples/` or small synthetic arrays. Write any files a test
creates under pytest's `tmp_path`.

## Data

| What | Needed for | Where it comes from |
|---|---|---|
| Nothing | the test suite, `mdebris samples`, `mdebris indices` | two Sentinel-2 chips bundled with the package |
| Sentinel-2 L2A | live detection, beach briefs, season runs | read over HTTP from Planetary Computer, no account |
| MARIDA, 1.1 GB | training and every `eval_*` script | downloaded on first use to `~/.cache/mdebris/marida`, resumable and checked against its md5; set `MDEBRIS_CACHE_DIR` to move it |
| `models/marida_spectral.joblib` | anything that scores pixels | written by `python scripts/train_marida.py`, never committed |
| `models/sargassum_calibration.json` | calibrated season runs | committed, written by `scripts/eval_calibration.py` |
| OpenStreetMap coastline | building segments | Overpass, through `make_bonaire_segments.py --fetch` |
| Model weights for OWLv2 and SAM 2 | the open-vocabulary path | downloaded on first use |

Do not commit a trained `.joblib` file. joblib is pickle-based, so loading one runs
code inside it, and `models/` is ignored for that reason. The calibrator is stored
as JSON breakpoints so the one artifact the runners need from the repository is
plain data. Do not commit imagery, downloaded datasets, or data that came to you
under terms that do not allow redistribution.

## Layout

The code is in `src/mdebris/`, the runs that produce every report and figure are in
`scripts/`, their outputs are in `docs/` and `assets/`, and the project site is in
`site/`. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has a table of modules and a
table of which script writes which file.

## Adding an island or a region

Open an issue with the "New region or island" template first, so the segments and
the control stretch can be agreed before anything runs. Then:

1. **Cut the coast into segments.** `scripts/make_bonaire_segments.py` is the
   pattern. Landmarks are OpenStreetMap nodes in a `LANDMARKS` table, segments are
   the arcs of the OSM coastline between consecutive landmarks in a `SEGMENTS`
   table, each with an exposure (windward or leeward) and a note. Include one
   stretch where sargassum should not arrive, as a control. The script writes the
   segments and an island polygon to `assets/`; keep the ODbL attribution it
   records in the file.
2. **Run a season.** `scripts/run_bonaire_season.py` takes `--segments`, `--island`,
   `--start`, `--end` and the output paths as arguments. Its area of interest is the
   `BONAIRE` bounding box at the top of the script, and its report names tile 19PEP,
   so another island needs its own box there. Try `--max-scenes 2` before a full
   run; the run resumes from its CSV if it stops.
3. **Test the segment file.** `tests/test_bonaire_segments.py` shows what to pin:
   every segment loads, has a name, sits on the island, is a plausible length, and
   exactly one is the control.
4. **Report it the same way.** Observability first (usable passes and the longest
   gap), then detections, then the per-pixel stationary check. Add the results to
   [docs/RESULTS.md](docs/RESULTS.md) under the islands marker, and say plainly
   that the detections have not been checked on the ground unless they have.

## Adding a spectral index

1. Write the function in `src/mdebris/indices/spectral.py`. It takes reflectance
   arrays in `[0, 1]` and returns float32, guards zero denominators the way the
   existing helpers do, and its docstring gives the formula and the reference with a
   DOI.
2. Register it in `INDEX_REGISTRY` with an `IndexSpec`: canonical band names in the
   order the function takes them, the attainable range, and the citation. The band
   ablation reads the band list to decide when the index has to be dropped.
3. Check whether it survives the Sentinel-2 reflectance offset. The module docstring
   of `src/mdebris/data/scaling.py` explains which indices care.
4. Add tests to `tests/test_indices_spectral.py`: a hand-computed value, the range,
   and the zero-denominator case.
5. Adding it to the classifier's features (`_INDEX_FEATURES` in
   `src/mdebris/models/spectral.py`) changes the feature columns, so the model has
   to be retrained and `eval_sargassum.py`, `eval_band_ablation.py` and
   `eval_calibration.py` rerun, with the reports in `docs/` committed alongside.

## How results are reported

These rules keep the numbers in this repository comparable and honest:

- Choose thresholds on MARIDA's validation split and score on the test split. If a
  number is tuned on the test split, label it as such.
- Give intervals from a bootstrap over test patches, not pixels.
- Keep negative results. A weak score stays in the tables with its interval.
- Every number in the README, the site and `docs/` must come from a report a script
  in `scripts/` writes. Update the report and the prose in the same pull request.
- A detection over a new coast is not ground truth. Say so until someone has
  checked it.

## Commits

Subjects describe the change in plain words, under 72 characters, in the imperative
("Add the Bonaire segments", not "Added"). Do not add generated-by notices, AI or tool
attribution trailers, or session links. No emojis in code, comments, docs or commit
messages.

## Pull request checklist

- `pytest -m "not network and not slow"`, `ruff check src tests scripts` and
  `ruff format --check src tests scripts` pass locally.
- New logic has a test that runs offline.
- No `.joblib` files, imagery, credentials or data under restrictive terms in the
  diff.
- Reports in `docs/` regenerated by their script if the numbers changed, and the
  README, `docs/RESULTS.md` and the site updated to match.
- `CHANGELOG.md` has an entry under Unreleased for anything a user would notice.

## Releases

1. Bump the version in `pyproject.toml` and `__version__` in
   `src/mdebris/__init__.py`.
2. In `CHANGELOG.md`, move the entries under Unreleased to a new heading for the
   version and date.
3. Commit, tag the commit `vX.Y.Z` and push the tag.

The release workflow checks that the tag matches the package version, builds the
sdist and wheel, and creates a GitHub release with that version's changelog section
as its notes.

## Questions

Questions and ideas are welcome in
[issues](https://github.com/danieltyukov/marine-debris-ml-model/issues).
