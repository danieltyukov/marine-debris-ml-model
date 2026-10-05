# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-10-05

The season runner works on any island, removes mapped mangrove before classifying, and
four Dutch Caribbean islands are run and compared. The Lac Bay result from 2.0 is
corrected: most of what it flagged there was the mangrove canopy edge.

### Added

- `scripts/run_island_season.py --island <key>`, driven by `assets/islands/<key>/island.toml`: name, tiles, relative orbits, the area read (pinned or derived from the island polygon and the surf zones), leeward control and figure windows. An island that needs two tiles or two orbits is run in parts. The engine is in `mdebris.coastal.runner`, `mdebris.coastal.islands`, `mdebris.coastal.report` and `mdebris.viz.season`.
- A granule rule folded in from the per-island wrappers: only the configured tiles and orbits, one granule per datatake, and only granules whose footprint covers at least 99% of every surf zone. Left-out granules are listed with the reason in each report.
- A mangrove mask: OpenStreetMap mangrove and other vegetated wetland, buffered 20 m, removed from the water side before classifying. On by default; `--no-mangrove-mask` restores 2.0. Seagrass and untagged wetland stay classifiable.
- A per-pixel persistent vegetation check (NDVI above 0.5 and B08 above 0.08 on 90% of 10 or more clear looks), reported per segment next to the stationary check, and per-pass `vegetated_pixels` columns. It removes nothing.
- The longest wait for a fully clear pass per segment, and `coast_at_once` for how often a whole coast is seen on the same pass, in `mdebris.coastal.season`.
- `scripts/make_island_segments.py`, with the builder in `mdebris.coastal.build`: segments, land polygons and the mangrove mask from OpenStreetMap for any configured island, in its own UTM zone, with one or two coastline rings and the main ring or every ring as land.
- Aruba, Curaçao and Sint Maarten: configs, OpenStreetMap data and season runs for January to August 2025, with `docs/islands.md` comparing all four islands and `scripts/compare_islands.py` to rebuild it.
- `docs/lac_bay_mangroves.md`, `docs/index_vs_model.md` and `docs/sentinel1_lac.md`, with `scripts/eval_mangrove_mask.py`, `scripts/eval_index_vs_model.py` and `scripts/eval_sentinel1_lac.py`.
- `scripts/eval_glint.py`: the glint angle at each island's windward coast and the offshore 1.6 um reflectance for every pass, in `docs/glint.csv`.
- `mdebris.data.search_items` and `scene_ref`, for searches that need the granule footprint.
- Project website in `site/`, deployed to GitHub Pages by a workflow, with the interactive report kept at `report.html`. The mdebris mark, logo, favicon and social card are in `site/` too.
- `docs/RESULTS.md` with every measurement, its protocol and what it does not show, and `docs/ARCHITECTURE.md` with the modules, the cascade, the per-pixel path, throughput and the design reasoning.
- Contributor guide, code of conduct, security policy, `CITATION.cff`, `.editorconfig`, code owners, issue templates (bug, feature, new region or island), a pull request template and Dependabot for pip and GitHub Actions.
- Release workflow: a `v*` tag builds the sdist and wheel and creates a GitHub release with that version's changelog section.

### Changed

- Bonaire's season rerun with the mangrove mask, same 64 passes. Lac Bay's flagged pixels fall from 1,167 to 204 (stationary from 25 to 1) and Cai to Boka Washikemba's from 285 to 69. No usable share, longest gap or longest wait for a fully clear pass changes: Lac Bay is still usable on 64% of passes, with a longest gap of 20 days and no fully observed pass from 12 March to 17 June. 151 of the 204 remaining Lac Bay pixels sit near canopy the mangrove map misses.
- Sun glint: `docs/islands.md` measures it per orbit and shows that a single glint-prone orbit undercounts what can be seen (Sint Maarten) and can silence the classifier (Aruba).
- Bonaire's segments and land polygon moved to `assets/islands/bonaire/`. `scripts/run_bonaire_season.py` and `scripts/make_bonaire_segments.py` keep their 2.0 flags and defaults as thin wrappers.
- The contributor guide's "add an island" section describes the config-driven way.
- The README is shorter: what it does, quick start, a results summary, key figures and limitations, with the long sections moved to `docs/RESULTS.md` and `docs/ARCHITECTURE.md`. It now leads with the validation-threshold scores (sargassum F1 0.940, debris F1 0.511, with their intervals) rather than the best-F1-on-test numbers.
- Package metadata: a description that matches what the package does now, project URLs for the site, issues, changelog and documentation, and more keywords.

### Fixed

- Lac Bay. The OpenStreetMap coastline runs round the landward edge of Lac Bay's mangrove forest, so 2.0 classified the canopy as water, and 93% of the 1,167 Lac Bay pixels it flagged sit on that canopy or within 20 m of it. The README, `docs/RESULTS.md`, `docs/bonaire_season.md`, the site and the runner's docstring described those flags as the shallow back-bay, the lagoon bottom and the reef flat, said they were not a stationary confuser and called sargassum held in the bay the plausible reading. That text is corrected everywhere.
- The README said row 3 of the classification figure showed a ship painted as debris. It is row 2.
- The README said B01 outranks every named index. In the shipped model's report it ranks third, behind B05 and MNDWI.

## [2.0.0] - 2026-09-09

A rewrite. 1.x was the TensorFlow 1.14 object detector from NASA-IMPACT/marine_debris_ML, which needed commercial Planet imagery and no longer installs on current Python.

### Added

- A PyTorch package, `mdebris`, for Python 3.11 and 3.12, with a typer CLI (`samples`, `indices`, `search`, `detect`, `evaluate`, `beaches`, `config`, `serve`, `export-geojson`) and a FastAPI service.
- Free Sentinel-2 L2A imagery through STAC (Planetary Computer, with earth-search as a fallback) read with windowed range requests, and two real sample chips bundled so nothing needs the network.
- Spectral indices (FDI, FAI, NDVI, NDWI, MNDWI, NDMI, RNDVI, PI, kNDVI), each carrying its citation, and a screening cascade that skipped 44% of detector calls on a coastal scene off Accra.
- Open-vocabulary detection with OWLv2 and box-prompted SAM 2 masks, with confuser classes in the prompts.
- A per-pixel gradient-boosting classifier trained on MARIDA's scene-grouped split, over 11 bands and seven indices.
- The same classifier scored as a sargassum detector: F1 0.940 (0.87 to 0.98) with thresholds chosen on validation, against 0.511 (0.27 to 0.78) for debris.
- `mdebris.coastal`: detections rolled onto named beach segments as coverage, affected shoreline and observability, with a cloudy Cancun scene where seven of ten segments come back blind.
- A comparison with LANOT's published sargassum operator on the four MARIDA tiles both cover, and a packaged subset of MARIDA pixels for testing cloud and shallow-water masks.
- A sensor band ablation with patch-bootstrap intervals, a calibration report with an isotonic remap stored as JSON breakpoints, and a season run over every Sentinel-2 pass over Bonaire from January to August 2025 with per-segment observability and a stationary-pixel check.
- Figures regenerated from code, an interactive report on GitHub Pages, and CI on Python 3.11 and 3.12 with 894 offline tests.

### Changed

- Reflectance conversion requires an explicit offset, derived from the scene's processing baseline, because a default in either direction silently corrupts half the archive.

### Removed

- The TensorFlow 1.14 stack, the vendored TensorFlow Object Detection API, the Planet-and-SQS inference pipeline and the 2019 screenshots.

### Fixed

- The README's debris figure of 0.515 came from an earlier training run; the shipped model's best F1 on test is 0.656, and 0.511 with the threshold chosen on validation.
- The Cancun brief's NDWI water gate removes dense sargassum. The Bonaire runner masks land with a coastline polygon instead, and observability is measured over the pixels that could have been seen.
- A beach segment outside a raster's projection domain crashed GEOS; it now reads as unobserved.

## [1.0.0] - 2021-10-03

The original TensorFlow 1.14 code: an SSD-ResNet101-FPN trained on hand-labelled Planet imagery, adapted from NASA-IMPACT/marine_debris_ML. It was published without a version number; 2.0.0 numbers it 1.0.0 in hindsight.

[Unreleased]: https://github.com/danieltyukov/marine-debris-ml-model/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/danieltyukov/marine-debris-ml-model/compare/7d7847d...v2.1.0
[2.0.0]: https://github.com/danieltyukov/marine-debris-ml-model/tree/7d7847d
[1.0.0]: https://github.com/danieltyukov/marine-debris-ml-model/tree/1c8578e
