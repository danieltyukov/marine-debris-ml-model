# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project website in `site/`, deployed to GitHub Pages by a workflow, with the interactive report kept at `report.html`. The mdebris mark, logo, favicon and social card are in `site/` too.
- `docs/RESULTS.md` with every measurement, its protocol and what it does not show, and `docs/ARCHITECTURE.md` with the modules, the cascade, the per-pixel path, throughput and the design reasoning.
- Contributor guide, code of conduct, security policy, `CITATION.cff`, `.editorconfig`, code owners, issue templates (bug, feature, new region or island), a pull request template and Dependabot for pip and GitHub Actions.
- Release workflow: a `v*` tag builds the sdist and wheel and creates a GitHub release with that version's changelog section.

### Changed

- The README is shorter: what it does, quick start, a results summary, key figures and limitations, with the long sections moved to `docs/RESULTS.md` and `docs/ARCHITECTURE.md`. It now leads with the validation-threshold scores (sargassum F1 0.940, debris F1 0.511, with their intervals) rather than the best-F1-on-test numbers.
- Package metadata: a description that matches what the package does now, project URLs for the site, issues, changelog and documentation, and more keywords.

### Fixed

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

[Unreleased]: https://github.com/danieltyukov/marine-debris-ml-model/compare/7d7847d...HEAD
[2.0.0]: https://github.com/danieltyukov/marine-debris-ml-model/tree/7d7847d
[1.0.0]: https://github.com/danieltyukov/marine-debris-ml-model/tree/1c8578e
