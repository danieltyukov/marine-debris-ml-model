"""Every Sentinel-2 pass over an island for a season, rolled onto named coast segments.

    python scripts/run_island_season.py --island bonaire
    python scripts/run_island_season.py --island curacao --part west --start 2025-01-01 --end 2025-08-31
    python scripts/run_island_season.py --island aruba --summary-only

The island is a folder under ``assets/islands/`` holding ``island.toml``, the segments,
the land polygons and the mangrove mask (``scripts/make_island_segments.py`` builds the
last three from OpenStreetMap). The configuration names the Sentinel-2 tiles and
relative orbits to read, the area read from each pass, the leeward control and the
figure windows; an island that needs two tiles or two orbits is run in parts, and with
no ``--part`` every part is run in turn.

Outputs, per part, with ``<prefix>`` the island key plus the part key (``bonaire``,
``curacao_west``): ``docs/<prefix>_season.csv`` (one row per segment per pass),
``docs/<prefix>_season.md`` and ``.json`` (the report), ``docs/<prefix>_persistence.npz``
(per-pixel counts) and three figures in ``assets/``.

The run is resumable: a scene already in the CSV is skipped, so a network failure
costs one scene, not the season. Persistence only counts passes read in one
invocation, so delete the CSV to start a season afresh. ``--summary-only`` rebuilds the
report and figures from the CSV and the npz without downloading anything.

The mangrove mask is on by default. ``--no-mangrove-mask`` classifies mangrove canopy
inside the surf zones as if it were water, which is what version 2.0 did; see
``docs/lac_bay_mangroves.md`` for why that is wrong.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import time
from pathlib import Path

from mdebris.coastal.islands import load_island
from mdebris.coastal.runner import SeasonInputs, SeasonOutputs, run_season

log = logging.getLogger("season")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared with ``run_bonaire_season.py``."""
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--model", type=Path, default=Path("models/marida_spectral.joblib"))
    parser.add_argument(
        "--calibrator", type=Path, default=Path("models/sargassum_calibration.json")
    )
    parser.add_argument("--calibration-json", type=Path, default=Path("docs/calibration.json"))
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--high", type=float, default=None, help="Override the sargassum edge.")
    parser.add_argument("--surf-zone-m", type=float, default=500.0)
    parser.add_argument(
        "--no-mangrove-mask",
        action="store_true",
        help="Classify mapped mangrove inside the surf zones, as version 2.0 did.",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip downloads; rebuild the report and figures from the CSV and npz.",
    )
    parser.add_argument(
        "--pass-figure-scene",
        default=None,
        help="Only redraw the pass figure for this scene id; nothing else runs.",
    )


def inputs_for(args: argparse.Namespace, island, part, *, segments=None, land=None) -> SeasonInputs:
    mangroves = None if args.no_mangrove_mask else island.mangroves_path
    if mangroves is not None and not mangroves.exists():
        raise SystemExit(
            f"{mangroves} is missing; build it with scripts/make_island_segments.py "
            f"--island {island.key} --fetch, or pass --no-mangrove-mask"
        )
    return SeasonInputs(
        island=island,
        part=part,
        start=args.start,
        end=args.end,
        segments_path=Path(segments or island.segments_path),
        island_path=Path(land or island.island_path),
        mangroves_path=mangroves,
        model=args.model,
        calibrator=None if args.no_calibration else args.calibrator,
        calibration_json=args.calibration_json,
        high=args.high,
        surf_zone_m=args.surf_zone_m,
        max_scenes=args.max_scenes,
        command=" ".join(["python", *map(shlex.quote, sys.argv)]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--island", required=True, help="Island key, folder or island.toml.")
    parser.add_argument("--part", default=None, help="Run one part only (default: all).")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    add_common_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    island = load_island(args.island)
    parts = [island.part(args.part)] if args.part is not None else list(island.parts)
    started = time.perf_counter()
    for part in parts:
        outputs = SeasonOutputs.for_prefix(
            part.prefix(island.key), docs=args.docs_dir, assets=args.assets_dir
        )
        run_season(
            inputs_for(args, island, part),
            outputs,
            summary_only=args.summary_only,
            pass_figure_scene=args.pass_figure_scene,
        )
    log.info("total %.1f min", (time.perf_counter() - started) / 60)


if __name__ == "__main__":
    main()
