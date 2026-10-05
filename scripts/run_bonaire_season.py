"""Every Sentinel-2 pass over Bonaire for a season, rolled onto named coast segments.

    python scripts/run_bonaire_season.py --start 2025-01-01 --end 2025-08-31

Kept so the commands in the documentation still run. It is
``scripts/run_island_season.py --island bonaire`` with the 2.0 flags: ``--segments``,
``--island`` (here the land polygon, as before) and one flag per output file. The island
configuration is ``assets/islands/bonaire/island.toml``; the engine is
``mdebris.coastal.runner``, whose docstring explains the method.

Two things changed in 2.1. Mapped mangrove is removed from the water side before
classifying (``--no-mangrove-mask`` restores the 2.0 behaviour). The 2.0 run classified
Lac Bay's mangrove canopy, because the OpenStreetMap coastline runs round the landward
edge of the forest, and 93% of the Lac Bay pixels it flagged sit on that canopy or
within 20 m of it: they were the mangrove edge, not the lagoon floor or sargassum held
in the bay (``docs/lac_bay_mangroves.md``). And the report adds a per-pixel persistent
vegetation check next to the stationary one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdebris.coastal.islands import load_island
from mdebris.coastal.runner import SeasonOutputs, run_season
from run_island_season import add_common_arguments, inputs_for

BONAIRE_DIR = Path("assets/islands/bonaire")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--segments", type=Path, default=BONAIRE_DIR / "segments.geojson")
    parser.add_argument("--island", type=Path, default=BONAIRE_DIR / "island.geojson")
    parser.add_argument("--csv", type=Path, default=Path("docs/bonaire_season.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/bonaire_season.md"))
    parser.add_argument("--figure", type=Path, default=Path("assets/bonaire_season.png"))
    parser.add_argument("--pass-figure", type=Path, default=Path("assets/bonaire_pass.png"))
    parser.add_argument("--persistence", type=Path, default=Path("docs/bonaire_persistence.npz"))
    parser.add_argument(
        "--persistence-figure", type=Path, default=Path("assets/bonaire_persistence.png")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    island = load_island("bonaire")
    outputs = SeasonOutputs(
        csv=args.csv,
        report=args.report,
        figure=args.figure,
        pass_figure=args.pass_figure,
        persistence=args.persistence,
        persistence_figure=args.persistence_figure,
    )
    run_season(
        inputs_for(args, island, island.part(), segments=args.segments, land=args.island),
        outputs,
        summary_only=args.summary_only,
        pass_figure_scene=args.pass_figure_scene,
    )


if __name__ == "__main__":
    main()
