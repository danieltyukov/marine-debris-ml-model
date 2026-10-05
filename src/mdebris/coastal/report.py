"""The Markdown report a season run writes next to its CSV.

Every number in it comes from the run's own rows and persistence counts. The prose
around the tables is the same for every island, apart from the island's name, its
leeward control and the caveats listed in its ``island.toml``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

__all__ = ["season_markdown"]


def _short(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%-d %b") if iso else "-"


def _wrap(text: str) -> str:
    import textwrap

    return textwrap.fill(
        " ".join(text.split()), width=88, break_long_words=False, break_on_hyphens=False
    )


def _or_dash(value: Any) -> str:
    return "-" if value is None else str(value)


def _relpath(path: Any) -> str:
    text = str(path)
    for marker in ("assets/", "docs/"):
        if marker in text:
            return text[text.index(marker) :]
    return text


def season_markdown(
    seasons: Sequence[Any],
    passes: Mapping[str, Mapping[str, Any]],
    points: Any,
    *,
    inputs: Any,
    segments: Sequence[Any],
    per_pixel: Mapping[str, Mapping[str, int]],
    segment_pixels: Mapping[str, Mapping[str, int]],
    dropped: Mapping[str, str],
    failures: Sequence[tuple[str, str]],
    outputs: Any,
    mangrove_mask: bool,
) -> str:
    """Render the report for one part of one island's season."""
    from mdebris.coastal.masks import (
        PERSISTENT_MIN_PASSES,
        PERSISTENT_SHARE,
        VEGETATED_B08,
        VEGETATED_NDVI,
    )
    from mdebris.coastal.runner import STATIONARY_MIN_PASSES, STATIONARY_SHARE

    island, part = inputs.island, inputs.part
    n_pass = len(passes)
    platforms = Counter(p["platform"] for p in passes.values())
    by_id = {s.segment_id: s for s in segments}
    title = f"{island.name}, {part.label}" if part.label else island.name
    tiles = " and ".join(part.tiles)
    orbit = (
        f" (relative orbit{'s' if len(part.orbits) > 1 else ''} {', '.join(part.orbits)})"
        if part.orbits
        else ""
    )

    land_line = (
        f"Land is removed with the OpenStreetMap coastline polygons plus a {island.land_buffer_m:.0f} m "
        "seaward strip, and mapped mangrove and other vegetated wetland is removed with it, plus a "
        f"{island.mangrove_buffer_m:.0f} m strip (`{_relpath(inputs.mangroves_path)}`)."
        if mangrove_mask
        else f"Land is removed with the OpenStreetMap coastline polygons plus a "
        f"{island.land_buffer_m:.0f} m seaward strip. The mangrove mask was off for this run, so "
        "mangrove canopy inside a surf zone was classified as if it were water."
    )
    platform_text = (
        ", " + ", ".join(f"{n} from {p}" for p, n in sorted(platforms.items())) if platforms else ""
    )
    intro = " ".join(
        [
            f"Produced by `{inputs.command}`." if inputs.command else "",
            f"{n_pass} passes over tile {tiles}{orbit} from {inputs.start} to {inputs.end}"
            f"{platform_text}. The search leaves out granules reported as 100% cloud, and keeps",
            f"one granule per datatake whose footprint covers at least {part.min_cover:.0%} of every",
            "surf zone. Segments are cut from the OpenStreetMap coastline at named landmarks",
            f"(`{_relpath(inputs.segments_path)}`); the surf zone is the water within",
            f"{inputs.surf_zone_m:.0f} m of each.",
            land_line,
            f"A pixel is called sargassum at calibrated probability >= {points.high:.2f}, which",
            f"after calibration means the model's own estimate is at least {points.high:.0%}; it is",
            f"uncertain in [{points.low:.3f}, {points.high:.2f}) and clear below. The lower edge keeps",
            "98% of true sargassum on MARIDA's validation split, and `docs/calibration.md` records",
            "what the upper edge achieves on the test split.",
        ]
    ).strip()
    lines = [
        f"# {title}: one season of Sentinel-2 passes",
        "",
        _wrap(intro),
        "",
        "## How often each stretch of coast could be seen",
        "",
        "| segment | exposure | passes | observed | partial | blind | usable | longest gap, days "
        "| median gap | longest wait for a fully clear pass | passes with sargassum "
        "| most front affected, m | pixels ever flagged | of which stationary |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in seasons:
        exposure = (
            by_id[s.segment_id].properties.get("exposure", "") if s.segment_id in by_id else ""
        )
        clear = (
            f"{s.longest_clear_gap_days} days ({_short(s.clear_gap_from)} to {_short(s.clear_gap_to)})"
            if s.longest_clear_gap_days is not None
            else "-"
        )
        px = per_pixel.get(s.segment_id)
        lines.append(
            f"| {s.name} | {exposure} | {s.n_passes} | {s.n_observed} | {s.n_partial} | {s.n_blind} | "
            f"{100 * s.usable_fraction:.0f}% | {_or_dash(s.longest_gap_days)} | "
            f"{_or_dash(s.median_gap_days)} | {clear} | {s.n_detection_passes} | "
            f"{s.max_affected_front_m:.0f} | "
            + (f"{px['flagged_pixels']:,} | {px['stationary_pixels']:,} |" if px else "- | - |")
        )
    lines += [
        "",
        "`usable` is observed plus partial: passes on which a detection means something.",
        "A segment is blind on a pass when more than 70% of the water side of its surf",
        "zone is cloud by the scene classification layer, partial above 20%. Every",
        "non-cloud pixel on the water side is classified, surf included: there is no NDWI",
        "water gate, because on MARIDA no dense sargassum pixel passes one. The longest",
        "gap is the most days between two consecutive usable looks; the longest wait for a",
        "fully clear pass is the same between two passes on which the segment was observed",
        "in full.",
        "",
        "## Checks on the detection columns",
        "",
        "Flagged pixels are not confirmed sargassum. Two per-pixel counts kept across the",
        "season name the flags that are probably something else.",
        "",
        f"- Stationary: flagged on at least {STATIONARY_SHARE:.0%} of the passes that could see it, with",
        f"  at least {STATIONARY_MIN_PASSES} such passes. Floating material moves; a reef, a wreck or a",
        "  structure does not.",
        "- Near persistent vegetation: on or within 20 m of a pixel that looked like leaf canopy",
        f"  (NDVI above {VEGETATED_NDVI}, B08 above {VEGETATED_B08}) on at least {PERSISTENT_SHARE:.0%} of at least",
        f"  {PERSISTENT_MIN_PASSES} clear looks. That is canopy the mangrove map missed, or its edge. The",
        "  check removes nothing; it only counts.",
        "",
        "| segment | water-side pixels | removed by the mangrove mask | persistent vegetation pixels "
        "| flagged pixels | of which near persistent vegetation |",
        "|---|---|---|---|---|---|",
    ]
    for s in seasons:
        px = per_pixel.get(s.segment_id, {})
        counts = segment_pixels.get(s.segment_id, {})
        lines.append(
            f"| {s.name} | {counts.get('water_side_pixels', 0):,} | "
            f"{counts.get('mangrove_masked_pixels', 0):,} | "
            f"{_or_dash(px.get('persistent_vegetation_pixels'))} | "
            f"{_or_dash(px.get('flagged_pixels'))} | {_or_dash(px.get('flagged_near_vegetation'))} |"
        )
    lines += [
        "",
        "Pixel counts are 10 m pixels on this run's grid. A pixel where two surf zones",
        f"overlap counts in both. `{_relpath(outputs.persistence_figure)}` shows where the",
        "flags sit.",
        "",
        "## Pass by pass",
        "",
        "| date | platform | tile cloud % | surf-zone cloud % | not water % | observed | partial "
        "| blind | sargassum pixels | segments with sargassum |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _scene_id, p in sorted(passes.items(), key=lambda kv: kv[1]["date"]):
        not_water = (
            100 * sum(p["not_water_fractions"]) / len(p["not_water_fractions"])
            if p["not_water_fractions"]
            else 0.0
        )
        lines.append(
            f"| {p['date']} | {p['platform']} | {p['tile_cloud_pct']:.0f} | "
            f"{100 * p['zone_cloud_fraction']:.0f} | {not_water:.0f} | "
            f"{p['observed']} | {p['partial']} | {p['blind']} | "
            f"{p['sargassum_pixels']} | {', '.join(p['affected']) or '-'} |"
        )
    if dropped:
        lines += ["", "## Granules not counted as passes", ""]
        lines += [f"- `{sid}`: {why}" for sid, why in dropped.items()]
    if failures:
        lines += ["", "## Passes that could not be read", ""]
        lines += [f"- `{sid}`: {why}" for sid, why in failures]
    lines += [
        "",
        "## What this does not say",
        "",
        "No pixel here has been checked against the beach. The classifier was trained and",
        "calibrated on MARIDA, whose sargassum pixels sit in the Gulf of Honduras and",
        "around Roatan, with a few hundred off Port-au-Prince and Santo Domingo; none is",
        f"from {island.name}. Reef flats, sand bottoms, lagoons and wrecks here are surfaces",
        "the model has never been graded on. The observability columns do not depend on",
        "the classifier at all and are the part of this report to trust first: they say",
        "how often an optical satellite could look at each stretch of coast, which is a",
        "property of the weather and the orbit, not of the model.",
        "",
    ]
    control = island.leeward_control
    if control and control in by_id:
        lines += [
            f"{by_id[control].name} is a leeward control. Sargassum arrives on the windward",
            "side; if that row fills with detections, the model is finding something other",
            "than sargassum.",
            "",
        ]
    if not part.comparable:
        lines += [
            "This run is a second look at part of the coast from another orbit. It is reported",
            "on its own and is not mixed into tables that compare islands on one orbit each.",
            "",
        ]
    for caveat in island.caveats:
        lines += [caveat, ""]
    for note in inputs.extra_notes:
        lines += [note, ""]
    return "\n".join(line for line in lines if line is not None)
