"""Every island's season side by side, per segment, from the season outputs.

    python scripts/compare_islands.py            # print the tables
    python scripts/compare_islands.py --write    # also rewrite them in docs/islands.md
    python scripts/compare_islands.py --write --figure assets/islands_observability.png

Reads ``docs/<prefix>_season.csv`` and ``.json`` for every part of every island under
``assets/islands/``; nothing is downloaded. An island run in two parts on two tiles of
the same orbit (Curaçao) is read as one island, its halves joined by date. A part marked
``comparable = false`` (Sint Maarten's second orbit) gets its own table and never enters
the comparison.

Besides each segment's usable share and longest gap, the table gives the longest wait
for a fully clear pass (two passes on which the segment was observed in full), and per
island how often the whole windward coast could be seen on the same pass. The tables
are written between the ``islands-table`` markers in ``docs/islands.md``; the prose
around them is left alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from mdebris.coastal.islands import list_islands, load_island
from mdebris.coastal.season import coast_at_once, summarize_history

START, END = "<!-- islands-table:start -->", "<!-- islands-table:end -->"


def _short(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%-d %b") if iso else "-"


def _read(docs: Path, prefix: str) -> tuple[list[dict], dict] | None:
    csv_path = docs / f"{prefix}_season.csv"
    json_path = docs / f"{prefix}_season.json"
    if not (csv_path.exists() and json_path.exists()):
        return None
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows, json.loads(json_path.read_text(encoding="utf-8"))


def _segment_lines(island, rows: list[dict], per_pixel: dict) -> tuple[list[str], list[dict]]:
    exposure = {s.segment_id: s.exposure for s in island.segments}
    lines, records = [], []
    for s in summarize_history(rows):
        px = per_pixel.get(s.segment_id, {})
        clear = (
            f"{s.longest_clear_gap_days} days ({_short(s.clear_gap_from)} to {_short(s.clear_gap_to)})"
            if s.longest_clear_gap_days is not None
            else "-"
        )
        near = px.get("flagged_near_vegetation")
        record = {
            "island": island.name,
            "segment_id": s.segment_id,
            "name": s.name,
            "exposure": exposure.get(s.segment_id, ""),
            "passes": s.n_passes,
            "usable": s.n_usable,
            "usable_fraction": s.usable_fraction,
            "longest_gap_days": s.longest_gap_days,
            "longest_clear_gap_days": s.longest_clear_gap_days,
            "clear_gap_from": s.clear_gap_from,
            "clear_gap_to": s.clear_gap_to,
            "passes_with_flags": s.n_detection_passes,
            "flagged_pixels": px.get("flagged_pixels"),
            "stationary_pixels": px.get("stationary_pixels"),
            "flagged_near_vegetation": near,
        }
        records.append(record)
        lines.append(
            f"| {island.name} | {s.name} | {record['exposure']} | "
            f"{100 * s.usable_fraction:.0f}% ({s.n_usable}/{s.n_passes}) | "
            f"{s.longest_gap_days if s.longest_gap_days is not None else '-'} | {clear} | "
            f"{s.n_detection_passes} | {px.get('flagged_pixels', 0):,} | "
            f"{px.get('stationary_pixels', 0):,} | {'-' if near is None else f'{near:,}'} |"
        )
    return lines, records


HEADER = [
    "| island | segment | exposure | usable | longest gap, days | longest wait for a fully clear pass "
    "| passes with flags | pixels ever flagged | of which stationary | of which near persistent vegetation |",
    "|---|---|---|---|---|---|---|---|---|---|",
]


def build(docs: Path) -> tuple[list[str], dict]:
    main, extra, missing, at_once = list(HEADER), [], [], []
    records: dict[str, list] = {"comparable": [], "other": []}
    for key in list_islands():
        island = load_island(key)
        rows: list[dict] = []
        per_pixel: dict = {}
        for part in [p for p in island.parts if p.comparable]:
            got = _read(docs, part.prefix(island.key))
            if got is None:
                missing.append(f"{island.name} ({part.prefix(island.key)})")
                continue
            rows += got[0]
            per_pixel.update(got[1].get("stationary", {}))
        if rows:
            lines, recs = _segment_lines(island, rows, per_pixel)
            main += lines
            records["comparable"] += recs
            windward = [s.segment_id for s in island.segments if s.exposure == "windward"]
            seen = coast_at_once(rows, windward)
            at_once.append(
                f"- {island.name}: on {seen.n_dates} pass dates, every windward segment was "
                f"usable on {seen.all_usable}, fully clear on {seen.all_clear}, and every one "
                f"was blind on {seen.none_usable}."
            )
        for part in [p for p in island.parts if not p.comparable]:
            got = _read(docs, part.prefix(island.key))
            if got is None:
                missing.append(f"{island.name} ({part.prefix(island.key)})")
                continue
            lines, recs = _segment_lines(island, got[0], got[1].get("stationary", {}))
            extra += [
                "",
                f"{island.name}, {part.label} (tile {', '.join(part.tiles)}, "
                f"orbit {', '.join(part.orbits)}), not part of the comparison:",
                "",
                *HEADER,
                *lines,
            ]
            records["other"] += [{**r, "part": part.key} for r in recs]
    out = [*main, "", "The whole windward coast on the same pass:", "", *at_once, *extra]
    if missing:
        out += ["", "Not run yet: " + ", ".join(missing) + "."]
    return out, records


def figure(records: list[dict], path: Path) -> None:
    """Usable share and the longest wait for a fully clear pass, per segment and island."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mdebris.viz.plots import save_figure

    rows = list(reversed(records))
    labels = [f"{r['island']}: {r['name']}" for r in rows]
    colours = {"windward": "#1f77b4", "leeward": "#9e9e9e"}
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 0.32 * len(rows) + 1.6), sharey=True, constrained_layout=True
    )
    y = range(len(rows))
    left.barh(
        y, [100 * r["usable_fraction"] for r in rows], color=[colours[r["exposure"]] for r in rows]
    )
    left.set_xlim(0, 100)
    left.set_xlabel("passes on which the segment could be used, %")
    right.barh(
        y,
        [r["longest_clear_gap_days"] or 0 for r in rows],
        color=[colours[r["exposure"]] for r in rows],
    )
    right.set_xlabel("longest wait for a fully clear pass, days")
    left.set_yticks(list(y))
    left.set_yticklabels(labels, fontsize=8)
    for ax in (left, right):
        ax.grid(axis="x", color="0.9")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "January to August 2025, one Sentinel-2 orbit per island (grey: leeward control)",
        fontsize=11,
    )
    save_figure(fig, path, tight=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--figure", type=Path, help="Also draw the summary figure here.")
    args = parser.parse_args()

    lines, records = build(args.docs_dir)
    text = "\n".join(lines)
    print(text)
    if args.write:
        page = args.docs_dir / "islands.md"
        body = page.read_text(encoding="utf-8") if page.exists() else f"{START}\n{END}\n"
        if START not in body or END not in body:
            raise SystemExit(f"{page} has no {START} ... {END} markers")
        head, rest = body.split(START, 1)
        _old, tail = rest.split(END, 1)
        page.write_text(f"{head}{START}\n{text}\n{END}{tail}", encoding="utf-8")
        (args.docs_dir / "islands.json").write_text(
            json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.figure:
        figure(records["comparable"], args.figure)


if __name__ == "__main__":
    main()
