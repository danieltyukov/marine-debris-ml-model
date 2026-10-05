"""A season of passes over the same segments, reduced to the numbers a proposal needs.

``append_history`` writes one row per segment per pass. That file answers "what did
each beach look like on each date". A monitoring proposal has to answer a different
question first: how often could the beach be seen at all, and how long did it go
unseen. Those are the gap statistics, and they are the honest size of the cloud
problem for a specific coast, measured rather than quoted from elsewhere.

This module is pure: it takes rows and returns dataclasses, so it is tested offline.
The rows are produced over the network by :mod:`mdebris.coastal.runner`, which
``scripts/run_island_season.py`` drives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np

from mdebris.coastal.segments import Observability

__all__ = ["CoastAtOnce", "SegmentSeason", "coast_at_once", "longest_gap", "summarize_history"]

_USABLE = {Observability.OBSERVED, Observability.PARTIAL}


@dataclass(slots=True)
class SegmentSeason:
    """How one segment fared over a run of passes."""

    segment_id: str
    name: str
    n_passes: int
    n_observed: int
    n_partial: int
    n_blind: int
    first_pass: str | None
    last_pass: str | None
    longest_gap_days: int | None
    median_gap_days: float | None
    n_detection_passes: int
    detection_dates: list[str] = field(default_factory=list)
    max_affected_front_m: float = 0.0
    # The longest stretch between two passes on which the segment was fully observed,
    # with its two ends. Partial looks see some of the water; this is the wait for a
    # pass that sees all of it.
    longest_clear_gap_days: int | None = None
    clear_gap_from: str | None = None
    clear_gap_to: str | None = None

    @property
    def n_usable(self) -> int:
        return self.n_observed + self.n_partial

    @property
    def usable_fraction(self) -> float:
        return self.n_usable / self.n_passes if self.n_passes else 0.0

    def to_row(self) -> dict[str, Any]:
        """Flat, JSON-safe record for a table."""
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "n_passes": self.n_passes,
            "n_observed": self.n_observed,
            "n_partial": self.n_partial,
            "n_blind": self.n_blind,
            "usable_fraction": self.usable_fraction,
            "first_pass": self.first_pass,
            "last_pass": self.last_pass,
            "longest_gap_days": self.longest_gap_days,
            "median_gap_days": self.median_gap_days,
            "n_detection_passes": self.n_detection_passes,
            "detection_dates": list(self.detection_dates),
            "max_affected_front_m": self.max_affected_front_m,
            "longest_clear_gap_days": self.longest_clear_gap_days,
            "clear_gap_from": self.clear_gap_from,
            "clear_gap_to": self.clear_gap_to,
        }


def longest_gap(dates: Iterable[str]) -> tuple[int, str, str] | None:
    """The longest wait between consecutive ISO dates: ``(days, from, to)``, or None."""
    ordered = sorted(dates)
    gaps = [
        ((date.fromisoformat(b) - date.fromisoformat(a)).days, a, b) for a, b in pairwise(ordered)
    ]
    return max(gaps) if gaps else None


def _iso_date(value: Any) -> str:
    text = str(value).strip()
    return date.fromisoformat(text[:10]).isoformat()


def _observability(value: Any) -> Observability:
    return value if isinstance(value, Observability) else Observability(str(value).strip().lower())


def summarize_history(rows: Iterable[Mapping[str, Any]]) -> list[SegmentSeason]:
    """Reduce per-pass rows to one :class:`SegmentSeason` per segment.

    Rows are grouped by ``segment_id`` in first-seen order and sorted by
    ``observed_on`` within each group. A repeated date for the same segment, which
    ``append_history`` produces on a rerun, keeps the last row and counts once.

    Gaps are measured between consecutive *usable* passes (observed or partial),
    in days. A blind pass does not shorten a gap: the beach was not seen.

    Detections count only on usable passes. A detection under a blind verdict is
    still a real positive, but it is not a look at the beach, and the summary is
    about looks.

    Raises:
        ValueError: If there are no rows.
    """
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    names: dict[str, str] = {}
    for row in rows:
        seg = str(row["segment_id"])
        when = _iso_date(row["observed_on"])
        grouped.setdefault(seg, {})[when] = row
        names.setdefault(seg, str(row.get("name") or seg))
    if not grouped:
        raise ValueError("no rows to summarise")

    seasons: list[SegmentSeason] = []
    for seg, by_date in grouped.items():
        dates = sorted(by_date)
        counts = dict.fromkeys(Observability, 0)
        usable_dates: list[str] = []
        clear_dates: list[str] = []
        detection_dates: list[str] = []
        max_front = 0.0
        for when in dates:
            row = by_date[when]
            obs = _observability(row["observability"])
            counts[obs] += 1
            if obs is Observability.OBSERVED:
                clear_dates.append(when)
            if obs in _USABLE:
                usable_dates.append(when)
                if int(row.get("detection_count", 0) or 0) > 0:
                    detection_dates.append(when)
                    max_front = max(max_front, float(row.get("affected_front_m", 0.0) or 0.0))

        gaps = [
            (date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in pairwise(usable_dates)
        ]
        clear = longest_gap(clear_dates)
        seasons.append(
            SegmentSeason(
                segment_id=seg,
                name=names[seg],
                n_passes=len(dates),
                n_observed=counts[Observability.OBSERVED],
                n_partial=counts[Observability.PARTIAL],
                n_blind=counts[Observability.BLIND],
                first_pass=dates[0],
                last_pass=dates[-1],
                longest_gap_days=max(gaps) if gaps else None,
                median_gap_days=float(np.median(gaps)) if gaps else None,
                n_detection_passes=len(detection_dates),
                detection_dates=detection_dates,
                max_affected_front_m=max_front,
                longest_clear_gap_days=clear[0] if clear else None,
                clear_gap_from=clear[1] if clear else None,
                clear_gap_to=clear[2] if clear else None,
            )
        )
    return seasons


@dataclass(frozen=True, slots=True)
class CoastAtOnce:
    """How often a set of segments could be seen on the same pass."""

    n_dates: int
    all_usable: int
    all_clear: int
    none_usable: int


def coast_at_once(rows: Iterable[Mapping[str, Any]], segment_ids: Iterable[str]) -> CoastAtOnce:
    """Count the passes on which every one, or none, of ``segment_ids`` could be seen.

    A segment with no row for a date (a granule that was not a pass on its tile) counts
    as blind on that date.
    """
    wanted = list(segment_ids)
    verdicts: dict[str, dict[str, Observability]] = {s: {} for s in wanted}
    for row in rows:
        seg = str(row["segment_id"])
        if seg in verdicts:
            verdicts[seg][_iso_date(row["observed_on"])] = _observability(row["observability"])
    dates = sorted({d for by_date in verdicts.values() for d in by_date})

    def seen(seg: str, when: str, ok: set[Observability]) -> bool:
        return verdicts[seg].get(when, Observability.BLIND) in ok

    clear = {Observability.OBSERVED}
    return CoastAtOnce(
        n_dates=len(dates),
        all_usable=sum(all(seen(s, d, _USABLE) for s in wanted) for d in dates),
        all_clear=sum(all(seen(s, d, clear) for s in wanted) for d in dates),
        none_usable=sum(not any(seen(s, d, _USABLE) for s in wanted) for d in dates),
    )
