"""Turn scene-wide detections into a per-beach-segment answer.

Everything upstream of this package answers "where is floating material in this
scene". That is not a question anyone schedules a crew on. A beach authority
manages named stretches of coast and needs to know which of them is affected
today, and, just as importantly, which ones nobody could see.

The distinction between *clean* and *not observed* is the reason this package
exists. Optical detection over the Caribbean is cloud-limited to the point where
LANOT, who run the closest comparable Sentinel-2 platform, report cloud fractions
above 90% and near-zero occurrences of a fully clear day. A product that reports
0% coverage on a 94%-cloud day is not conservative, it is wrong, and it is wrong
in the direction that sends nobody to a beach that needed clearing.
"""

from __future__ import annotations

from mdebris.coastal.segments import (
    BeachSegment,
    Observability,
    SegmentObservation,
    SegmentReport,
    aggregate_segments,
    append_history,
    load_segments,
    segment_cloud_fractions,
    surf_zone,
)

__all__ = [
    "BeachSegment",
    "Observability",
    "SegmentObservation",
    "SegmentReport",
    "aggregate_segments",
    "append_history",
    "load_segments",
    "segment_cloud_fractions",
    "surf_zone",
]
