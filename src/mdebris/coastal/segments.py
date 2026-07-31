"""Aggregate detections onto named stretches of coast.

Three numbers come out per segment, and they answer different questions:

``coverage``
    Detected area divided by surf-zone area. Comparable between segments of
    different length, so it ranks them.
``affected_front_m``
    How many metres of that segment's shoreline have detected material within
    the surf zone. This is the number a crew supervisor acts on, because crews
    are dispatched along a beach front, not over an area of water.
``observability``
    Whether the segment was actually seen. Reported separately from coverage
    rather than folded into it, because averaging a confidence into a load
    figure destroys the one distinction the user needs to make.

Geometry arrives in EPSG:4326 and every measurement is metric, so the work
happens in a projected CRS. UTM is chosen from the collective centroid of the
segments, which matches the native CRS of the Sentinel-2 scenes the detections
came from and keeps areas true to within a fraction of a percent for any single
municipality's coastline.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as date_type
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from mdebris.types import DetectionSet, SurfaceClass

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps rasterio/numpy off the import path
    import affine
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BLIND_ABOVE",
    "DEFAULT_PARTIAL_ABOVE",
    "DEFAULT_SURF_ZONE_M",
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

WGS84 = "EPSG:4326"

# Sargassum is transported in the nearshore by wind and Stokes drift, so material
# detected a few hundred metres offshore on the overpass is what lands. 500 m is a
# defensible default rather than a tuned one; no landfall validation exists to tune
# it against, and pretending otherwise would be the dishonest part.
DEFAULT_SURF_ZONE_M = 500.0

# Above 70% cloud over a segment there is not enough clear water left to call it
# either way. Between 20% and 70% a detection is still meaningful but an absence
# is not, which is what PARTIAL means.
DEFAULT_BLIND_ABOVE = 0.70
DEFAULT_PARTIAL_ABOVE = 0.20

# Classes that count as floating biomass for a beach-clearing customer. Debris is
# included because a beach authority clears whatever washes up; the label matters
# for reporting, not for whether a truck is sent.
DEFAULT_TARGET_LABELS: frozenset[SurfaceClass] = frozenset(
    {SurfaceClass.SARGASSUM, SurfaceClass.DEBRIS}
)


class Observability(StrEnum):
    """Whether the segment could be seen, kept separate from what was seen.

    ``BLIND`` is not a smaller number than ``OBSERVED``; it is a different kind of
    answer. Any consumer that sorts these as if they were ordered severities is
    misusing them, which is why this is a string enum and not a float.
    """

    OBSERVED = "observed"
    PARTIAL = "partial"
    BLIND = "blind"

    @property
    def is_actionable(self) -> bool:
        """True when an absence of detections means the beach is actually clear."""
        return self is Observability.OBSERVED


def _utm_epsg(lon: float, lat: float) -> str:
    """EPSG code of the UTM zone containing a point.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        An ``EPSG:326xx`` (north) or ``EPSG:327xx`` (south) code string.
    """
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _metric_crs(geoms: Sequence[Any]) -> str:
    """Pick one projected CRS for a collection of WGS84 geometries.

    A single CRS is used for the whole report so that intersections between
    segments and detections stay exact. Choosing per-segment would be more
    accurate in isolation and would silently break every cross-geometry
    operation.

    Args:
        geoms: Shapely geometries in EPSG:4326.

    Returns:
        A UTM EPSG code string.
    """
    merged = unary_union(list(geoms))
    centroid = merged.centroid
    epsg = _utm_epsg(centroid.x, centroid.y)
    west, _, east, _ = merged.bounds
    if east - west > 6.0:
        log.warning(
            "segments span %.1f degrees of longitude, wider than one UTM zone; "
            "areas away from %s are distorted",
            east - west,
            epsg,
        )
    return epsg


def _is_finite(geom: Any) -> bool:
    """True when every coordinate of a projected geometry is a real number.

    pyproj returns ``inf`` rather than raising when a point falls outside a
    projection's valid domain, and ``shapely.buffer`` on an infinite geometry
    segfaults GEOS instead of erroring. So the check has to happen before the
    buffer, not after it.
    """
    import math as _math

    if geom.is_empty:
        return False
    return all(_math.isfinite(v) for v in geom.bounds)


def _transformers(metric_crs: str):
    """Build the forward and inverse WGS84 <-> metric transform pair."""
    from pyproj import CRS, Transformer

    src, dst = CRS.from_user_input(WGS84), CRS.from_user_input(metric_crs)
    # always_xy pins both sides to (lon, lat) / (easting, northing). Without it
    # pyproj honours the EPSG:4326 authority axis order and hands back (lat, lon),
    # which mirrors every geometry through the diagonal.
    fwd = Transformer.from_crs(src, dst, always_xy=True).transform
    inv = Transformer.from_crs(dst, src, always_xy=True).transform
    return fwd, inv


@dataclass(frozen=True, slots=True)
class BeachSegment:
    """A named stretch of coast a customer manages as one unit.

    The geometry is normally a LineString along the shoreline, because that is how
    beach fronts are described and measured. A Polygon is accepted too, for
    customers who hold their beaches as parcels; the surf zone is then the buffer
    around the parcel rather than around a line.
    """

    segment_id: str
    geometry: Any  # shapely geometry in EPSG:4326
    name: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be a non-empty string")
        if self.geometry is None or self.geometry.is_empty:
            raise ValueError(f"segment {self.segment_id!r} has empty geometry")

    @property
    def label(self) -> str:
        """Human-facing name, falling back to the id when unnamed."""
        return self.name or self.segment_id


def load_segments(path: str | Path) -> list[BeachSegment]:
    """Read beach segments from a GeoJSON FeatureCollection.

    The id is taken from the feature's ``id`` member, or from a ``segment_id``,
    ``id`` or ``name`` property, in that order. A feature with none of those gets
    a positional id rather than being rejected, so a coastline exported from a
    GIS with no id column still loads.

    Args:
        path: GeoJSON file holding one feature per segment.

    Returns:
        Segments in file order.

    Raises:
        ValueError: If the file is not a FeatureCollection, or holds no usable
            features.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection, got {data.get('type')!r}")

    segments: list[BeachSegment] = []
    for i, feature in enumerate(data.get("features", [])):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        props = dict(feature.get("properties") or {})
        raw_id = (
            feature.get("id")
            or props.get("segment_id")
            or props.get("id")
            or props.get("name")
            or f"segment_{i:03d}"
        )
        segments.append(
            BeachSegment(
                segment_id=str(raw_id),
                geometry=shape(geometry),
                name=str(props.get("name", "")),
                properties=props,
            )
        )

    if not segments:
        raise ValueError(f"{path} contains no features with geometry")
    return segments


def surf_zone(
    segment: BeachSegment, metres: float = DEFAULT_SURF_ZONE_M, *, metric_crs: str | None = None
) -> Any:
    """The band of water a segment's arrivals come from, as a WGS84 polygon.

    Args:
        segment: Segment to buffer.
        metres: Buffer distance in metres.
        metric_crs: Projected CRS to buffer in. Defaults to the UTM zone of the
            segment centroid.

    Returns:
        A shapely Polygon in EPSG:4326.

    Raises:
        ValueError: If ``metres`` is not positive.
    """
    if metres <= 0:
        raise ValueError(f"surf zone must be a positive distance, got {metres}")
    crs = metric_crs or _metric_crs([segment.geometry])
    fwd, inv = _transformers(crs)
    return shapely_transform(inv, shapely_transform(fwd, segment.geometry).buffer(metres))


@dataclass(frozen=True, slots=True)
class SegmentObservation:
    """What one segment looked like on one date."""

    segment_id: str
    name: str
    observability: Observability
    cloud_fraction: float
    coverage: float
    detected_area_m2: float
    affected_front_m: float
    front_length_m: float
    detection_count: int
    observed_on: str | None = None
    scene_id: str | None = None
    geometry: Any | None = None  # the surf zone, EPSG:4326

    @property
    def affected_front_fraction(self) -> float:
        """Share of the segment's front with material offshore, 0 when unmeasurable."""
        return self.affected_front_m / self.front_length_m if self.front_length_m > 0 else 0.0

    def to_row(self) -> dict[str, Any]:
        """Flat record for CSV and tabular display."""
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "observed_on": self.observed_on or "",
            "observability": str(self.observability),
            "cloud_fraction": round(self.cloud_fraction, 4),
            "coverage": round(self.coverage, 6),
            "detected_area_m2": round(self.detected_area_m2, 1),
            "affected_front_m": round(self.affected_front_m, 1),
            "front_length_m": round(self.front_length_m, 1),
            "detection_count": self.detection_count,
            "scene_id": self.scene_id or "",
        }

    def to_feature(self) -> dict[str, Any]:
        """GeoJSON Feature of the surf zone, carrying the metrics as properties.

        Raises:
            ValueError: If no surf-zone geometry was attached.
        """
        if self.geometry is None:
            raise ValueError(f"segment {self.segment_id!r} has no surf-zone geometry")
        return {
            "type": "Feature",
            "id": self.segment_id,
            "geometry": mapping(self.geometry),
            "properties": self.to_row(),
        }


@dataclass(slots=True)
class SegmentReport:
    """Every segment for one observation date, plus how it was produced."""

    observations: list[SegmentObservation] = field(default_factory=list)
    observed_on: str | None = None
    scene_id: str | None = None
    surf_zone_m: float = DEFAULT_SURF_ZONE_M
    metric_crs: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self):
        return iter(self.observations)

    def ranked(self) -> list[SegmentObservation]:
        """Observed segments worst first, then partial, with blind segments last.

        Blind segments sort to the end deliberately. They are not low-risk, they
        are unknown, and a supervisor reading the top of this list should be
        looking at measurements rather than at gaps.
        """
        order = {Observability.OBSERVED: 0, Observability.PARTIAL: 1, Observability.BLIND: 2}
        return sorted(
            self.observations,
            key=lambda o: (order[o.observability], -o.coverage, -o.affected_front_m),
        )

    @property
    def blind(self) -> list[SegmentObservation]:
        """Segments no clear-sky pixel covered."""
        return [o for o in self.observations if o.observability is Observability.BLIND]

    @property
    def affected(self) -> list[SegmentObservation]:
        """Observed segments carrying any detected material."""
        return [
            o
            for o in self.observations
            if o.detection_count > 0 and o.observability is not Observability.BLIND
        ]

    def to_rows(self) -> list[dict[str, Any]]:
        return [o.to_row() for o in self.ranked()]

    def to_geojson(self) -> dict[str, Any]:
        """FeatureCollection of surf zones, one feature per segment."""
        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [o.to_feature() for o in self.ranked() if o.geometry is not None],
            "properties": {
                "count": len(self.observations),
                "observed_on": self.observed_on,
                "scene_id": self.scene_id,
                "surf_zone_m": self.surf_zone_m,
                "metric_crs": self.metric_crs,
                "blind_segments": len(self.blind),
                "affected_segments": len(self.affected),
                **self.meta,
            },
        }

    def write_geojson(self, path: str | Path, *, indent: int | None = 2) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_geojson(), indent=indent, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def write_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0])) if rows else None
            if writer is not None:
                writer.writeheader()
                writer.writerows(rows)
        return out


def _detection_union(ds: DetectionSet, labels: frozenset[SurfaceClass] | None, fwd) -> Any:
    """Union of the target detections' footprints, in the metric CRS.

    Individual detection polygons overlap after cross-tile merging, so summing
    their areas double-counts. Unioning first is what makes ``coverage`` a real
    fraction rather than a number that can exceed one.
    """
    from shapely.geometry import Polygon

    geoms = [
        d.geometry
        for d in ds.detections
        if d.geometry is not None and (labels is None or d.label in labels)
    ]
    if not geoms:
        return Polygon()
    return unary_union([shapely_transform(fwd, g) for g in geoms])


def aggregate_segments(
    ds: DetectionSet,
    segments: Sequence[BeachSegment],
    *,
    surf_zone_m: float = DEFAULT_SURF_ZONE_M,
    cloud_fractions: Mapping[str, float] | None = None,
    blind_above: float = DEFAULT_BLIND_ABOVE,
    partial_above: float = DEFAULT_PARTIAL_ABOVE,
    labels: Iterable[SurfaceClass] | None = DEFAULT_TARGET_LABELS,
    observed_on: str | None = None,
) -> SegmentReport:
    """Roll a scene's detections up onto named beach segments.

    A segment classed ``BLIND`` still reports whatever was detected over it. The
    detections are real; it is the *absence* of detections that becomes
    unreliable under cloud, so suppressing the positives would throw away good
    information to signal a caveat that the observability field already carries.

    Args:
        ds: Georeferenced detections. Detections without geometry are ignored.
        segments: Segments to report on.
        surf_zone_m: How far offshore to look, in metres.
        cloud_fractions: Cloud fraction per ``segment_id``, 0 to 1. Missing
            segments are treated as fully clear, so callers that have no cloud
            information get the old two-state behaviour rather than a crash.
        blind_above: Cloud fraction strictly above this is ``BLIND``.
        partial_above: Cloud fraction strictly above this is ``PARTIAL``.
        labels: Detection classes to count. ``None`` counts every class.
        observed_on: Observation date. Defaults to the scene datetime's date.

    Returns:
        One observation per input segment, in input order.

    Raises:
        ValueError: If ``segments`` is empty, ``surf_zone_m`` is not positive, or
            the thresholds are not ordered ``0 <= partial_above <= blind_above <= 1``.
    """
    if not segments:
        raise ValueError("no segments given")
    if surf_zone_m <= 0:
        raise ValueError(f"surf_zone_m must be positive, got {surf_zone_m}")
    if not 0.0 <= partial_above <= blind_above <= 1.0:
        raise ValueError(
            f"need 0 <= partial_above <= blind_above <= 1, "
            f"got partial_above={partial_above}, blind_above={blind_above}"
        )

    label_set = None if labels is None else frozenset(labels)
    metric_crs = _metric_crs([s.geometry for s in segments])
    fwd, inv = _transformers(metric_crs)

    detected = _detection_union(ds, label_set, fwd)
    # A shoreline point counts as affected when material sits within the surf-zone
    # distance of it, which is the same relation as the surf zone seen from the
    # other side. Buffering the detections once is far cheaper than testing every
    # segment against every detection.
    reach = detected.buffer(surf_zone_m) if not detected.is_empty else detected

    scene_id = ds.scene.scene_id if ds.scene else None
    when = observed_on
    if when is None and ds.scene is not None and ds.scene.datetime:
        when = str(ds.scene.datetime)[:10]

    clouds = dict(cloud_fractions or {})
    observations: list[SegmentObservation] = []
    for segment in segments:
        projected = shapely_transform(fwd, segment.geometry)
        zone = projected.buffer(surf_zone_m)
        zone_area = zone.area

        inside = detected.intersection(zone) if not detected.is_empty else detected
        detected_area = float(inside.area)
        # Points and lines have zero area, so a Polygon segment measures its own
        # extent while a LineString measures the buffered zone. Dividing by the
        # zone in both cases keeps coverage comparable across the two shapes.
        coverage = float(detected_area / zone_area) if zone_area > 0 else 0.0

        front_length = float(projected.length)
        if front_length > 0 and not reach.is_empty:
            affected_front = float(projected.intersection(reach).length)
        else:
            affected_front = 0.0

        count = sum(
            1
            for d in ds.detections
            if d.geometry is not None
            and (label_set is None or d.label in label_set)
            and shapely_transform(fwd, d.geometry).intersects(zone)
        )

        cloud = float(clouds.get(segment.segment_id, 0.0))
        if not 0.0 <= cloud <= 1.0:
            raise ValueError(
                f"cloud fraction for {segment.segment_id!r} must be in [0, 1], got {cloud}"
            )
        if cloud > blind_above:
            state = Observability.BLIND
        elif cloud > partial_above:
            state = Observability.PARTIAL
        else:
            state = Observability.OBSERVED

        observations.append(
            SegmentObservation(
                segment_id=segment.segment_id,
                name=segment.label,
                observability=state,
                cloud_fraction=cloud,
                coverage=coverage,
                detected_area_m2=detected_area,
                affected_front_m=affected_front,
                front_length_m=front_length,
                detection_count=count,
                observed_on=when,
                scene_id=scene_id,
                geometry=shapely_transform(inv, zone),
            )
        )

    return SegmentReport(
        observations=observations,
        observed_on=when,
        scene_id=scene_id,
        surf_zone_m=surf_zone_m,
        metric_crs=metric_crs,
        meta={"target_labels": sorted(str(x) for x in label_set) if label_set else "all"},
    )


def segment_cloud_fractions(
    segments: Sequence[BeachSegment],
    cloud_mask: NDArray[np.bool_],
    transform: affine.Affine,
    *,
    src_crs: Any,
    surf_zone_m: float = DEFAULT_SURF_ZONE_M,
) -> dict[str, float]:
    """Fraction of each segment's surf zone that a cloud mask marks unusable.

    This is what separates a real absence from an unobserved one, so it is worth
    the extra raster pass. Pixels outside the raster are counted as unobserved
    rather than clear: a segment at the edge of the AOI genuinely was not seen.

    Args:
        segments: Segments to measure.
        cloud_mask: Boolean array, True on cloud-contaminated pixels, matching
            the convention in ``mdebris.indices.masks``.
        transform: Affine mapping pixel ``(column, row)`` to ``src_crs``.
        src_crs: CRS of the raster. Sentinel-2 scenes are UTM.
        surf_zone_m: Surf-zone width in metres, matching ``aggregate_segments``.

    Returns:
        Cloud fraction per ``segment_id``, each in ``[0, 1]``. A segment whose
        surf zone falls entirely outside the raster maps to 1.0.

    Raises:
        ValueError: If ``cloud_mask`` is not two-dimensional.
    """
    import numpy as np
    from rasterio.features import geometry_mask

    mask = np.asarray(cloud_mask)
    if mask.ndim != 2:
        raise ValueError(f"cloud_mask must be 2-D, got shape {mask.shape}")
    cloudy = mask.astype(bool)
    height, width = cloudy.shape

    from pyproj import CRS

    raster_crs = CRS.from_user_input(src_crs)
    fwd, _ = _transformers(raster_crs.to_string())

    fractions: dict[str, float] = {}
    for segment in segments:
        projected = shapely_transform(fwd, segment.geometry)
        if not _is_finite(projected):
            # The segment lies outside the raster projection's domain entirely, so
            # this raster says nothing about it. That is ignorance, not clear sky.
            log.warning(
                "segment %r does not project into %s; recording it as unobserved",
                segment.segment_id,
                raster_crs.to_string(),
            )
            fractions[segment.segment_id] = 1.0
            continue
        zone = projected.buffer(surf_zone_m)
        # geometry_mask returns True *outside* the shapes by default; invert=True
        # flips it so True means inside, which is the sense wanted here.
        inside = geometry_mask(
            [mapping(zone)],
            out_shape=(height, width),
            transform=transform,
            invert=True,
            all_touched=True,
        )
        total = int(inside.sum())
        fractions[segment.segment_id] = 1.0 if total == 0 else float(cloudy[inside].sum() / total)
    return fractions


def append_history(report: SegmentReport, path: str | Path) -> Path:
    """Append a report to a running per-segment CSV, creating it if absent.

    The dated record is a product in its own right. A hotel association's standing
    complaint about sargassum coverage is that old photographs recirculate and get
    presented as today's beach; a satellite-timestamped series of measurements is
    the thing that answers that, and it only exists if every run is kept.

    Rows are appended, never deduplicated. A rerun of the same date appends again,
    which is visible in the file and recoverable, whereas silently overwriting a
    prior measurement is not.

    Args:
        report: Report to append.
        path: CSV file. Parent directories are created.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = report.to_rows()
    if not rows:
        return out
    exists = out.exists() and out.stat().st_size > 0
    with out.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return out


def _today() -> str:
    """Today as an ISO date string, isolated so tests can monkeypatch it."""
    return date_type.today().isoformat()
