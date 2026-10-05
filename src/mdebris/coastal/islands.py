"""Per-island configuration for the season runner and the segment builder.

Each island lives in its own folder, ``assets/islands/<key>/``, next to the files built
for it:

- ``island.toml``, the configuration read here;
- ``segments.geojson``, the named stretches of coast, built from OpenStreetMap;
- ``island.geojson``, the land polygons removed before anything is classified;
- ``mangroves.geojson``, the mapped mangrove and other vegetated wetland, removed from
  the water side as well (see :mod:`mdebris.coastal.masks` for why).

The configuration says everything that used to be hard-coded for Bonaire: the name to
print, which Sentinel-2 tiles and relative orbits to read, the area to read from each
pass, which segment is the leeward control, where the figures zoom in, and, for the
segment builder, the named landmarks the coast is cut at.

An island whose coast does not fit on one tile, or that sits in the overlap of two
orbits, is run in *parts*. A part is one complete run (one tile, one orbit filter, its
own CSV, report and persistence grid) over some or all of the segments. Curaçao is two
parts, west and east, on two tiles of the same orbit; Sint Maarten is one comparable
part on orbit R039 and a second, non-comparable one on orbit R139.

Parsing is pure and offline, so the configuration is tested without the network.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mdebris.types import GeoBBox

__all__ = [
    "DEFAULT_ISLANDS_DIR",
    "IslandConfig",
    "Landmark",
    "SeasonPart",
    "SegmentSpec",
    "Window",
    "derive_aoi",
    "list_islands",
    "load_island",
]

# assets/islands/ at the root of a source checkout. The runner and the builder are
# scripts run from a clone, so the configuration ships next to the data, not in the wheel.
DEFAULT_ISLANDS_DIR = Path(__file__).resolve().parents[3] / "assets" / "islands"

_LAND_MODES = ("main", "all")
_EXPOSURES = ("windward", "leeward")


@dataclass(frozen=True, slots=True)
class Window:
    """A lon/lat rectangle for a figure panel: ``west, north, east, south``."""

    title: str
    west: float
    north: float
    east: float
    south: float

    @property
    def corners(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((west, north), (east, south))``, the form the figure code crops with."""
        return (self.west, self.north), (self.east, self.south)


@dataclass(frozen=True, slots=True)
class Landmark:
    """A named point the coastline is cut at, snapped to the coast at build time."""

    key: str
    label: str
    lon: float
    lat: float
    osm: str = ""


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """One segment as the builder cuts it: the coast between two landmarks."""

    segment_id: str
    name: str
    start: str
    end: str
    exposure: str
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SeasonPart:
    """One complete season run over an island, or over part of it.

    Attributes:
        key: Suffix for the output files; ``""`` for an island run in one part.
        label: How the part is named in titles, e.g. ``"west, Watamula to Hato"``.
        tiles: Sentinel-2 MGRS tiles a granule may come from.
        orbits: Relative orbits (``"R082"``) a granule may come from; empty for any.
        segment_ids: Segments this part covers; empty for all of them.
        aoi: Area read from every pass. ``None`` derives it from the island polygon
            and the surf zones (see :func:`derive_aoi`).
        min_cover: A granule counts as a pass only if its data footprint covers at
            least this share of every segment's surf zone. Anything less reads as
            no-data over the rest, which the runner would count as cloud.
        comparable: False for a run that must not be mixed into cross-island tables,
            such as a second orbit over part of the coast.
        whole: Figure window for the overview panel; defaults to the area read.
        zoom: Figure window for the close-up panel.
        glint_point: ``(lon, lat)`` on the windward coast where
            ``scripts/eval_glint.py`` reads the sun and view angles.
        glint_box: ``(west, south, east, north)`` of open water a few kilometres
            offshore, where it reads the 1.6 um reflectance.
    """

    key: str
    label: str
    tiles: tuple[str, ...]
    orbits: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    aoi: GeoBBox | None = None
    min_cover: float = 0.99
    comparable: bool = True
    whole: Window | None = None
    zoom: Window | None = None
    glint_point: tuple[float, float] | None = None
    glint_box: tuple[float, float, float, float] | None = None

    def prefix(self, island_key: str) -> str:
        """File-name stem for this part's outputs, e.g. ``curacao_west``."""
        return f"{island_key}_{self.key}" if self.key else island_key

    def accepts(self, scene_id: str) -> bool:
        """Whether a granule id is on one of this part's tiles and orbits."""
        on_tile = any(f"_T{tile}_" in scene_id for tile in self.tiles)
        on_orbit = not self.orbits or any(f"_{orbit}_" in scene_id for orbit in self.orbits)
        return on_tile and on_orbit


@dataclass(frozen=True, slots=True)
class IslandConfig:
    """Everything the runner and the builder need to know about one island."""

    key: str
    name: str
    directory: Path
    utm_epsg: int
    parts: tuple[SeasonPart, ...]
    leeward_control: str | None = None
    # Sheltered lagoons on the windward coast (Lac Bay, Sint Joris Baai), reported apart
    # from the open coast in cross-island tables because their bottoms and shores differ.
    lagoons: tuple[str, ...] = ()
    land_buffer_m: float = 15.0
    mangrove_buffer_m: float = 20.0
    aoi_margin_m: float = 1000.0
    description: str = ""
    # Builder settings.
    osm_bbox: tuple[float, float, float, float] | None = None  # south, west, north, east
    segment_rings: int = 1
    land: str = "main"
    simplify_m: float = 15.0
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    segments: tuple[SegmentSpec, ...] = ()
    caveats: tuple[str, ...] = ()

    @property
    def segments_path(self) -> Path:
        return self.directory / "segments.geojson"

    @property
    def island_path(self) -> Path:
        return self.directory / "island.geojson"

    @property
    def mangroves_path(self) -> Path:
        return self.directory / "mangroves.geojson"

    def part(self, key: str | None = None) -> SeasonPart:
        """The part with this key; with no key, the island's only (or first) part."""
        if key is None:
            return self.parts[0]
        for part in self.parts:
            if part.key == key:
                return part
        known = ", ".join(repr(p.key) for p in self.parts)
        raise KeyError(f"{self.key} has no part {key!r}; parts: {known}")


def _window(raw: Any, title: str, where: str) -> Window:
    if isinstance(raw, dict):
        title = str(raw.get("title", title))
        box = raw.get("box")
    else:
        box = raw
    if not isinstance(box, list | tuple) or len(box) != 4:
        raise ValueError(f"{where}: a window is [west, north, east, south], got {box!r}")
    west, north, east, south = (float(v) for v in box)
    if not (west < east and south < north):
        raise ValueError(f"{where}: window {box!r} is not [west, north, east, south]")
    return Window(title=title, west=west, north=north, east=east, south=south)


def _bbox(raw: Any, where: str) -> GeoBBox:
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError(f"{where}: an aoi is [west, south, east, north], got {raw!r}")
    west, south, east, north = (float(v) for v in raw)
    if not (west < east and south < north):
        raise ValueError(f"{where}: aoi {raw!r} is not [west, south, east, north]")
    return GeoBBox(west=west, south=south, east=east, north=north)


def _floats(raw: Any, n: int, where: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or len(raw) != n:
        raise ValueError(f"{where}: expected {n} numbers, got {raw!r}")
    return tuple(float(v) for v in raw)


def _part(
    raw: dict[str, Any], island: str, figures: dict[str, Any], glint: dict[str, Any]
) -> SeasonPart:
    key = str(raw.get("key", ""))
    where = f"{island} part {key!r}"
    tiles = tuple(str(t).upper().removeprefix("T") for t in raw.get("tiles", ()))
    if not tiles:
        raise ValueError(f"{where}: give at least one Sentinel-2 tile in `tiles`")
    orbits = tuple(
        o if str(o).upper().startswith("R") else f"R{int(o):03d}" for o in raw.get("orbits", ())
    )
    min_cover = float(raw.get("min_cover", 0.99))
    if not 0.0 < min_cover <= 1.0:
        raise ValueError(f"{where}: min_cover must be in (0, 1], got {min_cover}")
    merged = {**figures, **raw.get("figures", {})}
    glint = {**glint, **raw.get("glint", {})}
    return SeasonPart(
        key=key,
        label=str(raw.get("label", "")),
        tiles=tiles,
        orbits=tuple(str(o).upper() for o in orbits),
        segment_ids=tuple(str(s) for s in raw.get("segments", ())),
        aoi=_bbox(raw["aoi"], where) if "aoi" in raw else None,
        min_cover=min_cover,
        comparable=bool(raw.get("comparable", True)),
        whole=_window(merged["whole"], "whole island", where) if "whole" in merged else None,
        zoom=_window(merged["zoom"], "close-up", where) if "zoom" in merged else None,
        glint_point=_floats(glint.get("point"), 2, f"{where} glint.point"),  # type: ignore[arg-type]
        glint_box=_floats(glint.get("offshore_box"), 4, f"{where} glint.offshore_box"),  # type: ignore[arg-type]
    )


def parse_island(blob: dict[str, Any], directory: Path) -> IslandConfig:
    """Build an :class:`IslandConfig` from a parsed ``island.toml``.

    Raises:
        ValueError: If a required field is missing or a value is out of range.
    """
    key = str(blob.get("key") or directory.name)
    for required in ("name", "utm_epsg"):
        if required not in blob:
            raise ValueError(f"{key}: island.toml has no `{required}`")
    season = blob.get("season", {})
    figures = blob.get("figures", {})
    raw_parts = season.get("parts") or []
    if not raw_parts:
        raise ValueError(f"{key}: island.toml has no [[season.parts]]")
    parts = tuple(_part(p, key, figures, blob.get("glint", {})) for p in raw_parts)
    if len({p.key for p in parts}) != len(parts):
        raise ValueError(f"{key}: two season parts share a key")

    osm = blob.get("osm", {})
    land = str(osm.get("land", "main"))
    if land not in _LAND_MODES:
        raise ValueError(f"{key}: osm.land must be one of {_LAND_MODES}, got {land!r}")
    landmarks = {
        k: Landmark(
            key=k,
            label=str(v.get("label", k)),
            lon=float(v["lon"]),
            lat=float(v["lat"]),
            osm=str(v.get("osm", "")),
        )
        for k, v in blob.get("landmarks", {}).items()
    }
    specs = []
    for raw in blob.get("segments", []):
        known = {"id", "name", "from", "to", "exposure", "note"}
        spec = SegmentSpec(
            segment_id=str(raw["id"]),
            name=str(raw["name"]),
            start=str(raw["from"]),
            end=str(raw["to"]),
            exposure=str(raw.get("exposure", "windward")),
            note=str(raw.get("note", "")),
            extra={k: v for k, v in raw.items() if k not in known},
        )
        if spec.exposure not in _EXPOSURES:
            raise ValueError(f"{key}: segment {spec.segment_id} has exposure {spec.exposure!r}")
        for end in (spec.start, spec.end):
            if end not in landmarks:
                raise ValueError(f"{key}: segment {spec.segment_id} names no landmark {end!r}")
        specs.append(spec)
    ids = {s.segment_id for s in specs}
    control = season.get("leeward_control")
    if control is not None and specs and control not in ids:
        raise ValueError(f"{key}: leeward_control {control!r} is not a segment id")
    lagoons = tuple(str(x) for x in season.get("lagoons", ()))
    if specs and set(lagoons) - ids:
        raise ValueError(f"{key}: lagoons {sorted(set(lagoons) - ids)} are not segment ids")
    for part in parts:
        unknown = set(part.segment_ids) - ids if specs else set()
        if unknown:
            raise ValueError(f"{key} part {part.key!r}: unknown segments {sorted(unknown)}")

    bbox = osm.get("bbox")
    return IslandConfig(
        key=key,
        name=str(blob["name"]),
        directory=directory,
        utm_epsg=int(blob["utm_epsg"]),
        parts=parts,
        leeward_control=str(control) if control is not None else None,
        lagoons=lagoons,
        land_buffer_m=float(blob.get("masks", {}).get("land_buffer_m", 15.0)),
        mangrove_buffer_m=float(blob.get("masks", {}).get("mangrove_buffer_m", 20.0)),
        aoi_margin_m=float(season.get("aoi_margin_m", 1000.0)),
        description=str(blob.get("description", "")).strip(),
        osm_bbox=tuple(float(v) for v in bbox) if bbox else None,  # type: ignore[arg-type]
        segment_rings=int(osm.get("segment_rings", 1)),
        land=land,
        simplify_m=float(osm.get("simplify_m", 15.0)),
        landmarks=landmarks,
        segments=tuple(specs),
        caveats=tuple(str(c).strip() for c in blob.get("caveats", [])),
    )


def load_island(name_or_path: str | Path, *, root: str | Path | None = None) -> IslandConfig:
    """Read an island's configuration.

    Args:
        name_or_path: An island key such as ``"bonaire"`` (looked up under ``root``),
            a folder holding ``island.toml``, or the path of an ``island.toml``.
        root: Folder of island folders. Defaults to ``assets/islands`` in the checkout.

    Raises:
        FileNotFoundError: If no ``island.toml`` is found.
    """
    path = Path(name_or_path)
    if path.suffix == ".toml":
        toml_path = path
    elif path.is_dir():
        toml_path = path / "island.toml"
    else:
        toml_path = Path(root or DEFAULT_ISLANDS_DIR) / str(name_or_path) / "island.toml"
    if not toml_path.is_file():
        known = ", ".join(list_islands(root)) or "none"
        raise FileNotFoundError(f"no island.toml at {toml_path}; islands found: {known}")
    blob = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return parse_island(blob, toml_path.parent)


def list_islands(root: str | Path | None = None) -> list[str]:
    """Keys of every island folder under ``root`` that holds an ``island.toml``."""
    base = Path(root or DEFAULT_ISLANDS_DIR)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "island.toml").is_file())


def derive_aoi(zones: list[Any], land: list[Any], *, margin_m: float, utm_epsg: int) -> GeoBBox:
    """The area to read from each pass: the surf zones, the land they touch, and a margin.

    Land polygons that do not touch any zone are left out, so an island run in parts
    does not pull the far half of the island (and the next tile) into each part's read.

    Args:
        zones: Surf-zone polygons in EPSG:4326.
        land: Land polygons in EPSG:4326.
        margin_m: Extra metres on every side.
        utm_epsg: Metric CRS to add the margin in.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform
    from shapely.ops import unary_union

    if not zones:
        raise ValueError("no surf zones to derive an area from")
    touching = [poly for poly in land if any(poly.intersects(z) for z in zones)]
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True).transform
    inv = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True).transform
    union = shapely_transform(fwd, unary_union([*zones, *touching]))
    west, south, east, north = shapely_transform(inv, union.envelope.buffer(margin_m)).bounds
    return GeoBBox(
        west=round(west, 4), south=round(south, 4), east=round(east, 4), north=round(north, 4)
    )
