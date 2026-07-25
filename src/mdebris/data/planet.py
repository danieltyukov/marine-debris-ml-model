"""Optional Planet connector, kept for parity with the legacy pipeline.

The legacy project could only run against Planet imagery, so a paid API key was a hard
prerequisite for detecting anything at all. That is no longer true: the default pipeline
runs on free Sentinel-2 through :mod:`mdebris.data.stac` with no credentials. This module
exists so that anyone who already has a Planet subscription can keep using it, and it is
deliberately small.

Planet's higher resolution (roughly 3 m for PlanetScope against 10 m for Sentinel-2) is a
real advantage for small debris patches. The trade-off is that Planet publishes only four
broad bands, so the spectral indices that separate plastic from sargassum (FDI needs B06
and B11) cannot be computed, and the cascade in :mod:`mdebris.pipeline` falls back to
running the detector on every tile.

The API key is read from the ``PL_API_KEY`` environment variable via
:data:`mdebris.config.settings` and is never written to disk or logged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mdebris.config import settings
from mdebris.types import GeoBBox, SceneRef, TileRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DATA_API",
    "DEFAULT_ITEM_TYPES",
    "TILE_API",
    "PlanetAuthError",
    "PlanetClient",
    "PlanetError",
]

DATA_API = "https://api.planet.com/data/v1"
TILE_API = "https://tiles.planet.com/data/v1"

# PlanetScope 4-band and 8-band surface reflectance scenes, the products the legacy
# pipeline consumed.
DEFAULT_ITEM_TYPES: tuple[str, ...] = ("PSScene",)


class PlanetError(RuntimeError):
    """A Planet API request failed."""


class PlanetAuthError(PlanetError):
    """No Planet API key is configured, or the configured key was rejected."""


class PlanetClient:
    """Thin wrapper over the Planet Data API v1.

    Constructing the client does not touch the network and does not require a key, so
    importing this module is always safe. The key is demanded at the first call that
    actually needs it, with a message that says how to set it.
    """

    def __init__(self, api_key: str | None = None, *, timeout: float = 30.0) -> None:
        """
        Args:
            api_key: Overrides ``PL_API_KEY``. Prefer the environment variable so the key
                never ends up in source control.
            timeout: Per-request timeout in seconds.
        """
        self._api_key = api_key or settings.planet_api_key
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"PlanetClient(configured={self.is_configured})"

    @property
    def is_configured(self) -> bool:
        """True when an API key is available. Check this before offering Planet in a UI."""
        return bool(self._api_key)

    @property
    def api_key(self) -> str:
        """The configured key.

        Raises:
            PlanetAuthError: if none is set, explaining where to get one.
        """
        if not self._api_key:
            raise PlanetAuthError(
                "no Planet API key configured. Planet imagery is commercial and optional; "
                "the default Sentinel-2 pipeline needs no credentials.\n"
                "To use Planet, set the key in your environment:\n"
                "    export PL_API_KEY=your_key_here\n"
                "Keys live at https://www.planet.com/account/#/user-settings"
            )
        return self._api_key

    def search(
        self,
        bbox: GeoBBox | Sequence[float],
        start: str,
        end: str,
        *,
        item_types: Sequence[str] = DEFAULT_ITEM_TYPES,
        max_cloud: float | None = None,
        limit: int = 10,
    ) -> list[SceneRef]:
        """Find Planet scenes over an area and date range.

        Args:
            bbox: Area of interest in EPSG:4326 degrees.
            start: Inclusive ISO start date, ``"2024-01-01"``.
            end: Exclusive ISO end date.
            item_types: Planet item types, e.g. ``("PSScene",)``.
            max_cloud: Maximum cloud fraction as a percentage, converted to Planet's
                0..1 ``cloud_cover`` field. Defaults to ``settings.max_cloud_cover``.
            limit: Maximum scenes to return.

        Returns:
            Scene references with ``source="planet"``.

        Raises:
            PlanetAuthError: if no key is set or the key is rejected.
            PlanetError: on any other API or transport failure.
        """
        import requests

        # Resolve the key before the request, so a missing key surfaces as
        # PlanetAuthError rather than being caught and re-wrapped as a transport failure.
        auth = (self.api_key, "")
        box = tuple(bbox.as_tuple() if isinstance(bbox, GeoBBox) else bbox)
        cloud = settings.max_cloud_cover if max_cloud is None else max_cloud
        payload = {
            "item_types": list(item_types),
            "filter": {
                "type": "AndFilter",
                "config": [
                    {
                        "type": "GeometryFilter",
                        "field_name": "geometry",
                        "config": _bbox_polygon(box),
                    },
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T00:00:00Z"},
                    },
                    {
                        "type": "RangeFilter",
                        "field_name": "cloud_cover",
                        "config": {"lte": float(cloud) / 100.0},
                    },
                ],
            },
        }
        try:
            resp = requests.post(
                f"{DATA_API}/quick-search",
                json=payload,
                auth=auth,
                params={"_page_size": min(limit, 250)},
                timeout=self.timeout,
            )
        except Exception as exc:
            raise PlanetError(f"Planet search failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code in (401, 403):
            raise PlanetAuthError(
                f"Planet rejected the API key (HTTP {resp.status_code}). "
                "Check PL_API_KEY and that your plan covers the requested item types."
            )
        if not resp.ok:
            raise PlanetError(f"Planet search failed: HTTP {resp.status_code}: {resp.text[:200]}")

        features = resp.json().get("features", [])[:limit]
        return [_to_scene_ref(f) for f in features]

    def tile_url(
        self,
        item_type: str,
        item_id: str,
        tile: TileRef,
        *,
        include_key: bool = True,
    ) -> str:
        """Build an XYZ tile URL for one Planet scene.

        Args:
            item_type: Planet item type, e.g. ``"PSScene"``.
            item_id: Planet scene id.
            tile: The slippy-map tile to fetch.
            include_key: Append ``?api_key=``. Set to ``False`` when the URL will be
                logged, shown in a UI or embedded in a shared map, and pass the key as an
                HTTP Basic credential instead.

        Returns:
            A PNG tile URL.

        Raises:
            PlanetAuthError: if ``include_key`` is set and no key is configured.
        """
        base = f"{TILE_API}/{item_type}/{item_id}/{tile.z}/{tile.x}/{tile.y}.png"
        return f"{base}?api_key={self.api_key}" if include_key else base

    def tile_url_template(self, item_type: str, item_id: str) -> str:
        """A ``{z}/{x}/{y}`` template for map libraries, with no key embedded."""
        return f"{TILE_API}/{item_type}/{item_id}/{{z}}/{{x}}/{{y}}.png"


def _bbox_polygon(box: Sequence[float]) -> dict[str, Any]:
    """A closed GeoJSON polygon ring for ``(west, south, east, north)``."""
    if len(box) != 4:
        raise ValueError(f"bbox needs 4 values (west, south, east, north), got {len(box)}")
    w, s, e, n = (float(v) for v in box)
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def _to_scene_ref(feature: dict[str, Any]) -> SceneRef:
    props = feature.get("properties", {})
    cloud = props.get("cloud_cover")
    return SceneRef(
        scene_id=feature.get("id", ""),
        collection=props.get("item_type", "PSScene"),
        datetime=props.get("acquired"),
        platform=props.get("satellite_id"),
        # Planet reports cloud cover as a 0..1 fraction; SceneRef stores a percentage so
        # that a Planet scene and a Sentinel-2 scene can be compared directly.
        cloud_cover=None if cloud is None else float(cloud) * 100.0,
        source="planet",
    )
