"""STAC search over free Sentinel-2 archives, with band names normalized across providers.

Two public STAC APIs serve the same Sentinel-2 L2A scenes under different asset keys.
Microsoft Planetary Computer names its assets after the ESA band ids (``B02``, ``B08``,
``SCL``); Element84's earth-search names them after the optical role (``blue``, ``nir``,
``scl``). Code that hardcodes either set silently breaks when the other endpoint is used,
which is the failure this module exists to prevent: callers always ask for a canonical
band (``Band.B04`` or the string ``"B04"``) and get back whichever asset the provider
actually published.

Signed URLs expire
------------------
Planetary Computer serves its blobs behind short-lived SAS tokens, on the order of an
hour. Hrefs are therefore signed at access time in :meth:`StacClient.asset_hrefs` and are
never cached: a token stored to disk and reused the next day returns HTTP 403. Persist the
scene id instead and re-resolve the assets, which is cheap. This is also why the STAC
client is opened without ``planetary_computer.sign_inplace`` as a ``modifier``: that would
stamp a token onto every item at search time, so the clock would start ticking during the
search rather than at the point of use.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from mdebris.config import settings
from mdebris.types import GeoBBox, SceneRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

    from pystac import Item
    from pystac_client import Client

__all__ = [
    "BAND_ALIASES",
    "BAND_GSD_M",
    "BAND_WAVELENGTH_NM",
    "PROVIDERS",
    "RGB_BANDS",
    "Band",
    "SceneNotFoundError",
    "StacClient",
    "StacError",
    "canonical_band",
    "get_scene_assets",
    "normalize_assets",
    "provider_for_endpoint",
    "search_scenes",
]


class StacError(RuntimeError):
    """A STAC request could not be completed.

    Raised instead of letting a ``requests`` or ``pystac`` traceback escape, so callers
    (CLI, API, notebooks) can show one actionable line rather than a stack of internals.
    """


class SceneNotFoundError(StacError):
    """No scene with the requested id exists in the searched collections."""


class Band(StrEnum):
    """Canonical Sentinel-2 band ids, the ESA naming used as the interchange vocabulary.

    ESA ids win over common names because they are unambiguous. "nir" alone does not say
    whether it means B08 (842 nm, 10 m) or B8A (865 nm, 20 m), and that distinction
    matters: the Floating Debris Index is defined against B08 and the narrow B8A gives a
    different answer.
    """

    B01 = "B01"  # coastal aerosol
    B02 = "B02"  # blue
    B03 = "B03"  # green
    B04 = "B04"  # red
    B05 = "B05"  # red edge 1
    B06 = "B06"  # red edge 2, the FDI baseline band
    B07 = "B07"  # red edge 3
    B08 = "B08"  # NIR wide
    B8A = "B8A"  # NIR narrow
    B09 = "B09"  # water vapour
    B10 = "B10"  # cirrus, L1C only, absent from L2A
    B11 = "B11"  # SWIR 1
    B12 = "B12"  # SWIR 2
    SCL = "SCL"  # scene classification layer
    AOT = "AOT"  # aerosol optical thickness
    WVP = "WVP"  # water vapour column
    VISUAL = "VISUAL"  # pre-rendered 8-bit true colour


RGB_BANDS: tuple[Band, Band, Band] = (Band.B04, Band.B03, Band.B02)

# Every asset key seen in the wild, lowercased, mapped onto a canonical band. Keys were
# read off live items from both endpoints rather than from documentation. Element84 also
# publishes a "<name>-jp2" variant of each band; those are deliberately absent so that
# normalization always resolves to the cloud-optimized GeoTIFF.
BAND_ALIASES: dict[str, Band] = {
    "b01": Band.B01, "coastal": Band.B01, "coastal_aerosol": Band.B01,
    "b02": Band.B02, "blue": Band.B02,
    "b03": Band.B03, "green": Band.B03,
    "b04": Band.B04, "red": Band.B04,
    "b05": Band.B05, "rededge1": Band.B05,
    "b06": Band.B06, "rededge2": Band.B06,
    "b07": Band.B07, "rededge3": Band.B07,
    "b08": Band.B08, "nir": Band.B08,
    "b8a": Band.B8A, "b08a": Band.B8A, "nir08": Band.B8A,
    "b09": Band.B09, "nir09": Band.B09,
    "b10": Band.B10, "cirrus": Band.B10,
    "b11": Band.B11, "swir16": Band.B11, "swir1": Band.B11,
    "b12": Band.B12, "swir22": Band.B12, "swir2": Band.B12,
    "scl": Band.SCL,
    "aot": Band.AOT,
    "wvp": Band.WVP,
    "visual": Band.VISUAL, "tci": Band.VISUAL,
}  # fmt: skip

# Native ground sample distance in metres. Needed whenever bands of different resolution
# are stacked onto one grid, which is most of the time: FDI mixes B06 (20 m) with B08
# (10 m), so one of them always has to be resampled.
BAND_GSD_M: dict[Band, int] = {
    Band.B01: 60,
    Band.B02: 10,
    Band.B03: 10,
    Band.B04: 10,
    Band.B05: 20,
    Band.B06: 20,
    Band.B07: 20,
    Band.B08: 10,
    Band.B8A: 20,
    Band.B09: 60,
    Band.B10: 60,
    Band.B11: 20,
    Band.B12: 20,
    Band.SCL: 20,
    Band.AOT: 10,
    Band.WVP: 10,
    Band.VISUAL: 10,
}

# Sentinel-2A central wavelengths in nanometres, used by the index formulas that
# interpolate a baseline between bands (FDI, FAI).
BAND_WAVELENGTH_NM: dict[Band, float] = {
    Band.B01: 442.7,
    Band.B02: 492.4,
    Band.B03: 559.8,
    Band.B04: 664.6,
    Band.B05: 704.1,
    Band.B06: 740.5,
    Band.B07: 782.8,
    Band.B08: 832.8,
    Band.B8A: 864.7,
    Band.B09: 945.1,
    Band.B10: 1373.5,
    Band.B11: 1613.7,
    Band.B12: 2202.4,
}

# Endpoint host fragment to the ``SceneRef.source`` label recorded on every result.
PROVIDERS: dict[str, str] = {
    "planetarycomputer.microsoft.com": "planetary-computer",
    "earth-search.aws.element84.com": "element84",
    "catalogue.dataspace.copernicus.eu": "cdse",
}

_MISSING_DEPS = (
    "STAC access needs the optional data extra. Install it with:\n"
    '    pip install "mdebris[data]"'
)


def provider_for_endpoint(endpoint: str) -> str:
    """Label the provider behind a STAC URL, or ``'unknown'`` for a self-hosted catalog."""
    for fragment, name in PROVIDERS.items():
        if fragment in endpoint:
            return name
    return "unknown"


def canonical_band(name: str | Band) -> Band:
    """Resolve any provider spelling of a band to its canonical :class:`Band`.

    Accepts ESA ids, Element84 role names and mixed case, so ``"b8a"``, ``"B8A"`` and
    ``"nir08"`` all land on :attr:`Band.B8A`.

    Raises:
        KeyError: if the name matches no known band, listing the accepted spellings.
    """
    if isinstance(name, Band):
        return name
    key = str(name).strip().lower().replace("-", "_")
    try:
        return BAND_ALIASES[key]
    except KeyError:
        known = ", ".join(sorted(BAND_ALIASES))
        raise KeyError(f"unknown band {name!r}; expected one of: {known}") from None


def normalize_assets(assets: Iterable[str]) -> dict[Band, str]:
    """Map canonical bands onto the provider's own asset keys.

    Works off asset keys alone, so it is provider-agnostic and needs no network: hand it
    ``item.assets`` from either endpoint and it reports which key holds which band.
    Unrecognized keys (metadata documents, thumbnails, JPEG2000 duplicates) are dropped.

    When two keys resolve to the same band, the first in iteration order wins.
    """
    out: dict[Band, str] = {}
    for key in assets:
        try:
            band = canonical_band(key)
        except KeyError:
            continue
        out.setdefault(band, key)
    return out


class StacClient:
    """A STAC API client that hides endpoint differences, signing and failover.

    The primary endpoint is tried first; on a transport or protocol failure the fallback
    endpoint is tried before giving up, so a Planetary Computer outage degrades to
    earth-search instead of taking the pipeline down.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        fallback_endpoint: str | None = None,
        collection: str | None = None,
        sign: bool | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            endpoint: Primary STAC API root. Defaults to ``settings.stac_endpoint``.
            fallback_endpoint: Tried when the primary fails. Pass ``""`` to disable.
            collection: Default collection for searches.
            sign: Force signing on or off. ``None`` signs only for Planetary Computer,
                which is the only supported endpoint whose assets require it.
            timeout: Per-request timeout in seconds.
        """
        self.endpoint = endpoint or settings.stac_endpoint
        fb = settings.stac_fallback_endpoint if fallback_endpoint is None else fallback_endpoint
        self.fallback_endpoint = fb or None
        self.collection = collection or settings.stac_collection
        self.timeout = timeout
        self._sign_override = sign
        self._clients: dict[str, Client] = {}
        self._items: dict[str, tuple[Item, str]] = {}

    def __repr__(self) -> str:
        return f"StacClient(endpoint={self.endpoint!r}, fallback={self.fallback_endpoint!r})"

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Primary then fallback, de-duplicated, in the order they will be tried."""
        if self.fallback_endpoint and self.fallback_endpoint != self.endpoint:
            return (self.endpoint, self.fallback_endpoint)
        return (self.endpoint,)

    def signs(self, endpoint: str) -> bool:
        """Whether assets from ``endpoint`` need a signature before they can be read."""
        if self._sign_override is not None:
            return self._sign_override
        return provider_for_endpoint(endpoint) == "planetary-computer"

    # -- search ------------------------------------------------------------------

    def search_items(
        self,
        bbox: GeoBBox | Sequence[float],
        start: str,
        end: str,
        *,
        collection: str | None = None,
        max_cloud: float | None = None,
        limit: int = 10,
        sort_by_cloud: bool = True,
    ) -> list[Item]:
        """Search for scenes and return raw STAC Items.

        Use this when you need the full item (geometry, all properties); use
        :meth:`search` when a :class:`~mdebris.types.SceneRef` is enough.
        """
        # Validate before touching the network, so a malformed bbox is reported as a
        # malformed bbox rather than hiding behind a connection error from each endpoint.
        box = _as_bbox(bbox)
        errors: list[str] = []
        for endpoint in self.endpoints:
            try:
                items = self._search_one(
                    endpoint,
                    bbox=bbox,
                    start=start,
                    end=end,
                    collection=collection,
                    max_cloud=max_cloud,
                    limit=limit,
                )
            except StacError as exc:
                errors.append(f"  {endpoint}: {exc}")
                continue
            for item in items:
                self._items[item.id] = (item, endpoint)
            if sort_by_cloud:
                items.sort(key=lambda i: i.properties.get("eo:cloud_cover") or 0.0)
            return items
        raise StacError("every STAC endpoint failed:\n" + "\n".join(errors))

    def search(
        self,
        bbox: GeoBBox | Sequence[float],
        start: str,
        end: str,
        *,
        collection: str | None = None,
        max_cloud: float | None = None,
        limit: int = 10,
    ) -> list[SceneRef]:
        """Search for scenes and return provenance records, cheapest cloud cover first."""
        items = self.search_items(
            bbox,
            start,
            end,
            collection=collection,
            max_cloud=max_cloud,
            limit=limit,
        )
        return [self._to_scene_ref(item) for item in items]

    def _search_one(
        self,
        endpoint: str,
        *,
        bbox: GeoBBox | Sequence[float],
        start: str,
        end: str,
        collection: str | None,
        max_cloud: float | None,
        limit: int,
    ) -> list[Item]:
        client = self._open(endpoint)
        box = tuple(bbox.as_tuple() if isinstance(bbox, GeoBBox) else bbox)
        if len(box) != 4:
            raise StacError(f"bbox needs 4 values (west, south, east, north), got {len(box)}")
        cloud = settings.max_cloud_cover if max_cloud is None else max_cloud
        query = {"eo:cloud_cover": {"lt": float(cloud)}} if cloud is not None else None
        try:
            search = client.search(
                collections=[collection or self.collection],
                bbox=box,
                datetime=f"{start}/{end}",
                query=query,
                max_items=limit,
            )
            return list(search.items())
        except Exception as exc:
            raise StacError(_describe(exc)) from exc

    def _open(self, endpoint: str) -> Client:
        if endpoint in self._clients:
            return self._clients[endpoint]
        try:
            from pystac_client import Client as _Client
        except ImportError as exc:
            raise StacError(_MISSING_DEPS) from exc
        try:
            client = _Client.open(endpoint, timeout=self.timeout)
        except Exception as exc:
            raise StacError(f"cannot reach {endpoint}: {_describe(exc)}") from exc
        self._clients[endpoint] = client
        return client

    def _to_scene_ref(self, item: Item) -> SceneRef:
        endpoint = self._items.get(item.id, (item, self.endpoint))[1]
        props = item.properties
        return SceneRef(
            scene_id=item.id,
            collection=item.collection_id or self.collection,
            datetime=props.get("datetime"),
            platform=props.get("platform"),
            cloud_cover=props.get("eo:cloud_cover"),
            source=provider_for_endpoint(endpoint),
        )

    # -- assets ------------------------------------------------------------------

    def get_item(self, scene_id: str, *, collection: str | None = None) -> Item:
        """Fetch one item by id, reusing the result of an earlier search when possible."""
        if scene_id in self._items:
            return self._items[scene_id][0]
        errors: list[str] = []
        for endpoint in self.endpoints:
            client = self._open(endpoint)
            try:
                found = list(
                    client.search(
                        collections=[collection or self.collection],
                        ids=[scene_id],
                        max_items=1,
                    ).items()
                )
            except Exception as exc:
                errors.append(f"  {endpoint}: {_describe(exc)}")
                continue
            if found:
                self._items[scene_id] = (found[0], endpoint)
                return found[0]
        if errors:
            raise StacError(f"could not look up scene {scene_id!r}:\n" + "\n".join(errors))
        raise SceneNotFoundError(
            f"no scene {scene_id!r} in collection "
            f"{collection or self.collection!r} on {', '.join(self.endpoints)}"
        )

    def asset_hrefs(
        self,
        scene: str | Item | SceneRef,
        bands: Iterable[str | Band] | None = None,
        *,
        collection: str | None = None,
    ) -> dict[str, str]:
        """Resolve canonical band names to ready-to-read URLs.

        Hrefs are signed here, at the moment of use, because Planetary Computer tokens
        expire within about an hour. Call this again rather than storing the result.

        Args:
            scene: A scene id, a ``SceneRef``, or an already-fetched STAC Item.
            bands: Canonical or provider band names to return. ``None`` returns every
                band the scene publishes.
            collection: Collection to search when ``scene`` is an id.

        Returns:
            Canonical band name to href, e.g. ``{"B04": "https://...?st=..."}``.

        Raises:
            KeyError: if a requested band is not published for this scene.
        """
        item = self._resolve_item(scene, collection=collection)
        endpoint = self._items.get(item.id, (item, self.endpoint))[1]
        available = normalize_assets(item.assets)

        if bands is None:
            wanted = list(available)
        else:
            wanted = [canonical_band(b) for b in bands]
            missing = [b for b in wanted if b not in available]
            if missing:
                raise KeyError(
                    f"scene {item.id!r} does not publish "
                    f"{', '.join(str(b) for b in missing)}; "
                    f"available: {', '.join(sorted(str(b) for b in available))}"
                )

        sign = _signer() if self.signs(endpoint) else None
        out: dict[str, str] = {}
        for band in wanted:
            href = item.assets[available[band]].href
            out[str(band)] = sign(href) if sign else href
        return out

    def _resolve_item(self, scene: str | Item | SceneRef, *, collection: str | None) -> Item:
        if isinstance(scene, SceneRef):
            return self.get_item(scene.scene_id, collection=collection or scene.collection)
        if isinstance(scene, str):
            return self.get_item(scene, collection=collection)
        if hasattr(scene, "assets"):
            return scene
        raise TypeError(f"expected a scene id, SceneRef or STAC Item, got {type(scene).__name__}")


def _signer():
    """Return ``planetary_computer.sign``, or explain how to install it."""
    try:
        import planetary_computer
    except ImportError as exc:
        raise StacError(
            "Planetary Computer assets are behind short-lived SAS tokens and cannot be "
            f"read unsigned.\n{_MISSING_DEPS}"
        ) from exc
    return planetary_computer.sign


# pystac_client wraps transport failures in APIError, so the exception type alone does not
# say whether the network is down. These fragments appear in the wrapped urllib3 message.
_OFFLINE_MARKERS = (
    "max retries exceeded",
    "failed to resolve",
    "name or service not known",
    "connection refused",
    "connection reset",
    "temporary failure in name resolution",
    "timed out",
    "network is unreachable",
)


def _describe(exc: BaseException) -> str:
    """One readable line for an exception raised somewhere inside the HTTP or STAC stack."""
    name = type(exc).__name__
    text = str(exc).strip() or "no detail"
    lowered = text.lower()
    offline = any(t in name for t in ("Connection", "Timeout", "DNS", "SSL", "Socket")) or any(
        m in lowered for m in _OFFLINE_MARKERS
    )
    if offline:
        return f"network unavailable ({name}): {text}"
    return f"{name}: {text}"


# -- module-level convenience ----------------------------------------------------

_default_client: StacClient | None = None


def _client(endpoint: str | None) -> StacClient:
    """Reuse one client for the default endpoint so repeated calls reuse HTTP sessions."""
    global _default_client
    if endpoint is not None:
        return StacClient(endpoint)
    if _default_client is None:
        _default_client = StacClient()
    return _default_client


def search_scenes(
    bbox: GeoBBox | Sequence[float],
    start: str,
    end: str,
    *,
    collection: str | None = None,
    max_cloud: float | None = None,
    limit: int = 10,
    endpoint: str | None = None,
) -> list[SceneRef]:
    """Find Sentinel-2 scenes over an area and date range, least cloudy first.

    Args:
        bbox: Area of interest, a :class:`~mdebris.types.GeoBBox` or a
            ``(west, south, east, north)`` sequence in EPSG:4326 degrees.
        start: Inclusive ISO start date, e.g. ``"2024-01-01"``.
        end: Inclusive ISO end date.
        collection: STAC collection, defaults to ``settings.stac_collection``.
        max_cloud: Maximum ``eo:cloud_cover`` percent. Defaults to
            ``settings.max_cloud_cover``; pass ``100`` to accept any cloud cover.
        limit: Maximum scenes to return.
        endpoint: Override the STAC API root.

    Returns:
        Scene references carrying id, date, platform and cloud cover.

    Raises:
        StacError: if no configured endpoint could be reached.

    Example:
        >>> from mdebris.types import GeoBBox
        >>> accra = GeoBBox(-0.35, 5.45, -0.05, 5.65)
        >>> scenes = search_scenes(accra, "2024-01-01", "2024-06-30")  # doctest: +SKIP
    """
    return _client(endpoint).search(
        bbox, start, end, collection=collection, max_cloud=max_cloud, limit=limit
    )


def get_scene_assets(
    scene: str | Item | SceneRef,
    bands: Iterable[str | Band] | None = None,
    *,
    collection: str | None = None,
    endpoint: str | None = None,
) -> dict[str, str]:
    """Resolve a scene's bands to signed, ready-to-read URLs.

    The returned URLs are short lived when the source is Planetary Computer. Re-resolve
    rather than caching them; see the module docstring.

    Example:
        >>> hrefs = get_scene_assets(scene_id, ["B04", "B08"])  # doctest: +SKIP
        >>> sorted(hrefs)                                       # doctest: +SKIP
        ['B04', 'B08']
    """
    return _client(endpoint).asset_hrefs(scene, bands, collection=collection)
