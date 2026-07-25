"""Tests for STAC search and, mainly, for band-name normalization.

The normalization layer is the part that silently corrupts results when it is wrong: ask
for "B08" against Element84, get an asset that does not exist or, worse, the wrong band,
and every downstream index is computed from the wrong wavelength while the pipeline
reports success. So it is tested exhaustively and offline, against asset key lists
captured verbatim from live items on both endpoints.
"""

from __future__ import annotations

import pytest

from mdebris.data.stac import (
    BAND_ALIASES,
    BAND_GSD_M,
    BAND_WAVELENGTH_NM,
    Band,
    SceneNotFoundError,
    StacClient,
    StacError,
    canonical_band,
    normalize_assets,
    provider_for_endpoint,
    search_scenes,
)
from mdebris.types import GeoBBox, SceneRef

# Captured from a live item, S2A_MSIL2A_20240527T100601_R022_T30NZM_20240527T182116.
PC_ASSET_KEYS = [
    "AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12",
    "B8A", "SCL", "WVP", "datastrip-metadata", "granule-metadata", "inspire-metadata",
    "preview", "product-metadata", "rendered_preview", "safe-manifest", "tilejson", "visual",
]  # fmt: skip

# Captured from the same scene on Element84 earth-search, S2A_30NZM_20240527_0_L2A.
E84_ASSET_KEYS = [
    "aot", "aot-jp2", "blue", "blue-jp2", "coastal", "coastal-jp2", "granule_metadata",
    "green", "green-jp2", "nir", "nir-jp2", "nir08", "nir08-jp2", "nir09", "nir09-jp2",
    "red", "red-jp2", "rededge1", "rededge1-jp2", "rededge2", "rededge2-jp2", "rededge3",
    "rededge3-jp2", "scl", "scl-jp2", "swir16", "swir16-jp2", "swir22", "swir22-jp2",
    "thumbnail", "tileinfo_metadata", "visual", "visual-jp2", "wvp", "wvp-jp2",
]  # fmt: skip

ACCRA = GeoBBox(-0.35, 5.45, -0.05, 5.65)


class FakeAsset:
    def __init__(self, href: str) -> None:
        self.href = href


class FakeItem:
    """The slice of a pystac Item this module touches."""

    def __init__(self, item_id: str, keys: list[str], **props: object) -> None:
        self.id = item_id
        self.collection_id = "sentinel-2-l2a"
        self.assets = {k: FakeAsset(f"https://example.invalid/{item_id}/{k}.tif") for k in keys}
        self.properties = {
            "datetime": "2024-05-27T10:06:01Z",
            "platform": "Sentinel-2A",
            "eo:cloud_cover": 12.2,
            **props,
        }


# -- canonical_band ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("B04", Band.B04),
        ("b04", Band.B04),
        ("  B04 ", Band.B04),
        ("red", Band.B04),
        ("RED", Band.B04),
        ("blue", Band.B02),
        ("green", Band.B03),
        ("coastal", Band.B01),
        ("nir", Band.B08),
        ("nir08", Band.B8A),
        ("B8A", Band.B8A),
        ("b8a", Band.B8A),
        ("B08A", Band.B8A),
        ("nir09", Band.B09),
        ("rededge1", Band.B05),
        ("rededge2", Band.B06),
        ("rededge3", Band.B07),
        ("swir16", Band.B11),
        ("swir1", Band.B11),
        ("swir22", Band.B12),
        ("swir2", Band.B12),
        ("scl", Band.SCL),
        ("SCL", Band.SCL),
        ("visual", Band.VISUAL),
        ("tci", Band.VISUAL),
    ],
)
def test_canonical_band_resolves_provider_spellings(alias: str, expected: Band) -> None:
    assert canonical_band(alias) is expected


def test_canonical_band_is_idempotent_on_band_members() -> None:
    for band in Band:
        assert canonical_band(band) is band
        assert canonical_band(str(band)) is band


def test_canonical_band_rejects_unknown_names_with_a_useful_message() -> None:
    with pytest.raises(KeyError) as excinfo:
        canonical_band("panchromatic")
    message = str(excinfo.value)
    assert "panchromatic" in message
    assert "swir16" in message  # the error lists the accepted spellings


def test_nir_never_silently_resolves_to_the_narrow_band() -> None:
    """B08 and B8A are 22 nm apart and FDI is defined against B08, not B8A."""
    assert canonical_band("nir") is Band.B08
    assert canonical_band("nir08") is Band.B8A
    assert canonical_band("nir") is not canonical_band("nir08")


# -- normalize_assets -------------------------------------------------------------


def test_normalize_assets_planetary_computer() -> None:
    mapping = normalize_assets(PC_ASSET_KEYS)
    assert mapping[Band.B02] == "B02"
    assert mapping[Band.B04] == "B04"
    assert mapping[Band.B08] == "B08"
    assert mapping[Band.B8A] == "B8A"
    assert mapping[Band.B11] == "B11"
    assert mapping[Band.SCL] == "SCL"


def test_normalize_assets_element84() -> None:
    mapping = normalize_assets(E84_ASSET_KEYS)
    assert mapping[Band.B02] == "blue"
    assert mapping[Band.B04] == "red"
    assert mapping[Band.B08] == "nir"
    assert mapping[Band.B8A] == "nir08"
    assert mapping[Band.B11] == "swir16"
    assert mapping[Band.B12] == "swir22"
    assert mapping[Band.SCL] == "scl"


def test_both_providers_expose_the_same_canonical_bands() -> None:
    """The whole point: identical canonical keys regardless of endpoint."""
    pc = set(normalize_assets(PC_ASSET_KEYS))
    e84 = set(normalize_assets(E84_ASSET_KEYS))
    assert pc == e84


def test_normalize_assets_prefers_cogs_over_jpeg2000() -> None:
    """Element84 publishes a -jp2 twin of every band; reading those is far slower."""
    mapping = normalize_assets(E84_ASSET_KEYS)
    assert not any(value.endswith("-jp2") for value in mapping.values())


def test_normalize_assets_drops_metadata_and_thumbnails() -> None:
    mapping = normalize_assets(PC_ASSET_KEYS)
    assert "tilejson" not in mapping.values()
    assert "rendered_preview" not in mapping.values()
    assert Band.B10 not in mapping  # cirrus is L1C only, absent from L2A


def test_normalize_assets_is_empty_for_an_unrelated_catalog() -> None:
    assert normalize_assets(["cog_default", "metadata", "thumbnail"]) == {}


def test_normalize_assets_keeps_the_first_key_when_two_resolve_alike() -> None:
    assert normalize_assets(["B04", "red"])[Band.B04] == "B04"
    assert normalize_assets(["red", "B04"])[Band.B04] == "red"


# -- band reference tables --------------------------------------------------------


def test_every_alias_points_at_a_real_band() -> None:
    assert set(BAND_ALIASES.values()) <= set(Band)


def test_aliases_are_lowercase_so_lookup_cannot_miss() -> None:
    assert all(key == key.lower() for key in BAND_ALIASES)


def test_every_band_has_a_ground_sample_distance() -> None:
    assert set(BAND_GSD_M) == set(Band)
    assert set(BAND_GSD_M.values()) <= {10, 20, 60}


def test_ten_metre_bands_are_the_ones_sentinel_2_actually_ships_at_ten_metres() -> None:
    ten_metre = {b for b, gsd in BAND_GSD_M.items() if gsd == 10}
    assert {Band.B02, Band.B03, Band.B04, Band.B08} <= ten_metre
    assert Band.B11 not in ten_metre
    assert Band.SCL not in ten_metre


def test_wavelengths_increase_with_band_number() -> None:
    ordered = [Band.B01, Band.B02, Band.B03, Band.B04, Band.B05, Band.B06, Band.B07]
    values = [BAND_WAVELENGTH_NM[b] for b in ordered]
    assert values == sorted(values)
    assert BAND_WAVELENGTH_NM[Band.B08] < BAND_WAVELENGTH_NM[Band.B8A]
    assert BAND_WAVELENGTH_NM[Band.B11] < BAND_WAVELENGTH_NM[Band.B12]


# -- provider detection -----------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://planetarycomputer.microsoft.com/api/stac/v1", "planetary-computer"),
        ("https://earth-search.aws.element84.com/v1", "element84"),
        ("https://catalogue.dataspace.copernicus.eu/stac", "cdse"),
        ("https://stac.example.org/v1", "unknown"),
    ],
)
def test_provider_for_endpoint(endpoint: str, expected: str) -> None:
    assert provider_for_endpoint(endpoint) == expected


def test_only_planetary_computer_assets_are_signed_by_default() -> None:
    pc = StacClient("https://planetarycomputer.microsoft.com/api/stac/v1")
    assert pc.signs(pc.endpoint) is True
    e84 = StacClient("https://earth-search.aws.element84.com/v1", fallback_endpoint="")
    assert e84.signs(e84.endpoint) is False


def test_signing_can_be_forced_off_for_a_mirror() -> None:
    client = StacClient("https://planetarycomputer.microsoft.com/api/stac/v1", sign=False)
    assert client.signs(client.endpoint) is False


# -- client wiring ----------------------------------------------------------------


def test_endpoints_are_primary_then_fallback() -> None:
    client = StacClient("https://a.example/v1", fallback_endpoint="https://b.example/v1")
    assert client.endpoints == ("https://a.example/v1", "https://b.example/v1")


def test_fallback_can_be_disabled() -> None:
    assert StacClient("https://a.example/v1", fallback_endpoint="").endpoints == (
        "https://a.example/v1",
    )


def test_identical_fallback_is_not_tried_twice() -> None:
    client = StacClient("https://a.example/v1", fallback_endpoint="https://a.example/v1")
    assert client.endpoints == ("https://a.example/v1",)


def test_asset_hrefs_normalizes_an_element84_item_without_network() -> None:
    client = StacClient("https://earth-search.aws.element84.com/v1", fallback_endpoint="")
    item = FakeItem("S2A_30NZM_20240527_0_L2A", E84_ASSET_KEYS)
    hrefs = client.asset_hrefs(item, ["B04", "B08", "SCL"])
    assert sorted(hrefs) == ["B04", "B08", "SCL"]
    assert hrefs["B04"].endswith("/red.tif")
    assert hrefs["B08"].endswith("/nir.tif")
    assert hrefs["SCL"].endswith("/scl.tif")


def test_asset_hrefs_accepts_provider_names_as_input_too() -> None:
    client = StacClient("https://earth-search.aws.element84.com/v1", fallback_endpoint="")
    item = FakeItem("x", E84_ASSET_KEYS)
    assert client.asset_hrefs(item, ["red"]) == client.asset_hrefs(item, ["B04"])


def test_asset_hrefs_returns_every_band_when_none_are_requested() -> None:
    client = StacClient("https://earth-search.aws.element84.com/v1", fallback_endpoint="")
    hrefs = client.asset_hrefs(FakeItem("x", E84_ASSET_KEYS))
    assert set(hrefs) == {str(b) for b in normalize_assets(E84_ASSET_KEYS)}


def test_asset_hrefs_names_the_missing_band_and_lists_what_is_there() -> None:
    client = StacClient("https://earth-search.aws.element84.com/v1", fallback_endpoint="")
    item = FakeItem("x", ["red", "green", "blue"])
    with pytest.raises(KeyError) as excinfo:
        client.asset_hrefs(item, ["B04", "B11"])
    message = str(excinfo.value)
    assert "B11" in message
    assert "B04" in message  # listed as available, so the user can see what went wrong


def test_asset_hrefs_rejects_a_type_it_cannot_resolve() -> None:
    client = StacClient(fallback_endpoint="")
    with pytest.raises(TypeError, match="scene id, SceneRef or STAC Item"):
        client.asset_hrefs(42)  # type: ignore[arg-type]


def test_unreachable_endpoints_raise_stac_error_not_a_raw_traceback() -> None:
    client = StacClient("https://stac.invalid.localhost/v1", fallback_endpoint="")
    with pytest.raises(StacError) as excinfo:
        client.search(ACCRA, "2024-01-01", "2024-06-30")
    message = str(excinfo.value)
    assert "stac.invalid.localhost" in message
    assert "network unavailable" in message


def test_both_endpoints_are_reported_when_both_fail() -> None:
    client = StacClient(
        "https://one.invalid.localhost/v1",
        fallback_endpoint="https://two.invalid.localhost/v1",
    )
    with pytest.raises(StacError) as excinfo:
        client.search(ACCRA, "2024-01-01", "2024-06-30")
    message = str(excinfo.value)
    assert "one.invalid.localhost" in message
    assert "two.invalid.localhost" in message


def test_scene_not_found_is_a_stac_error() -> None:
    assert issubclass(SceneNotFoundError, StacError)


def test_bad_bbox_is_rejected_before_any_request() -> None:
    """Reported as a bad bbox, not as an endpoint failure, even with a dead endpoint."""
    client = StacClient("https://stac.invalid.localhost/v1", fallback_endpoint="")
    with pytest.raises(StacError, match="bbox needs 4 values"):
        client.search([1.0, 2.0, 3.0], "2024-01-01", "2024-06-30")
    with pytest.raises(StacError, match="must be four numbers"):
        client.search(["west", "south", "east", "north"], "2024-01-01", "2024-06-30")  # type: ignore[list-item]


def test_geobbox_and_a_plain_tuple_are_interchangeable() -> None:
    assert ACCRA.as_tuple() == (-0.35, 5.45, -0.05, 5.65)


# -- network ----------------------------------------------------------------------


@pytest.mark.network
def test_search_scenes_finds_sentinel_2_over_accra() -> None:
    scenes = search_scenes(ACCRA, "2024-01-01", "2024-06-30", max_cloud=20, limit=5)
    assert scenes
    assert all(isinstance(s, SceneRef) for s in scenes)
    assert all(s.collection == "sentinel-2-l2a" for s in scenes)
    assert all(s.source == "planetary-computer" for s in scenes)
    assert all(s.cloud_cover is not None and s.cloud_cover < 20 for s in scenes)
    assert [s.cloud_cover for s in scenes] == sorted(s.cloud_cover for s in scenes)


@pytest.mark.network
def test_the_same_scene_normalizes_identically_on_both_providers() -> None:
    """One acquisition, two catalogs, two naming schemes, one canonical answer."""
    pc = StacClient("https://planetarycomputer.microsoft.com/api/stac/v1", fallback_endpoint="")
    e84 = StacClient("https://earth-search.aws.element84.com/v1", fallback_endpoint="")
    wanted = ["B02", "B03", "B04", "B06", "B08", "B11", "SCL"]

    pc_scenes = pc.search(ACCRA, "2024-05-01", "2024-06-01", max_cloud=20, limit=1)
    e84_scenes = e84.search(ACCRA, "2024-05-01", "2024-06-01", max_cloud=20, limit=1)
    assert pc_scenes and e84_scenes

    pc_hrefs = pc.asset_hrefs(pc_scenes[0], wanted)
    e84_hrefs = e84.asset_hrefs(e84_scenes[0], wanted)
    assert sorted(pc_hrefs) == sorted(e84_hrefs) == sorted(wanted)
    # Planetary Computer hrefs must arrive signed; earth-search is public.
    assert all("?" in href for href in pc_hrefs.values())
    assert all("blob.core.windows.net" in href for href in pc_hrefs.values())
    assert all("sentinel-cogs.s3" in href for href in e84_hrefs.values())


@pytest.mark.network
def test_a_failing_primary_falls_through_to_the_fallback() -> None:
    client = StacClient(
        "https://stac.invalid.localhost/v1",
        fallback_endpoint="https://earth-search.aws.element84.com/v1",
    )
    scenes = client.search(ACCRA, "2024-01-01", "2024-06-30", max_cloud=20, limit=2)
    assert scenes
    assert all(s.source == "element84" for s in scenes)


@pytest.mark.network
def test_looking_up_a_nonexistent_scene_id_says_so() -> None:
    client = StacClient(fallback_endpoint="")
    with pytest.raises(SceneNotFoundError):
        client.get_item("S2A_MSIL2A_19000101T000000_R000_T00XXX_19000101T000000")
