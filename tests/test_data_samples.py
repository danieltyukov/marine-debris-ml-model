"""Tests for the bundled sample chips, and for the Planet connector's import safety.

These tests are the reason the chips exist: they run with no network, no credentials and
no downloads, against real Sentinel-2 pixels. If they pass, the geometry, band naming and
radiometry the rest of the pipeline depends on are all real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdebris.data.planet import PlanetAuthError, PlanetClient, PlanetError
from mdebris.data.samples import (
    CHIP_PIXELS,
    HOTSPOTS,
    SAMPLE_BANDS,
    SAMPLES_DIR,
    SCL_WATER,
    SampleError,
    list_samples,
    load_sample,
    sample_bbox,
    sample_meta,
    sample_scene,
    to_reflectance,
)
from mdebris.data.stac import Band
from mdebris.types import GeoBBox, SceneRef, TileRef

SAMPLE_NAMES = list_samples()


# -- bundling ---------------------------------------------------------------------


def test_chips_are_bundled_for_every_hotspot() -> None:
    assert sorted(h.name for h in HOTSPOTS) == SAMPLE_NAMES


def test_every_chip_has_a_provenance_sidecar() -> None:
    for name in SAMPLE_NAMES:
        assert (SAMPLES_DIR / f"{name}.json").is_file()


def test_the_bundle_stays_small_enough_to_live_in_git() -> None:
    """Samples ship inside the wheel, so a bloated chip is everyone's download."""
    total = sum(p.stat().st_size for p in SAMPLES_DIR.iterdir() if p.is_file())
    assert total < 20 * 1024 * 1024, f"samples total {total / 1e6:.1f} MB"


def test_unknown_sample_names_list_what_is_available() -> None:
    with pytest.raises(SampleError) as excinfo:
        load_sample("atlantis")
    message = str(excinfo.value)
    assert "atlantis" in message
    for name in SAMPLE_NAMES:
        assert name in message


# -- chip contents ----------------------------------------------------------------


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_chip_carries_the_bands_the_pipeline_needs(name: str) -> None:
    bands, meta = load_sample(name)
    assert set(bands) == {str(b) for b in SAMPLE_BANDS}
    assert meta["bands"] == [str(b) for b in SAMPLE_BANDS]


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_all_bands_share_one_grid(name: str) -> None:
    """A 20 m band resampled onto the 10 m grid means a pixel index means one thing."""
    bands, _ = load_sample(name)
    shapes = {array.shape for array in bands.values()}
    assert shapes == {(CHIP_PIXELS, CHIP_PIXELS)}


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_reflectance_is_uint16_and_scl_is_a_class_code(name: str) -> None:
    bands, meta = load_sample(name)
    for band in SAMPLE_BANDS:
        array = bands[str(band)]
        if band is Band.SCL:
            assert array.dtype == np.uint8
            # The L2A scene classification legend runs 0..11.
            assert array.max() <= 11
        else:
            assert array.dtype == np.uint16
            reflectance = to_reflectance(array, meta)
            assert reflectance.min() > -0.15  # nodata sits at the offset, not below it
            assert reflectance.max() < 2.0


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_chips_are_almost_entirely_valid_pixels(name: str) -> None:
    """A window that fell off the edge of a Sentinel-2 tile would be mostly zeros."""
    bands, _ = load_sample(name)
    assert float((bands["B08"] == 0).mean()) < 0.02


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_chips_actually_contain_sea(name: str) -> None:
    """These are marine samples; a chip of pure land would test nothing useful."""
    bands, _ = load_sample(name)
    water = float((bands["SCL"] == SCL_WATER).mean())
    assert water > 0.10, f"{name} is only {water:.0%} water"


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_water_is_dark_in_the_near_infrared(name: str) -> None:
    """A physical check that bands are not transposed or mislabelled.

    Water absorbs NIR almost completely and reflects green, so NDWI is positive over sea
    and negative over land. If B03 and B08 were swapped, or a 20 m band were pasted onto
    the wrong grid, this relationship would not hold.
    """
    bands, _ = load_sample(name)
    scl = bands["SCL"]
    green = bands["B03"].astype("float32")
    nir = bands["B08"].astype("float32")
    ndwi = (green - nir) / np.maximum(green + nir, 1.0)

    sea = scl == SCL_WATER
    assert np.median(ndwi[sea]) > 0.0
    assert np.median(nir[sea]) < np.median(green[sea])

    # SCL 4 and 5 are vegetated and bare land, which behave the opposite way.
    land = np.isin(scl, [4, 5])
    if land.sum() > 1000:
        assert np.median(ndwi[land]) < np.median(ndwi[sea])


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_water_is_dark_in_absolute_reflectance(name: str) -> None:
    """Checks the scale and offset together, and the 20 m to 10 m resampling.

    Clear water reflects a few percent at most beyond the visible. Getting a plausible
    absolute number requires applying the -0.1 baseline offset as well as the 1e-4 scale;
    dividing by 10000 alone puts water at 0.12 in the NIR, which no sea does.
    """
    bands, meta = load_sample(name)
    sea = bands["SCL"] == SCL_WATER
    for band in ("B08", "B11"):
        reflectance = to_reflectance(bands[band], meta)[sea]
        assert 0.0 <= np.median(reflectance) < 0.06, band


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_sidecar_records_the_baseline_reflectance_offset(name: str) -> None:
    """Both chips postdate processing baseline 04.00, so both carry the -1000 offset."""
    meta = sample_meta(name)
    assert meta["reflectance_scale"] == pytest.approx(1e-4)
    assert meta["reflectance_offset"] == pytest.approx(-0.1)
    assert meta["processing_baseline_offset_applied"] is True


def test_to_reflectance_applies_scale_and_offset() -> None:
    meta = {"reflectance_scale": 1e-4, "reflectance_offset": -0.1}
    values = np.array([1000, 2000, 11000], dtype="uint16")
    # abs= is needed because the first element is exactly zero, where a relative
    # tolerance has nothing to be relative to and float32 rounding still shows up.
    assert to_reflectance(values, meta) == pytest.approx([0.0, 0.1, 1.0], abs=1e-6)


def test_to_reflectance_defaults_to_no_offset_for_older_products() -> None:
    """Pre-baseline-04.00 scenes have no offset, and a bare scale must still work."""
    assert to_reflectance(np.array([5000], dtype="uint16"), {}) == pytest.approx([0.5])


# -- georeferencing ---------------------------------------------------------------


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_chip_is_georeferenced_in_a_utm_projection(name: str) -> None:
    _, meta = load_sample(name)
    assert meta["crs"].startswith("EPSG:326") or meta["crs"].startswith("EPSG:327")
    assert len(meta["transform"]) == 6
    assert meta["gsd_m"] == 10
    # Affine order is (a, b, c, d, e, f): a is the pixel width, e the pixel height,
    # negative because rows run north to south.
    assert meta["transform"][0] == pytest.approx(10.0)
    assert meta["transform"][4] == pytest.approx(-10.0)


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_chip_bounds_are_a_valid_geobbox_around_its_hotspot(name: str) -> None:
    bbox = sample_bbox(name)
    assert isinstance(bbox, GeoBBox)
    lon, lat = sample_meta(name)["centre_lonlat"]
    assert bbox.west < lon < bbox.east
    assert bbox.south < lat < bbox.north


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_chip_covers_roughly_five_kilometres(name: str) -> None:
    """512 pixels at 10 m is 5.12 km, which is about 0.046 degrees of latitude."""
    bbox = sample_bbox(name)
    assert 0.035 < bbox.north - bbox.south < 0.06


# -- provenance -------------------------------------------------------------------


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_provenance_identifies_the_source_scene(name: str) -> None:
    scene = sample_scene(name)
    assert isinstance(scene, SceneRef)
    assert scene.scene_id.startswith("S2")
    assert "MSIL2A" in scene.scene_id
    assert scene.collection == "sentinel-2-l2a"
    assert scene.datetime is not None and scene.datetime.startswith("20")
    assert scene.cloud_cover is not None and scene.cloud_cover < 25


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_provenance_records_the_copernicus_licence(name: str) -> None:
    """Redistributing Sentinel-2 derivatives requires stating the terms and attribution."""
    meta = sample_meta(name)
    assert "Copernicus" in meta["license"]
    assert "Copernicus" in meta["attribution"]


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_provenance_explains_why_the_site_was_chosen(name: str) -> None:
    assert len(sample_meta(name)["description"]) > 40


def test_sidecar_and_geotiff_agree_on_the_grid() -> None:
    for name in SAMPLE_NAMES:
        _, meta = load_sample(name)
        sidecar = sample_meta(name)
        assert meta["crs"] == sidecar["crs"]
        assert list(meta["transform"]) == pytest.approx(sidecar["transform"])


def test_a_missing_sidecar_is_reported_clearly(tmp_path: Path) -> None:
    from mdebris.data import samples as samples_module

    original = samples_module.SAMPLES_DIR
    samples_module.SAMPLES_DIR = tmp_path
    try:
        with pytest.raises(SampleError, match="provenance sidecar"):
            samples_module.sample_meta("accra")
    finally:
        samples_module.SAMPLES_DIR = original


# -- hotspots ---------------------------------------------------------------------


def test_hotspot_search_boxes_contain_their_own_centres() -> None:
    for spot in HOTSPOTS:
        bbox = spot.bbox
        assert bbox.west < spot.lon < bbox.east
        assert bbox.south < spot.lat < bbox.north


def test_hotspots_are_where_they_claim_to_be() -> None:
    by_name = {h.name: h for h in HOTSPOTS}
    assert by_name["accra"].lon == pytest.approx(-0.20)  # Gulf of Guinea, Ghana
    assert by_name["accra"].lat == pytest.approx(5.55)
    assert by_name["limassol"].lon == pytest.approx(33.0)  # Akrotiri Bay, Cyprus
    assert by_name["limassol"].lat == pytest.approx(34.6)


# -- planet connector -------------------------------------------------------------


def test_planet_imports_and_constructs_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing an optional commercial connector must never break the default path."""
    monkeypatch.setattr("mdebris.config.settings.planet_api_key", None)
    client = PlanetClient()
    assert client.is_configured is False
    assert "configured=False" in repr(client)


def test_planet_explains_how_to_configure_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mdebris.config.settings.planet_api_key", None)
    client = PlanetClient()
    with pytest.raises(PlanetAuthError) as excinfo:
        client.search(GeoBBox(-1, 5, 0, 6), "2024-01-01", "2024-02-01")
    message = str(excinfo.value)
    assert "PL_API_KEY" in message
    assert "optional" in message


def test_planet_auth_error_is_a_planet_error() -> None:
    assert issubclass(PlanetAuthError, PlanetError)


def test_planet_tile_urls_can_be_built_without_leaking_the_key() -> None:
    client = PlanetClient(api_key="test-key-not-real")
    tile = TileRef(x=3, y=4, z=10)
    with_key = client.tile_url("PSScene", "20240101_000000_00_1234", tile)
    assert with_key.endswith("/10/3/4.png?api_key=test-key-not-real")
    without = client.tile_url("PSScene", "20240101_000000_00_1234", tile, include_key=False)
    assert "api_key" not in without
    assert client.tile_url_template("PSScene", "abc").endswith("/{z}/{x}/{y}.png")
    assert "api_key" not in client.tile_url_template("PSScene", "abc")


def test_planet_client_accepts_an_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mdebris.config.settings.planet_api_key", None)
    assert PlanetClient(api_key="explicit").is_configured is True


# -- network ----------------------------------------------------------------------


@pytest.mark.network
def test_fetch_sample_chips_reproduces_a_chip(tmp_path: Path) -> None:
    """Re-cuts the Accra chip from live imagery and checks it against the bundled one."""
    from mdebris.data.samples import fetch_sample_chips

    accra = next(h for h in HOTSPOTS if h.name == "accra")
    written = fetch_sample_chips([accra], dest=tmp_path)
    assert len(written) == 1
    assert written[0].stat().st_size < 10 * 1024 * 1024

    import rasterio

    with rasterio.open(written[0]) as src:
        assert (src.width, src.height) == (CHIP_PIXELS, CHIP_PIXELS)
        assert list(src.descriptions) == [str(b) for b in SAMPLE_BANDS]
        assert src.transform.a == pytest.approx(10.0)
    assert (tmp_path / "accra.json").is_file()
