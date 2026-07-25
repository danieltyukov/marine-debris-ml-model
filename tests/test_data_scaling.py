"""Tests for reflectance scaling.

The whole point of this module is that getting it wrong is silent: no crash, no obviously
broken mask, just every band shifted by 0.1. So the tests assert the numbers directly, and
assert the two failure directions separately, because applying -0.1 to a pre-2022 scene is
just as wrong as omitting it on a modern one.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdebris.data.scaling import (
    BOA_OFFSET,
    NON_REFLECTANCE_BANDS,
    REFLECTANCE_SCALE,
    reflectance_params_for_baseline,
    reflectance_params_for_item,
    scale_bands,
    to_reflectance,
)
from mdebris.data.stac import Band

# Shapes copied from live items. Element84 publishes raster:bands per asset with different
# scales; Planetary Computer publishes none and only carries the baseline property.
E84_ASSETS = {
    "red": {"raster:bands": [{"scale": 0.0001, "offset": -0.1, "data_type": "uint16"}]},
    "nir": {"raster:bands": [{"scale": 0.0001, "offset": -0.1, "data_type": "uint16"}]},
    "scl": {"raster:bands": [{"data_type": "uint8", "spatial_resolution": 20}]},
    "aot": {"raster:bands": [{"scale": 0.001, "offset": 0, "data_type": "uint16"}]},
    "wvp": {"raster:bands": [{"scale": 0.001, "offset": 0, "unit": "cm"}]},
}
PC_ASSETS = {"B04": {}, "B08": {}, "SCL": {}}


def item(assets: dict, baseline: str | None = "05.10") -> dict:
    props = {} if baseline is None else {"s2:processing_baseline": baseline}
    return {"assets": assets, "properties": props}


# -- to_reflectance ---------------------------------------------------------------


def test_to_reflectance_applies_scale_and_offset() -> None:
    values = np.array([1000, 2000, 11000], dtype="uint16")
    assert to_reflectance(values) == pytest.approx([0.0, 0.1, 1.0], abs=1e-6)


def test_to_reflectance_returns_float32() -> None:
    """Reflectance stays float32; float64 would double every array in the pipeline."""
    assert to_reflectance(np.array([1234], dtype="uint16")).dtype == np.float32


def test_to_reflectance_honours_an_explicit_zero_offset() -> None:
    assert to_reflectance(np.array([5000], dtype="uint16"), offset=0.0) == pytest.approx([0.5])


def test_the_two_encodings_differ_by_exactly_the_offset() -> None:
    values = np.array([1200, 3000, 8000], dtype="uint16")
    modern = to_reflectance(values, offset=BOA_OFFSET)
    legacy = to_reflectance(values, offset=0.0)
    assert (legacy - modern) == pytest.approx([0.1, 0.1, 0.1], abs=1e-6)


def test_the_naive_conversion_is_what_makes_ocean_look_absurd() -> None:
    """Documents the actual bug: raw DN 1150 over deep water."""
    water_dn = np.array([1150], dtype="uint16")
    assert to_reflectance(water_dn) == pytest.approx([0.015], abs=1e-6)  # plausible sea
    assert to_reflectance(water_dn, offset=0.0) == pytest.approx([0.115], abs=1e-6)  # absurd


def test_to_reflectance_accepts_a_plain_list() -> None:
    assert to_reflectance([1000, 2000]) == pytest.approx([0.0, 0.1], abs=1e-6)


# -- baseline derivation ----------------------------------------------------------


@pytest.mark.parametrize("baseline", ["04.00", "05.10", "05.11", "06.00", 5.1])
def test_baseline_04_and_later_gets_the_offset(baseline: str | float) -> None:
    assert reflectance_params_for_baseline(baseline) == (REFLECTANCE_SCALE, BOA_OFFSET)


@pytest.mark.parametrize("baseline", ["02.07", "03.00", "03.01", "03.99", 2.0])
def test_baseline_before_04_gets_no_offset(baseline: str | float) -> None:
    """Applying -0.1 to an old scene breaks it in the opposite direction."""
    assert reflectance_params_for_baseline(baseline) == (REFLECTANCE_SCALE, 0.0)


def test_the_cutover_is_exactly_at_04_00() -> None:
    assert reflectance_params_for_baseline("03.99")[1] == 0.0
    assert reflectance_params_for_baseline("04.00")[1] == BOA_OFFSET


@pytest.mark.parametrize("baseline", [None, "", "not-a-number"])
def test_an_unknown_baseline_assumes_a_modern_product(baseline: str | None) -> None:
    assert reflectance_params_for_baseline(baseline) == (REFLECTANCE_SCALE, BOA_OFFSET)


# -- item derivation --------------------------------------------------------------


def test_planetary_computer_items_fall_back_to_the_baseline_property() -> None:
    """PC publishes no raster:bands, so the baseline is the only signal."""
    assert reflectance_params_for_item(item(PC_ASSETS, "05.10")) == (REFLECTANCE_SCALE, -0.1)
    assert reflectance_params_for_item(item(PC_ASSETS, "03.01")) == (REFLECTANCE_SCALE, 0.0)


def test_explicit_raster_bands_metadata_wins_over_the_baseline_heuristic() -> None:
    """A provider that states its encoding is believed, even against the baseline rule."""
    harmonised = {"red": {"raster:bands": [{"scale": 0.0001, "offset": 0.0}]}}
    assert reflectance_params_for_item(item(harmonised, "05.10")) == (0.0001, 0.0)


def test_element84_metadata_is_read_correctly() -> None:
    assert reflectance_params_for_item(item(E84_ASSETS, "05.10")) == (0.0001, -0.1)


def test_derivation_ignores_aot_whose_scale_is_ten_times_larger() -> None:
    """AOT carries scale 0.001. Reading it as reflectance is a 10x error."""
    aot_first = {"aot": E84_ASSETS["aot"], "red": E84_ASSETS["red"]}
    assert reflectance_params_for_item(item(aot_first, "05.10")) == (0.0001, -0.1)


def test_derivation_ignores_scl_which_publishes_no_scale() -> None:
    scl_only = {"scl": E84_ASSETS["scl"]}
    # No reflectance asset states a scale, so it falls through to the baseline.
    assert reflectance_params_for_item(item(scl_only, "03.01")) == (REFLECTANCE_SCALE, 0.0)


def test_derivation_ignores_unrecognized_asset_keys() -> None:
    noise = {"thumbnail": {"raster:bands": [{"scale": 1.0}]}, "red": E84_ASSETS["red"]}
    assert reflectance_params_for_item(item(noise, "05.10")) == (0.0001, -0.1)


def test_an_item_with_no_usable_metadata_assumes_a_modern_product() -> None:
    assert reflectance_params_for_item({}) == (REFLECTANCE_SCALE, BOA_OFFSET)


def test_derivation_works_on_an_object_with_attributes_not_just_a_dict() -> None:
    """pystac Items expose assets and properties as attributes."""

    class FakeAsset:
        def __init__(self, extra: dict) -> None:
            self.extra_fields = extra

    class FakeItem:
        def __init__(self) -> None:
            self.assets = {"red": FakeAsset({"raster:bands": [{"scale": 0.0001, "offset": -0.1}]})}
            self.properties = {"s2:processing_baseline": "05.10"}

    assert reflectance_params_for_item(FakeItem()) == (0.0001, -0.1)


# -- scale_bands ------------------------------------------------------------------


def test_scale_bands_converts_reflectance_and_leaves_scl_alone() -> None:
    bands = {
        "B04": np.array([2000], dtype="uint16"),
        "B08": np.array([1150], dtype="uint16"),
        "SCL": np.array([6], dtype="uint8"),
    }
    out = scale_bands(bands)
    assert out["B04"] == pytest.approx([0.1], abs=1e-6)
    assert out["B08"] == pytest.approx([0.015], abs=1e-6)
    assert out["SCL"].dtype == np.uint8
    assert out["SCL"] == pytest.approx([6])


def test_scl_is_never_scaled_under_any_parameters() -> None:
    """A class code run through a reflectance conversion becomes meaningless."""
    scl = np.array([0, 4, 6, 9, 11], dtype="uint8")
    out = scale_bands({"SCL": scl}, scale=1e-4, offset=-0.1)
    assert np.array_equal(out["SCL"], scl)
    assert out["SCL"].dtype == np.uint8


def test_every_non_reflectance_band_passes_through_untouched() -> None:
    values = np.array([500], dtype="uint16")
    bands = {str(b): values for b in NON_REFLECTANCE_BANDS}
    out = scale_bands(bands)
    for name in bands:
        assert np.array_equal(out[name], values), name


def test_scl_is_in_the_excluded_set_and_reflectance_bands_are_not() -> None:
    assert Band.SCL in NON_REFLECTANCE_BANDS
    for band in (Band.B02, Band.B04, Band.B06, Band.B08, Band.B11, Band.B12):
        assert band not in NON_REFLECTANCE_BANDS


def test_scale_bands_accepts_provider_band_names() -> None:
    out = scale_bands({"red": np.array([2000], dtype="uint16"), "scl": np.array([6], "uint8")})
    assert out["red"] == pytest.approx([0.1], abs=1e-6)
    assert out["scl"].dtype == np.uint8


def test_scale_bands_leaves_unrecognized_layers_alone() -> None:
    """An unknown layer is not known to be reflectance, so converting it would be a guess."""
    mask = np.array([1], dtype="uint8")
    assert np.array_equal(scale_bands({"my_custom_mask": mask})["my_custom_mask"], mask)


def test_scale_bands_preserves_keys_and_does_not_mutate_its_input() -> None:
    original = np.array([2000], dtype="uint16")
    bands = {"B04": original, "SCL": np.array([6], dtype="uint8")}
    out = scale_bands(bands)
    assert set(out) == set(bands)
    assert bands["B04"].dtype == np.uint16
    assert bands["B04"][0] == 2000


def test_the_module_docstring_examples_are_true() -> None:
    """Doctests here are the reference for a conversion that is easy to get wrong."""
    import doctest

    from mdebris.data import scaling

    results = doctest.testmod(scaling, verbose=False)
    assert results.failed == 0, f"{results.failed} of {results.attempted} doctests failed"
    assert results.attempted > 0


# -- end to end on the bundled chips ----------------------------------------------


def test_the_bundled_chips_round_trip_through_the_shared_helper() -> None:
    """The sidecar values and the shared converter agree on real pixels."""
    from mdebris.data.samples import list_samples, load_sample, sample_reflectance

    for name in list_samples():
        raw, meta = load_sample(name)
        scaled, _ = sample_reflectance(name)
        expected = to_reflectance(
            raw["B08"], scale=meta["reflectance_scale"], offset=meta["reflectance_offset"]
        )
        assert np.allclose(scaled["B08"], expected)
        assert scaled["SCL"].dtype == np.uint8
        sea = raw["SCL"] == 6
        assert 0.0 <= np.median(scaled["B08"][sea]) < 0.06
