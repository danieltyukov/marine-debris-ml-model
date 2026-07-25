"""Tests for mdebris.indices.spectral.

Three kinds of assertion here, in descending order of how much they would catch:

1. analytic cases with hand-computed answers, which are the only thing that catches a
   transcribed formula being subtly wrong,
2. properties that must hold for every registered index (dtype, declared range, NaN
   propagation, warning silence), which catch a new index being added carelessly,
3. registry and dispatch behaviour.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mdebris.indices.spectral import (
    BAND_ALIASES,
    FDI_WAVELENGTHS_NM,
    FDI_WAVELENGTHS_NM_USGS,
    INDEX_REGISTRY,
    available_indices,
    compute_indices,
    fai,
    fdi,
    fdi_baseline_slope,
    kndvi,
    mndwi,
    ndmi,
    ndvi,
    ndwi,
    normalize_bands,
    plastic_index,
    rndvi,
)

# Bands that between them satisfy every registered index.
ALL_BANDS = ("green", "red", "rededge2", "nir", "swir1")


def _full_bands(seed: int = 0, shape: tuple[int, ...] = (8, 8)) -> dict[str, np.ndarray]:
    """Random but physically plausible reflectance for every band an index needs."""
    rng = np.random.default_rng(seed)
    return {b: rng.uniform(0.0, 1.0, size=shape).astype(np.float32) for b in ALL_BANDS}


def _pathological_bands() -> dict[str, np.ndarray]:
    """Bands exercising every numerical hazard a real L2A tile contains.

    Column by column: NaN no-data, an all-zero pixel, a pixel whose bands sum to zero
    because atmospheric correction went negative, and an ordinary bright pixel.
    """
    return {
        "nir": np.array([np.nan, 0.0, -0.05, 0.40], dtype=np.float32),
        "red": np.array([0.10, 0.0, 0.05, 0.20], dtype=np.float32),
        "green": np.array([0.0, 0.0, 0.03, np.nan], dtype=np.float32),
        "rededge2": np.array([0.05, 0.0, 0.0, np.nan], dtype=np.float32),
        "swir1": np.array([0.0, 0.0, -0.02, 0.10], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Analytic cases
# ---------------------------------------------------------------------------


def test_ndvi_exact_half():
    assert ndvi(np.float32(0.6), np.float32(0.2)) == pytest.approx(0.5)


def test_ndwi_exact_half():
    # (0.3 - 0.1) / (0.3 + 0.1)
    assert ndwi(np.float32(0.3), np.float32(0.1)) == pytest.approx(0.5)


def test_mndwi_exact():
    # (0.30 - 0.05) / (0.30 + 0.05)
    assert mndwi(np.float32(0.30), np.float32(0.05)) == pytest.approx(0.25 / 0.35, rel=1e-6)


def test_ndmi_exact():
    # (0.40 - 0.10) / (0.40 + 0.10)
    assert ndmi(np.float32(0.40), np.float32(0.10)) == pytest.approx(0.6)


def test_rndvi_is_negated_ndvi():
    nir = np.array([0.6, 0.1, 0.35], dtype=np.float32)
    red = np.array([0.2, 0.4, 0.35], dtype=np.float32)
    np.testing.assert_allclose(rndvi(red, nir), -ndvi(nir, red), rtol=0, atol=0)


def test_plastic_index_exact():
    # 0.6 / (0.6 + 0.2)
    assert plastic_index(np.float32(0.6), np.float32(0.2)) == pytest.approx(0.75)


def test_plastic_index_is_rescaled_ndvi():
    """PI == (NDVI + 1) / 2 identically; the test pins that they cannot drift apart."""
    bands = _full_bands(seed=3)
    np.testing.assert_allclose(
        plastic_index(bands["nir"], bands["red"]),
        (ndvi(bands["nir"], bands["red"]) + 1.0) / 2.0,
        rtol=1e-6,
        atol=1e-7,
    )


def test_kndvi_exact():
    # NDVI = 0.5, so kNDVI = tanh(0.25)
    assert kndvi(np.float32(0.6), np.float32(0.2)) == pytest.approx(math.tanh(0.25), rel=1e-6)


def test_kndvi_equals_tanh_ndvi_squared():
    bands = _full_bands(seed=5)
    nd = ndvi(bands["nir"], bands["red"])
    np.testing.assert_allclose(
        kndvi(bands["nir"], bands["red"]), np.tanh(nd**2), rtol=1e-6, atol=1e-7
    )


def test_fdi_baseline_slope_is_paper_value():
    """10 * (833 - 665) / (1610 - 665) = 1.7778, the Biermann et al. 2020 constant."""
    assert fdi_baseline_slope() == pytest.approx(10.0 * 168.0 / 945.0)
    assert fdi_baseline_slope() == pytest.approx(1.777778, rel=1e-5)


def test_fdi_hand_computed():
    """nir=0.15, rededge2=0.10, swir1=0.05.

    baseline = 0.10 + (0.05 - 0.10) * 16/9 = 0.10 - 0.088889 = 0.011111
    FDI      = 0.15 - 0.011111 = 0.138889
    """
    value = fdi(np.float32(0.15), np.float32(0.10), np.float32(0.05))
    assert value == pytest.approx(0.05 + 0.8 / 9.0, rel=1e-6)
    assert value == pytest.approx(0.1388889, rel=1e-5)


def test_fdi_is_zero_on_a_flat_spectrum():
    """When swir1 equals the baseline band, the baseline collapses to that band."""
    assert fdi(np.float32(0.07), np.float32(0.07), np.float32(0.07)) == pytest.approx(0.0)


def test_fdi_usgs_wavelengths_change_the_slope():
    """The 842 nm convention is a materially different index, not a rounding detail."""
    assert fdi_baseline_slope(FDI_WAVELENGTHS_NM_USGS) == pytest.approx(10.0 * 177.0 / 945.0)
    default = fdi(np.float32(0.15), np.float32(0.10), np.float32(0.05))
    usgs = fdi(
        np.float32(0.15), np.float32(0.10), np.float32(0.05), wavelengths=FDI_WAVELENGTHS_NM_USGS
    )
    assert usgs != pytest.approx(default, rel=1e-3)
    # Larger slope, lower baseline for swir1 < rededge2, so a larger FDI.
    assert usgs > default


def test_fdi_custom_wavelengths_round_trip():
    assert fdi_baseline_slope(dict(FDI_WAVELENGTHS_NM)) == pytest.approx(fdi_baseline_slope())


def test_fdi_rejects_degenerate_wavelengths():
    with pytest.raises(ValueError, match="must differ"):
        fdi_baseline_slope({"red": 665.0, "nir": 833.0, "swir1": 665.0})


def test_fdi_rejects_unknown_baseline_band():
    with pytest.raises(ValueError, match="baseline_band"):
        fdi(0.1, 0.1, 0.1, baseline_band="B05")  # type: ignore[arg-type]


def test_fdi_b04_variant_differs_from_b06():
    """Substituting B04 for B06 is a different number and must not be silently equated."""
    bands = {"nir": 0.15, "rededge2": 0.10, "red": 0.06, "swir1": 0.05}
    b06 = fdi(bands["nir"], bands["rededge2"], bands["swir1"])
    b04 = fdi(bands["nir"], bands["red"], bands["swir1"], baseline_band="B04")
    assert float(b06) != pytest.approx(float(b04), rel=1e-3)


def test_fai_hand_computed():
    """nir=0.15, red=0.10, swir1=0.05, slope = 168/945 = 0.177778 (no factor of 10).

    baseline = 0.10 + (0.05 - 0.10) * 0.177778 = 0.091111
    FAI      = 0.15 - 0.091111 = 0.058889
    """
    value = fai(np.float32(0.15), np.float32(0.10), np.float32(0.05))
    assert value == pytest.approx(0.05 + 0.08 / 9.0, rel=1e-6)
    assert value == pytest.approx(0.0588889, rel=1e-5)


def test_fai_is_fdi_without_the_factor_of_ten():
    """The only arithmetic difference between the two is the scale factor."""
    nir, low, swir1 = np.float32(0.15), np.float32(0.10), np.float32(0.05)
    scaled_baseline = low + (swir1 - low) * (fdi_baseline_slope() / 10.0)
    assert fai(nir, low, swir1) == pytest.approx(float(nir - scaled_baseline), rel=1e-6)


def test_fdi_is_more_swir_sensitive_than_fai():
    """The factor of 10 is what makes FDI respond to the SWIR contrast plastic shows."""
    nir, low = np.float32(0.12), np.float32(0.06)
    lo_swir, hi_swir = np.float32(0.02), np.float32(0.06)
    fdi_swing = abs(float(fdi(nir, low, lo_swir) - fdi(nir, low, hi_swir)))
    fai_swing = abs(float(fai(nir, low, lo_swir) - fai(nir, low, hi_swir)))
    assert fdi_swing == pytest.approx(10.0 * fai_swing, rel=1e-4)


# ---------------------------------------------------------------------------
# Properties over every registered index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_output_is_float32_and_shape_preserving(name):
    bands = _full_bands(seed=1, shape=(6, 7))
    result = INDEX_REGISTRY[name].compute(bands)
    assert result.dtype == np.float32
    assert result.shape == (6, 7)


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_output_within_declared_valid_range(name):
    """The range recorded on each IndexSpec must actually bound its output."""
    spec = INDEX_REGISTRY[name]
    lo, hi = spec.valid_range
    # Include the extremes, since several indices attain their bounds only at 0 and 1.
    bands = _full_bands(seed=2, shape=(64, 64))
    for band in bands:
        bands[band][0, 0] = 0.0
        bands[band][0, 1] = 1.0
    bands["nir"][1, 0] = 1.0
    bands["red"][1, 0] = 0.0
    bands["swir1"][1, 0] = 0.0
    bands["rededge2"][1, 0] = 1.0
    result = spec.compute(bands)
    finite = result[np.isfinite(result)]
    assert finite.min() >= lo, f"{name} fell below its declared minimum {lo}"
    assert finite.max() <= hi, f"{name} exceeded its declared maximum {hi}"


def test_normalized_indices_stay_in_unit_interval():
    bands = _full_bands(seed=4, shape=(32, 32))
    for values in (
        ndvi(bands["nir"], bands["red"]),
        ndwi(bands["green"], bands["nir"]),
        mndwi(bands["green"], bands["swir1"]),
        ndmi(bands["nir"], bands["swir1"]),
        rndvi(bands["red"], bands["nir"]),
    ):
        assert np.all(values >= -1.0)
        assert np.all(values <= 1.0)


def test_kndvi_in_zero_to_one():
    bands = _full_bands(seed=6, shape=(32, 32))
    values = kndvi(bands["nir"], bands["red"])
    assert np.all(values >= 0.0)
    assert np.all(values < 1.0)
    # tanh(1) is the true supremum, reached only at |NDVI| = 1.
    assert values.max() <= math.tanh(1.0)


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_nan_propagates(name):
    """No-data must stay no-data. A silent 0 here reads as ordinary water downstream."""
    spec = INDEX_REGISTRY[name]
    bands = _full_bands(seed=7, shape=(4, 4))
    for band in spec.bands:
        clean = {b: v.copy() for b, v in bands.items()}
        clean[band][2, 3] = np.nan
        result = spec.compute(clean)
        assert np.isnan(result[2, 3]), f"{name} swallowed a NaN in {band}"
        assert np.isfinite(result[0, 0]), f"{name} spread NaN beyond the affected pixel"


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_no_warnings_on_nan_zeros_and_negatives(name):
    """Not one RuntimeWarning, on any of the three hazards real tiles contain."""
    spec = INDEX_REGISTRY[name]
    bands = _pathological_bands()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = spec.compute(bands)
    assert not caught, f"{name} raised {[str(w.message) for w in caught]}"
    assert result.dtype == np.float32


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_never_returns_infinity(name):
    """Every value is finite or NaN. An inf would poison any downstream statistic."""
    spec = INDEX_REGISTRY[name]
    result = spec.compute(_pathological_bands())
    assert not np.any(np.isinf(result)), f"{name} produced an infinity"


def test_zero_denominator_gives_nan_not_zero():
    """An all-zero pixel is undefined, not neutral."""
    zero = np.zeros(3, dtype=np.float32)
    assert np.all(np.isnan(ndvi(zero, zero)))
    assert np.all(np.isnan(ndwi(zero, zero)))
    assert np.all(np.isnan(plastic_index(zero, zero)))
    assert np.all(np.isnan(kndvi(zero, zero)))


def test_denominator_cancelling_to_zero_gives_nan():
    """Negative reflectance can cancel a positive band exactly; that is still undefined."""
    nir = np.float32(0.05)
    red = np.float32(-0.05)
    assert np.isnan(ndvi(nir, red))
    assert np.isnan(plastic_index(nir, red))


def test_negative_reflectance_is_computed_not_rejected():
    """Slightly negative L2A reflectance is normal and must not be clipped away."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fdi(np.float32(0.10), np.float32(-0.01), np.float32(-0.02))
    assert not caught
    assert np.isfinite(value)
    # 0.10 - (-0.01 + (-0.02 + 0.01) * 16/9)
    assert value == pytest.approx(0.10 + 0.01 + 0.16 / 9.0, rel=1e-5)


def test_baseline_indices_do_not_warn_on_nan():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert np.isnan(fdi(np.nan, 0.1, 0.05))
        assert np.isnan(fai(0.1, np.nan, 0.05))
    assert not caught


def test_broadcasting_against_a_scalar():
    nir = np.full((3, 3), 0.6, dtype=np.float32)
    result = ndvi(nir, 0.2)
    assert result.shape == (3, 3)
    assert np.allclose(result, 0.5)


def test_integer_input_is_accepted():
    """Callers occasionally pass raw integers; they must not silently integer-divide."""
    result = ndvi(np.array([6], dtype=np.int16), np.array([2], dtype=np.int16))
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_keys_match_spec_names():
    for key, spec in INDEX_REGISTRY.items():
        assert key == spec.name


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_every_index_is_cited(name):
    """A number that lands in exported GeoJSON has to be traceable to a paper."""
    spec = INDEX_REGISTRY[name]
    assert spec.citation.strip(), f"{name} has no citation"
    assert any(str(year) in spec.citation for year in range(1970, 2027)), (
        f"{name} citation carries no year: {spec.citation}"
    )
    assert spec.description.strip(), f"{name} has no description"
    assert callable(spec.func)
    assert spec.bands


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_registry_bands_are_canonical(name):
    """A band name in a spec must be one normalize_bands can actually produce."""
    for band in INDEX_REGISTRY[name].bands:
        assert BAND_ALIASES.get(band) == band, f"{name} declares non-canonical band {band!r}"


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_registry_valid_range_is_ordered(name):
    lo, hi = INDEX_REGISTRY[name].valid_range
    assert lo < hi


def test_expected_indices_are_registered():
    assert {
        "FDI",
        "FDI_B04",
        "FAI",
        "NDVI",
        "NDWI",
        "MNDWI",
        "NDMI",
        "RNDVI",
        "PI",
        "KNDVI",
    } == set(INDEX_REGISTRY)


def test_index_spec_available_and_missing():
    spec = INDEX_REGISTRY["FDI"]
    assert spec.available({"nir": 1, "rededge2": 1, "swir1": 1})
    assert not spec.available({"nir": 1, "swir1": 1})
    with pytest.raises(KeyError, match="rededge2"):
        spec.compute({"nir": np.float32(0.1), "swir1": np.float32(0.1)})


# ---------------------------------------------------------------------------
# Band-name normalization and compute_indices dispatch
# ---------------------------------------------------------------------------


def test_normalize_bands_accepts_esa_ids_and_stac_names():
    resolved = normalize_bands(
        {"B03": [0.1], "red": [0.2], "B06": [0.3], "nir": [0.4], "swir16": [0.5]}
    )
    assert set(resolved) == {"green", "red", "rededge2", "nir", "swir1"}
    assert all(v.dtype == np.float32 for v in resolved.values())


def test_normalize_bands_is_case_insensitive():
    assert set(normalize_bands({"b04": [0.1], "NIR": [0.2]})) == {"red", "nir"}


def test_normalize_bands_drops_unknown_keys():
    """Passing a whole asset dictionary, SCL and visual included, must not raise."""
    resolved = normalize_bands({"B04": [0.1], "SCL": [4], "visual": [0], "AOT": [0]})
    assert set(resolved) == {"red"}


def test_normalize_bands_is_idempotent():
    once = normalize_bands({"B04": [0.1], "B08": [0.2]})
    assert set(normalize_bands(once)) == set(once)


def test_compute_indices_skips_indices_with_missing_bands():
    """The whole point: a scene without B06 still yields everything else."""
    bands = {"B08": np.float32([0.4]), "B04": np.float32([0.2])}
    result = compute_indices(bands)
    assert set(result) == {"NDVI", "RNDVI", "PI", "KNDVI"}
    assert "FDI" not in result
    assert "NDWI" not in result


def test_compute_indices_falls_back_to_the_b04_fdi_variant():
    bands = {"B08": np.float32([0.4]), "B04": np.float32([0.2]), "B11": np.float32([0.1])}
    result = compute_indices(bands)
    assert "FDI_B04" in result
    assert "FDI" not in result


def test_compute_indices_computes_everything_when_all_bands_present():
    result = compute_indices(_full_bands(seed=8, shape=(5, 5)))
    assert set(result) == set(INDEX_REGISTRY)
    assert all(v.shape == (5, 5) for v in result.values())


def test_compute_indices_with_no_usable_bands_returns_empty():
    assert compute_indices({"SCL": np.zeros(4, dtype=np.uint8)}) == {}


def test_compute_indices_honours_an_explicit_subset():
    result = compute_indices(_full_bands(seed=9, shape=(3, 3)), ["NDVI", "FDI"])
    assert set(result) == {"NDVI", "FDI"}


def test_compute_indices_subset_still_skips_missing_bands():
    result = compute_indices({"B08": np.float32([0.4]), "B04": np.float32([0.2])}, ["NDVI", "FDI"])
    assert set(result) == {"NDVI"}


def test_compute_indices_raises_on_an_unknown_name():
    """A missing band is a data fact and is skipped; a bad name is a bug and raises."""
    with pytest.raises(KeyError, match="NDXI"):
        compute_indices(_full_bands(seed=10, shape=(2, 2)), ["NDXI"])


def test_compute_indices_matches_calling_the_function_directly():
    bands = _full_bands(seed=11, shape=(4, 4))
    result = compute_indices(bands, ["FDI", "NDVI"])
    np.testing.assert_array_equal(
        result["FDI"], fdi(bands["nir"], bands["rededge2"], bands["swir1"])
    )
    np.testing.assert_array_equal(result["NDVI"], ndvi(bands["nir"], bands["red"]))


def test_available_indices_reflects_the_band_set():
    assert available_indices({"B08": [0.4], "B04": [0.2]}) == ["NDVI", "RNDVI", "PI", "KNDVI"]
    assert available_indices({}) == []
