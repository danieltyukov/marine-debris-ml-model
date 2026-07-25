"""Spectral indices for floating-material detection on Sentinel-2 reflectance.

Every function here is pure numpy over float32 surface-reflectance arrays scaled to
``[0, 1]``, i.e. Sentinel-2 L2A digital numbers divided by 10000. Nothing in this module
imports torch, rasterio or the project config, so the cheap prescreen stage of the
cascade can run anywhere the arrays can be materialised.

Why several indices rather than one: the hard part of marine-litter detection is not
finding bright pixels on dark water, it is deciding whether a bright pixel is plastic,
Sargassum, sea foam, a ship wake or a sediment plume. Those confusers separate along
different spectral axes, so the cascade needs a small panel of indices and not a single
number. FDI is the load-bearing one; the rest exist to reject the look-alikes.

Sign and dtype conventions used throughout:

* inputs are cast to ``float32``; outputs are always ``float32``,
* no-data is ``NaN`` and propagates through every index,
* a zero denominator yields ``NaN`` rather than 0, because 0/0 is genuinely undefined
  and a silent 0 would read as "ordinary water" downstream,
* no function ever emits a numpy ``RuntimeWarning``, since these run per-tile over whole
  scenes and a warning storm is indistinguishable from a real problem in a log.

Wavelength constants are documented per index because the literature is not consistent
about them. See ``FDI_WAVELENGTHS_NM`` for the one case where the disagreement changes
results materially.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BAND_ALIASES",
    "FDI_WAVELENGTHS_NM",
    "FDI_WAVELENGTHS_NM_USGS",
    "INDEX_REGISTRY",
    "S2A_CENTRAL_WAVELENGTHS_NM",
    "FloatArray",
    "IndexSpec",
    "available_indices",
    "compute_indices",
    "fai",
    "fdi",
    "fdi_baseline_slope",
    "kndvi",
    "mndwi",
    "ndmi",
    "ndvi",
    "ndwi",
    "normalize_bands",
    "plastic_index",
    "rndvi",
]

FloatArray = NDArray[np.float32]

# ---------------------------------------------------------------------------
# Band naming
# ---------------------------------------------------------------------------

# Canonical band names used by every function and by IndexSpec.bands. They follow the
# STAC "common name" vocabulary rather than the ESA band ids, because a name says what a
# band measures and "B06" does not.
CANONICAL_BANDS: tuple[str, ...] = (
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "watervapour",
    "cirrus",
    "swir1",
    "swir2",
)

# Alias -> canonical. Lookup is case-insensitive. Two naming schemes are in circulation
# for the same product: Planetary Computer exposes ESA band ids ("B04", "B11") while
# Element 84 earth-search exposes STAC common names ("red", "swir16"). Accepting both
# means callers do not have to care which STAC endpoint the scene came from.
BAND_ALIASES: dict[str, str] = {
    "b01": "coastal",
    "coastal": "coastal",
    "b02": "blue",
    "blue": "blue",
    "b03": "green",
    "green": "green",
    "b04": "red",
    "red": "red",
    "b05": "rededge1",
    "rededge1": "rededge1",
    "red_edge1": "rededge1",
    "b06": "rededge2",
    "rededge2": "rededge2",
    "red_edge2": "rededge2",
    "red2": "rededge2",
    "b07": "rededge3",
    "rededge3": "rededge3",
    "red_edge3": "rededge3",
    "b08": "nir",
    "nir": "nir",
    "b8a": "nir08",
    "b08a": "nir08",
    "nir08": "nir08",
    "b09": "watervapour",
    "nir09": "watervapour",
    "watervapour": "watervapour",
    "water_vapor": "watervapour",
    "b10": "cirrus",
    "cirrus": "cirrus",
    "b11": "swir1",
    "swir16": "swir1",
    "swir1": "swir1",
    "b12": "swir2",
    "swir22": "swir2",
    "swir2": "swir2",
}

# Sentinel-2A MSI central wavelengths in nm, spectral-response-weighted, from the ESA
# Sentinel-2 MSI Spectral Response Functions. Sentinel-2B differs by under 3 nm in every
# band, which is far below the precision any index here depends on. Kept for reference
# and for callers that want to build their own baselines; the indices below use the
# rounded literature values instead, see FDI_WAVELENGTHS_NM.
S2A_CENTRAL_WAVELENGTHS_NM: dict[str, float] = {
    "coastal": 442.7,
    "blue": 492.4,
    "green": 559.8,
    "red": 664.6,
    "rededge1": 704.1,
    "rededge2": 740.5,
    "rededge3": 782.8,
    "nir": 832.8,
    "nir08": 864.7,
    "watervapour": 945.1,
    "cirrus": 1373.5,
    "swir1": 1613.7,
    "swir2": 2202.4,
}

# Wavelengths as they appear in Biermann et al. 2020 and in the MARIDA benchmark
# (Kikaki et al. 2022), which is the reference implementation most of the marine-litter
# literature compares against. These are rounded band centres, not the SRF-weighted
# values above.
FDI_WAVELENGTHS_NM: dict[str, float] = {"red": 665.0, "nir": 833.0, "swir1": 1610.0}

# The other convention in circulation. Several public implementations (for example the
# Digital Earth Africa floating-debris notebook) take lambda_NIR = 842 nm from the USGS
# EROS Sentinel-2 band table instead of 833 nm. That is a 5.4 percent change in the
# baseline slope (1.7778 -> 1.8730), so FDI thresholds are not transferable between the
# two conventions. Exposed so the difference can be measured rather than argued about.
FDI_WAVELENGTHS_NM_USGS: dict[str, float] = {"red": 665.0, "nir": 842.0, "swir1": 1610.0}

# Factor appearing in the Biermann et al. 2020 FDI equation. It is not a unit conversion
# and it is not part of the FAI equation that FDI was derived from; it is in the
# published formula, so it is reproduced here. See fdi() for the consequence.
_FDI_SCALE = 10.0


# ---------------------------------------------------------------------------
# Numerically safe primitives
# ---------------------------------------------------------------------------


def _f32(a: ArrayLike) -> FloatArray:
    """Cast to float32 without copying an array that is already float32."""
    return np.asarray(a, dtype=np.float32)


# Denominators at or below this magnitude are treated as undefined rather than divided
# by. Sentinel-2 L2A reflectance is quantised at 1e-4, so a denominator this small
# carries no signal, and the guard is what makes every index total: for finite inputs the
# output is always either finite or NaN, never +/-inf.
_MIN_DENOMINATOR = np.float32(1e-12)


def _safe_divide(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Elementwise division with undefined results mapped to NaN and no warnings.

    Real L2A tiles contain no-data (NaN), exact zeros over deep shadow, and small
    negative reflectance where atmospheric correction over-subtracted. All three reach
    the ratio indices, and numpy's default behaviour for them is a RuntimeWarning plus
    an ``inf`` that survives silently into downstream statistics. Here a vanishing
    denominator produces NaN and nothing is warned about.
    """
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    out = np.full(shape, np.nan, dtype=np.float32)
    # A NaN denominator fails this test too, so those elements are skipped rather than
    # divided into a NaN. errstate covers the remaining ways float division can flag.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        usable = np.abs(denominator) > _MIN_DENOMINATOR
        np.divide(numerator, denominator, out=out, where=usable)
    return out


def _normalized_difference(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """``(a - b) / (a + b)``, NaN-safe. Range ``[-1, 1]`` for non-negative inputs.

    NaN where ``a + b`` vanishes, meaning exactly zero or below the ``_MIN_DENOMINATOR``
    guard. Note that non-zero bands can still sum to zero once atmospheric correction
    has produced negative reflectance, so this is not only a no-data path.
    """
    x, y = _f32(a), _f32(b)
    with np.errstate(invalid="ignore", over="ignore"):
        return _safe_divide(x - y, x + y)


def _baseline_difference(
    nir: ArrayLike, low: ArrayLike, swir1: ArrayLike, slope: float
) -> FloatArray:
    """Shared kernel of FDI and FAI: NIR minus a linear low-to-SWIR1 baseline.

    ``slope`` is the fraction of the ``low`` to ``swir1`` span at which the baseline is
    evaluated. This is plain arithmetic with no division, so NaN propagates on its own
    and no error state can be raised.
    """
    n, lo, sw = _f32(nir), _f32(low), _f32(swir1)
    baseline = lo + (sw - lo) * np.float32(slope)
    return np.asarray(n - baseline, dtype=np.float32)


# ---------------------------------------------------------------------------
# Baseline-subtraction indices
# ---------------------------------------------------------------------------


def fdi_baseline_slope(wavelengths: Mapping[str, float] | None = None) -> float:
    """Slope term of the FDI baseline, ``(l_nir - l_red) / (l_swir1 - l_red) * 10``.

    With the default wavelengths this is ``10 * (833 - 665) / (1610 - 665) = 1.77778``.
    Note it exceeds 1, so the FDI baseline is an extrapolation past SWIR1 rather than an
    interpolation between the two endpoints. That is what the published equation says.
    """
    w = FDI_WAVELENGTHS_NM if wavelengths is None else wavelengths
    span = w["swir1"] - w["red"]
    if span == 0:
        raise ValueError("swir1 and red wavelengths must differ")
    return float((w["nir"] - w["red"]) / span * _FDI_SCALE)


def fdi(
    nir: ArrayLike,
    red2: ArrayLike,
    swir1: ArrayLike,
    *,
    baseline_band: Literal["B06", "B04"] = "B06",
    wavelengths: Mapping[str, float] | None = None,
) -> FloatArray:
    """Floating Debris Index. The primary index for floating plastic.

    Formula (Biermann et al. 2020, equation 1)::

        FDI      = R_NIR - R'_NIR
        R'_NIR   = R_RE2 + (R_SWIR1 - R_RE2)
                          * ((l_NIR - l_RED) / (l_SWIR1 - l_RED)) * 10

    Sentinel-2 band mapping (the paper's convention, ``baseline_band="B06"``):

    ===========  =====  ==============  ==========
    Symbol       Band   Wavelength      Resolution
    ===========  =====  ==============  ==========
    ``R_RE2``    B06    740 nm          20 m
    ``R_NIR``    B08    833 nm          10 m
    ``R_SWIR1``  B11    1610 nm         20 m
    ===========  =====  ==============  ==========

    B06 and B11 must be resampled to the 10 m grid before this is called.

    Two conventions this implementation pins down explicitly, because published
    implementations differ and the differences are not cosmetic:

    1. **Which band is the baseline's lower endpoint.** The paper uses red edge 2 (B06)
       as the reflectance endpoint while keeping ``l_RED = 665 nm`` (B04) inside the
       wavelength ratio. That mismatch is in the published equation, not a transcription
       error here: FDI was derived from FAI by substituting the red-edge band for the
       red band, and the wavelength term was carried over unchanged. Passing
       ``baseline_band="B04"`` swaps the reflectance endpoint to B04 for scenes where
       B06 was not retrieved, which is a common substitution but produces systematically
       different values. FDI thresholds are not transferable between the two, so the
       registry exposes them as two separate indices, ``FDI`` and ``FDI_B04``.
    2. **The NIR wavelength.** 833 nm here, matching Biermann et al. and the MARIDA
       benchmark. Some implementations use 842 nm from the USGS EROS band table, which
       changes the slope from 1.7778 to 1.8730. Pass ``FDI_WAVELENGTHS_NM_USGS`` to
       reproduce those.

    Interpretation: positive FDI means NIR sits above the baseline, which is what
    floating material does and open water does not. Open water lands near zero (order
    0.005 for the reflectance magnitudes typical of L2A coastal scenes) and floating
    debris an order of magnitude higher. FDI alone does not separate plastic from
    Sargassum, both of which are FDI-positive; pair it with NDVI, which is high for
    algae and low for plastic.

    Args:
        nir: B08 reflectance in ``[0, 1]``.
        red2: Baseline endpoint reflectance. B06 by default, B04 when
            ``baseline_band="B04"``.
        swir1: B11 reflectance in ``[0, 1]``.
        baseline_band: Which band ``red2`` actually holds. Recorded for provenance and
            documentation; it does not change the arithmetic, since the substitution
            convention in the literature keeps the wavelength term fixed.
        wavelengths: Override the ``red``/``nir``/``swir1`` wavelength constants in nm.

    Returns:
        float32 array. Theoretical range for reflectance in ``[0, 1]`` is
        ``[-1.78, 1.78]`` with the default wavelengths, symmetric about zero at
        plus or minus the baseline slope. Real values sit in roughly
        ``[-0.05, 0.2]``.

    Reference:
        Biermann, L., Clewley, D., Martinez-Vicente, V., Topouzelis, K. (2020).
        Finding Plastic Patches in Coastal Waters using Optical Satellite Data.
        Scientific Reports 10, 5364. doi:10.1038/s41598-020-62298-z
    """
    if baseline_band not in ("B06", "B04"):
        raise ValueError(f"baseline_band must be 'B06' or 'B04', got {baseline_band!r}")
    return _baseline_difference(nir, red2, swir1, fdi_baseline_slope(wavelengths))


def fai(
    nir: ArrayLike,
    red: ArrayLike,
    swir1: ArrayLike,
    *,
    wavelengths: Mapping[str, float] | None = None,
) -> FloatArray:
    """Floating Algae Index. The index FDI was derived from.

    Formula::

        FAI    = R_NIR - R'_NIR
        R'_NIR = R_RED + (R_SWIR - R_RED) * (l_NIR - l_RED) / (l_SWIR - l_RED)

    Sentinel-2 band mapping: RED = B04 (665 nm), NIR = B08 (833 nm),
    SWIR1 = B11 (1610 nm). Hu 2009 defined FAI on MODIS bands 1, 2 and 5
    (645, 859 and 1240 nm); the Sentinel-2 adaptation used here is the standard one.

    FAI is FDI with two changes: the baseline's lower endpoint is red (B04) rather than
    red edge 2 (B06), and there is no factor of 10, so the slope is a genuine
    interpolation weight of 0.1778 rather than 1.7778. The practical consequence is that
    FAI is far less sensitive to SWIR1 than FDI, which is why FDI responds more strongly
    to plastic: plastic holds more SWIR reflectance than a wet algal mat, and the
    amplified SWIR term is what makes the two separable.

    Elevated FAI means floating biomass. Used here as a Sargassum discriminator: a patch
    that is high in both FDI and FAI with high NDVI is far more likely to be algae than
    plastic.

    Returns:
        float32 array. Theoretical range ``[-1, 1]`` for reflectance in ``[0, 1]``;
        real values sit near zero for water and reach roughly 0.05 over dense algae.

    Reference:
        Hu, C. (2009). A novel ocean color index to detect floating algae in the global
        oceans. Remote Sensing of Environment 113(10), 2118-2129.
        doi:10.1016/j.rse.2009.05.012
    """
    w = FDI_WAVELENGTHS_NM if wavelengths is None else wavelengths
    span = w["swir1"] - w["red"]
    if span == 0:
        raise ValueError("swir1 and red wavelengths must differ")
    slope = float((w["nir"] - w["red"]) / span)
    return _baseline_difference(nir, red, swir1, slope)


# ---------------------------------------------------------------------------
# Normalized-difference indices
# ---------------------------------------------------------------------------


def ndvi(nir: ArrayLike, red: ArrayLike) -> FloatArray:
    """Normalized Difference Vegetation Index, ``(NIR - RED) / (NIR + RED)``.

    Sentinel-2 band mapping: NIR = B08, RED = B04.

    The single most useful companion to FDI. Both plastic and Sargassum are
    FDI-positive, but Sargassum carries the chlorophyll red edge and plastic does not,
    so NDVI is what breaks the tie. Biermann et al. 2020 separate their classes in the
    FDI-NDVI plane rather than on FDI alone.

    Returns:
        float32 array in ``[-1, 1]`` for non-negative reflectance. Open water is
        negative, floating plastic mildly positive, dense algae strongly positive.
        NaN where ``NIR + RED`` vanishes.

    Reference:
        Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). Monitoring
        vegetation systems in the Great Plains with ERTS. NASA SP-351, 309-317.
    """
    return _normalized_difference(nir, red)


def ndwi(green: ArrayLike, nir: ArrayLike) -> FloatArray:
    """Normalized Difference Water Index, ``(GREEN - NIR) / (GREEN + NIR)``.

    Sentinel-2 band mapping: GREEN = B03, NIR = B08.

    This is McFeeters' water-delineation NDWI, not Gao's vegetation-moisture index of
    the same acronym (that one is ``ndmi`` below). Water absorbs strongly in NIR and
    reflects in green, so water is positive and land is negative; zero is the
    conventional cut and the one the cascade defaults to.

    Returns:
        float32 array in ``[-1, 1]`` for non-negative reflectance. NaN where
        ``GREEN + NIR`` vanishes.

    Reference:
        McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI)
        in the delineation of open water features. International Journal of Remote
        Sensing 17(7), 1425-1432. doi:10.1080/01431169608948714
    """
    return _normalized_difference(green, nir)


def mndwi(green: ArrayLike, swir1: ArrayLike) -> FloatArray:
    """Modified Normalized Difference Water Index, ``(GREEN - SWIR1) / (GREEN + SWIR1)``.

    Sentinel-2 band mapping: GREEN = B03, SWIR1 = B11.

    Xu's modification replaces NIR with SWIR1, which suppresses built-up land that
    McFeeters' NDWI misclassifies as water. It is the more reliable water mask near
    ports and urban coastline, which is exactly where coastal debris accumulates. It is
    also less confused by floating material than NDWI, since a floating patch raises NIR
    much more than it raises SWIR1.

    Returns:
        float32 array in ``[-1, 1]`` for non-negative reflectance. NaN where
        ``GREEN + SWIR1`` vanishes.

    Reference:
        Xu, H. (2006). Modification of normalised difference water index (NDWI) to
        enhance open water features in remotely sensed imagery. International Journal of
        Remote Sensing 27(14), 3025-3033. doi:10.1080/01431160600589179
    """
    return _normalized_difference(green, swir1)


def ndmi(nir: ArrayLike, swir1: ArrayLike) -> FloatArray:
    """Normalized Difference Moisture Index, ``(NIR - SWIR1) / (NIR + SWIR1)``.

    Sentinel-2 band mapping: NIR = B08, SWIR1 = B11. Also published as NDII and, in
    Gao's original wording, as NDWI, which collides with McFeeters' index of the same
    name; NDMI is used here to keep the two apart.

    Included as a plastic-versus-Sargassum discriminator. Liquid water absorbs strongly
    at 1610 nm, so a water-laden algal mat is dark in SWIR1 and gives high NDMI, whereas
    a dry polymer sheet retains SWIR1 reflectance and gives lower NDMI at comparable
    NIR. This is the same physical contrast that the amplified SWIR term in FDI exploits,
    expressed as a bounded ratio that does not depend on the FDI scaling convention.
    Treat it as a supporting feature: the separation is reported in the literature but
    it is weaker and more variable than the FDI-NDVI split.

    Returns:
        float32 array in ``[-1, 1]`` for non-negative reflectance. NaN where
        ``NIR + SWIR1`` vanishes.

    Reference:
        Gao, B.-C. (1996). NDWI: A normalized difference water index for remote sensing
        of vegetation liquid water from space. Remote Sensing of Environment 58(3),
        257-266. doi:10.1016/S0034-4257(96)00067-3
    """
    return _normalized_difference(nir, swir1)


def rndvi(red: ArrayLike, nir: ArrayLike) -> FloatArray:
    """Reversed NDVI, ``(RED - NIR) / (RED + NIR)``.

    Sentinel-2 band mapping: RED = B04, NIR = B08. Numerically this is ``-NDVI``; it is
    kept as a named index because Themistocleous et al. report it as one of their
    plastic-detection indices and keeping the name makes their thresholds directly
    usable rather than requiring a mental sign flip at every comparison.

    Returns:
        float32 array in ``[-1, 1]`` for non-negative reflectance. NaN where
        ``RED + NIR`` vanishes.

    Reference:
        Themistocleous, K., Papoutsa, C., Michaelides, S., Hadjimitsis, D. (2020).
        Investigating Detection of Floating Plastic Litter from Space Using Sentinel-2
        Imagery. Remote Sensing 12(16), 2648. doi:10.3390/rs12162648
    """
    return _normalized_difference(red, nir)


# ---------------------------------------------------------------------------
# Ratio and kernel indices
# ---------------------------------------------------------------------------


def plastic_index(nir: ArrayLike, red: ArrayLike) -> FloatArray:
    """Plastic Index, ``PI = NIR / (NIR + RED)``.

    Sentinel-2 band mapping: NIR = B08, RED = B04.

    Themistocleous et al. designed PI specifically for large floating plastic targets.
    It is a rescaled NDVI (``PI = (NDVI + 1) / 2`` exactly), so it carries no
    information NDVI does not; it is implemented under its published name so that
    thresholds quoted in the plastic-detection literature can be applied without
    re-deriving them. The paper reports plastic near PI 0.9 and water near 0.5.

    Returns:
        float32 array in ``[0, 1]`` for non-negative reflectance. Values outside that
        range are possible where atmospheric correction produced negative reflectance.
        NaN where ``NIR + RED`` vanishes.

    Reference:
        Themistocleous, K., Papoutsa, C., Michaelides, S., Hadjimitsis, D. (2020).
        Investigating Detection of Floating Plastic Litter from Space Using Sentinel-2
        Imagery. Remote Sensing 12(16), 2648. doi:10.3390/rs12162648
    """
    n, r = _f32(nir), _f32(red)
    with np.errstate(invalid="ignore", over="ignore"):
        return _safe_divide(n, n + r)


def kndvi(nir: ArrayLike, red: ArrayLike) -> FloatArray:
    """Kernel NDVI, ``kNDVI = tanh(NDVI^2)``.

    Sentinel-2 band mapping: NIR = B08, RED = B04.

    Camps-Valls et al. define kNDVI as ``tanh(((NIR - RED) / (2*sigma))^2)`` for an RBF
    kernel with length-scale sigma. Setting ``sigma = 0.5 * (NIR + RED)``, the choice the
    paper recommends as a sensible default, collapses it to ``tanh(NDVI^2)``, which is
    the form implemented here and the one used in the marine-debris literature.

    Squaring makes kNDVI insensitive to the sign of NDVI, so it does not separate
    vegetation from water the way NDVI does. Its value here is different: it saturates
    much later than NDVI, so it keeps contrast among the strongly vegetated pixels where
    NDVI has flattened out, which helps grade dense Sargassum mats rather than merely
    flagging them.

    Returns:
        float32 array in ``[0, tanh(1)] = [0, 0.7616]``, hence within ``[0, 1)``.
        NaN wherever NDVI is NaN.

    Reference:
        Camps-Valls, G., Campos-Taberner, M., Moreno-Martinez, A., Walther, S., Duveiller,
        G., Cescatti, A., Mahecha, M.D., Munoz-Mari, J., Garcia-Haro, F.J., Guanter, L.,
        Jung, M., Gamon, J.A., Reichstein, M., Running, S.W. (2021). A unified vegetation
        index for quantifying the terrestrial biosphere. Science Advances 7(9), eabc7447.
        doi:10.1126/sciadv.abc7447
    """
    nd = _normalized_difference(nir, red)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.asarray(np.tanh(nd * nd), dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """Everything the pipeline needs to compute and interpret one index generically.

    Carrying the citation on the spec rather than only in the docstring is deliberate:
    index values end up as properties on exported GeoJSON detections, and a reader of
    that file needs to know which paper's definition produced the number.
    """

    name: str
    func: Callable[..., FloatArray]
    bands: tuple[str, ...]
    """Canonical band names, in the positional order ``func`` expects them."""
    valid_range: tuple[float, float]
    """Attainable output range for reflectance in ``[0, 1]``, used for sanity checks."""
    citation: str
    description: str = ""
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    """Fixed keyword arguments baked into this registry entry, e.g. the FDI variant."""

    def available(self, bands: Mapping[str, Any]) -> bool:
        """True when every required band is present in an already-normalized mapping."""
        return all(b in bands for b in self.bands)

    def compute(self, bands: Mapping[str, ArrayLike]) -> FloatArray:
        """Call ``func`` with the required bands pulled from a normalized mapping."""
        missing = [b for b in self.bands if b not in bands]
        if missing:
            raise KeyError(f"index {self.name} needs missing band(s): {', '.join(missing)}")
        return self.func(*(bands[b] for b in self.bands), **self.kwargs)


_BIERMANN_2020 = (
    "Biermann et al. (2020), Sci. Rep. 10:5364, doi:10.1038/s41598-020-62298-z"
)
_THEMISTOCLEOUS_2020 = (
    "Themistocleous et al. (2020), Remote Sens. 12(16):2648, doi:10.3390/rs12162648"
)

INDEX_REGISTRY: dict[str, IndexSpec] = {
    "FDI": IndexSpec(
        name="FDI",
        func=fdi,
        bands=("nir", "rededge2", "swir1"),
        valid_range=(-2.0, 2.0),
        citation=_BIERMANN_2020,
        description="Floating Debris Index, B06 baseline as published. Primary debris index.",
    ),
    "FDI_B04": IndexSpec(
        name="FDI_B04",
        func=fdi,
        bands=("nir", "red", "swir1"),
        valid_range=(-2.0, 2.0),
        citation=_BIERMANN_2020,
        description=(
            "Floating Debris Index with B04 substituted for the B06 baseline endpoint. "
            "Fallback when B06 is unavailable; values are not comparable to FDI."
        ),
        kwargs={"baseline_band": "B04"},
    ),
    "FAI": IndexSpec(
        name="FAI",
        func=fai,
        bands=("nir", "red", "swir1"),
        valid_range=(-1.0, 1.0),
        citation="Hu (2009), Remote Sens. Environ. 113(10):2118, doi:10.1016/j.rse.2009.05.012",
        description="Floating Algae Index. Sargassum discriminator.",
    ),
    "NDVI": IndexSpec(
        name="NDVI",
        func=ndvi,
        bands=("nir", "red"),
        valid_range=(-1.0, 1.0),
        citation="Rouse et al. (1974), NASA SP-351:309-317",
        description="Vegetation index. Separates algae from plastic in the FDI-NDVI plane.",
    ),
    "NDWI": IndexSpec(
        name="NDWI",
        func=ndwi,
        bands=("green", "nir"),
        valid_range=(-1.0, 1.0),
        citation="McFeeters (1996), Int. J. Remote Sens. 17(7):1425, doi:10.1080/01431169608948714",
        description="Open-water delineation. Drives the cascade water mask.",
    ),
    "MNDWI": IndexSpec(
        name="MNDWI",
        func=mndwi,
        bands=("green", "swir1"),
        valid_range=(-1.0, 1.0),
        citation="Xu (2006), Int. J. Remote Sens. 27(14):3025, doi:10.1080/01431160600589179",
        description="Water delineation robust to built-up coastline.",
    ),
    "NDMI": IndexSpec(
        name="NDMI",
        func=ndmi,
        bands=("nir", "swir1"),
        valid_range=(-1.0, 1.0),
        citation="Gao (1996), Remote Sens. Environ. 58(3):257, doi:10.1016/S0034-4257(96)00067-3",
        description="Moisture contrast. Wet algal mat versus dry polymer.",
    ),
    "RNDVI": IndexSpec(
        name="RNDVI",
        func=rndvi,
        bands=("red", "nir"),
        valid_range=(-1.0, 1.0),
        citation=_THEMISTOCLEOUS_2020,
        description="Reversed NDVI as published for plastic detection. Equals -NDVI.",
    ),
    "PI": IndexSpec(
        name="PI",
        func=plastic_index,
        bands=("nir", "red"),
        valid_range=(0.0, 1.0),
        citation=_THEMISTOCLEOUS_2020,
        description="Plastic Index. Rescaled NDVI, kept for published thresholds.",
    ),
    "KNDVI": IndexSpec(
        name="KNDVI",
        func=kndvi,
        bands=("nir", "red"),
        valid_range=(0.0, 1.0),
        citation="Camps-Valls et al. (2021), Sci. Adv. 7(9):eabc7447, doi:10.1126/sciadv.abc7447",
        description="Kernel NDVI. Keeps contrast where NDVI saturates over dense algae.",
    ),
}


def normalize_bands(bands: Mapping[str, ArrayLike]) -> dict[str, FloatArray]:
    """Map band keys to canonical names and cast every array to float32.

    Accepts ESA band ids ("B04"), STAC common names ("red", "swir16") and canonical
    names, case-insensitively. Keys that match no alias are dropped rather than raising,
    so a band dictionary that also carries "SCL" or "visual" can be passed straight
    through. When two aliases of the same band are present the later key wins, following
    ordinary dict semantics.
    """
    out: dict[str, FloatArray] = {}
    for key, array in bands.items():
        canonical = BAND_ALIASES.get(key.strip().lower())
        if canonical is not None:
            out[canonical] = _f32(array)
    return out


def available_indices(bands: Mapping[str, ArrayLike]) -> list[str]:
    """Names of every registered index computable from ``bands``, in registry order."""
    resolved = normalize_bands(bands)
    return [name for name, spec in INDEX_REGISTRY.items() if spec.available(resolved)]


def compute_indices(
    bands: Mapping[str, ArrayLike], names: Sequence[str] | None = None
) -> dict[str, FloatArray]:
    """Compute every requested index whose required bands are present.

    Indices whose bands are missing are skipped silently. That is the point: band
    availability varies by product and by STAC endpoint, and a scene that happens to
    lack B06 should still yield NDVI and NDWI rather than aborting the whole prescreen.
    An unknown index *name*, by contrast, raises, because that is a caller bug and not a
    property of the data.

    Args:
        bands: Band arrays keyed by ESA id, STAC common name or canonical name. All
            arrays must broadcast against each other.
        names: Which indices to compute. ``None`` means every registered index.

    Returns:
        Index name to float32 array, containing only the indices that were computable.

    Raises:
        KeyError: if ``names`` contains a name that is not in ``INDEX_REGISTRY``.
    """
    resolved = normalize_bands(bands)
    wanted: Iterable[str] = INDEX_REGISTRY if names is None else names
    out: dict[str, FloatArray] = {}
    for name in wanted:
        spec = INDEX_REGISTRY.get(name)
        if spec is None:
            raise KeyError(
                f"unknown index {name!r}; available: {', '.join(sorted(INDEX_REGISTRY))}"
            )
        if spec.available(resolved):
            out[name] = spec.compute(resolved)
    return out
