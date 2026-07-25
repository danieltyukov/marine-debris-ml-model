"""Converting stored Sentinel-2 integers into surface reflectance.

L2A does not store reflectance. It stores integers, and turning them back into physical
units is not just a division, because ESA changed the encoding partway through the
archive. From processing baseline 04.00 (25 January 2022) products carry
``BOA_ADD_OFFSET = -1000``, so:

    reflectance = dn * 0.0001 - 0.1        baseline 04.00 and later
    reflectance = dn * 0.0001              baseline 03.xx and earlier

Getting this wrong is quiet rather than loud. Dividing a modern scene by 10000 puts open
ocean at about 0.12 in the near infrared, which no sea on Earth does, but nothing raises:
every band simply shifts by +0.1.

Which indices care
------------------
Not all of them, and the split is worth knowing before trusting a number:

- FDI and FAI are unaffected. They are differences of differences, so a constant added to
  every band cancels exactly.
- NDVI, NDWI, MNDWI, PI and kNDVI change magnitude, because the offset cancels in the
  numerator but not in the denominator. Their *sign* survives, so a threshold at zero
  (``NDWI > 0`` for water) still gives the same mask.
- Any absolute threshold on a ratio index, or on reflectance itself, is simply wrong.

So the failure mode is not a crash and not a garbled mask. It is a magnitude threshold
that was tuned on one encoding and silently misapplied to the other.

Provider notes, all verified against live items rather than documentation
------------------------------------------------------------------------
- **Element84 earth-search** publishes explicit ``raster:bands`` scale and offset per
  asset, and they are authoritative. Verified on ``S2A_30NYM_20240430_0_L2A``: the
  reflectance bands carry ``scale 0.0001, offset -0.1``; ``SCL`` carries no scale at all;
  ``AOT`` and ``WVP`` carry ``scale 0.001, offset 0``. This is why the derivation below
  looks specifically at a reflectance band. Reading whichever asset happens to come first
  can pick up AOT's 0.001 and be wrong by a factor of ten.
- **Planetary Computer** does not publish ``raster:bands`` on its Sentinel-2 items, so the
  offset has to come from ``s2:processing_baseline``. Verified present and equal to
  ``"05.10"`` on current scenes.
- Neither provider was observed pre-harmonising the offset away, but a mirror could, which
  is why explicit metadata always wins over the baseline rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mdebris.data.stac import Band, canonical_band

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

__all__ = [
    "BOA_OFFSET",
    "BOA_OFFSET_BASELINE",
    "NON_REFLECTANCE_BANDS",
    "REFLECTANCE_SCALE",
    "reflectance_params_for_baseline",
    "reflectance_params_for_item",
    "scale_bands",
    "to_reflectance",
]

# Quantification value for L2A reflectance: stored integers are reflectance * 10000.
REFLECTANCE_SCALE = 1e-4

# BOA_ADD_OFFSET expressed in reflectance units, applied from baseline 04.00 onward.
BOA_OFFSET = -0.1
BOA_OFFSET_BASELINE = 4.0

# Assets that are not reflectance and must never be converted as if they were. SCL is a
# class code, AOT is an optical thickness, WVP is a water vapour column in centimetres,
# and VISUAL is an already-rendered 8-bit image. Each has its own units, and the reflect-
# ance offset is meaningless for all of them.
NON_REFLECTANCE_BANDS: frozenset[Band] = frozenset({Band.SCL, Band.AOT, Band.WVP, Band.VISUAL})


def to_reflectance(
    array: Any,
    *,
    scale: float = REFLECTANCE_SCALE,
    offset: float,
) -> np.ndarray:
    """Convert stored integers to surface reflectance as ``dn * scale + offset``.

    ``offset`` is deliberately required and has no default. -0.1 is correct for Sentinel-2
    L2A under processing baseline 04.00 or later (everything acquired since January 2022)
    and wrong for older scenes, which need 0.0. A default either way would silently corrupt
    half the archive, in whichever direction it was chosen, without ever raising. Derive the
    value with :func:`reflectance_params_for_item` or
    :func:`reflectance_params_for_baseline` and pass it in.

    Args:
        array: Stored band values, any integer or float dtype.
        scale: Multiplier, ``1e-4`` for L2A reflectance.
        offset: Additive term in reflectance units, ``-0.1`` from baseline 04.00.

    Returns:
        A float32 array of surface reflectance, nominally in ``[0, 1]``.

        The arithmetic is done in float32 rather than promoting to float64, because a
        full Sentinel-2 band is 120 megapixels and the promotion would double peak
        memory. The cost is rounding of order 1e-8, which is four orders of magnitude
        below the 1e-4 quantisation of the data itself, so a stored value that should map
        to exactly 0.0 can come back as -7e-09.

    Example:
        >>> values = np.array([1000, 2000, 11000], dtype="uint16")
        >>> bool(np.allclose(to_reflectance(values, offset=BOA_OFFSET), [0.0, 0.1, 1.0]))
        True
        >>> bool(np.allclose(to_reflectance(values, offset=0.0), [0.1, 0.2, 1.1]))
        True
    """
    return np.asarray(array, dtype="float32") * np.float32(scale) + np.float32(offset)


def reflectance_params_for_baseline(baseline: str | float | None) -> tuple[float, float]:
    """Derive ``(scale, offset)`` from a Sentinel-2 processing baseline.

    Args:
        baseline: The ``s2:processing_baseline`` property, e.g. ``"05.10"`` or ``"03.01"``.
            ``None`` or an unparseable value falls back to the module defaults, which
            assume a modern product.

    Returns:
        ``(scale, offset)`` ready to pass to :func:`to_reflectance`.

    Example:
        >>> reflectance_params_for_baseline("05.10")
        (0.0001, -0.1)
        >>> reflectance_params_for_baseline("03.01")
        (0.0001, 0.0)
    """
    if baseline is None or baseline == "":
        return REFLECTANCE_SCALE, BOA_OFFSET
    try:
        value = float(baseline)
    except (TypeError, ValueError):
        return REFLECTANCE_SCALE, BOA_OFFSET
    offset = BOA_OFFSET if value >= BOA_OFFSET_BASELINE else 0.0
    return REFLECTANCE_SCALE, offset


def reflectance_params_for_item(item: Any) -> tuple[float, float]:
    """Derive ``(scale, offset)`` for a STAC Item, preferring the provider's own metadata.

    Resolution order:

    1. Explicit ``raster:bands`` scale and offset on a **reflectance** asset. Publishers
       that state their encoding are believed, including if they pre-harmonised the offset
       away. The asset is chosen by canonical band so that AOT (``scale 0.001``) and SCL
       (no scale) cannot be mistaken for reflectance.
    2. The ``s2:processing_baseline`` property, via
       :func:`reflectance_params_for_baseline`.
    3. Module defaults, assuming a modern product.

    Accepts a pystac ``Item`` or any object exposing ``assets`` and ``properties``,
    including a plain dict, so it can be exercised without a network round trip.

    Example:
        >>> reflectance_params_for_item({"properties": {"s2:processing_baseline": "03.01"}})
        (0.0001, 0.0)
    """
    explicit = _explicit_params(_field(item, "assets") or {})
    if explicit is not None:
        return explicit
    properties = _field(item, "properties") or {}
    return reflectance_params_for_baseline(_field(properties, "s2:processing_baseline"))


def scale_bands(
    bands: Mapping[str, Any],
    *,
    scale: float = REFLECTANCE_SCALE,
    offset: float = BOA_OFFSET,
    skip: Iterable[str | Band] = NON_REFLECTANCE_BANDS,
) -> dict[str, np.ndarray]:
    """Convert a whole band stack, leaving non-reflectance layers untouched.

    This is the safe way to scale a scene, because it is the one call site that cannot
    forget to exclude SCL. Skipped bands are passed through unchanged and keep their
    original dtype, so ``SCL`` stays a uint8 class code.

    Args:
        bands: Canonical band name to array, as returned by the sample and scene readers.
        scale: Reflectance multiplier.
        offset: Reflectance offset. See :func:`to_reflectance` on the default.
        skip: Bands to pass through. Defaults to :data:`NON_REFLECTANCE_BANDS`.

    Returns:
        A new dict with the same keys.

    Example:
        >>> out = scale_bands({"B08": np.array([2000], dtype="uint16"),
        ...                    "SCL": np.array([6], dtype="uint8")})
        >>> bool(np.isclose(out["B08"][0], 0.1)), int(out["SCL"][0])
        (True, 6)
    """
    excluded = set()
    for name in skip:
        try:
            excluded.add(canonical_band(name))
        except KeyError:
            continue
    return {
        name: array
        if _is_excluded(name, excluded)
        else to_reflectance(array, scale=scale, offset=offset)
        for name, array in bands.items()
    }


def _is_excluded(name: str, excluded: set[Band]) -> bool:
    try:
        return canonical_band(name) in excluded
    except KeyError:
        # An unrecognized layer is not known to be reflectance, so leave it alone.
        return True


def _explicit_params(assets: Mapping[str, Any]) -> tuple[float, float] | None:
    """Read scale and offset off a reflectance asset, or ``None`` if none publishes them."""
    for key, asset in assets.items():
        try:
            band = canonical_band(key)
        except KeyError:
            continue
        if band in NON_REFLECTANCE_BANDS:
            continue
        extra = _field(asset, "extra_fields") or asset
        entries = _field(extra, "raster:bands") or _field(extra, "bands")
        if not entries:
            continue
        first = entries[0]
        scale = _field(first, "scale")
        if scale is None:
            continue
        return float(scale), float(_field(first, "offset") or 0.0)
    return None


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an attribute.

    Lets a pystac ``Item``, a plain dict and a test stub all be handled by one code path,
    which is what keeps the derivation testable without a network round trip.
    """
    if obj is None:
        return default
    if hasattr(obj, "get") and not hasattr(obj, "extra_fields"):
        return obj.get(key, default)
    return getattr(obj, key, default)
