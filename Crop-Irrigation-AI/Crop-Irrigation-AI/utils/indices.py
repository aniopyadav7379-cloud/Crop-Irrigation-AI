"""
utils/indices.py
─────────────────────────────────────────────────────────────────────────────
Pure-NumPy spectral index library.
All functions accept and return float32 NumPy arrays.
Division by zero is handled via np.errstate and produces np.nan.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np


_EPS = 1e-10   # small epsilon prevents true divide-by-zero


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > _EPS, a / b, np.nan)
    return result.astype(np.float32)


# ── Vegetation ─────────────────────────────────────────────────────────────────

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Vegetation Index.
    NDVI = (NIR - RED) / (NIR + RED)   ∈ [-1, 1]
    Sentinel-2: NIR=B08, RED=B04
    """
    return _safe_div(nir - red, nir + red)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
        g: float = 2.5, c1: float = 6.0, c2: float = 7.5,
        l: float = 1.0) -> np.ndarray:
    """
    Enhanced Vegetation Index — reduces canopy background noise.
    EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)
    Sentinel-2: NIR=B08, RED=B04, BLUE=B02
    """
    denom = nir + c1 * red - c2 * blue + l
    return _safe_div(g * (nir - red), denom)


def savi(nir: np.ndarray, red: np.ndarray, l: float = 0.5) -> np.ndarray:
    """
    Soil-Adjusted Vegetation Index.
    SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    """
    return _safe_div((nir - red) * (1 + l), nir + red + l)


def lai(ndvi_arr: np.ndarray) -> np.ndarray:
    """
    Empirical Leaf Area Index from NDVI (Baret & Guyot 1991).
    LAI = -log((0.69 - NDVI) / 0.59) / 0.91
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        inner = (0.69 - ndvi_arr) / 0.59
        result = np.where(inner > 0, -np.log(inner) / 0.91, np.nan)
    return result.astype(np.float32)


# ── Water / Moisture ───────────────────────────────────────────────────────────

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Water Index (Gao 1996) — open-water bodies.
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    Sentinel-2: GREEN=B03, NIR=B08
    """
    return _safe_div(green - nir, green + nir)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Moisture Index — vegetation water content.
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    Sentinel-2: NIR=B8A, SWIR1=B11
    """
    return _safe_div(nir - swir1, nir + swir1)


def lswi(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Land Surface Water Index.
    LSWI = (NIR - SWIR2) / (NIR + SWIR2)
    Sentinel-2: NIR=B8A, SWIR2=B12
    """
    return _safe_div(nir - swir2, nir + swir2)


def msi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Moisture Stress Index — higher value = more stress.
    MSI = SWIR1 / NIR
    Sentinel-2: SWIR1=B11, NIR=B8A
    """
    return _safe_div(swir1, nir)


# ── Crop Water Stress Index ────────────────────────────────────────────────────

def cwsi(
    lst: np.ndarray,
    t_wet: float,
    t_dry: float,
) -> np.ndarray:
    """
    Crop Water Stress Index (Jackson et al. 1981).
    CWSI = (LST - T_wet) / (T_dry - T_wet)   ∈ [0, 1]

    Parameters
    ----------
    lst   : Land Surface Temperature array (°C)
    t_wet : lower baseline — temperature of a well-watered canopy (°C)
    t_dry : upper baseline — temperature of a fully stressed canopy (°C)
    """
    denom = t_dry - t_wet
    result = _safe_div(lst - t_wet, np.full_like(lst, denom))
    return np.clip(result, 0.0, 1.0)


# ── Soil / Bare-earth ──────────────────────────────────────────────────────────

def bsi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
) -> np.ndarray:
    """
    Bare Soil Index.
    BSI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    """
    num = (swir1 + red) - (nir + blue)
    denom = (swir1 + red) + (nir + blue)
    return _safe_div(num, denom)


def ndsi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Snow Index — masks snow/ice pixels.
    NDSI = (GREEN - SWIR1) / (GREEN + SWIR1)
    """
    return _safe_div(green - swir1, green + swir1)


# ── SAR-derived soil moisture proxy ───────────────────────────────────────────

def sar_soil_moisture_proxy(
    vv: np.ndarray,
    vh: np.ndarray,
    vv_dry: float = -15.0,
    vv_wet: float = -5.0,
) -> np.ndarray:
    """
    Empirical soil-moisture proxy from Sentinel-1 backscatter.
    Normalises VV backscatter (dB) between dry and wet reference values.
    Returns volumetric moisture estimate ∈ [0, 1].

    Parameters
    ----------
    vv      : VV backscatter in dB
    vh      : VH backscatter in dB (used for cross-pol ratio)
    vv_dry  : reference VV backscatter for dry soil (dB)
    vv_wet  : reference VV backscatter for saturated soil (dB)
    """
    sm = _safe_div(vv - vv_dry, np.full_like(vv, vv_wet - vv_dry))
    return np.clip(sm, 0.0, 1.0)


# ── Evapotranspiration helpers ─────────────────────────────────────────────────

def et_fraction(et_actual: np.ndarray, et_reference: np.ndarray) -> np.ndarray:
    """
    Evaporative Fraction = ETa / ET0.
    Values < 0.5 indicate water stress.
    """
    return np.clip(_safe_div(et_actual, et_reference), 0.0, 1.2)


# ── Composite feature vector ───────────────────────────────────────────────────

def compute_all_indices(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    nir_broad: np.ndarray,  # B8A
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convenience function — computes the full index stack used for
    crop classification and stress detection.

    Returns
    -------
    dict mapping index name → 2-D float32 array
    """
    _ndvi = ndvi(nir, red)
    return {
        "NDVI":  _ndvi,
        "EVI":   evi(nir, red, blue),
        "SAVI":  savi(nir, red),
        "LAI":   lai(_ndvi),
        "NDWI":  ndwi(green, nir),
        "NDMI":  ndmi(nir_broad, swir1),
        "LSWI":  lswi(nir_broad, swir2),
        "MSI":   msi(swir1, nir_broad),
        "BSI":   bsi(blue, red, nir, swir1),
    }
