"""
tests/test_indices.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for utils/indices.py spectral index calculations.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pytest

from utils.indices import (
    ndvi, evi, savi, ndwi, ndmi, msi, cwsi,
    sar_soil_moisture_proxy, compute_all_indices, _safe_div,
)


pytestmark = pytest.mark.unit


class TestSafeDiv:
    def test_normal_division(self):
        a = np.array([4.0, 9.0])
        b = np.array([2.0, 3.0])
        result = _safe_div(a, b)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_division_by_zero_returns_nan(self):
        a = np.array([1.0])
        b = np.array([0.0])
        result = _safe_div(a, b)
        assert np.isnan(result[0])


class TestNDVI:
    def test_ndvi_range(self):
        nir = np.array([0.5, 0.8, 0.3])
        red = np.array([0.1, 0.2, 0.3])
        result = ndvi(nir, red)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_ndvi_known_value(self):
        nir = np.array([0.6])
        red = np.array([0.2])
        result = ndvi(nir, red)
        expected = (0.6 - 0.2) / (0.6 + 0.2)
        np.testing.assert_allclose(result, [expected], rtol=1e-5)

    def test_ndvi_identical_bands_is_zero(self):
        x = np.array([0.4, 0.4])
        result = ndvi(x, x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-6)


class TestEVI:
    def test_evi_finite_for_valid_inputs(self):
        nir  = np.array([0.5])
        red  = np.array([0.2])
        blue = np.array([0.1])
        result = evi(nir, red, blue)
        assert np.isfinite(result[0])


class TestSAVI:
    def test_savi_with_l_zero_equals_ndvi(self):
        nir = np.array([0.6])
        red = np.array([0.3])
        result = savi(nir, red, l=0.0)
        expected = ndvi(nir, red)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestNDWI:
    def test_ndwi_water_pixel_positive(self):
        # Open water: green reflectance > NIR reflectance
        green = np.array([0.3])
        nir   = np.array([0.05])
        result = ndwi(green, nir)
        assert result[0] > 0


class TestNDMI:
    def test_ndmi_range(self):
        nir   = np.array([0.5, 0.7])
        swir1 = np.array([0.2, 0.6])
        result = ndmi(nir, swir1)
        assert np.all(result >= -1) and np.all(result <= 1)


class TestMSI:
    def test_msi_higher_swir_means_more_stress(self):
        nir = np.array([0.5])
        swir_low  = np.array([0.1])
        swir_high = np.array([0.4])
        low_stress  = msi(swir_low, nir)
        high_stress = msi(swir_high, nir)
        assert high_stress[0] > low_stress[0]


class TestCWSI:
    def test_cwsi_well_watered_canopy_near_zero(self):
        lst = np.array([25.0])
        result = cwsi(lst, t_wet=25.0, t_dry=40.0)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_cwsi_stressed_canopy_near_one(self):
        lst = np.array([40.0])
        result = cwsi(lst, t_wet=25.0, t_dry=40.0)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_cwsi_clipped_to_unit_interval(self):
        lst = np.array([100.0, -50.0])
        result = cwsi(lst, t_wet=25.0, t_dry=40.0)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestSARSoilMoisture:
    def test_dry_soil_near_zero(self):
        vv = np.array([-15.0])
        vh = np.array([-20.0])
        result = sar_soil_moisture_proxy(vv, vh, vv_dry=-15.0, vv_wet=-5.0)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_wet_soil_near_one(self):
        vv = np.array([-5.0])
        vh = np.array([-10.0])
        result = sar_soil_moisture_proxy(vv, vh, vv_dry=-15.0, vv_wet=-5.0)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_clipped_to_unit_interval(self):
        vv = np.array([10.0, -40.0])
        vh = np.array([-10.0, -10.0])
        result = sar_soil_moisture_proxy(vv, vh)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestComputeAllIndices:
    def test_returns_all_expected_keys(self):
        shape = (4, 4)
        bands = {k: np.random.uniform(0.05, 0.5, shape).astype(np.float32)
                 for k in ["blue", "green", "red", "nir", "nir_broad", "swir1", "swir2"]}
        result = compute_all_indices(**bands)

        expected_keys = {"NDVI", "EVI", "SAVI", "LAI", "NDWI", "NDMI", "LSWI", "MSI", "BSI"}
        assert set(result.keys()) == expected_keys
        for arr in result.values():
            assert arr.shape == shape
            assert arr.dtype == np.float32
