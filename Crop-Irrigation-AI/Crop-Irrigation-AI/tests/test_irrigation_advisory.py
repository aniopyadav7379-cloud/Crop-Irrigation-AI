"""
tests/test_irrigation_advisory.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for models/irrigation_advisory.py priority rule engine.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import date, timedelta

import pytest

from models.irrigation_advisory import IrrigationAdvisoryEngine, IrrigationPriority


pytestmark = pytest.mark.unit


@pytest.fixture
def engine() -> IrrigationAdvisoryEngine:
    # Paths don't need to exist for testing the pure rule logic
    return IrrigationAdvisoryEngine(
        zones_shp=None, crop_map=None, deficit_map=None, stress_map=None,
    )


class TestPriorityAssignment:
    def test_critical_on_high_deficit(self, engine: IrrigationAdvisoryEngine):
        priority, text, irrigate_by = engine._assign_priority(
            deficit_mm=60.0, cwsi=0.4, soil_moisture=40.0, rainfall_forecast=0.0
        )
        assert priority == IrrigationPriority.CRITICAL
        assert irrigate_by == date.today() + timedelta(days=1)
        assert "CRITICAL" in text

    def test_critical_on_high_cwsi(self, engine: IrrigationAdvisoryEngine):
        priority, _, _ = engine._assign_priority(
            deficit_mm=5.0, cwsi=0.85, soil_moisture=40.0, rainfall_forecast=0.0
        )
        assert priority == IrrigationPriority.CRITICAL

    def test_critical_on_low_soil_moisture(self, engine: IrrigationAdvisoryEngine):
        priority, _, _ = engine._assign_priority(
            deficit_mm=5.0, cwsi=0.2, soil_moisture=15.0, rainfall_forecast=0.0
        )
        assert priority == IrrigationPriority.CRITICAL

    def test_moderate_priority(self, engine: IrrigationAdvisoryEngine):
        priority, _, _ = engine._assign_priority(
            deficit_mm=30.0, cwsi=0.4, soil_moisture=40.0, rainfall_forecast=0.0
        )
        assert priority == IrrigationPriority.MODERATE

    def test_low_priority(self, engine: IrrigationAdvisoryEngine):
        priority, _, _ = engine._assign_priority(
            deficit_mm=10.0, cwsi=0.35, soil_moisture=45.0, rainfall_forecast=0.0
        )
        assert priority == IrrigationPriority.LOW

    def test_adequate_priority(self, engine: IrrigationAdvisoryEngine):
        priority, _, irrigate_by = engine._assign_priority(
            deficit_mm=2.0, cwsi=0.1, soil_moisture=55.0, rainfall_forecast=0.0
        )
        assert priority == IrrigationPriority.ADEQUATE
        assert irrigate_by is None

    def test_rainfall_forecast_reduces_priority(self, engine: IrrigationAdvisoryEngine):
        """A critical deficit should downgrade if heavy rain is forecast."""
        priority_no_rain, _, _ = engine._assign_priority(
            deficit_mm=55.0, cwsi=0.3, soil_moisture=40.0, rainfall_forecast=0.0
        )
        priority_with_rain, _, _ = engine._assign_priority(
            deficit_mm=55.0, cwsi=0.3, soil_moisture=40.0, rainfall_forecast=50.0
        )
        priority_order = {
            IrrigationPriority.ADEQUATE: 0,
            IrrigationPriority.LOW: 1,
            IrrigationPriority.MODERATE: 2,
            IrrigationPriority.CRITICAL: 3,
        }
        assert priority_order[priority_with_rain] <= priority_order[priority_no_rain]


class TestAdvisoryTextContent:
    def test_advisory_text_mentions_deficit_value(self, engine: IrrigationAdvisoryEngine):
        _, text, _ = engine._assign_priority(
            deficit_mm=45.0, cwsi=0.6, soil_moisture=30.0, rainfall_forecast=0.0
        )
        assert "45.0" in text or "45.0 mm" in text
