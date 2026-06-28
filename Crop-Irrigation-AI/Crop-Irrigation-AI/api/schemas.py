"""
api/schemas.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 request and response schemas for the FastAPI REST layer.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Generic responses ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


# ── Pipeline trigger ───────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    start_date:   Optional[date] = None
    end_date:     Optional[date] = None
    force_rerun:  bool = False


class PipelineRunResponse(BaseModel):
    task_id:  str
    status:   str
    message:  str


# ── Irrigation advisory ────────────────────────────────────────────────────────

class AdvisoryRequest(BaseModel):
    as_of_date: Optional[date] = None
    zone_ids:   Optional[list[str]] = None   # None = all zones


class ZoneAdvisoryResponse(BaseModel):
    zone_id:           str
    zone_name:         str
    priority:          str
    water_deficit_mm:  float
    cwsi_mean:         float
    soil_moisture_pct: float
    dominant_crop:     str
    area_ha:           float
    irrigate_by:       Optional[date]
    rainfall_forecast_mm: float
    advisory_text:     str
    generated_date:    date


class AdvisoryListResponse(BaseModel):
    count:     int
    as_of:     date
    advisories: list[ZoneAdvisoryResponse]


# ── Map endpoints ──────────────────────────────────────────────────────────────

class MapMetaResponse(BaseModel):
    name:      str
    path:      str
    crs:       str
    width:     int
    height:    int
    bounds:    list[float]   # [west, south, east, north]
    generated: Optional[str]


# ── Statistics ─────────────────────────────────────────────────────────────────

class CropAreaStats(BaseModel):
    crop_name:   str
    class_id:    int
    area_ha:     float
    pixel_count: int
    pct_of_total: float


class StressStats(BaseModel):
    no_stress_pct:   float
    mild_pct:        float
    moderate_pct:    float
    severe_pct:      float
    mean_cwsi:       float


class DashboardStatsResponse(BaseModel):
    command_area_ha:       float
    total_irrigated_ha:    float
    mean_water_deficit_mm: float
    critical_zones:        int
    crop_distribution:     list[CropAreaStats]
    stress_distribution:   StressStats
    as_of:                 date


# ── Weather ────────────────────────────────────────────────────────────────────

class WeatherDayResponse(BaseModel):
    date:         date
    temp_max_c:   float
    temp_min_c:   float
    precip_mm:    float
    et0_mm:       float
    humidity_pct: float
    wind_speed_ms: float
    description:  str


class WeatherForecastResponse(BaseModel):
    location:  str
    days:      list[WeatherDayResponse]
