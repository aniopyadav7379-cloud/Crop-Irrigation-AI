"""
api/routes.py
─────────────────────────────────────────────────────────────────────────────
FastAPI routers for the Crop-Irrigation-AI REST API.

Prefix  /api/v1
────────────────
GET   /health
POST  /pipeline/run
GET   /advisory
GET   /advisory/{zone_id}
GET   /stats/dashboard
GET   /stats/crops
GET   /stats/stress
GET   /maps/meta
GET   /weather/forecast
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import rasterio
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from config import settings
from utils import logger
from models import (
    CropClassifier,
    MoistureStressDetector,
    WaterDeficitModel,
    IrrigationAdvisoryEngine,
    IrrigationPriority,
    CROP_LABELS,
)
from api.schemas import (
    HealthResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    AdvisoryRequest,
    AdvisoryListResponse,
    ZoneAdvisoryResponse,
    DashboardStatsResponse,
    CropAreaStats,
    StressStats,
    MapMetaResponse,
    WeatherForecastResponse,
    WeatherDayResponse,
)

router = APIRouter(prefix=settings.api.api_prefix)


# ── Dependency: lazy-loaded model singletons ──────────────────────────────────

_crop_clf:      Optional[CropClassifier]        = None
_stress_det:    Optional[MoistureStressDetector] = None
_deficit_model: Optional[WaterDeficitModel]     = None


def _get_crop_clf() -> CropClassifier:
    global _crop_clf
    if _crop_clf is None:
        _crop_clf = CropClassifier()
    return _crop_clf


def _get_stress_det() -> MoistureStressDetector:
    global _stress_det
    if _stress_det is None:
        _stress_det = MoistureStressDetector()
    return _stress_det


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness probe for load-balancers and uptime monitors."""
    return HealthResponse(
        version=settings.project_version,
        models_loaded=settings.saved_models_dir.exists(),
    )


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _run_full_pipeline(start: date, end: date, force: bool) -> None:
    """Background task: download → preprocess → infer → advisory."""
    from main import run_pipeline
    try:
        run_pipeline(start_date=start, end_date=end, force_rerun=force)
    except Exception as exc:
        logger.error(f"Pipeline background task failed: {exc}")


@router.post("/pipeline/run", response_model=PipelineRunResponse, tags=["Pipeline"])
def trigger_pipeline(req: PipelineRunRequest, bg: BackgroundTasks):
    """
    Trigger the full preprocessing + inference pipeline asynchronously.
    Returns a task_id for status tracking (extend with Celery for production).
    """
    task_id = str(uuid.uuid4())
    bg.add_task(
        _run_full_pipeline,
        req.start_date or date.today(),
        req.end_date   or date.today(),
        req.force_rerun,
    )
    return PipelineRunResponse(
        task_id=task_id,
        status="queued",
        message="Pipeline started in background. Check logs for progress.",
    )


# ── Advisory ───────────────────────────────────────────────────────────────────

def _load_advisories(zone_ids: Optional[list[str]] = None):
    engine = IrrigationAdvisoryEngine()
    advisories = engine.generate_advisories()
    if zone_ids:
        advisories = [a for a in advisories if a.zone_id in zone_ids]
    return advisories


@router.get("/advisory", response_model=AdvisoryListResponse, tags=["Advisory"])
def get_all_advisories(
    zone_ids: Optional[str] = Query(None, description="Comma-separated zone IDs"),
):
    """Return irrigation advisories for all (or selected) command-area zones."""
    ids = zone_ids.split(",") if zone_ids else None
    advisories = _load_advisories(ids)
    return AdvisoryListResponse(
        count=len(advisories),
        as_of=date.today(),
        advisories=[ZoneAdvisoryResponse(**a.model_dump()) for a in advisories],
    )


@router.get("/advisory/{zone_id}", response_model=ZoneAdvisoryResponse, tags=["Advisory"])
def get_zone_advisory(zone_id: str):
    """Return the irrigation advisory for a single zone."""
    advisories = _load_advisories([zone_id])
    if not advisories:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    return ZoneAdvisoryResponse(**advisories[0].model_dump())


# ── Statistics ─────────────────────────────────────────────────────────────────

@router.get("/stats/dashboard", response_model=DashboardStatsResponse, tags=["Statistics"])
def get_dashboard_stats():
    """Aggregate statistics for the main dashboard."""
    crop_map_path = settings.crop_map_path
    deficit_map_path = settings.irrigation_map_path
    stress_map_path = settings.stress_map_path

    if not crop_map_path.exists():
        raise HTTPException(status_code=503, detail="Maps not yet generated. Run pipeline first.")

    # Crop distribution
    with rasterio.open(crop_map_path) as src:
        crop_arr = src.read(1).astype(np.float32)
        transform = src.transform

    pixel_area_ha = abs(transform.a * transform.e) / 10_000
    total_valid = int(np.sum(crop_arr != 255))

    crop_stats: list[CropAreaStats] = []
    for class_id, label in CROP_LABELS.items():
        count = int(np.sum(crop_arr == class_id))
        crop_stats.append(CropAreaStats(
            crop_name=label,
            class_id=class_id,
            area_ha=round(count * pixel_area_ha, 1),
            pixel_count=count,
            pct_of_total=round(100.0 * count / max(total_valid, 1), 2),
        ))

    # Water deficit
    mean_deficit = 0.0
    if deficit_map_path.exists():
        with rasterio.open(deficit_map_path) as s:
            d = s.read(1).astype(np.float32)
            d[d == s.nodata] = np.nan
            mean_deficit = float(np.nanmean(d))

    # Stress distribution
    stress_dist = StressStats(
        no_stress_pct=0, mild_pct=0, moderate_pct=0, severe_pct=0, mean_cwsi=0.5
    )
    if stress_map_path.exists():
        with rasterio.open(stress_map_path) as s:
            st = s.read(1).astype(np.float32)
            valid_st = st[st != 255]
            n = max(len(valid_st), 1)
            stress_dist = StressStats(
                no_stress_pct=round(100 * np.sum(valid_st == 0) / n, 1),
                mild_pct=round(100 * np.sum(valid_st == 1) / n, 1),
                moderate_pct=round(100 * np.sum(valid_st == 2) / n, 1),
                severe_pct=round(100 * np.sum(valid_st == 3) / n, 1),
                mean_cwsi=round(float(np.mean(valid_st) / 3), 3),
            )

    # Advisory counts
    advisories = _load_advisories()
    critical = sum(1 for a in advisories if a.priority == IrrigationPriority.CRITICAL)

    command_area = round(total_valid * pixel_area_ha, 1)

    return DashboardStatsResponse(
        command_area_ha=command_area,
        total_irrigated_ha=round(command_area * 0.65, 1),
        mean_water_deficit_mm=round(mean_deficit, 2),
        critical_zones=critical,
        crop_distribution=crop_stats,
        stress_distribution=stress_dist,
        as_of=date.today(),
    )


# ── Map metadata ───────────────────────────────────────────────────────────────

@router.get("/maps/meta", response_model=list[MapMetaResponse], tags=["Maps"])
def get_map_metadata():
    """Return metadata for available output raster maps."""
    map_configs = [
        ("crop_map",     settings.crop_map_path),
        ("stress_map",   settings.stress_map_path),
        ("deficit_map",  settings.irrigation_map_path),
    ]
    results: list[MapMetaResponse] = []
    for name, path in map_configs:
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            b = src.bounds
            results.append(MapMetaResponse(
                name=name,
                path=str(path),
                crs=str(src.crs),
                width=src.width,
                height=src.height,
                bounds=[b.left, b.bottom, b.right, b.top],
                generated=src.tags().get("generated", None),
            ))
    return results


# ── Weather (stub — replace with live OWM / ERA5 call) ────────────────────────

@router.get("/weather/forecast", response_model=WeatherForecastResponse, tags=["Weather"])
def get_weather_forecast(location: str = "Command Area"):
    """7-day weather forecast (stub data — wire to OWM in production)."""
    from datetime import timedelta
    days = []
    for i in range(7):
        d = date.today() + timedelta(days=i)
        days.append(WeatherDayResponse(
            date=d,
            temp_max_c=36.0 - i * 0.5,
            temp_min_c=26.0,
            precip_mm=14.0 if i == 2 else (8.0 if i == 3 else 0.0),
            et0_mm=5.2,
            humidity_pct=72.0,
            wind_speed_ms=3.1,
            description="Partly cloudy" if i % 2 == 0 else "Light rain expected",
        ))
    return WeatherForecastResponse(location=location, days=days)
