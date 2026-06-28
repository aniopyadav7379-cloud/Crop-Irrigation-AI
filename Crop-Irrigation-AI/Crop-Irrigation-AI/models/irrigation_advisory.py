"""
models/irrigation_advisory.py
─────────────────────────────────────────────────────────────────────────────
Rule-based + ML-augmented irrigation advisory engine.

For each command-area zone the engine:
  1. Aggregates pixel-level water deficit, CWSI, and soil-moisture maps
  2. Applies FAO-56 / CWSI thresholds from settings
  3. Adjusts for upcoming rainfall forecast
  4. Assigns a priority (critical / moderate / low / adequate)
  5. Emits a structured IrrigationAdvisory pydantic object per zone
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pydantic import BaseModel, Field
from shapely.geometry import mapping

from config import settings
from utils import logger
from models.water_deficit import WaterDeficitModel, CROP_KC
from models.crop_classification import CROP_LABELS


# ── Enums & DTOs ───────────────────────────────────────────────────────────────

class IrrigationPriority(str, Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    LOW      = "low"
    ADEQUATE = "adequate"


class IrrigationAdvisory(BaseModel):
    zone_id:           str
    zone_name:         str
    priority:          IrrigationPriority
    water_deficit_mm:  float = Field(..., description="Mean water deficit across zone (mm)")
    cwsi_mean:         float = Field(..., description="Mean CWSI across zone [0–1]")
    soil_moisture_pct: float = Field(..., description="Mean SAR soil moisture (%)")
    dominant_crop:     str
    area_ha:           float
    irrigate_by:       Optional[date]
    rainfall_forecast_mm: float
    advisory_text:     str
    generated_date:    date = Field(default_factory=date.today)


class ZoneSummary(BaseModel):
    zone_id:   str
    zone_name: str
    area_ha:   float
    dominant_crop: str
    pixel_count:   int


# ── Advisory Engine ────────────────────────────────────────────────────────────

class IrrigationAdvisoryEngine:
    """
    Consumes classified maps + stress indices and produces zone-level advisories.
    """

    def __init__(
        self,
        zones_shp:       Path | None = None,
        crop_map:        Path | None = None,
        deficit_map:     Path | None = None,
        stress_map:      Path | None = None,
        soil_moisture_map: Path | None = None,
    ):
        self.zones_shp          = zones_shp or settings.boundary_shp
        self.crop_map           = crop_map  or settings.crop_map_path
        self.deficit_map        = deficit_map or settings.irrigation_map_path
        self.stress_map         = stress_map  or settings.stress_map_path
        self.soil_moisture_map  = soil_moisture_map  # optional SAR-derived SM

    # ── Map readers ────────────────────────────────────────────────────────────

    def _zone_stats(
        self,
        raster_path: Path,
        zone_geom,
        nodata: float = -9999.0,
        band: int = 1,
    ) -> dict[str, float]:
        """Extract mean / std / coverage for a raster within a zone polygon."""
        geoms = [mapping(zone_geom)]
        try:
            with rasterio.open(raster_path) as src:
                clipped, _ = rio_mask(src, geoms, crop=True, nodata=nodata)
                data = clipped[band - 1].astype(np.float32)
        except Exception:
            return {"mean": np.nan, "std": np.nan, "coverage_pct": 0.0}

        valid = data[data != nodata]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            return {"mean": np.nan, "std": np.nan, "coverage_pct": 0.0}

        return {
            "mean":         float(np.mean(valid)),
            "std":          float(np.std(valid)),
            "coverage_pct": 100.0 * len(valid) / data.size,
        }

    def _dominant_crop(self, zone_geom) -> tuple[str, int]:
        """Return (dominant crop label, pixel count) for a zone."""
        geoms = [mapping(zone_geom)]
        try:
            with rasterio.open(self.crop_map) as src:
                clipped, _ = rio_mask(src, geoms, crop=True, nodata=255)
                data = clipped[0].astype(np.uint8)
        except Exception:
            return "Unknown", 0

        valid = data[data != 255]
        if len(valid) == 0:
            return "Unknown", 0

        values, counts = np.unique(valid, return_counts=True)
        dominant_class = int(values[np.argmax(counts)])
        return CROP_LABELS.get(dominant_class, "Other"), int(np.max(counts))

    # ── Priority rule engine ───────────────────────────────────────────────────

    def _assign_priority(
        self,
        deficit_mm:   float,
        cwsi:         float,
        soil_moisture: float,
        rainfall_forecast: float,
    ) -> tuple[IrrigationPriority, str, Optional[date]]:
        """
        Apply threshold rules to assign irrigation priority.

        Returns
        -------
        (priority, advisory_text, irrigate_by date)
        """
        cfg = settings.irrigation

        # Effective deficit after rainfall
        effective_deficit = max(0.0, deficit_mm - rainfall_forecast)

        if (
            effective_deficit >= cfg.water_deficit_critical_mm
            or cwsi            >= cfg.cwsi_critical
            or soil_moisture   <= cfg.soil_moisture_critical_pct
        ):
            return (
                IrrigationPriority.CRITICAL,
                (
                    f"⚠ CRITICAL: Water deficit of {effective_deficit:.1f} mm detected. "
                    f"CWSI={cwsi:.2f} indicates severe stress. "
                    "Irrigate within the next 24 hours to prevent irreversible yield loss."
                ),
                date.today() + timedelta(days=1),
            )

        if (
            effective_deficit >= cfg.water_deficit_moderate_mm
            or cwsi            >= cfg.cwsi_moderate
            or soil_moisture   <= cfg.soil_moisture_moderate_pct
        ):
            return (
                IrrigationPriority.MODERATE,
                (
                    f"Water deficit of {effective_deficit:.1f} mm. "
                    f"CWSI={cwsi:.2f} — moderate stress building. "
                    f"Plan irrigation within {cfg.advisory_lead_days} days."
                ),
                date.today() + timedelta(days=cfg.advisory_lead_days),
            )

        if cwsi >= cfg.cwsi_low:
            return (
                IrrigationPriority.LOW,
                (
                    f"Mild water deficit ({effective_deficit:.1f} mm). "
                    "Monitor daily — irrigate if no significant rainfall within 5 days."
                ),
                date.today() + timedelta(days=5),
            )

        return (
            IrrigationPriority.ADEQUATE,
            (
                "Soil moisture is adequate. "
                "No irrigation required at this time. Continue monitoring."
            ),
            None,
        )

    # ── Main advisory generation ───────────────────────────────────────────────

    def generate_advisories(
        self,
        weather_forecast: pd.DataFrame | None = None,
        zone_id_col: str = "zone_id",
        zone_name_col: str = "zone_name",
        area_ha_col: str | None = "area_ha",
    ) -> list[IrrigationAdvisory]:
        """
        Generate one IrrigationAdvisory per zone in the command-area shapefile.

        Parameters
        ----------
        weather_forecast : DataFrame with columns [date, precip_mm, et0_mm, …]
                           and DatetimeIndex or 'date' column.
                           If None, rainfall_forecast_mm = 0.
        zone_id_col      : column for zone identifier
        zone_name_col    : column for zone display name
        area_ha_col      : column for zone area in hectares (optional)

        Returns
        -------
        list of IrrigationAdvisory — one per zone
        """
        zones = gpd.read_file(self.zones_shp)

        # Reproject zones to map CRS
        with rasterio.open(self.deficit_map) as src:
            map_crs = src.crs
        if zones.crs != map_crs:
            zones = zones.to_crs(map_crs)

        # Rainfall forecast for the next N days
        forecast_rain = 0.0
        if weather_forecast is not None:
            future = weather_forecast[
                weather_forecast.index >= pd.Timestamp.today()
            ]
            forecast_rain = float(
                future["precip_mm"].head(settings.irrigation.advisory_lead_days).sum()
            )

        advisories: list[IrrigationAdvisory] = []

        for _, row in zones.iterrows():
            geom = row.geometry
            zid  = str(row.get(zone_id_col, "Z_unknown"))
            zname = str(row.get(zone_name_col, zid))
            area  = float(row.get(area_ha_col, 0.0)) if area_ha_col else 0.0

            # Aggregate raster statistics within zone
            deficit_stats = self._zone_stats(self.deficit_map, geom)
            deficit_mm    = deficit_stats["mean"] if not np.isnan(deficit_stats["mean"]) else 0.0

            stress_stats  = self._zone_stats(self.stress_map, geom)
            cwsi_mean     = stress_stats["mean"] if not np.isnan(stress_stats["mean"]) else 0.5

            sm_mean = 50.0  # default if no SAR SM map
            if self.soil_moisture_map and Path(self.soil_moisture_map).exists():
                sm_stats = self._zone_stats(self.soil_moisture_map, geom)
                if not np.isnan(sm_stats["mean"]):
                    sm_mean = sm_stats["mean"] * 100.0   # convert 0–1 → %

            dominant_crop, pixel_count = self._dominant_crop(geom)

            priority, text, irrigate_by = self._assign_priority(
                deficit_mm, cwsi_mean, sm_mean, forecast_rain
            )

            advisory = IrrigationAdvisory(
                zone_id=zid,
                zone_name=zname,
                priority=priority,
                water_deficit_mm=round(deficit_mm, 2),
                cwsi_mean=round(cwsi_mean, 3),
                soil_moisture_pct=round(sm_mean, 1),
                dominant_crop=dominant_crop,
                area_ha=round(area, 1),
                irrigate_by=irrigate_by,
                rainfall_forecast_mm=round(forecast_rain, 1),
                advisory_text=text,
            )
            advisories.append(advisory)
            logger.debug(f"Zone {zid}: {priority.value.upper()}  deficit={deficit_mm:.1f}mm")

        n_critical = sum(1 for a in advisories if a.priority == IrrigationPriority.CRITICAL)
        logger.info(
            f"Generated {len(advisories)} advisories  |  "
            f"{n_critical} critical zones"
        )
        return advisories

    # ── Export helpers ─────────────────────────────────────────────────────────

    def to_dataframe(self, advisories: list[IrrigationAdvisory]) -> pd.DataFrame:
        return pd.DataFrame([a.model_dump() for a in advisories])

    def to_geojson(
        self,
        advisories: list[IrrigationAdvisory],
        out_path: Path,
        zone_id_col: str = "zone_id",
    ) -> Path:
        """Merge advisories back onto the zone shapefile and export as GeoJSON."""
        df = self.to_dataframe(advisories)
        zones = gpd.read_file(self.zones_shp)

        merged = zones.merge(df, left_on=zone_id_col, right_on="zone_id", how="left")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_file(out_path, driver="GeoJSON")
        logger.info(f"Advisory GeoJSON written → {out_path}")
        return out_path

    def to_json(self, advisories: list[IrrigationAdvisory], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [a.model_dump(mode="json") for a in advisories],
                f, indent=2, default=str,
            )
        logger.info(f"Advisory JSON written → {out_path}")
        return out_path

    def to_csv(self, advisories: list[IrrigationAdvisory], out_path: Path) -> Path:
        df = self.to_dataframe(advisories)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Advisory CSV written → {out_path}")
        return out_path
