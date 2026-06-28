"""
config/settings.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Crop-Irrigation-AI system.
All environment overrides go in a .env file at the project root.
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models" / "saved"


class SatelliteSettings(BaseSettings):
    """Satellite data acquisition parameters."""
    model_config = SettingsConfigDict(extra="ignore")

    sentinel_user: str = Field("", validation_alias="SENTINELSAT_USER")
    sentinel_pass: str = Field("", validation_alias="SENTINELSAT_PASS")
    sentinel_api_url: str = "https://apihub.copernicus.eu/apihub"

    # Sentinel-2 bands used for index computation
    s2_optical_bands: list[str] = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
    # Sentinel-1 polarisations
    s1_polarisations: list[str] = ["VV", "VH"]

    spatial_resolution_m: int = 10       # target resampling resolution
    cloud_cover_threshold: float = 20.0  # max acceptable cloud cover %
    revisit_days: int = 5                # days between satellite passes


class WeatherSettings(BaseSettings):
    """ERA5 / OpenWeatherMap parameters."""
    era5_dataset: str = "reanalysis-era5-land"
    era5_variables: list[str] = [
        "2m_temperature",
        "total_precipitation",
        "potential_evaporation",
        "soil_water_layer_1",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    model_config = SettingsConfigDict(extra="ignore")

    owm_api_key: str = Field("", validation_alias="OWM_API_KEY")
    forecast_days: int = 7


class ModelSettings(BaseSettings):
    """ML model hyper-parameter defaults."""
    # Random-Forest crop classifier
    rf_n_estimators: int = 300
    rf_max_depth: int = 20
    rf_min_samples_leaf: int = 4
    rf_n_jobs: int = -1
    rf_random_state: int = 42

    # CNN moisture-stress detector
    cnn_input_channels: int = 6          # number of spectral bands as input
    cnn_num_classes: int = 4             # {no_stress, mild, moderate, severe}
    cnn_batch_size: int = 32
    cnn_epochs: int = 50
    cnn_learning_rate: float = 1e-3
    cnn_image_patch_size: int = 64       # spatial patch size in pixels

    # Water-deficit regression
    wdr_model_type: str = "xgboost"      # xgboost | lightgbm | catboost
    wdr_n_estimators: int = 500
    wdr_max_depth: int = 8
    wdr_learning_rate: float = 0.05

    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.1


class IrrigationSettings(BaseSettings):
    """Advisory engine thresholds (all in mm unless noted)."""
    # Crop Water Stress Index thresholds
    cwsi_critical: float = 0.80          # irrigate immediately
    cwsi_moderate: float = 0.55          # irrigate within 3 days
    cwsi_low: float = 0.30               # monitor

    # Soil moisture thresholds (% volumetric)
    soil_moisture_critical_pct: float = 20.0
    soil_moisture_moderate_pct: float = 35.0

    # Water-deficit thresholds (mm)
    water_deficit_critical_mm: float = 50.0
    water_deficit_moderate_mm: float = 25.0

    # Rainfall to skip irrigation
    rainfall_skip_threshold_mm: float = 10.0

    # Advisory lead time
    advisory_lead_days: int = 3
    sms_alert_enabled: bool = False       # set True in production


class DatabaseSettings(BaseSettings):
    """PostgreSQL + Redis connection strings."""
    model_config = SettingsConfigDict(extra="ignore")

    postgres_dsn: str = Field(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/crop_irrigation",
        validation_alias="DATABASE_URL",
    )
    redis_url: str = Field("redis://localhost:6379/0", validation_alias="REDIS_URL")
    pool_size: int = 10
    max_overflow: int = 20


class APISettings(BaseSettings):
    """FastAPI server settings."""
    model_config = SettingsConfigDict(extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = Field(False, validation_alias="DEBUG")
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]
    api_prefix: str = "/api/v1"
    jwt_secret: str = Field("change-me-in-production", validation_alias="JWT_SECRET")
    jwt_expire_minutes: int = 60 * 24    # 24 hours


class Settings(BaseSettings):
    """Root settings — aggregates all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-configs
    satellite: SatelliteSettings = SatelliteSettings()
    weather: WeatherSettings = WeatherSettings()
    model: ModelSettings = ModelSettings()
    irrigation: IrrigationSettings = IrrigationSettings()
    db: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()

    # Project-level
    project_name: str = "Crop-Irrigation-AI"
    project_version: str = "1.0.0"
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    # Derived paths (not overridable via env)
    @property
    def satellite_dir(self) -> Path:
        return DATA_DIR / "satellite_data"

    @property
    def weather_dir(self) -> Path:
        return DATA_DIR / "weather_data"

    @property
    def ground_truth_csv(self) -> Path:
        return DATA_DIR / "ground_truth.csv"

    @property
    def boundary_shp(self) -> Path:
        return DATA_DIR / "command_area_boundary.shp"

    @property
    def saved_models_dir(self) -> Path:
        return MODELS_DIR

    @property
    def crop_map_path(self) -> Path:
        return OUTPUTS_DIR / "crop_map.tif"

    @property
    def stress_map_path(self) -> Path:
        return OUTPUTS_DIR / "stress_map.tif"

    @property
    def irrigation_map_path(self) -> Path:
        return OUTPUTS_DIR / "irrigation_map.tif"

    @property
    def reports_dir(self) -> Path:
        return OUTPUTS_DIR / "reports"


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = Settings()
