# 🛰️ Crop-Irrigation-AI

**AI-powered satellite irrigation advisory system** — fuses Sentinel-2 optical imagery, Sentinel-1 SAR, and weather reanalysis data to deliver zone-level crop classification, moisture-stress detection, and irrigation recommendations across a command area.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 What it does

| Capability | Method |
|---|---|
| **Crop classification** | Random-Forest ensemble on optical + SAR + spectral-index features (Optuna-tuned, SHAP-explainable) |
| **Moisture-stress detection** | Lightweight CNN over 64×64 spectral patches → 4-class stress map |
| **Water-deficit estimation** | XGBoost/LightGBM regression fusing FAO-56 ETc, SAR soil moisture, and weather |
| **Irrigation advisory** | Rule engine combining CWSI, deficit, soil moisture, and rainfall forecast → per-zone priority + lead time |
| **Dashboard** | Streamlit app with interactive Folium maps, Plotly charts, advisory tables |
| **REST API** | FastAPI service exposing advisories, stats, and map metadata as JSON |

---

## 🏗️ Architecture

```
Sentinel-2 (optical) ──┐
                        ├──► Preprocessing ──► Feature Extraction ──► ML Models ──► Advisory Engine ──► Dashboard / API
Sentinel-1 (SAR)    ────┤                                                │
                        │                                                ├─ Crop Classifier (RF)
ERA5 weather ───────────┘                                                ├─ Moisture Stress (CNN)
                                                                          └─ Water Deficit (XGBoost)
```

---

## 📂 Project Structure

```
Crop-Irrigation-AI/
│
├── data/
│   ├── satellite_data/              # Raw + processed Sentinel-1/2 scenes
│   ├── weather_data/                # ERA5 / OWM daily weather CSVs
│   ├── ground_truth.csv             # Field survey points (crop labels)
│   └── command_area_boundary.shp    # Zone polygons (6-zone sample command area)
│
├── preprocessing/
│   ├── optical_preprocessing.py     # Sentinel-2: cloud mask, reflectance, reproject, clip, index stack
│   ├── sar_preprocessing.py         # Sentinel-1: calibration, Lee speckle filter, dB, soil-moisture proxy
│   └── feature_extraction.py        # Pixel-wise feature matrix builder for ML models
│
├── models/
│   ├── crop_classification.py       # Random-Forest + Optuna tuning + SHAP explainability
│   ├── moisture_stress.py           # PyTorch CNN, patch sampler, full-scene inference
│   ├── water_deficit.py             # XGBoost/LightGBM regression, FAO-56 ETc helpers
│   └── irrigation_advisory.py       # Zone-level rule engine → IrrigationAdvisory objects
│
├── api/
│   ├── app.py                       # FastAPI application factory
│   ├── routes.py                    # /health /advisory /stats /maps /weather endpoints
│   └── schemas.py                   # Pydantic request/response models
│
├── dashboard/
│   ├── app.py                       # Streamlit multi-page app (Dashboard/Maps/Advisory/Analytics/Settings)
│   ├── maps.py                      # Folium interactive raster + zone overlays
│   └── visualization.py             # Plotly charts (KPIs, donuts, bars, heatmaps, time series)
│
├── config/
│   └── settings.py                  # Pydantic-settings: satellite, weather, model, irrigation thresholds
│
├── utils/
│   ├── logger.py                    # Loguru console + rotating file + JSON sinks
│   ├── geo_utils.py                 # Reproject / clip / resample / stack / read / write helpers
│   └── indices.py                   # NDVI, EVI, SAVI, NDWI, NDMI, MSI, CWSI, SAR soil-moisture proxy
│
├── outputs/
│   ├── crop_map.tif                 # Generated crop classification raster
│   ├── stress_map.tif               # Generated moisture-stress raster
│   ├── irrigation_map.tif           # Generated water-deficit raster
│   └── reports/                     # CSV / JSON / GeoJSON advisory exports
│
├── notebooks/
│   ├── data_analysis.ipynb          # EDA: NDVI distributions, ground-truth balance, weather trends
│   └── model_training.ipynb         # End-to-end training walkthrough for all 3 models
│
├── tests/                           # 48 pytest unit/integration tests (indices, geo_utils, models, advisory)
├── requirements.txt
├── pytest.ini
├── .env.example
├── main.py                          # CLI: run / train-crop / train-stress / train-deficit / serve-api / serve-dashboard
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone <repo-url> && cd Crop-Irrigation-AI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in Sentinel/weather credentials
```

### 2. Run the dashboard (uses bundled sample data)

```bash
python main.py serve-dashboard
# → http://localhost:8501
```

### 3. Run the REST API

```bash
python main.py serve-api --reload
# → http://localhost:8000/docs   (interactive Swagger UI)
```

### 4. Run the full satellite-to-advisory pipeline

```bash
python main.py run --start 2026-06-01 --end 2026-06-15
```

### 5. Train models from scratch

```bash
# Crop classifier (with Optuna hyper-parameter search)
python preprocessing/feature_extraction.py \
    --optical data/satellite_data/processed/<scene>/<scene>_stack.tif \
    --sar     data/satellite_data/processed_sar/<scene>/<scene>_SAR_stack.tif \
    --index-dir data/satellite_data/processed/<scene> \
    --out training_features.npz

python main.py train-crop --features training_features.npz --tune

# Water-deficit regressor
python main.py train-deficit --features deficit_features.npz --model-type xgboost

# Moisture-stress CNN
python main.py train-stress --patches stress_patches.npz
```

---

## 🧪 Testing

```bash
pytest                          # full suite (48 tests)
pytest -m unit                  # fast unit tests only
pytest --cov --cov-report=html  # coverage report → outputs/reports/coverage_html/
```

---

## ⚙️ Configuration

All thresholds and credentials live in `config/settings.py`, overridable via `.env`:

```ini
SENTINELSAT_USER=your_copernicus_username
SENTINELSAT_PASS=your_copernicus_password
OWM_API_KEY=your_openweathermap_key
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/crop_irrigation
```

Key irrigation thresholds (`IrrigationSettings`):

| Threshold | Default | Meaning |
|---|---|---|
| `cwsi_critical` | 0.80 | CWSI above this → irrigate within 24h |
| `cwsi_moderate` | 0.55 | → irrigate within 3 days |
| `water_deficit_critical_mm` | 50 mm | Deficit triggering critical priority |
| `soil_moisture_critical_pct` | 20% | SAR soil moisture floor |
| `rainfall_skip_threshold_mm` | 10 mm | Forecast rain that can defer irrigation |

---

## 🔬 Methodology Notes

- **Spectral indices** (`utils/indices.py`): NDVI, EVI, SAVI, LAI, NDWI, NDMI, LSWI, MSI, BSI — all pure NumPy, division-safe.
- **CWSI** follows Jackson et al. (1981): `(LST − T_wet) / (T_dry − T_wet)`, clipped to [0, 1].
- **SAR soil moisture** is an empirical VV-backscatter normalization between dry/wet reference dB values — a proxy, not a calibrated retrieval; swap in a Water Cloud Model or Dubois model for production accuracy.
- **Crop coefficients (Kc)** use FAO-56 mid-season values per crop class for ETc = Kc × ET₀.
- **Speckle filtering** uses an adaptive Lee filter (5×5 default window).

---

## 🛣️ Roadmap

- [ ] Swap rule-based advisory for a learned ranking model using historical irrigation outcomes
- [ ] Add Celery + Redis for true async pipeline orchestration (replacing BackgroundTasks)
- [ ] SMS/WhatsApp advisory delivery integration
- [ ] Multi-season historical trend storage in PostgreSQL/TimescaleDB
- [ ] Sub-field tile-level (Sentinel-2 10m → drone imagery) super-resolution stress mapping

---

## 📄 License

MIT — see `LICENSE`.

## 🙏 Acknowledgements

Built on Copernicus Sentinel-1/2 (ESA), ERA5-Land (Copernicus Climate Data Store), and FAO-56 crop-water methodology.
