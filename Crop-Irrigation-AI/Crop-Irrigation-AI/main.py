"""
main.py
─────────────────────────────────────────────────────────────────────────────
Crop-Irrigation-AI — master pipeline orchestrator.

Pipeline stages
───────────────
1. Acquire     — download Sentinel-1 / Sentinel-2 scenes for the date window
2. Preprocess  — optical + SAR preprocessing → feature stacks
3. Extract     — build pixel-wise feature matrix
4. Infer       — crop classification, moisture stress, water deficit maps
5. Advise      — generate zone-level irrigation advisories
6. Report      — export CSV / JSON / GeoJSON to outputs/reports

Usage
─────
    python main.py run                       # run full pipeline, today's date
    python main.py run --start 2026-06-01 --end 2026-06-15
    python main.py train-crop --features data/training_features.npz
    python main.py train-stress --patches data/stress_patches.npz
    python main.py train-deficit --features data/deficit_features.npz
    python main.py serve-api                 # launch FastAPI server
    python main.py serve-dashboard            # launch Streamlit dashboard
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np

from config import settings
from utils import logger
from preprocessing import OpticalPreprocessor, SARPreprocessor, FeatureExtractor
from models import (
    CropClassifier,
    MoistureStressDetector,
    WaterDeficitModel,
    IrrigationAdvisoryEngine,
)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    start_date: date | None = None,
    end_date: date | None = None,
    force_rerun: bool = False,
) -> dict:
    """
    Execute the full crop-irrigation-AI pipeline end to end.

    Returns
    -------
    summary dict with paths to generated outputs and advisory counts.
    """
    start_date = start_date or date.today()
    end_date   = end_date or date.today()
    t0 = datetime.now()

    logger.info("=" * 70)
    logger.info(f"  CROP-IRRIGATION-AI PIPELINE  |  {start_date} → {end_date}")
    logger.info("=" * 70)

    summary: dict = {"start_date": str(start_date), "end_date": str(end_date)}

    # ── Stage 1: Acquire ──────────────────────────────────────────────────────
    logger.info("[1/6] Acquiring satellite data …")
    optical_pp = OpticalPreprocessor()
    sar_pp = SARPreprocessor()

    optical_products = optical_pp.download(start_date=start_date, end_date=end_date)
    if not optical_products:
        logger.warning(
            "No new optical products downloaded (credentials missing or "
            "no new acquisitions). Checking for existing local scenes …"
        )
        optical_products = sorted(settings.satellite_dir.glob("*.SAFE")) + \
                            sorted(settings.satellite_dir.glob("*.zip"))

    if not optical_products:
        raise RuntimeError(
            "No Sentinel-2 products available locally or remotely. "
            "Configure SENTINELSAT_USER/SENTINELSAT_PASS or place "
            "pre-downloaded scenes in data/satellite_data/."
        )

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    logger.info("[2/6] Preprocessing optical imagery …")
    optical_stack = optical_pp.run(optical_products[0])

    sar_products = sorted(settings.satellite_dir.glob("*S1*.zip")) + \
                   sorted(settings.satellite_dir.glob("*S1*sigma0*.tif"))
    sar_stack = None
    if sar_products:
        logger.info("[2/6] Preprocessing SAR imagery …")
        sar_stack = sar_pp.run(sar_products[0])
    else:
        logger.warning("No SAR products found — soil-moisture layer will be skipped.")

    index_dir = optical_stack.parent

    # ── Stage 3: Extract features ─────────────────────────────────────────────
    logger.info("[3/6] Extracting feature matrix …")
    if sar_stack is None:
        raise RuntimeError(
            "SAR stack required for feature extraction. "
            "Provide Sentinel-1 data in data/satellite_data/."
        )

    extractor = FeatureExtractor(optical_stack, sar_stack, index_dir)
    raster_meta = extractor.get_raster_meta()
    X_inference = extractor.extract_inference_features(scale=False)

    # ── Stage 4: Inference ────────────────────────────────────────────────────
    logger.info("[4/6] Running model inference …")

    crop_clf = CropClassifier()
    if crop_clf.model is None:
        raise RuntimeError(
            "Crop classifier not trained. Run: "
            "python main.py train-crop --features <path>"
        )
    crop_map = crop_clf.predict_to_raster(X_inference, raster_meta)
    summary["crop_map"] = str(settings.crop_map_path)

    stress_detector = MoistureStressDetector()
    if stress_detector.model is not None:
        stress_map = stress_detector.predict_to_raster(optical_stack)
        summary["stress_map"] = str(settings.stress_map_path)
    else:
        logger.warning("Moisture-stress model not trained — skipping stress map.")

    deficit_model = WaterDeficitModel()
    if deficit_model.model is not None:
        deficit_map = deficit_model.predict_to_raster(X_inference, raster_meta)
        summary["deficit_map"] = str(settings.irrigation_map_path)
    else:
        logger.warning("Water-deficit model not trained — skipping deficit map.")

    # ── Stage 5: Advisory ─────────────────────────────────────────────────────
    logger.info("[5/6] Generating irrigation advisories …")
    advisory_engine = IrrigationAdvisoryEngine()
    advisories = advisory_engine.generate_advisories()
    summary["n_advisories"] = len(advisories)
    summary["n_critical"] = sum(1 for a in advisories if a.priority.value == "critical")

    # ── Stage 6: Reports ──────────────────────────────────────────────────────
    logger.info("[6/6] Exporting reports …")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    date_tag = end_date.strftime("%Y%m%d")

    advisory_engine.to_csv(advisories, settings.reports_dir / f"advisory_{date_tag}.csv")
    advisory_engine.to_json(advisories, settings.reports_dir / f"advisory_{date_tag}.json")
    try:
        advisory_engine.to_geojson(advisories, settings.reports_dir / f"advisory_{date_tag}.geojson")
    except Exception as exc:
        logger.warning(f"GeoJSON export skipped: {exc}")

    elapsed = (datetime.now() - t0).total_seconds()
    summary["elapsed_seconds"] = round(elapsed, 1)

    logger.info("=" * 70)
    logger.success(
        f"  PIPELINE COMPLETE in {elapsed:.1f}s  |  "
        f"{summary['n_advisories']} advisories  |  "
        f"{summary['n_critical']} critical zones"
    )
    logger.info("=" * 70)
    return summary


# ── Training helpers ───────────────────────────────────────────────────────────

def train_crop_model(features_path: Path, tune: bool = False) -> dict:
    """Train the crop classifier from a saved .npz feature file (X, y)."""
    data = np.load(features_path)
    X, y = data["X"], data["y"]

    clf = CropClassifier(feature_names=[
        *["B02", "B03", "B04", "B08", "B8A", "B11", "B12"],
        *["NDVI", "EVI", "SAVI", "LAI", "NDWI", "NDMI", "LSWI", "MSI", "BSI"],
        *["VV_dB", "VH_dB", "RVI", "SoilMoisture"],
    ])

    params = None
    if tune:
        params = clf.tune(X, y, n_trials=30)

    metrics = clf.train(X, y, params=params, cv_folds=settings.model.cv_folds)
    logger.info(f"Training report:\n{metrics.get('classification_report', '')}")

    # SHAP explainability
    shap_df = clf.explain(X, n_samples=500, out_dir=settings.reports_dir)
    logger.info(f"Top features:\n{shap_df.head(10).to_string(index=False)}")

    return metrics


def train_stress_model(patches_path: Path) -> list[dict]:
    """Train the moisture-stress CNN from a saved .npz patches file."""
    data = np.load(patches_path)
    patches, labels = data["patches"], data["labels"]

    detector = MoistureStressDetector(in_channels=patches.shape[1])
    history = detector.train(patches, labels)
    return history


def train_deficit_model(features_path: Path, model_type: str = "xgboost") -> dict:
    """Train the water-deficit regressor from a saved .npz feature file."""
    data = np.load(features_path)
    X, y = data["X"], data["y"]

    model = WaterDeficitModel(model_type=model_type)
    metrics = model.train(X, y, cv_folds=settings.model.cv_folds)
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Crop-Irrigation-AI command-line interface."""
    pass


@cli.command("run")
@click.option("--start", "start_str", default=None, help="Start date YYYY-MM-DD")
@click.option("--end",   "end_str",   default=None, help="End date YYYY-MM-DD")
@click.option("--force", is_flag=True, default=False, help="Force re-run even if outputs exist")
def cli_run(start_str, end_str, force):
    """Run the full satellite-to-advisory pipeline."""
    start = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else None
    end   = datetime.strptime(end_str,   "%Y-%m-%d").date() if end_str else None
    summary = run_pipeline(start_date=start, end_date=end, force_rerun=force)
    click.echo(summary)


@cli.command("train-crop")
@click.option("--features", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--tune", is_flag=True, default=False, help="Run Optuna hyper-parameter search")
def cli_train_crop(features, tune):
    """Train the crop classification model."""
    train_crop_model(features, tune=tune)


@cli.command("train-stress")
@click.option("--patches", type=click.Path(exists=True, path_type=Path), required=True)
def cli_train_stress(patches):
    """Train the moisture-stress CNN."""
    train_stress_model(patches)


@cli.command("train-deficit")
@click.option("--features", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-type", type=click.Choice(["xgboost", "lightgbm"]), default="xgboost")
def cli_train_deficit(features, model_type):
    """Train the water-deficit regression model."""
    train_deficit_model(features, model_type=model_type)


@cli.command("serve-api")
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--reload", is_flag=True, default=False)
def cli_serve_api(host, port, reload):
    """Launch the FastAPI REST server."""
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
    )


@cli.command("serve-dashboard")
def cli_serve_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "dashboard" / "app.py"),
    ])


if __name__ == "__main__":
    cli()
