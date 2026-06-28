"""
models/moisture_stress.py
─────────────────────────────────────────────────────────────────────────────
Moisture-stress detection using a lightweight CNN on spectral image patches.

Architecture
────────────
Input  : (batch, C, 64, 64) — C = number of input channels (bands + indices)
Block 1: Conv2d(C  → 32, 3×3) → BN → ReLU → MaxPool(2)
Block 2: Conv2d(32 → 64, 3×3) → BN → ReLU → MaxPool(2)
Block 3: Conv2d(64 → 128, 3×3) → BN → ReLU → AdaptiveAvgPool
Head   : Linear(128 → 64) → ReLU → Dropout(0.4) → Linear(64 → n_classes)

Stress classes
──────────────
0 – No stress
1 – Mild stress
2 – Moderate stress
3 – Severe stress
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, f1_score
import rasterio
from tqdm import tqdm

from config import settings
from utils import logger, write_band

STRESS_LABELS = {0: "No stress", 1: "Mild", 2: "Moderate", 3: "Severe"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CNN Architecture ───────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MoistureStressCNN(nn.Module):
    """Lightweight CNN for per-patch moisture-stress classification."""

    def __init__(self, in_channels: int = 6, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(in_channels, 32, pool=True),
            _ConvBlock(32, 64, pool=True),
            _ConvBlock(64, 128, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Patch sampler ──────────────────────────────────────────────────────────────

class PatchSampler:
    """
    Extract fixed-size spatial patches from a multi-band raster.
    Used for both training-sample generation and inference tiling.
    """

    def __init__(
        self,
        raster_path: Path,
        patch_size: int = 64,
        stride: int = 32,
    ):
        self.raster_path = raster_path
        self.patch_size = patch_size
        self.stride = stride

    def get_patches(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Tile the raster into overlapping patches.

        Returns
        -------
        patches     : (n_patches, C, patch_size, patch_size) float32
        top_lefts   : list of (row, col) origin pixel for each patch
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read().astype(np.float32)   # (C, H, W)

        C, H, W = data.shape
        P = self.patch_size
        S = self.stride
        patches, coords = [], []

        for r in range(0, H - P + 1, S):
            for c in range(0, W - P + 1, S):
                patch = data[:, r:r + P, c:c + P]
                patches.append(patch)
                coords.append((r, c))

        return np.stack(patches), coords

    def reconstruct_map(
        self,
        predictions: np.ndarray,
        coords: list[tuple[int, int]],
        h: int,
        w: int,
        n_classes: int,
    ) -> np.ndarray:
        """
        Reconstruct a per-pixel probability map from patch predictions
        using average pooling over overlapping patches.

        Parameters
        ----------
        predictions : (n_patches, n_classes) softmax probabilities
        coords      : patch top-left (row, col) coordinates
        h, w        : output map height and width
        n_classes   : number of classes

        Returns
        -------
        prob_map : (n_classes, H, W) float32
        """
        P = self.patch_size
        accum = np.zeros((n_classes, h, w), dtype=np.float32)
        count = np.zeros((1, h, w), dtype=np.float32)

        for (r, c), proba in zip(coords, predictions):
            accum[:, r:r + P, c:c + P] += proba[:, None, None]
            count[:, r:r + P, c:c + P] += 1.0

        count = np.maximum(count, 1e-6)
        return (accum / count).astype(np.float32)


# ── Trainer / Predictor ────────────────────────────────────────────────────────

class MoistureStressDetector:
    """
    End-to-end training, evaluation, and inference for moisture-stress CNN.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        in_channels: int | None = None,
        num_classes: int = 4,
    ):
        self.model_path = model_path or (
            settings.saved_models_dir / "moisture_stress_cnn.pt"
        )
        self.in_channels = in_channels or settings.model.cnn_input_channels
        self.num_classes = num_classes
        self.model: Optional[MoistureStressCNN] = None
        self._load_if_exists()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if self.model_path.exists():
            self.model = MoistureStressCNN(self.in_channels, self.num_classes)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=DEVICE)
            )
            self.model.to(DEVICE).eval()
            logger.info(f"Loaded moisture-stress CNN from {self.model_path}")

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Moisture-stress CNN saved → {self.model_path}")

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        val_split: float = 0.15,
    ) -> list[dict]:
        """
        Train the CNN on labelled patches.

        Parameters
        ----------
        patches    : (N, C, H, W) float32
        labels     : (N,) int64
        epochs     : training epochs
        batch_size : mini-batch size
        lr         : initial learning rate
        val_split  : fraction for validation

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        epochs     = epochs     or settings.model.cnn_epochs
        batch_size = batch_size or settings.model.cnn_batch_size
        lr         = lr         or settings.model.cnn_learning_rate

        X_t = torch.from_numpy(patches).float()
        y_t = torch.from_numpy(labels).long()
        dataset = TensorDataset(X_t, y_t)

        val_size  = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        self.model = MoistureStressCNN(self.in_channels, self.num_classes).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        history = []
        for epoch in range(1, epochs + 1):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(Xb)

            # ── Validate ──
            self.model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(DEVICE)
                    logits = self.model(Xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    val_preds.append(preds)
                    val_trues.append(yb.numpy())

            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            val_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)

            epoch_metrics = {
                "epoch":      epoch,
                "train_loss": train_loss / train_size,
                "val_f1":     val_f1,
                "lr":         scheduler.get_last_lr()[0],
            }
            history.append(epoch_metrics)
            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs}  |  "
                    f"loss={epoch_metrics['train_loss']:.4f}  |  "
                    f"val_F1={val_f1:.4f}"
                )

        self.save()
        return history

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_patches(
        self,
        patches: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Run inference on a batch of patches.

        Returns
        -------
        proba : (n_patches, n_classes) softmax probabilities
        """
        self._require_model()
        self.model.eval()
        results = []
        for i in range(0, len(patches), batch_size):
            Xb = torch.from_numpy(patches[i:i + batch_size]).float().to(DEVICE)
            proba = torch.softmax(self.model(Xb), dim=1).cpu().numpy()
            results.append(proba)
        return np.concatenate(results, axis=0)

    def predict_to_raster(
        self,
        raster_path: Path,
        out_stress_map: Path | None = None,
        patch_size: int = 64,
        stride: int = 32,
    ) -> np.ndarray:
        """
        Full-scene moisture-stress inference from a multi-band GeoTIFF.

        Returns
        -------
        stress_class_map : (H, W) uint8 — dominant stress class per pixel
        """
        self._require_model()

        sampler = PatchSampler(raster_path, patch_size, stride)
        patches, coords = sampler.get_patches()
        logger.info(f"Running stress inference on {len(patches)} patches …")

        probas = self.predict_patches(patches)

        with rasterio.open(raster_path) as src:
            h, w = src.height, src.width
            profile = src.profile.copy()

        prob_map = sampler.reconstruct_map(probas, coords, h, w, self.num_classes)
        stress_map = prob_map.argmax(axis=0).astype(np.uint8)

        out_path = out_stress_map or settings.stress_map_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(stress_map, 1)
            dst.update_tags(
                classes=str(STRESS_LABELS),
                nodata_value="255",
            )

        logger.success(f"Stress map written → {out_path}")
        return stress_map

    # ── Guard ──────────────────────────────────────────────────────────────────

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model not trained or loaded. "
                "Call .train() or ensure model_path exists."
            )
