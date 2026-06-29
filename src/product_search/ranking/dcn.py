from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CrossLayer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.weight, std=0.03)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scalar + self.bias + x


class TinyDCN(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.cross1 = CrossLayer(dimension)
        self.cross2 = CrossLayer(dimension)
        self.deep = nn.Sequential(
            nn.Linear(dimension, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 24),
            nn.ReLU(),
        )
        self.output = nn.Linear(dimension + 24, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        crossed = self.cross1(x, x)
        crossed = self.cross2(x, crossed)
        deep = self.deep(x)
        return self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)


@dataclass
class DCNRanker:
    epochs: int = 7
    batch_size: int = 256
    learning_rate: float = 0.003
    seed: int = 42
    patience: int = 2
    use_amp: bool = True

    def fit(
        self,
        frame: pd.DataFrame,
        features: list[str],
        label: str = "clicked",
        validation: pd.DataFrame | None = None,
    ) -> "DCNRanker":
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.reset_peak_memory_stats()
        self.features_ = list(features)
        self.scaler_ = StandardScaler()
        x = self.scaler_.fit_transform(
            frame[self.features_].fillna(0).to_numpy(dtype=np.float32)
        )
        y = frame[label].to_numpy(dtype=np.float32)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled_ = bool(self.use_amp and self.device_.type == "cuda")
        self.model_ = TinyDCN(x.shape[1]).to(self.device_)
        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negative / positive, device=self.device_)
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled_)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        validation_tensors = None
        if validation is not None and not validation.empty:
            validation_x = self.scaler_.transform(
                validation[self.features_].fillna(0).to_numpy(dtype=np.float32)
            )
            validation_y = validation[label].to_numpy(dtype=np.float32)
            validation_tensors = (
                torch.from_numpy(validation_x).to(self.device_),
                torch.from_numpy(validation_y).to(self.device_),
            )

        self.history_ = []
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(self.epochs):
            self.model_.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device_.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled_,
                ):
                    logits = self.model_(xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

            validation_loss = float("nan")
            if validation_tensors is not None:
                self.model_.eval()
                vx, vy = validation_tensors
                with torch.no_grad(), torch.amp.autocast(
                    device_type=self.device_.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled_,
                ):
                    validation_loss = float(loss_fn(self.model_(vx), vy).detach().cpu())
                monitored = validation_loss
            else:
                monitored = float(np.mean(losses))
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                }
            )
            if monitored < best_loss - 1e-6:
                best_loss = monitored
                best_state = deepcopy(self.state_dict_cpu())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if validation_tensors is not None and epochs_without_improvement >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.epochs_trained_ = len(self.history_)
        self.peak_vram_bytes_ = (
            int(torch.cuda.max_memory_allocated(self.device_))
            if self.device_.type == "cuda"
            else 0
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.scaler_.transform(
            frame[self.features_].fillna(0).to_numpy(dtype=np.float32)
        )
        self.model_.eval()
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device_.type,
            dtype=torch.float16,
            enabled=self.amp_enabled_,
        ):
            logits = self.model_(torch.from_numpy(x).to(self.device_))
            return torch.sigmoid(logits.float()).cpu().numpy()

    def state_dict_cpu(self) -> dict:
        return {
            key: value.detach().cpu() for key, value in self.model_.state_dict().items()
        }
