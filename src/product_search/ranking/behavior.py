from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TemporalCalibratedBehaviorModel:
    """Behavior model trained on past blocks and calibrated on a later validation block."""

    seed: int = 42
    class_weight: str | None = "balanced"

    def fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        features: list[str],
        label: str = "clicked",
    ) -> "TemporalCalibratedBehaviorModel":
        self.features_ = list(features)
        self.base_model_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight=self.class_weight,
                        random_state=self.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        x_train = train[self.features_].fillna(0)
        y_train = train[label].to_numpy(dtype=int)
        if np.unique(y_train).size < 2:
            raise ValueError("Behavior training data must contain both classes")
        self.base_model_.fit(x_train, y_train)

        base_probability = self.base_model_.predict_proba(validation[self.features_].fillna(0))[:, 1]
        y_validation = validation[label].to_numpy(dtype=int)
        clipped = np.clip(base_probability, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        self.calibrator_ = None
        if np.unique(y_validation).size >= 2:
            calibrator = LogisticRegression(max_iter=500, random_state=self.seed, solver="lbfgs")
            calibrator.fit(logits, y_validation)
            self.calibrator_ = calibrator
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        base_probability = self.base_model_.predict_proba(frame[self.features_].fillna(0))[:, 1]
        if self.calibrator_ is None:
            calibrated = base_probability
        else:
            clipped = np.clip(base_probability, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            calibrated = self.calibrator_.predict_proba(logits)[:, 1]
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - calibrated, calibrated])

    def predict_uncalibrated(self, frame: pd.DataFrame) -> np.ndarray:
        return self.base_model_.predict_proba(frame[self.features_].fillna(0))[:, 1]
