from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker


@dataclass
class LambdaMARTRanker:
    n_estimators: int = 50
    seed: int = 42

    def fit(self, frame: pd.DataFrame, features: list[str], label: str = "relevance") -> "LambdaMARTRanker":
        ordered = frame.sort_values(["query_id", "product_id"]).copy()
        groups = ordered.groupby("query_id", sort=False).size().to_numpy()
        self.features_ = features
        self.model_ = XGBRanker(
            objective="rank:ndcg",
            n_estimators=self.n_estimators,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.seed,
            n_jobs=1,
            tree_method="hist",
        )
        self.model_.fit(ordered[features], ordered[label], group=groups, verbose=False)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(frame[self.features_])

    def save(self, model_path: str | Path, metadata_path: str | Path) -> None:
        """Persist the booster with XGBoost's stable model format.

        XGBoost guarantees backward compatibility for saved models, but not for Python memory
        snapshots.  Feature order and wrapper metadata are stored separately as plain JSON.
        """
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{model_path.name}.",
            suffix=model_path.suffix or ".json",
            dir=model_path.parent,
            delete=False,
        )
        handle.close()
        temp_model = Path(handle.name)
        try:
            self.model_.save_model(temp_model)
            os.replace(temp_model, model_path)
        finally:
            temp_model.unlink(missing_ok=True)
        payload = {
            "format": "xgboost-stable-model",
            "features": list(self.features_),
            "n_estimators": int(self.n_estimators),
            "seed": int(self.seed),
        }
        temp_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
        temp_metadata.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_metadata, metadata_path)

    @classmethod
    def load(
        cls, model_path: str | Path, metadata_path: str | Path
    ) -> "LambdaMARTRanker":
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "xgboost-stable-model":
            raise RuntimeError("Unsupported LambdaMART persistence format")
        instance = cls(
            n_estimators=int(metadata.get("n_estimators", 0)),
            seed=int(metadata.get("seed", 0)),
        )
        instance.features_ = list(metadata["features"])
        instance.model_ = XGBRanker()
        instance.model_.load_model(model_path)
        return instance
