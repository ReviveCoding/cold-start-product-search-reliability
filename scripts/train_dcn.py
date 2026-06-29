from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from product_search.config import load_config
from product_search.evaluation.metrics import prediction_metrics, ranking_report
from product_search.pipeline import _gate_config
from product_search.policy.gate import apply_coverage_overreach_gate
from product_search.provenance import (
    atomic_joblib_dump,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from product_search.ranking.dcn import DCNRanker


def _fit_platt(scores: np.ndarray, labels: np.ndarray, seed: int):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if np.unique(labels).size < 2:
        return None
    return LogisticRegression(max_iter=500, random_state=seed).fit(logits, labels)


def _calibrate(scores: np.ndarray, calibrator) -> np.ndarray:
    if calibrator is None:
        return np.clip(scores, 1e-6, 1 - 1e-6)
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return np.clip(calibrator.predict_proba(logits)[:, 1], 1e-6, 1 - 1e-6)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the optional PyTorch DCN behavioral challenger."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    rank_cfg = config.raw["ranking"]
    epochs = args.epochs or int(rank_cfg["dcn_epochs"])
    batch_size = args.batch_size or int(rank_cfg["dcn_batch_size"])
    learning_rate = args.learning_rate or float(rank_cfg["learning_rate"])
    seed = args.seed if args.seed is not None else config.seed

    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    test = pd.read_csv(args.test)
    candidates = pd.read_csv(args.candidates)
    features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    missing = {
        name: sorted(set(features) - set(frame.columns))
        for name, frame in {
            "train": train,
            "validation": validation,
            "test": test,
            "candidates": candidates,
        }.items()
        if set(features) - set(frame.columns)
    }
    if missing:
        raise ValueError(f"DCN feature contract mismatch: {missing}")

    model = DCNRanker(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    ).fit(train, features, label="clicked", validation=validation)

    raw_validation = model.predict_proba(validation)
    calibrator = _fit_platt(raw_validation, validation.clicked.to_numpy(dtype=int), seed)
    train_score = _calibrate(model.predict_proba(train), calibrator)
    test_score = _calibrate(model.predict_proba(test), calibrator)
    candidate_score = _calibrate(model.predict_proba(candidates), calibrator)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        output / "dcn_train_predictions.csv",
        pd.DataFrame({"row_id": train.row_id.astype(int), "behavior_score": train_score}),
    )
    atomic_write_csv(
        output / "dcn_test_predictions.csv",
        pd.DataFrame({"row_id": test.row_id.astype(int), "behavior_score": test_score}),
    )
    atomic_write_csv(output / "dcn_training_history.csv", pd.DataFrame(model.history_))
    atomic_joblib_dump(output / "dcn_scaler.joblib", model.scaler_)
    atomic_joblib_dump(output / "dcn_calibrator.joblib", calibrator)
    _atomic_torch_save(
        output / "dcn_state.pt",
        {
            "state_dict": model.state_dict_cpu(),
            "features": features,
            "epochs_trained": model.epochs_trained_,
        },
    )

    evaluated_logged = test.copy()
    evaluated_logged["behavior_score"] = np.asarray(test_score, dtype=float)
    evaluated_candidates = candidates.copy()
    evaluated_candidates["behavior_score"] = np.asarray(candidate_score, dtype=float)
    evaluated_candidates = apply_coverage_overreach_gate(
        evaluated_candidates, _gate_config(config.raw)
    )
    metrics = {
        **{
            f"behavior_{key}": value
            for key, value in prediction_metrics(evaluated_logged).items()
        },
        **ranking_report(evaluated_candidates, seed=seed),
    }
    atomic_write_json(output / "dcn_challenger_metrics.json", metrics)
    atomic_write_csv(
        output / "dcn_evaluated_candidates.csv",
        evaluated_candidates[
            [
                "query_id",
                "product_id",
                "behavior_score",
                "qrsbt_boost",
                "gate_action",
                "gate_reason",
                "final_score",
            ]
        ],
    )

    runtime = time.perf_counter() - started
    metadata = {
        "device": str(model.device_),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(model.device_)
            if model.device_.type == "cuda"
            else None
        ),
        "precision_mode": "amp_fp16" if model.amp_enabled_ else "float32",
        "peak_vram_bytes": int(model.peak_vram_bytes_),
        "runtime_seconds": runtime,
        "training_rows_per_second": float(len(train) * model.epochs_trained_ / max(runtime, 1e-9)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "candidate_rows": int(len(candidates)),
        "feature_count": int(len(features)),
        "epochs_requested": epochs,
        "epochs_trained": int(model.epochs_trained_),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "state_sha256": sha256_file(output / "dcn_state.pt"),
        "metrics_file": "dcn_challenger_metrics.json",
    }
    atomic_write_json(output / "dcn_training_metadata.json", metadata)

    baseline_path = output.parent / "metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report_rows = [
            ("ROC-AUC", baseline["behavior_roc_auc"], metrics["behavior_roc_auc"]),
            ("Brier score", baseline["behavior_brier"], metrics["behavior_brier"]),
            ("ECE", baseline["behavior_ece"], metrics["behavior_ece"]),
            ("Final NDCG@10", baseline["final_ndcg_at_10"], metrics["final_ndcg_at_10"]),
            (
                "Cold NDCG@10",
                baseline["cold_ndcg_at_10_final"],
                metrics["cold_ndcg_at_10_final"],
            ),
            (
                "Warm NDCG@10",
                baseline["warm_ndcg_at_10_final"],
                metrics["warm_ndcg_at_10_final"],
            ),
            (
                "Irrelevant exposure@10",
                baseline["irrelevant_exposure_final"],
                metrics["irrelevant_exposure_final"],
            ),
        ]
        lines = [
            "# Advanced Behavioral Challenger Report",
            "",
            "The optional PyTorch DCN challenger uses the exact feature contract exported by the ",
            "temporally calibrated logistic champion and is calibrated on the same isolated ",
            "validation block.",
            "",
            "| Metric | Calibrated logistic champion | Calibrated DCN challenger |",
            "|---|---:|---:|",
        ]
        lines.extend(
            f"| {name} | {base:.4f} | {challenger:.4f} |"
            for name, base, challenger in report_rows
        )
        lines.extend(
            [
                "",
                "The challenger is not promoted automatically. It must also pass dynamic-feedback ",
                "and release guardrails before replacing the champion.",
            ]
        )
        reports = output.parent / "reports"
        reports.mkdir(exist_ok=True)
        atomic_write_text(
            reports / "advanced_challenger_report.md", "\n".join(lines) + "\n"
        )

    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
