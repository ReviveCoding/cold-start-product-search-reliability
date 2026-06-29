from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def dcg(relevance: Iterable[float], k: int) -> float:
    values = np.asarray(list(relevance), dtype=float)[:k]
    if len(values) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2))
    return float(np.sum((2**values - 1) / discounts))


def _sort_ranked(group: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if "product_id" in group.columns:
        return group.sort_values([score_col, "product_id"], ascending=[False, True])
    return group.sort_values(score_col, ascending=False, kind="mergesort")


def _judged_mask(frame: pd.DataFrame) -> np.ndarray:
    if "judged" in frame.columns:
        return frame["judged"].fillna(0).astype(bool).to_numpy()
    return frame["relevance"].notna().to_numpy()


def _sort_ideal(group: pd.DataFrame) -> pd.DataFrame:
    judged = group.loc[_judged_mask(group)].copy()
    if "product_id" in judged.columns:
        return judged.sort_values(["relevance", "product_id"], ascending=[False, True])
    return judged.sort_values("relevance", ascending=False, kind="mergesort")


def _ranked_relevance(group: pd.DataFrame, score_col: str, k: int) -> np.ndarray:
    ranked = _sort_ranked(group, score_col).head(k)
    return ranked["relevance"].fillna(0.0).to_numpy(dtype=float)


def _query_ndcg(group: pd.DataFrame, score_col: str, k: int = 10) -> float | None:
    ideal = _sort_ideal(group)
    denom = dcg(ideal.relevance, k)
    return dcg(_ranked_relevance(group, score_col, k), k) / denom if denom > 0 else None


def ndcg_at_k(frame: pd.DataFrame, score_col: str, k: int = 10) -> float:
    values = [
        value
        for _, group in frame.groupby("query_id", sort=False)
        if (value := _query_ndcg(group, score_col, k)) is not None
    ]
    return float(np.mean(values)) if values else 0.0


def _query_cohort_ndcg(
    group: pd.DataFrame,
    score_col: str,
    cohort_col: str,
    cohort_value: int,
    k: int,
) -> float | None:
    judged = _judged_mask(group)
    mask = (group[cohort_col].to_numpy() == cohort_value) & judged
    if not mask.any():
        return None
    ranked = _sort_ranked(group, score_col).head(k)
    ranked_judged = _judged_mask(ranked)
    ranked_gain = np.where(
        (ranked[cohort_col].to_numpy() == cohort_value) & ranked_judged,
        ranked.relevance.fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    ideal_values = group.loc[mask, "relevance"].sort_values(ascending=False).to_numpy(dtype=float)
    denom = dcg(ideal_values, k)
    return dcg(ranked_gain, k) / denom if denom > 0 else None


def cohort_ndcg(
    frame: pd.DataFrame,
    score_col: str,
    cohort_col: str,
    cohort_value: int,
    k: int = 10,
) -> float:
    values = []
    for _, group in frame.groupby("query_id", sort=False):
        value = _query_cohort_ndcg(group, score_col, cohort_col, cohort_value, k)
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def judgment_coverage_at_k(frame: pd.DataFrame, score_col: str, k: int = 10) -> float:
    rates = []
    for _, group in frame.groupby("query_id", sort=False):
        top = _sort_ranked(group, score_col).head(k)
        rates.append(float(_judged_mask(top).mean()))
    return float(np.mean(rates)) if rates else 0.0


def unjudged_exposure_at_k(frame: pd.DataFrame, score_col: str, k: int = 10) -> float:
    return 1.0 - judgment_coverage_at_k(frame, score_col, k)


def irrelevant_exposure_at_k(frame: pd.DataFrame, score_col: str, k: int = 10) -> float:
    rates = []
    for _, group in frame.groupby("query_id", sort=False):
        top = _sort_ranked(group, score_col).head(k)
        judged = _judged_mask(top)
        if judged.any():
            rates.append(float((top.loc[judged, "relevance"] == 0).mean()))
    return float(np.mean(rates)) if rates else 0.0


def relation_exposure_at_k(
    frame: pd.DataFrame, score_col: str, relation: str, k: int = 10
) -> float:
    rates = []
    for _, group in frame.groupby("query_id", sort=False):
        top = _sort_ranked(group, score_col).head(k)
        judged = _judged_mask(top)
        if judged.any():
            rates.append(float((top.loc[judged, "relation"].astype(str) == relation).mean()))
    return float(np.mean(rates)) if rates else 0.0


def cold_relevant_exposure_at_k(frame: pd.DataFrame, score_col: str, k: int = 10) -> float:
    rates = []
    for _, group in frame.groupby("query_id", sort=False):
        top = _sort_ranked(group, score_col).head(k)
        judged = _judged_mask(top)
        rates.append(
            float(
                (
                    (top.zero_history.to_numpy() == 1)
                    & judged
                    & (top.relevance.fillna(0.0).to_numpy(dtype=float) >= 2)
                ).mean()
            )
        )
    return float(np.mean(rates)) if rates else 0.0


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability < high if high < 1 else probability <= high
        )
        if mask.any():
            error += mask.mean() * abs(
                float(y_true[mask].mean()) - float(probability[mask].mean())
            )
    return float(error)


def prediction_metrics(
    frame: pd.DataFrame, probability_col: str = "behavior_score"
) -> dict[str, float]:
    y = frame.clicked.to_numpy(dtype=int)
    p = np.clip(frame[probability_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    return {
        "roc_auc": auc,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


def _bootstrap_query_deltas(
    deltas: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _query_ndcg_deltas(
    frame: pd.DataFrame, baseline_col: str, candidate_col: str
) -> np.ndarray:
    deltas = []
    for _, group in frame.groupby("query_id", sort=False):
        baseline = _query_ndcg(group, baseline_col, 10)
        candidate = _query_ndcg(group, candidate_col, 10)
        if baseline is not None and candidate is not None:
            deltas.append(candidate - baseline)
    return np.asarray(deltas, dtype=float)


def _query_cohort_ndcg_deltas(
    frame: pd.DataFrame, baseline_col: str, candidate_col: str, cohort_value: int
) -> np.ndarray:
    deltas = []
    for _, group in frame.groupby("query_id", sort=False):
        baseline = _query_cohort_ndcg(group, baseline_col, "zero_history", cohort_value, 10)
        candidate = _query_cohort_ndcg(group, candidate_col, "zero_history", cohort_value, 10)
        if baseline is not None and candidate is not None:
            deltas.append(candidate - baseline)
    return np.asarray(deltas, dtype=float)


def _query_irrelevant_deltas(
    frame: pd.DataFrame, baseline_col: str, candidate_col: str
) -> np.ndarray:
    deltas = []
    for _, group in frame.groupby("query_id", sort=False):
        baseline_top = _sort_ranked(group, baseline_col).head(10)
        candidate_top = _sort_ranked(group, candidate_col).head(10)
        baseline_judged = _judged_mask(baseline_top)
        candidate_judged = _judged_mask(candidate_top)
        if not baseline_judged.any() or not candidate_judged.any():
            continue
        deltas.append(
            float((candidate_top.loc[candidate_judged, "relevance"] == 0).mean())
            - float((baseline_top.loc[baseline_judged, "relevance"] == 0).mean())
        )
    return np.asarray(deltas, dtype=float)

def ranking_report(
    frame: pd.DataFrame, *, bootstrap_samples: int = 300, seed: int = 42
) -> dict[str, float]:
    report = {
        "semantic_ndcg_at_10": ndcg_at_k(frame, "semantic_rank_score", 10),
        "base_ndcg_at_10": ndcg_at_k(frame, "base_score", 10),
        "final_ndcg_at_10": ndcg_at_k(frame, "final_score", 10),
        "cold_ndcg_at_10_base": cohort_ndcg(frame, "base_score", "zero_history", 1, 10),
        "cold_ndcg_at_10_final": cohort_ndcg(frame, "final_score", "zero_history", 1, 10),
        "warm_ndcg_at_10_base": cohort_ndcg(frame, "base_score", "zero_history", 0, 10),
        "warm_ndcg_at_10_final": cohort_ndcg(frame, "final_score", "zero_history", 0, 10),
        "irrelevant_exposure_base": irrelevant_exposure_at_k(frame, "base_score", 10),
        "irrelevant_exposure_final": irrelevant_exposure_at_k(frame, "final_score", 10),
        "complement_exposure_base": relation_exposure_at_k(frame, "base_score", "complement", 10),
        "complement_exposure_final": relation_exposure_at_k(frame, "final_score", "complement", 10),
        "cold_relevant_exposure_base": cold_relevant_exposure_at_k(frame, "base_score", 10),
        "cold_relevant_exposure_final": cold_relevant_exposure_at_k(frame, "final_score", 10),
        "judgment_coverage_base": judgment_coverage_at_k(frame, "base_score", 10),
        "judgment_coverage_final": judgment_coverage_at_k(frame, "final_score", 10),
        "unjudged_exposure_base": unjudged_exposure_at_k(frame, "base_score", 10),
        "unjudged_exposure_final": unjudged_exposure_at_k(frame, "final_score", 10),
        "boost_rate": float((frame.qrsbt_boost > 0).mean()),
        "block_rate": float((frame.gate_action == "BLOCK").mean()),
        "fallback_rate": float((frame.gate_action == "FALLBACK").mean()),
    }

    overall_low, overall_high = _bootstrap_query_deltas(
        _query_ndcg_deltas(frame, "base_score", "final_score"),
        samples=bootstrap_samples,
        seed=seed - 1,
    )
    cold_low, cold_high = _bootstrap_query_deltas(
        _query_cohort_ndcg_deltas(frame, "base_score", "final_score", 1),
        samples=bootstrap_samples,
        seed=seed,
    )
    warm_low, warm_high = _bootstrap_query_deltas(
        _query_cohort_ndcg_deltas(frame, "base_score", "final_score", 0),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    irr_low, irr_high = _bootstrap_query_deltas(
        _query_irrelevant_deltas(frame, "base_score", "final_score"),
        samples=bootstrap_samples,
        seed=seed + 2,
    )
    report.update(
        {
            "overall_ndcg_delta_ci_low": overall_low,
            "overall_ndcg_delta_ci_high": overall_high,
            "cold_ndcg_lift_ci_low": cold_low,
            "cold_ndcg_lift_ci_high": cold_high,
            "warm_ndcg_delta_ci_low": warm_low,
            "warm_ndcg_delta_ci_high": warm_high,
            "irrelevant_exposure_delta_ci_low": irr_low,
            "irrelevant_exposure_delta_ci_high": irr_high,
        }
    )
    return report
