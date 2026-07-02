from __future__ import annotations

import pandas as pd

from product_search.evaluation import sensitivity


class _DynamicResult:
    def __init__(self, summary: dict[str, float]) -> None:
        self.summary = summary


def _raw_config() -> dict:
    return {
        "seed": 42,
        "qrsbt": {
            "semantic_threshold": 0.18,
            "confidence_threshold": 0.80,
            "compatibility_threshold": 0.32,
            "irrelevant_risk_threshold": 0.60,
            "min_support": 2,
            "max_boost": 0.01275,
            "promotion_window": 5,
            "max_promotions_per_query": 1,
            "promotion_mode": "boundary_entry_only",
        },
        "ranking": {
            "semantic_weight": 0.65,
            "behavior_weight": 0.35,
        },
        "retrieval": {"final_k": 10},
        "simulation": {
            "days": 10,
            "traffic_per_day": 120,
            "replications": 30,
        },
        "release": {
            "min_cold_ndcg_lift": 0.005,
            "min_cold_ndcg_lift_ci_low": 0.0,
            "max_warm_ndcg_drop": 0.01,
            "max_irrelevant_exposure_increase": 0.0,
            "max_dynamic_irrelevant_exposure_increase": 0.0,
            "min_worst_scenario_utility_delta": 0.0,
            "min_p10_scenario_replication_utility_delta": 0.0,
        },
    }


def _static_metrics() -> dict[str, float]:
    return {
        "base_ndcg_at_10": 0.50,
        "final_ndcg_at_10": 0.505,
        "cold_ndcg_at_10_base": 0.50,
        "cold_ndcg_at_10_final": 0.505,
        "cold_ndcg_lift_ci_low": 0.001,
        "cold_ndcg_lift_ci_high": 0.009,
        "warm_ndcg_at_10_base": 0.50,
        "warm_ndcg_at_10_final": 0.50,
        "warm_ndcg_delta_ci_low": 0.0,
        "warm_ndcg_delta_ci_high": 0.0,
        "irrelevant_exposure_base": 0.0,
        "irrelevant_exposure_final": 0.0,
        "cold_relevant_exposure_base": 0.0,
        "cold_relevant_exposure_final": 0.0,
    }


def test_default_dynamic_finalists_cover_the_admission_cliff() -> None:
    assert sensitivity.DEFAULT_DYNAMIC_FINALISTS == (
        0.01125,
        0.012,
        0.0125,
        0.01275,
        0.013,
        0.01325,
    )


def test_dynamic_exposure_is_reported_and_blocks_sensitivity_pass(
    monkeypatch,
) -> None:
    frame = pd.DataFrame({"placeholder": [1]})
    dynamic_summary = {
        "base_irrelevant_exposure": 10.0,
        "qrsbt_irrelevant_exposure": 11.0,
        "mean_scenario_replication_utility_delta": 1.0,
        "worst_scenario_utility_delta": 1.0,
        "p10_scenario_replication_utility_delta": 1.0,
    }

    monkeypatch.setattr(
        sensitivity,
        "apply_coverage_overreach_gate",
        lambda source, _gate: source.copy(),
    )
    monkeypatch.setattr(
        sensitivity,
        "ranking_report",
        lambda _candidate, **_kwargs: _static_metrics(),
    )
    monkeypatch.setattr(
        sensitivity,
        "run_dynamic_simulation",
        lambda _candidate, **_kwargs: _DynamicResult(dynamic_summary),
    )

    result, summary = sensitivity.evaluate_policy_sensitivity(
        frame,
        _raw_config(),
        boosts=(0.01275,),
        dynamic_finalists=(0.01275,),
        bootstrap_samples=1,
    )

    assert result.loc[0, "dynamic_irrelevant_exposure_delta"] == 1.0
    assert summary["status"] == "FAIL"
