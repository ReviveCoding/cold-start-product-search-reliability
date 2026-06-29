from types import SimpleNamespace

import pandas as pd
import pytest

import product_search.evaluation.sensitivity as sensitivity


def _config() -> dict:
    return {
        "seed": 42,
        "retrieval": {"final_k": 10},
        "ranking": {"semantic_weight": 0.65, "behavior_weight": 0.35},
        "qrsbt": {
            "semantic_threshold": 0.18,
            "confidence_threshold": 0.18,
            "compatibility_threshold": 0.32,
            "irrelevant_risk_threshold": 0.60,
            "min_support": 2,
            "max_boost": 0.015,
            "promotion_window": 5,
            "max_promotions_per_query": 1,
        },
        "simulation": {"days": 2, "traffic_per_day": 10, "replications": 2},
        "release": {
            "min_cold_ndcg_lift": 0.005,
            "min_cold_ndcg_lift_ci_low": 0.0,
            "max_warm_ndcg_drop": 0.01,
            "max_irrelevant_exposure_increase": 0.0,
            "min_worst_scenario_utility_delta": 0.0,
            "min_p10_scenario_replication_utility_delta": 0.0,
        },
    }


def test_parse_policy_values_validates_range_and_deduplicates():
    assert sensitivity.parse_policy_values("0.015,0.012,0.015", label="boosts") == (
        0.012,
        0.015,
    )
    with pytest.raises(ValueError, match="boosts"):
        sensitivity.parse_policy_values("", label="boosts")
    with pytest.raises(ValueError, match="boosts"):
        sensitivity.parse_policy_values("1.2", label="boosts")


def test_policy_sensitivity_evaluates_finalists_and_selected_gate(monkeypatch):
    frame = pd.DataFrame({"query_id": [1], "product_id": [10]})

    def fake_gate(candidate, config):
        return candidate.assign(test_boost=config.max_boost)

    def fake_report(candidate, **kwargs):
        del kwargs
        boost = float(candidate.test_boost.iloc[0])
        return {
            "final_ndcg_at_10": 0.80 + boost,
            "base_ndcg_at_10": 0.80,
            "cold_ndcg_at_10_final": 0.60 + boost,
            "cold_ndcg_at_10_base": 0.60,
            "cold_ndcg_lift_ci_low": boost - 0.001,
            "cold_ndcg_lift_ci_high": boost + 0.001,
            "warm_ndcg_at_10_final": 0.80 - boost / 2,
            "warm_ndcg_at_10_base": 0.80,
            "warm_ndcg_delta_ci_low": -boost,
            "warm_ndcg_delta_ci_high": 0.0,
            "irrelevant_exposure_final": 0.04,
            "irrelevant_exposure_base": 0.04,
            "cold_relevant_exposure_final": 0.36,
            "cold_relevant_exposure_base": 0.35,
        }

    def fake_dynamic(candidate, **kwargs):
        del candidate, kwargs
        return SimpleNamespace(
            summary={
                "mean_scenario_replication_utility_delta": 3.0,
                "worst_scenario_utility_delta": 2.0,
                "p10_scenario_replication_utility_delta": 1.0,
            }
        )

    monkeypatch.setattr(sensitivity, "apply_coverage_overreach_gate", fake_gate)
    monkeypatch.setattr(sensitivity, "ranking_report", fake_report)
    monkeypatch.setattr(sensitivity, "run_dynamic_simulation", fake_dynamic)

    result, summary = sensitivity.evaluate_policy_sensitivity(
        frame,
        _config(),
        boosts=(0.012, 0.015),
        dynamic_finalists=(0.012, 0.015),
        bootstrap_samples=10,
    )
    assert len(result) == 2
    assert summary["status"] == "PASS"
    assert summary["selected_max_boost"] == 0.015
    markdown = sensitivity.sensitivity_markdown(result, 0.015)
    assert "0.015" in markdown
    assert "yes" in markdown


def test_policy_sensitivity_rejects_inconsistent_finalists(monkeypatch):
    del monkeypatch
    with pytest.raises(ValueError, match="finalist"):
        sensitivity.evaluate_policy_sensitivity(
            pd.DataFrame(),
            _config(),
            boosts=(0.015,),
            dynamic_finalists=(0.012,),
        )
