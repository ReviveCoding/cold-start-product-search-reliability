from copy import deepcopy

from product_search.policy.release import evaluate_release


THRESHOLDS = {
    "max_overall_ndcg_drop": 0.005,
    "max_overall_ndcg_drop_ci": 0.015,
    "min_cold_ndcg_lift": 0.0,
    "min_cold_ndcg_lift_ci_low": -0.01,
    "max_warm_ndcg_drop": 0.02,
    "max_warm_ndcg_drop_ci": 0.03,
    "max_irrelevant_exposure_increase": 0.0,
    "max_irrelevant_exposure_ci_high": 0.01,
    "min_ope_ess": 50,
    "max_ope_dr_abs_error": 0.05,
    "min_ope_support_overlap": 0.99,
    "max_dynamic_relevant_discovery_drop_rate": 0.02,
    "max_dynamic_false_warmup_increase": 0,
    "max_dynamic_irrelevant_exposure_increase": 0,
    "min_worst_scenario_utility_delta": -2.0,
    "min_p10_scenario_replication_utility_delta": -5.0,
    "min_judgment_coverage_at_10": 0.95,
    "min_future_behavior_roc_auc": 0.65,
    "max_future_behavior_brier": 0.20,
    "max_future_behavior_ece": 0.10,
    "max_future_logged_ndcg_drop": 0.02,
    "max_relation_eligibility_brier": 0.10,
    "max_relation_eligibility_ece": 0.10,
}


def _inputs():
    metrics = {
        "final_ndcg_at_10": 0.81,
        "base_ndcg_at_10": 0.80,
        "overall_ndcg_delta_ci_low": -0.002,
        "cold_ndcg_at_10_final": 0.61,
        "cold_ndcg_at_10_base": 0.60,
        "warm_ndcg_at_10_final": 0.80,
        "warm_ndcg_at_10_base": 0.80,
        "irrelevant_exposure_final": 0.04,
        "irrelevant_exposure_base": 0.05,
        "cold_ndcg_lift_ci_low": -0.005,
        "warm_ndcg_delta_ci_low": -0.005,
        "irrelevant_exposure_delta_ci_high": 0.0,
        "judgment_coverage_base": 1.0,
        "judgment_coverage_final": 1.0,
        "future_behavior_roc_auc": 0.75,
        "future_behavior_brier": 0.13,
        "future_behavior_ece": 0.03,
        "future_logged_base_ndcg_at_10": 0.80,
        "future_logged_final_ndcg_at_10": 0.80,
        "relation_eligibility_brier": 0.02,
        "relation_eligibility_ece": 0.03,
    }
    ope = {
        "effective_sample_size": 500,
        "dr": 0.2,
        "dr_abs_error": 0.01,
        "dr_covers_true_value": 1.0,
        "support_overlap_rate": 1.0,
    }
    dynamic = {
        "base_relevant_discovery": 100,
        "qrsbt_relevant_discovery": 99,
        "base_false_warmup": 1,
        "qrsbt_false_warmup": 1,
        "base_irrelevant_exposure": 20,
        "qrsbt_irrelevant_exposure": 15,
        "worst_scenario_utility_delta": 0.0,
        "p10_scenario_replication_utility_delta": 0.0,
    }
    return metrics, ope, dynamic


def test_release_launches_when_all_gates_pass():
    metrics, ope, dynamic = _inputs()
    assert evaluate_release(metrics, ope, dynamic, THRESHOLDS)["status"] == "LAUNCH"


def test_release_holds_on_dynamic_relevance_regression():
    metrics, ope, dynamic = _inputs()
    dynamic = deepcopy(dynamic)
    dynamic["qrsbt_relevant_discovery"] = 90
    result = evaluate_release(metrics, ope, dynamic, THRESHOLDS)
    assert result["status"] == "HOLD"
    assert not result["gates"]["dynamic_relevant_discovery_non_inferiority"]


def test_release_holds_on_overall_ranking_regression():
    metrics, ope, dynamic = _inputs()
    metrics = deepcopy(metrics)
    metrics["final_ndcg_at_10"] = 0.75
    metrics["overall_ndcg_delta_ci_low"] = -0.06
    result = evaluate_release(metrics, ope, dynamic, THRESHOLDS)
    assert result["status"] == "HOLD"
    assert not result["gates"]["overall_non_inferiority"]


def test_release_holds_when_semantic_judgment_coverage_is_too_low():
    metrics, ope, dynamic = _inputs()
    metrics["judgment_coverage_final"] = 0.5
    result = evaluate_release(metrics, ope, dynamic, THRESHOLDS)
    assert result["status"] == "HOLD"
    assert not result["gates"]["semantic_judgment_coverage"]


def test_release_holds_on_untouched_future_block_failure():
    metrics, ope, dynamic = _inputs()
    metrics["future_behavior_ece"] = 0.25
    result = evaluate_release(metrics, ope, dynamic, THRESHOLDS)
    assert result["status"] == "HOLD"
    assert not result["gates"]["future_behavior_calibration"]


def test_release_holds_on_relation_confidence_miscalibration():
    metrics, ope, dynamic = _inputs()
    metrics["relation_eligibility_brier"] = 0.25
    result = evaluate_release(metrics, ope, dynamic, THRESHOLDS)
    assert result["status"] == "HOLD"
    assert not result["gates"]["relation_eligibility_brier"]


def test_release_holds_when_policy_sensitivity_fails():
    metrics, ope, dynamic = _inputs()
    decision = evaluate_release(
        metrics,
        ope,
        dynamic,
        THRESHOLDS,
        policy_sensitivity={"status": "FAIL", "selected_max_boost": 0.015},
    )
    assert decision["status"] == "HOLD"
    assert decision["gates"]["policy_sensitivity"] is False
