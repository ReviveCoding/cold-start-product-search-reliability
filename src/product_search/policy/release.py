from __future__ import annotations

import numpy as np


def evaluate_release(
    metrics: dict,
    ope: dict,
    dynamic: dict,
    thresholds: dict,
    system: dict | None = None,
    policy_sensitivity: dict | None = None,
) -> dict:
    overall_delta = metrics["final_ndcg_at_10"] - metrics["base_ndcg_at_10"]
    cold_lift = metrics["cold_ndcg_at_10_final"] - metrics["cold_ndcg_at_10_base"]
    warm_delta = metrics["warm_ndcg_at_10_final"] - metrics["warm_ndcg_at_10_base"]
    irrelevant_delta = metrics["irrelevant_exposure_final"] - metrics["irrelevant_exposure_base"]
    base_discovery = max(float(dynamic["base_relevant_discovery"]), 1.0)
    dynamic_discovery_drop_rate = max(
        0.0,
        (float(dynamic["base_relevant_discovery"]) - float(dynamic["qrsbt_relevant_discovery"]))
        / base_discovery,
    )
    dynamic_false_warmup_increase = (
        float(dynamic["qrsbt_false_warmup"]) - float(dynamic["base_false_warmup"])
    )
    dynamic_irrelevant_increase = (
        float(dynamic["qrsbt_irrelevant_exposure"])
        - float(dynamic["base_irrelevant_exposure"])
    )

    gates = {
        "overall_non_inferiority": overall_delta >= -thresholds["max_overall_ndcg_drop"],
        "overall_ci_non_inferiority": metrics["overall_ndcg_delta_ci_low"]
        >= -thresholds["max_overall_ndcg_drop_ci"],
        "cold_point_improvement": cold_lift >= thresholds["min_cold_ndcg_lift"],
        "cold_ci_non_inferiority": metrics["cold_ndcg_lift_ci_low"]
        >= thresholds["min_cold_ndcg_lift_ci_low"],
        "warm_non_inferiority": warm_delta >= -thresholds["max_warm_ndcg_drop"],
        "warm_ci_non_inferiority": metrics["warm_ndcg_delta_ci_low"]
        >= -thresholds["max_warm_ndcg_drop_ci"],
        "irrelevant_exposure_guardrail": irrelevant_delta
        <= thresholds["max_irrelevant_exposure_increase"],
        "irrelevant_ci_guardrail": metrics["irrelevant_exposure_delta_ci_high"]
        <= thresholds["max_irrelevant_exposure_ci_high"],
        "ope_effective_sample_size": ope["effective_sample_size"] >= thresholds["min_ope_ess"],
        "ope_dr_finite": bool(np.isfinite(ope["dr"])),
        "ope_dr_accuracy": ope["dr_abs_error"] <= thresholds["max_ope_dr_abs_error"],
        "ope_dr_interval_coverage": bool(ope["dr_covers_true_value"]),
        "ope_support_overlap": ope["support_overlap_rate"] >= thresholds["min_ope_support_overlap"],
        "dynamic_relevant_discovery_non_inferiority": dynamic_discovery_drop_rate
        <= thresholds["max_dynamic_relevant_discovery_drop_rate"],
        "dynamic_false_warmup_guardrail": dynamic_false_warmup_increase
        <= thresholds["max_dynamic_false_warmup_increase"],
        "dynamic_irrelevant_exposure_guardrail": dynamic_irrelevant_increase
        <= thresholds["max_dynamic_irrelevant_exposure_increase"],
        "dynamic_worst_scenario_utility": dynamic["worst_scenario_utility_delta"]
        >= thresholds["min_worst_scenario_utility_delta"],
        "dynamic_p10_replication_utility": dynamic[
            "p10_scenario_replication_utility_delta"
        ] >= thresholds["min_p10_scenario_replication_utility_delta"],
        "semantic_judgment_coverage": min(
            float(metrics.get("judgment_coverage_base", 1.0)),
            float(metrics.get("judgment_coverage_final", 1.0)),
        ) >= thresholds["min_judgment_coverage_at_10"],
        "future_behavior_discrimination": float(metrics["future_behavior_roc_auc"])
        >= thresholds["min_future_behavior_roc_auc"],
        "future_behavior_brier": float(metrics["future_behavior_brier"])
        <= thresholds["max_future_behavior_brier"],
        "future_behavior_calibration": float(metrics["future_behavior_ece"])
        <= thresholds["max_future_behavior_ece"],
        "future_logged_ranking_non_inferiority": (
            float(metrics["future_logged_final_ndcg_at_10"])
            - float(metrics["future_logged_base_ndcg_at_10"])
        ) >= -thresholds["max_future_logged_ndcg_drop"],
        "relation_eligibility_brier": float(metrics["relation_eligibility_brier"])
        <= thresholds["max_relation_eligibility_brier"],
        "relation_eligibility_calibration": float(metrics["relation_eligibility_ece"])
        <= thresholds["max_relation_eligibility_ece"],
    }
    if policy_sensitivity is not None:
        gates["policy_sensitivity"] = policy_sensitivity.get("status") == "PASS"
    if system is not None and "max_serving_p95_ms" in thresholds:
        gates["serving_latency_p95"] = (
            float(system["p95_latency_ms"]) <= float(thresholds["max_serving_p95_ms"])
        )
        gates["serving_no_fallback"] = int(system.get("fallbacks", 0)) == 0
        gates["serving_result_contract"] = bool(system.get("result_contract_pass", False))
    hold_gates = {
        "overall_non_inferiority",
        "overall_ci_non_inferiority",
        "warm_non_inferiority",
        "warm_ci_non_inferiority",
        "irrelevant_exposure_guardrail",
        "irrelevant_ci_guardrail",
        "ope_dr_finite",
        "ope_support_overlap",
        "dynamic_relevant_discovery_non_inferiority",
        "dynamic_false_warmup_guardrail",
        "dynamic_irrelevant_exposure_guardrail",
        "dynamic_p10_replication_utility",
        "serving_no_fallback",
        "serving_result_contract",
        "semantic_judgment_coverage",
        "future_behavior_discrimination",
        "future_behavior_brier",
        "future_behavior_calibration",
        "future_logged_ranking_non_inferiority",
        "relation_eligibility_brier",
        "relation_eligibility_calibration",
        "policy_sensitivity",
    }
    failed = [name for name, passed in gates.items() if not passed]
    if not failed:
        status = "LAUNCH"
    elif any(name in hold_gates for name in failed):
        status = "HOLD"
    else:
        status = "ITERATE"
    return {
        "status": status,
        "gates": gates,
        "failed_gates": failed,
        "diagnostics": {
            "overall_ndcg_delta": overall_delta,
            "cold_ndcg_lift": cold_lift,
            "warm_ndcg_delta": warm_delta,
            "irrelevant_exposure_delta": irrelevant_delta,
            "dynamic_relevant_discovery_drop_rate": dynamic_discovery_drop_rate,
            "dynamic_false_warmup_increase": dynamic_false_warmup_increase,
            "dynamic_irrelevant_exposure_increase": dynamic_irrelevant_increase,
            "ope_dr_abs_error": float(ope["dr_abs_error"]),
            "worst_scenario_utility_delta": float(dynamic["worst_scenario_utility_delta"]),
            "p10_scenario_replication_utility_delta": float(
                dynamic["p10_scenario_replication_utility_delta"]
            ),
            "judgment_coverage_at_10": min(
                float(metrics.get("judgment_coverage_base", 1.0)),
                float(metrics.get("judgment_coverage_final", 1.0)),
            ),
            "future_behavior_roc_auc": float(metrics["future_behavior_roc_auc"]),
            "future_behavior_brier": float(metrics["future_behavior_brier"]),
            "future_behavior_ece": float(metrics["future_behavior_ece"]),
            "future_logged_ndcg_delta": float(metrics["future_logged_final_ndcg_at_10"])
            - float(metrics["future_logged_base_ndcg_at_10"]),
            "relation_eligibility_brier": float(metrics["relation_eligibility_brier"]),
            "relation_eligibility_ece": float(metrics["relation_eligibility_ece"]),
            **(
                {
                    "policy_sensitivity_status": policy_sensitivity.get("status"),
                    "policy_sensitivity_selected_max_boost": float(
                        policy_sensitivity["selected_max_boost"]
                    ),
                }
                if policy_sensitivity is not None
                else {}
            ),
            **(
                {
                    "serving_p95_latency_ms": float(system["p95_latency_ms"]),
                    "serving_fallbacks": int(system.get("fallbacks", 0)),
                }
                if system is not None
                else {}
            ),
        },
    }
