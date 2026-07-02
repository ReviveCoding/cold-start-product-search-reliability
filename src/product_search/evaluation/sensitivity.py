from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..evaluation.metrics import ranking_report
from ..policy.gate import GateConfig, apply_coverage_overreach_gate
from ..simulation.dynamic import run_dynamic_simulation

DEFAULT_BOOSTS = (
    0.003,
    0.005,
    0.008,
    0.010,
    0.01125,
    0.012,
    0.0125,
    0.01275,
    0.013,
    0.01325,
    0.0135,
    0.015,
)
DEFAULT_DYNAMIC_FINALISTS = (
    0.01125,
    0.012,
    0.0125,
    0.01275,
    0.013,
    0.01325,
)


def gate_config_from_raw(raw: dict) -> GateConfig:
    qcfg = raw["qrsbt"]
    rank_cfg = raw["ranking"]
    return GateConfig(
        semantic_threshold=float(qcfg["semantic_threshold"]),
        confidence_threshold=float(qcfg["confidence_threshold"]),
        compatibility_threshold=float(qcfg["compatibility_threshold"]),
        irrelevant_risk_threshold=float(qcfg["irrelevant_risk_threshold"]),
        min_support=int(qcfg["min_support"]),
        max_boost=float(qcfg["max_boost"]),
        semantic_weight=float(rank_cfg["semantic_weight"]),
        behavior_weight=float(rank_cfg["behavior_weight"]),
        top_k=int(raw["retrieval"]["final_k"]),
        promotion_window=int(qcfg["promotion_window"]),
        max_promotions_per_query=int(qcfg["max_promotions_per_query"]),
        promotion_mode=str(qcfg.get("promotion_mode", "in_window")),
        boost_allocation_mode=str(qcfg.get("boost_allocation_mode", "fixed_cap")),
    )


def parse_policy_values(value: str, *, label: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or any(item < 0 or item > 1 for item in values):
        raise ValueError(f"{label} must contain one or more values in [0, 1]")
    return values


def evaluate_policy_sensitivity(
    frame: pd.DataFrame,
    raw_config: dict,
    *,
    boosts: tuple[float, ...] = DEFAULT_BOOSTS,
    dynamic_finalists: tuple[float, ...] = DEFAULT_DYNAMIC_FINALISTS,
    bootstrap_samples: int = 300,
) -> tuple[pd.DataFrame, dict]:
    if not set(dynamic_finalists).issubset(set(boosts)):
        raise ValueError("Every dynamic finalist must also appear in the static boost grid")
    base_gate = gate_config_from_raw(raw_config)
    simulation = raw_config["simulation"]
    seed = int(raw_config.get("seed", 42))
    rows: list[dict[str, float]] = []
    for boost in boosts:
        candidate = apply_coverage_overreach_gate(frame, replace(base_gate, max_boost=boost))
        static = ranking_report(
            candidate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        dynamic = None
        if boost in dynamic_finalists:
            dynamic = run_dynamic_simulation(
                candidate,
                days=int(simulation["days"]),
                traffic_per_day=int(simulation["traffic_per_day"]),
                seed=seed,
                replications=int(simulation["replications"]),
            ).summary
        rows.append(
            {
                "max_boost": boost,
                "overall_ndcg_delta": static["final_ndcg_at_10"] - static["base_ndcg_at_10"],
                "cold_ndcg_delta": static["cold_ndcg_at_10_final"] - static["cold_ndcg_at_10_base"],
                "cold_ndcg_ci_low": static["cold_ndcg_lift_ci_low"],
                "cold_ndcg_ci_high": static["cold_ndcg_lift_ci_high"],
                "warm_ndcg_delta": static["warm_ndcg_at_10_final"] - static["warm_ndcg_at_10_base"],
                "warm_ndcg_ci_low": static["warm_ndcg_delta_ci_low"],
                "warm_ndcg_ci_high": static["warm_ndcg_delta_ci_high"],
                "irrelevant_exposure_delta": static["irrelevant_exposure_final"]
                - static["irrelevant_exposure_base"],
                "cold_relevant_exposure_delta": static["cold_relevant_exposure_final"]
                - static["cold_relevant_exposure_base"],
                "dynamic_irrelevant_exposure_delta": (
                    dynamic["qrsbt_irrelevant_exposure"] - dynamic["base_irrelevant_exposure"]
                    if dynamic is not None
                    else float("nan")
                ),
                "mean_scenario_replication_utility_delta": (
                    dynamic["mean_scenario_replication_utility_delta"]
                    if dynamic is not None
                    else float("nan")
                ),
                "worst_scenario_utility_delta": (
                    dynamic["worst_scenario_utility_delta"] if dynamic is not None else float("nan")
                ),
                "p10_scenario_replication_utility_delta": (
                    dynamic["p10_scenario_replication_utility_delta"]
                    if dynamic is not None
                    else float("nan")
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values("max_boost").reset_index(drop=True)
    selected = float(raw_config["qrsbt"]["max_boost"])
    selected_rows = result[result.max_boost.sub(selected).abs() < 1e-12]
    if selected_rows.empty:
        raise RuntimeError("Configured qrsbt.max_boost is not present in the sensitivity grid")
    selected_metrics = selected_rows.iloc[0].to_dict()
    dynamic_keys = (
        "dynamic_irrelevant_exposure_delta",
        "mean_scenario_replication_utility_delta",
        "worst_scenario_utility_delta",
        "p10_scenario_replication_utility_delta",
    )
    if any(not np.isfinite(float(selected_metrics[key])) for key in dynamic_keys):
        raise RuntimeError("Configured qrsbt.max_boost must be a dynamic finalist")

    release = raw_config["release"]
    passed = (
        float(selected_metrics["cold_ndcg_delta"]) >= float(release["min_cold_ndcg_lift"])
        and float(selected_metrics["cold_ndcg_ci_low"])
        >= float(release["min_cold_ndcg_lift_ci_low"])
        and float(selected_metrics["warm_ndcg_delta"]) >= -float(release["max_warm_ndcg_drop"])
        and float(selected_metrics["irrelevant_exposure_delta"])
        <= float(release["max_irrelevant_exposure_increase"])
        and float(selected_metrics["dynamic_irrelevant_exposure_delta"])
        <= float(release["max_dynamic_irrelevant_exposure_increase"])
        and float(selected_metrics["worst_scenario_utility_delta"])
        >= float(release["min_worst_scenario_utility_delta"])
        and float(selected_metrics["p10_scenario_replication_utility_delta"])
        >= float(release["min_p10_scenario_replication_utility_delta"])
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "selected_max_boost": selected,
        "static_candidates": len(result),
        "dynamic_finalists": list(dynamic_finalists),
        "bootstrap_samples": int(bootstrap_samples),
        "dynamic_days": int(simulation["days"]),
        "dynamic_traffic_per_day": int(simulation["traffic_per_day"]),
        "dynamic_replications": int(simulation["replications"]),
        "dynamic_scenarios": 3,
        "selected_metrics": {key: float(value) for key, value in selected_metrics.items()},
    }
    return result, summary


def sensitivity_markdown(frame: pd.DataFrame, selected: float) -> str:
    def number(value: float, *, digits: int = 6) -> str:
        return "—" if not np.isfinite(value) else f"{value:+.{digits}f}"

    lines = [
        "# Q-RSBT Policy Sensitivity",
        "",
        "The static sweep changes only the bounded `max_boost` policy parameter. Full",
        "release-scale dynamic simulation is limited to Pareto finalists. Every candidate",
        "uses the same frozen candidate evidence and deterministic random seeds.",
        "",
        "| max_boost | cold NDCG delta | warm NDCG delta | cold exposure delta | mean utility delta | worst utility delta | p10 utility delta | selected |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            f"{row.max_boost:.3f} | {number(row.cold_ndcg_delta)} | "
            f"{number(row.warm_ndcg_delta)} | "
            f"{number(row.cold_relevant_exposure_delta)} | "
            f"{number(row.mean_scenario_replication_utility_delta, digits=2)} | "
            f"{number(row.worst_scenario_utility_delta, digits=2)} | "
            f"{number(row.p10_scenario_replication_utility_delta, digits=2)} | "
            f"{'yes' if abs(row.max_boost - selected) < 1e-12 else 'no'} |"
        )
    lines.extend(
        [
            "",
            "A dash means the setting was statically dominated and was not sent through the",
            "expensive full dynamic replay.",
            "",
        ]
    )
    return "\n".join(lines)
