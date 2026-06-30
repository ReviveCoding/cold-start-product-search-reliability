from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    intercept: float
    relevance_weight: float
    quality_weight: float
    popularity_weight: float
    novelty_bonus: float


DEFAULT_SCENARIOS = (
    SimulationScenario("conservative", -2.7, 0.85, 0.70, 0.06, 0.00),
    SimulationScenario("neutral", -2.4, 0.90, 0.80, 0.05, 0.10),
    SimulationScenario("exploratory", -2.2, 0.95, 0.80, 0.04, 0.25),
)


@dataclass
class DynamicSimulationResult:
    daily: pd.DataFrame
    summary: dict[str, float]


@dataclass(frozen=True)
class _QueryArrays:
    state_indices: np.ndarray
    product_ids: np.ndarray
    base_score: np.ndarray
    final_score: np.ndarray
    relevance: np.ndarray
    quality: np.ndarray
    zero_history: np.ndarray


def _top_indices(
    score: np.ndarray,
    product_ids: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Return top-k with a complete deterministic ordering."""

    count = min(k, len(score))

    if count == 0:
        return np.empty(0, dtype=int)

    order = np.lexsort(
        (
            product_ids.astype(int),
            -score,
        )
    )

    return order[:count]


def run_dynamic_simulation(
    ranked_frame: pd.DataFrame,
    *,
    days: int = 14,
    traffic_per_day: int = 160,
    seed: int = 42,
    replications: int = 1,
    scenarios: tuple[SimulationScenario, ...] = DEFAULT_SCENARIOS,
) -> DynamicSimulationResult:
    required = {
        "query_id",
        "product_id",
        "base_score",
        "final_score",
        "relevance",
        "quality",
        "zero_history",
    }
    missing = sorted(required - set(ranked_frame.columns))
    if missing:
        raise ValueError(f"Dynamic simulation missing columns: {missing}")
    if days < 1 or traffic_per_day < 1 or replications < 1:
        raise ValueError("days, traffic_per_day, and replications must be positive")

    frame = ranked_frame.copy()
    if frame.duplicated(["query_id", "product_id"]).any():
        raise ValueError("Dynamic simulation requires unique query-product candidates")

    frame = frame.sort_values(
        ["query_id", "product_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    product_ids = np.sort(frame.product_id.astype(int).unique())
    product_to_state = {int(pid): idx for idx, pid in enumerate(product_ids)}
    query_arrays: dict[int, _QueryArrays] = {}
    for query_id, group in frame.groupby("query_id", sort=True):
        pids = group.product_id.to_numpy(dtype=int)
        query_arrays[int(query_id)] = _QueryArrays(
            state_indices=np.asarray([product_to_state[int(pid)] for pid in pids], dtype=int),
            product_ids=pids,
            base_score=group.base_score.to_numpy(dtype=float),
            final_score=group.final_score.to_numpy(dtype=float),
            relevance=group.relevance.to_numpy(dtype=float),
            quality=group.quality.to_numpy(dtype=float),
            zero_history=group.zero_history.to_numpy(dtype=int),
        )

    query_ids = np.asarray(sorted(query_arrays), dtype=int)
    policies = {"base": "base_score", "qrsbt_gate": "final_score"}
    rows: list[dict] = []
    for replication in range(replications):
        # A large odd stride keeps deterministic replications independent while preserving exact
        # common random numbers between the two policies within each replication.
        rng = np.random.default_rng(seed + 1009 * replication)
        traffic_queries = rng.choice(
            query_ids, size=(len(scenarios), days, traffic_per_day), replace=True
        )
        click_uniform = rng.random((len(scenarios), days, traffic_per_day, 10))
        for scenario_index, scenario in enumerate(scenarios):
            product_state = {policy: np.zeros(len(product_ids), dtype=float) for policy in policies}
            for day in range(days):
                for policy, score_name in policies.items():
                    relevant_discovery = 0
                    irrelevant_exposure = 0
                    false_warmup = 0
                    cold_to_warm = 0
                    clicks = 0
                    state = product_state[policy]
                    for event_index, query_id in enumerate(traffic_queries[scenario_index, day]):
                        query = query_arrays[int(query_id)]
                        static_score = getattr(query, score_name)
                        popularity = state[query.state_indices]
                        dynamic_score = static_score + scenario.popularity_weight * np.log1p(
                            popularity
                        )
                        top_local = _top_indices(dynamic_score, query.product_ids, 10)
                        if top_local.size == 0:
                            continue
                        top_state_indices = query.state_indices[top_local]
                        top_relevance = query.relevance[top_local]
                        top_quality = query.quality[top_local]
                        top_cold = query.zero_history[top_local]
                        positions = np.arange(1, len(top_local) + 1, dtype=float)
                        examination = 1.0 / np.log2(positions + 1.5)
                        novelty = scenario.novelty_bonus * top_cold
                        click_logit = (
                            scenario.intercept
                            + scenario.relevance_weight * top_relevance
                            + scenario.quality_weight * top_quality
                            + np.log(examination)
                            + novelty
                        )
                        click_probability = 1.0 / (1.0 + np.exp(-click_logit))
                        clicked = (
                            click_uniform[scenario_index, day, event_index, : len(top_local)]
                            < click_probability
                        )
                        before = state[top_state_indices].copy()
                        state[top_state_indices] += clicked.astype(float)
                        after = state[top_state_indices]

                        clicks += int(clicked.sum())
                        relevant_discovery += int(
                            ((top_cold == 1) & (top_relevance >= 2) & clicked).sum()
                        )
                        irrelevant_exposure += int((top_relevance == 0).sum())
                        false_warmup += int(
                            (
                                (top_cold == 1) & (top_relevance == 0) & (before < 3) & (after >= 3)
                            ).sum()
                        )
                        cold_to_warm += int(
                            (
                                (top_cold == 1) & (top_relevance >= 2) & (before < 3) & (after >= 3)
                            ).sum()
                        )

                    utility = relevant_discovery - 0.20 * irrelevant_exposure - 2.0 * false_warmup
                    rows.append(
                        {
                            "replication": replication,
                            "scenario": scenario.name,
                            "day": day + 1,
                            "policy": policy,
                            "relevant_discovery": relevant_discovery,
                            "irrelevant_exposure": irrelevant_exposure,
                            "false_warmup": false_warmup,
                            "cold_to_warm": cold_to_warm,
                            "clicks": clicks,
                            "utility": utility,
                        }
                    )

    daily = pd.DataFrame(rows)
    replication_policy = daily.groupby(["replication", "policy"], as_index=False).sum(
        numeric_only=True
    )
    aggregate = replication_policy.groupby("policy").mean(numeric_only=True)
    scenario_replication = daily.groupby(["replication", "scenario", "policy"], as_index=False).sum(
        numeric_only=True
    )
    deltas: list[dict[str, float | str | int]] = []
    for (replication, scenario_name), group in scenario_replication.groupby(
        ["replication", "scenario"], sort=False
    ):
        view = group.set_index("policy")
        deltas.append(
            {
                "replication": int(replication),
                "scenario": str(scenario_name),
                "relevant_delta": float(
                    view.loc["qrsbt_gate", "relevant_discovery"]
                    - view.loc["base", "relevant_discovery"]
                ),
                "irrelevant_delta": float(
                    view.loc["qrsbt_gate", "irrelevant_exposure"]
                    - view.loc["base", "irrelevant_exposure"]
                ),
                "utility_delta": float(
                    view.loc["qrsbt_gate", "utility"] - view.loc["base", "utility"]
                ),
            }
        )
    delta_frame = pd.DataFrame(deltas)
    scenario_means = delta_frame.groupby("scenario").mean(numeric_only=True)
    summary = {
        "base_relevant_discovery": float(aggregate.loc["base", "relevant_discovery"]),
        "qrsbt_relevant_discovery": float(aggregate.loc["qrsbt_gate", "relevant_discovery"]),
        "base_irrelevant_exposure": float(aggregate.loc["base", "irrelevant_exposure"]),
        "qrsbt_irrelevant_exposure": float(aggregate.loc["qrsbt_gate", "irrelevant_exposure"]),
        "base_false_warmup": float(aggregate.loc["base", "false_warmup"]),
        "qrsbt_false_warmup": float(aggregate.loc["qrsbt_gate", "false_warmup"]),
        "base_cold_to_warm": float(aggregate.loc["base", "cold_to_warm"]),
        "qrsbt_cold_to_warm": float(aggregate.loc["qrsbt_gate", "cold_to_warm"]),
        "base_utility": float(aggregate.loc["base", "utility"]),
        "qrsbt_utility": float(aggregate.loc["qrsbt_gate", "utility"]),
        "worst_scenario_relevant_delta": float(scenario_means.relevant_delta.min()),
        "worst_scenario_irrelevant_delta": float(scenario_means.irrelevant_delta.max()),
        "worst_scenario_utility_delta": float(scenario_means.utility_delta.min()),
        "p10_scenario_replication_utility_delta": float(delta_frame.utility_delta.quantile(0.10)),
        "mean_scenario_replication_utility_delta": float(delta_frame.utility_delta.mean()),
        "scenario_count": float(len(scenarios)),
        "replications": float(replications),
    }
    return DynamicSimulationResult(daily=daily, summary=summary)
