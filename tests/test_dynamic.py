import pandas as pd

from product_search.simulation.dynamic import run_dynamic_simulation


def _frame():
    rows = []
    for query_id in range(3):
        for product_id in range(12):
            rows.append(
                {
                    "query_id": query_id,
                    "product_id": product_id,
                    "base_score": 1 - product_id / 20,
                    "final_score": 1 - product_id / 20,
                    "relevance": 3 if product_id < 3 else (1 if product_id < 8 else 0),
                    "quality": 0.7,
                    "zero_history": int(product_id % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_common_random_numbers_make_identical_policies_identical():
    result = run_dynamic_simulation(_frame(), days=2, traffic_per_day=20, seed=9)
    summary = result.summary
    assert summary["base_relevant_discovery"] == summary["qrsbt_relevant_discovery"]
    assert summary["base_irrelevant_exposure"] == summary["qrsbt_irrelevant_exposure"]
    assert summary["base_false_warmup"] == summary["qrsbt_false_warmup"]
    assert summary["worst_scenario_utility_delta"] == 0


def test_dynamic_simulation_is_seed_deterministic():
    left = run_dynamic_simulation(_frame(), days=3, traffic_per_day=24, seed=21)
    right = run_dynamic_simulation(_frame(), days=3, traffic_per_day=24, seed=21)
    assert left.summary == right.summary
    pd.testing.assert_frame_equal(left.daily, right.daily, check_exact=True)


def test_dynamic_multi_replication_summary_uses_scenario_means_and_tail_metric():
    result = run_dynamic_simulation(
        _frame(), days=2, traffic_per_day=20, seed=9, replications=3
    )
    assert result.summary["replications"] == 3.0
    assert "p10_scenario_replication_utility_delta" in result.summary
    assert set(result.daily["replication"]) == {0, 1, 2}
