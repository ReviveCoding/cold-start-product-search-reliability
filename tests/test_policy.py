import pandas as pd

from product_search.policy.gate import GateConfig, apply_coverage_overreach_gate


def _frame(**overrides):
    values = {
        "query_id": [1],
        "dense_score": [0.8],
        "qrsbt_confidence": [0.9],
        "qrsbt_support": [5],
        "qrsbt_compatibility": [0.9],
        "qrsbt_irrelevant_probability": [0.05],
        "zero_history": [1],
        "qrsbt_ctr_lower": [0.7],
        "qrsbt_purchase_lower": [0.2],
        "semantic_rank_score": [1.0],
        "behavior_score": [0.2],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_gate_blocks_incompatible_transfer():
    out = apply_coverage_overreach_gate(_frame(qrsbt_compatibility=[0.0]), GateConfig())
    assert out.qrsbt_boost.iloc[0] == 0
    assert out.gate_action.iloc[0] == "BLOCK"


def test_gate_blocks_high_irrelevant_risk():
    out = apply_coverage_overreach_gate(_frame(qrsbt_irrelevant_probability=[0.95]), GateConfig())
    assert out.qrsbt_boost.iloc[0] == 0
    assert out.gate_reason.iloc[0] == "irrelevant_risk"


def test_boost_is_bounded():
    out = apply_coverage_overreach_gate(
        _frame(qrsbt_ctr_lower=[10.0], qrsbt_purchase_lower=[10.0]),
        GateConfig(max_boost=0.25),
    )
    assert out.qrsbt_boost.iloc[0] <= 0.25
    assert out.final_score.iloc[0] <= out.base_score.iloc[0] + 0.25


def test_gate_limits_promotions_per_query_and_window():
    frame = pd.DataFrame(
        {
            "query_id": [1] * 4,
            "product_id": [10, 11, 12, 13],
            "dense_score": [0.9] * 4,
            "qrsbt_confidence": [0.9, 0.8, 0.7, 0.6],
            "qrsbt_support": [5] * 4,
            "qrsbt_compatibility": [0.9] * 4,
            "qrsbt_irrelevant_probability": [0.05] * 4,
            "qrsbt_relevance_probability": [0.9, 0.8, 0.7, 0.6],
            "zero_history": [1] * 4,
            "qrsbt_ctr_lower": [0.7] * 4,
            "qrsbt_purchase_lower": [0.2] * 4,
            "semantic_rank_score": [1.0, 0.9, 0.8, 0.7],
            "behavior_score": [0.4, 0.3, 0.2, 0.1],
        }
    )
    out = apply_coverage_overreach_gate(
        frame,
        GateConfig(top_k=1, promotion_window=1, max_promotions_per_query=1),
    )
    assert int((out.gate_action == "BOOST").sum()) == 1
    assert out.loc[out.base_rank > 2, "qrsbt_boost"].eq(0).all()


# Boundary-entry-only regression tests


def _boundary_entry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [1, 2],
            "dense_score": [0.90, 0.90],
            "semantic_rank_score": [0.90, 0.10],
            "behavior_score": [0.90, 0.10],
            "zero_history": [0, 1],
            "qrsbt_confidence": [0.00, 1.00],
            "qrsbt_support": [0, 8],
            "qrsbt_compatibility": [0.00, 1.00],
            "qrsbt_irrelevant_probability": [0.00, 0.00],
            "qrsbt_ctr_lower": [0.00, 10.00],
            "qrsbt_purchase_lower": [0.00, 10.00],
        }
    )


def test_boundary_entry_only_promotes_a_real_top_k_entry():
    out = apply_coverage_overreach_gate(
        _boundary_entry_frame(),
        GateConfig(
            top_k=1,
            promotion_window=1,
            max_boost=0.75,
            promotion_mode="boundary_entry_only",
        ),
    )

    cold = out.loc[out.product_id.eq(2)].iloc[0]

    assert cold.base_rank == 2
    assert cold.qrsbt_boost > 0
    assert cold.final_score > out.loc[out.product_id.eq(1), "final_score"].iloc[0]


def test_boundary_entry_only_rejects_an_insufficient_boost():
    out = apply_coverage_overreach_gate(
        _boundary_entry_frame(),
        GateConfig(
            top_k=1,
            promotion_window=1,
            max_boost=0.40,
            promotion_mode="boundary_entry_only",
        ),
    )

    cold = out.loc[out.product_id.eq(2)].iloc[0]

    assert cold.qrsbt_boost == 0
    assert cold.gate_reason == "no_top_k_entry"


def test_boundary_entry_only_does_not_reorder_existing_top_k_cold_item():
    frame = _boundary_entry_frame()
    frame.loc[frame.product_id.eq(2), "semantic_rank_score"] = 0.95
    frame.loc[frame.product_id.eq(2), "behavior_score"] = 0.95

    out = apply_coverage_overreach_gate(
        frame,
        GateConfig(
            top_k=1,
            promotion_window=1,
            max_boost=0.75,
            promotion_mode="boundary_entry_only",
        ),
    )

    cold = out.loc[out.product_id.eq(2)].iloc[0]

    assert cold.base_rank == 1
    assert cold.qrsbt_boost == 0
