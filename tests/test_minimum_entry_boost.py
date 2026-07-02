from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from product_search.config import validate_config
from product_search.policy.gate import GateConfig, apply_coverage_overreach_gate


def _boundary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": [10, 20, 30],
            "dense_score": [0.9, 0.9, 0.9],
            "semantic_rank_score": [0.9, 0.6, 0.1],
            "behavior_score": [0.9, 0.6, 0.1],
            "zero_history": [0, 0, 1],
            "qrsbt_confidence": [0.0, 0.0, 1.0],
            "qrsbt_support": [0, 0, 8],
            "qrsbt_compatibility": [0.0, 0.0, 1.0],
            "qrsbt_irrelevant_probability": [0.0, 0.0, 0.0],
            "qrsbt_ctr_lower": [0.0, 0.0, 1.0],
            "qrsbt_purchase_lower": [0.0, 0.0, 1.0],
        }
    )


def _rank_of(frame: pd.DataFrame, product_id: int) -> int:
    ordered = frame.sort_values(
        ["final_score", "product_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return int(ordered.index[ordered["product_id"].eq(product_id)][0]) + 1


def test_minimum_entry_uses_less_than_cap_and_enters_top_k() -> None:
    fixed = apply_coverage_overreach_gate(
        _boundary_frame(),
        GateConfig(
            top_k=2,
            promotion_window=1,
            max_boost=0.8,
            promotion_mode="boundary_entry_only",
            boost_allocation_mode="fixed_cap",
        ),
    )
    minimum = apply_coverage_overreach_gate(
        _boundary_frame(),
        GateConfig(
            top_k=2,
            promotion_window=1,
            max_boost=0.8,
            promotion_mode="boundary_entry_only",
            boost_allocation_mode="minimum_entry",
        ),
    )
    fixed_boost = float(fixed.loc[fixed.product_id.eq(30), "qrsbt_boost"].iloc[0])
    minimum_boost = float(minimum.loc[minimum.product_id.eq(30), "qrsbt_boost"].iloc[0])

    assert fixed_boost == pytest.approx(0.8)
    assert 0.0 < minimum_boost < fixed_boost
    assert _rank_of(minimum, 30) == 2


def test_minimum_entry_requires_boundary_mode_and_one_promotion() -> None:
    with pytest.raises(ValueError, match="boundary_entry_only"):
        apply_coverage_overreach_gate(
            _boundary_frame(),
            GateConfig(
                top_k=2,
                promotion_window=1,
                max_boost=0.8,
                promotion_mode="in_window",
                boost_allocation_mode="minimum_entry",
            ),
        )
    with pytest.raises(ValueError, match="max_promotions_per_query=1"):
        apply_coverage_overreach_gate(
            _boundary_frame(),
            GateConfig(
                top_k=2,
                promotion_window=1,
                max_boost=0.8,
                max_promotions_per_query=2,
                promotion_mode="boundary_entry_only",
                boost_allocation_mode="minimum_entry",
            ),
        )


def test_config_accepts_and_rejects_minimum_entry_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "smoke.yaml").read_text(encoding="utf-8"))

    accepted = deepcopy(raw)
    accepted["qrsbt"]["boost_allocation_mode"] = "minimum_entry"
    validate_config(accepted)

    invalid = deepcopy(raw)
    invalid["qrsbt"]["boost_allocation_mode"] = "unsupported"
    with pytest.raises(ValueError, match="boost_allocation_mode"):
        validate_config(invalid)


def test_minimum_entry_preserves_unreachable_boundary_rejection() -> None:
    result = apply_coverage_overreach_gate(
        _boundary_frame(),
        GateConfig(
            top_k=2,
            promotion_window=1,
            max_boost=0.1,
            promotion_mode="boundary_entry_only",
            boost_allocation_mode="minimum_entry",
        ),
    )
    candidate = result.loc[result.product_id.eq(30)].iloc[0]

    assert float(candidate["qrsbt_boost"]) == pytest.approx(0.0)
    assert bool(candidate["boundary_entry_rejected"])
    assert candidate["gate_action"] == "SHRINK"
    assert candidate["gate_reason"] == "no_top_k_entry"


def test_minimum_entry_advances_final_score_past_boundary_tie() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [1, 2],
            "dense_score": [0.9, 0.9],
            "semantic_rank_score": [0.9, 0.1],
            "behavior_score": [0.9, 0.1],
            "zero_history": [0, 1],
            "qrsbt_confidence": [0.0, 1.0],
            "qrsbt_support": [0, 8],
            "qrsbt_compatibility": [0.0, 1.0],
            "qrsbt_irrelevant_probability": [0.0, 0.0],
            "qrsbt_ctr_lower": [0.0, 1.0],
            "qrsbt_purchase_lower": [0.0, 1.0],
        }
    )
    result = apply_coverage_overreach_gate(
        frame,
        GateConfig(
            top_k=1,
            promotion_window=1,
            max_boost=0.75,
            promotion_mode="boundary_entry_only",
            boost_allocation_mode="minimum_entry",
        ),
    )
    candidate = result.loc[result.product_id.eq(2)].iloc[0]

    assert float(candidate["qrsbt_boost"]) > 0.5
    assert candidate["gate_action"] == "BOOST"
    assert _rank_of(result, 2) == 1
