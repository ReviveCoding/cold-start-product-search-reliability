from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateConfig:
    semantic_threshold: float = 0.18
    confidence_threshold: float = 0.20
    compatibility_threshold: float = 0.45
    irrelevant_risk_threshold: float = 0.45
    min_support: int = 2
    max_boost: float = 0.015
    semantic_weight: float = 0.65
    behavior_weight: float = 0.35
    top_k: int = 10
    promotion_window: int = 5
    max_promotions_per_query: int = 1
    promotion_mode: str = "in_window"


def _percentile_by_query(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("query_id", sort=False)[column].rank(method="average", pct=True)


def compose_base_score(frame: pd.DataFrame, config: GateConfig) -> pd.DataFrame:
    result = frame.copy()
    semantic_component = _percentile_by_query(result, "semantic_rank_score")
    behavior_component = _percentile_by_query(result, "behavior_score")
    total_weight = max(config.semantic_weight + config.behavior_weight, 1e-12)
    result["semantic_component"] = semantic_component
    result["behavior_component"] = behavior_component
    result["base_score"] = (
        config.semantic_weight * semantic_component + config.behavior_weight * behavior_component
    ) / total_weight
    return result


def _canonical_descending_rank(
    frame: pd.DataFrame,
    score_column: str,
) -> pd.Series:
    """Return one-based query ranks with an explicit stable tie-break."""

    required = {"query_id", score_column}
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(f"Canonical ranking missing columns: {missing}")

    working = pd.DataFrame(
        {
            "query_id": frame["query_id"].to_numpy(),
            "_score": frame[score_column].to_numpy(),
            "_row_position": np.arange(
                len(frame),
                dtype=np.int64,
            ),
        },
        index=frame.index,
    )

    if "product_id" in frame.columns:
        product_id = pd.to_numeric(
            frame["product_id"],
            errors="raise",
        )

        if product_id.isna().any():
            raise ValueError("Canonical ranking requires non-null product_id values.")

        working["_tie_break"] = product_id.astype("int64")
    else:
        fingerprint_columns = sorted(column for column in frame.columns if column != score_column)

        working["_tie_break"] = pd.util.hash_pandas_object(
            frame.loc[:, fingerprint_columns],
            index=False,
        ).astype("uint64")

    ordered = working.sort_values(
        [
            "query_id",
            "_score",
            "_tie_break",
            "_row_position",
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()

    ordered["_canonical_rank"] = ordered.groupby("query_id", sort=False).cumcount().add(1)

    rank_values = np.empty(
        len(working),
        dtype=np.int64,
    )

    rank_values[
        ordered["_row_position"].to_numpy(
            dtype=np.int64,
        )
    ] = ordered["_canonical_rank"].to_numpy(
        dtype=np.int64,
    )

    return pd.Series(
        rank_values,
        index=frame.index,
        dtype="int64",
    )


def apply_coverage_overreach_gate(frame: pd.DataFrame, config: GateConfig) -> pd.DataFrame:
    required = {
        "dense_score",
        "semantic_rank_score",
        "behavior_score",
        "zero_history",
        "qrsbt_confidence",
        "qrsbt_support",
        "qrsbt_compatibility",
        "qrsbt_irrelevant_probability",
        "qrsbt_ctr_lower",
        "qrsbt_purchase_lower",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Coverage-overreach gate missing columns: {missing}")
    if config.promotion_mode not in {"in_window", "boundary_entry_only"}:
        raise ValueError("promotion_mode must be one of: in_window, boundary_entry_only")

    result = compose_base_score(frame, config)
    cold = result["zero_history"] == 1
    semantic_ok = result["dense_score"] >= config.semantic_threshold
    confidence_ok = result["qrsbt_confidence"] >= config.confidence_threshold
    support_ok = result["qrsbt_support"] >= config.min_support
    compatibility_ok = result["qrsbt_compatibility"] >= config.compatibility_threshold
    relevance_risk_ok = result["qrsbt_irrelevant_probability"] <= config.irrelevant_risk_threshold
    eligible = (
        cold & semantic_ok & confidence_ok & support_ok & compatibility_ok & relevance_risk_ok
    )

    conservative_utility = 0.75 * result["qrsbt_ctr_lower"] + 0.25 * result["qrsbt_purchase_lower"]
    raw_boost = result["qrsbt_confidence"] * np.maximum(conservative_utility, 0.0)
    proposed_boost = np.where(eligible, np.minimum(raw_boost, config.max_boost), 0.0)
    result["base_rank"] = _canonical_descending_rank(
        result,
        "base_score",
    )
    if config.promotion_mode == "boundary_entry_only":
        in_promotion_window = result["base_rank"].gt(config.top_k) & result["base_rank"].le(
            config.top_k + config.promotion_window
        )
    else:
        in_promotion_window = result["base_rank"] <= (config.top_k + config.promotion_window)

    candidate_for_promotion = eligible & in_promotion_window & (proposed_boost > 0)

    # A bounded intervention budget prevents a large set of cold candidates from
    # simultaneously displacing warm, highly relevant products.  Selection is
    # deterministic so batch and serving paths remain reproducible.
    result["_proposed_boost"] = proposed_boost
    result["_promotion_priority"] = (
        result["_proposed_boost"]
        + 1e-3 * result.get("qrsbt_relevance_probability", 0.0)
        + 1e-6 * result["qrsbt_confidence"]
    )
    selected = pd.Series(False, index=result.index)
    eligible_rows = result.loc[candidate_for_promotion].copy()
    if not eligible_rows.empty:
        sort_columns = ["query_id", "_promotion_priority"]
        ascending = [True, False]
        if "product_id" in eligible_rows.columns:
            sort_columns.append("product_id")
            ascending.append(True)
        eligible_rows = eligible_rows.sort_values(
            sort_columns,
            ascending=ascending,
            kind="mergesort",
        )
        chosen_index = (
            eligible_rows.groupby("query_id", sort=False)
            .head(config.max_promotions_per_query)
            .index
        )
        selected.loc[chosen_index] = True

    boundary_entry_rejected = pd.Series(False, index=result.index)
    if config.promotion_mode == "boundary_entry_only":
        tentative_score = result["base_score"] + np.where(
            selected,
            proposed_boost,
            0.0,
        )
        tentative_rank = _canonical_descending_rank(
            result.assign(
                _tentative_score=tentative_score,
            ),
            "_tentative_score",
        )
        boundary_entry_rejected = selected & tentative_rank.gt(config.top_k)
        selected = selected & ~boundary_entry_rejected

    result["boundary_entry_rejected"] = boundary_entry_rejected
    result["qrsbt_boost"] = np.where(selected, proposed_boost, 0.0)

    blocked = cold & (~compatibility_ok | ~relevance_risk_ok | ~semantic_ok)
    promotion_budgeted = candidate_for_promotion & ~selected
    outside_window = eligible & ~in_promotion_window
    shrunk = (
        cold
        & ~blocked
        & (result.qrsbt_support > 0)
        & (~eligible | promotion_budgeted | outside_window)
    )
    fallback = cold & ~blocked & (result.qrsbt_support == 0)
    result["gate_action"] = np.select(
        [selected, blocked, shrunk, fallback],
        ["BOOST", "BLOCK", "SHRINK", "FALLBACK"],
        default="NATIVE",
    )
    result["gate_reason"] = np.select(
        [
            selected,
            cold & ~semantic_ok,
            cold & ~relevance_risk_ok,
            cold & ~compatibility_ok,
            cold & ~support_ok,
            cold & ~confidence_ok,
            outside_window,
            promotion_budgeted,
            fallback,
        ],
        [
            "eligible_reliable_transfer",
            "semantic_threshold",
            "irrelevant_risk",
            "compatibility",
            "insufficient_support",
            "low_confidence",
            "outside_promotion_window",
            "promotion_budget",
            "no_substitute_support",
        ],
        default="native_history",
    )
    result = result.drop(columns=["_proposed_boost", "_promotion_priority"])
    if config.promotion_mode == "boundary_entry_only":
        result.loc[boundary_entry_rejected, "gate_reason"] = "no_top_k_entry"

    result["final_score"] = result["base_score"] + result["qrsbt_boost"]
    return result
