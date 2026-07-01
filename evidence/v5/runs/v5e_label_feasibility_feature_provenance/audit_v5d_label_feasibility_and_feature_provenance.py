from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


decision_path = Path(sys.argv[1]).resolve()
completion_path = Path(sys.argv[2]).resolve()
output_root = Path(sys.argv[3]).resolve()
expected_head = sys.argv[4]

EXPECTED_ACTIONS = (
    "PLACE_AT_1",
    "PLACE_AT_2",
    "PLACE_AT_3",
    "PLACE_AT_5",
    "PLACE_AT_10",
)

EXPECTED_RANKS = (1, 2, 3, 5, 10)

IDENTIFIER_COLUMNS = {
    "seed",
    "proposal_index",
    "query_id",
    "product_id",
}

REQUIRED_ACTION_COLUMNS = {
    *IDENTIFIER_COLUMNS,
    "action",
    "requested_rank",
    "achieved_rank",
    "relevant_discovery_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "mean_scenario_replication_utility_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
}

REQUIRED_PROPOSAL_COLUMNS = {
    *IDENTIFIER_COLUMNS,
    "teacher_or_oracle_columns_used",
}

FORBIDDEN_PROPOSAL_COLUMNS = {
    "relevance",
    "teacher_selected",
    "teacher_action",
    "teacher_label",
    "oracle_action",
    "oracle_target_score",
    "target_score",
    "final_score",
    "qrsbt_boost",
    "required_boost",
    "achieved_rank",
    "requested_rank",
    "relevant_discovery_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "mean_scenario_replication_utility_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
}

OUTCOME_COLUMNS = (
    "relevant_discovery_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "mean_scenario_replication_utility_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")

    temp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def bool_false(value: Any) -> bool:
    return value is False or isinstance(value, np.bool_) and not bool(value)


decision = json.loads(
    decision_path.read_text(encoding="utf-8-sig")
)

completion = json.loads(
    completion_path.read_text(encoding="utf-8-sig")
)

if (
    decision.get("status")
    != "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
):
    raise RuntimeError(
        "Unexpected V5-D corpus decision status."
    )

if decision.get("baseline_commit") != expected_head:
    raise RuntimeError(
        "V5-D corpus decision baseline mismatch."
    )

if (
    completion.get("status")
    != "V5D_FINAL_TRAINING_CORPUS_VERIFIED_WITH_BOOLEAN_CONFIRMATION_CONTRACT"
):
    raise RuntimeError(
        "Unexpected V5-D completion contract status."
    )

if completion.get("baseline_commit") != expected_head:
    raise RuntimeError(
        "V5-D completion contract baseline mismatch."
    )

if int(decision.get("action_count", -1)) != 720:
    raise RuntimeError(
        "V5-D corpus decision action count mismatch."
    )

if int(decision.get("proposal_count", -1)) != 144:
    raise RuntimeError(
        "V5-D corpus decision proposal count mismatch."
    )

if decision.get("model_trained") is not False:
    raise RuntimeError(
        "V5-D corpus unexpectedly fitted a model."
    )

if decision.get("threshold_selected") is not False:
    raise RuntimeError(
        "V5-D corpus unexpectedly selected a threshold."
    )

if completion.get("confirmation_seeds_executed") is not False:
    raise RuntimeError(
        "Completion contract indicates confirmation execution."
    )

if completion.get("calibration_seeds_executed") not in ([], None):
    raise RuntimeError(
        "Completion contract indicates calibration execution."
    )

action_effects_path = Path(
    decision["action_effects_path"]
).resolve()

proposal_manifest_path = Path(
    decision["proposal_manifest_path"]
).resolve()

daily_path = Path(
    decision["daily_path"]
).resolve()

summary_path = Path(
    decision["summary_by_action_path"]
).resolve()

for path in (
    action_effects_path,
    proposal_manifest_path,
    daily_path,
    summary_path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Required V5-D artifact missing: {path}"
        )

actions = pd.read_csv(action_effects_path)
proposals = pd.read_csv(proposal_manifest_path)
daily = pd.read_csv(daily_path)
summary = pd.read_csv(summary_path)

require_columns(
    actions,
    REQUIRED_ACTION_COLUMNS,
    label="V5-D action labels",
)

require_columns(
    proposals,
    REQUIRED_PROPOSAL_COLUMNS,
    label="V5-D proposal manifest",
)

for column in OUTCOME_COLUMNS:
    actions[column] = pd.to_numeric(
        actions[column],
        errors="raise",
    )

if not np.isfinite(
    actions.loc[:, OUTCOME_COLUMNS].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Action-effect labels contain non-finite outcomes."
    )

if len(actions) != 720:
    raise RuntimeError(
        f"Expected 720 action labels, found {len(actions)}."
    )

if len(proposals) != 144:
    raise RuntimeError(
        f"Expected 144 proposals, found {len(proposals)}."
    )

if proposals.duplicated(
    ["seed", "proposal_index"]
).any():
    raise RuntimeError(
        "Proposal manifest contains duplicate seed/proposal keys."
    )

if actions.duplicated(
    ["seed", "proposal_index", "action"]
).any():
    raise RuntimeError(
        "Action label table contains duplicate context-action keys."
    )

if tuple(
    sorted(
        str(value)
        for value in actions["action"].unique()
    )
) != tuple(sorted(EXPECTED_ACTIONS)):
    raise RuntimeError(
        "Action space differs from the pre-registered V5 schema."
    )

if tuple(
    sorted(
        int(value)
        for value in actions["requested_rank"].unique()
    )
) != EXPECTED_RANKS:
    raise RuntimeError(
        "Requested ranks differ from the pre-registered V5 schema."
    )

if not actions["requested_rank"].eq(
    actions["achieved_rank"]
).all():
    raise RuntimeError(
        "At least one V5-D action violated exact rank placement."
    )

if not all(
    bool_false(value)
    for value in proposals[
        "teacher_or_oracle_columns_used"
    ].tolist()
):
    raise RuntimeError(
        "Proposal manifest indicates teacher or oracle input usage."
    )

forbidden_present = sorted(
    FORBIDDEN_PROPOSAL_COLUMNS
    & set(proposals.columns)
)

if forbidden_present:
    raise RuntimeError(
        "Outcome or oracle fields appeared in proposal manifest: "
        f"{forbidden_present}"
    )

context_key = [
    "seed",
    "proposal_index",
    "query_id",
    "product_id",
]

joined = actions.merge(
    proposals,
    on=context_key,
    how="inner",
    validate="many_to_one",
    suffixes=("", "_proposal"),
)

if len(joined) != len(actions):
    raise RuntimeError(
        "Action labels do not join one-to-one with proposals."
    )

seed_context_counts = (
    proposals.groupby("seed", as_index=False)
    .agg(
        proposal_count=("proposal_index", "size"),
        unique_query_count=("query_id", "nunique"),
    )
)

if len(seed_context_counts) != 18:
    raise RuntimeError(
        "Final corpus does not contain exactly 18 training seeds."
    )

if not seed_context_counts["proposal_count"].eq(8).all():
    raise RuntimeError(
        "At least one seed does not have exactly eight contexts."
    )

if not seed_context_counts["unique_query_count"].eq(8).all():
    raise RuntimeError(
        "At least one seed contains duplicate sampled query IDs."
    )

context_action_counts = (
    actions.groupby(
        ["seed", "proposal_index"],
        as_index=False,
    )
    .agg(
        action_count=("action", "size"),
        unique_action_count=("action", "nunique"),
    )
)

if not context_action_counts["action_count"].eq(5).all():
    raise RuntimeError(
        "At least one context does not contain five action labels."
    )

if not context_action_counts["unique_action_count"].eq(5).all():
    raise RuntimeError(
        "At least one context has duplicated action labels."
    )

daily_context_action_counts = (
    daily.groupby(
        ["seed", "proposal_index", "action"],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "daily_row_count"})
)

daily_join = actions.loc[
    :,
    ["seed", "proposal_index", "action"],
].merge(
    daily_context_action_counts,
    on=["seed", "proposal_index", "action"],
    how="left",
    validate="one_to_one",
)

if daily_join["daily_row_count"].isna().any():
    raise RuntimeError(
        "At least one action label lacks daily replay rows."
    )

if not daily_join["daily_row_count"].eq(150).all():
    raise RuntimeError(
        "Daily replay rows per action are not uniformly 150."
    )

numeric_proposal_columns = [
    column
    for column in proposals.columns
    if (
        column not in IDENTIFIER_COLUMNS
        and column != "teacher_or_oracle_columns_used"
        and pd.api.types.is_numeric_dtype(proposals[column])
    )
]

feature_inventory_rows = []

for column in numeric_proposal_columns:
    values = pd.to_numeric(
        proposals[column],
        errors="coerce",
    )

    finite_values = values[
        np.isfinite(values.to_numpy(dtype=float))
    ]

    feature_inventory_rows.append(
        {
            "feature": column,
            "dtype": str(proposals[column].dtype),
            "row_count": int(len(values)),
            "missing_count": int(values.isna().sum()),
            "finite_count": int(len(finite_values)),
            "unique_count": int(finite_values.nunique()),
            "constant": bool(
                finite_values.nunique() <= 1
            ),
            "min": (
                float(finite_values.min())
                if len(finite_values)
                else None
            ),
            "p01": (
                float(finite_values.quantile(0.01))
                if len(finite_values)
                else None
            ),
            "median": (
                float(finite_values.median())
                if len(finite_values)
                else None
            ),
            "p99": (
                float(finite_values.quantile(0.99))
                if len(finite_values)
                else None
            ),
            "max": (
                float(finite_values.max())
                if len(finite_values)
                else None
            ),
        }
    )

feature_inventory = pd.DataFrame(
    feature_inventory_rows
)

categorical_proposal_columns = [
    column
    for column in proposals.columns
    if (
        column not in IDENTIFIER_COLUMNS
        and column != "teacher_or_oracle_columns_used"
        and column not in numeric_proposal_columns
    )
]

categorical_inventory_rows = []

for column in categorical_proposal_columns:
    values = proposals[column].astype("string")

    categorical_inventory_rows.append(
        {
            "feature": column,
            "dtype": str(proposals[column].dtype),
            "row_count": int(len(values)),
            "missing_count": int(values.isna().sum()),
            "unique_count": int(values.nunique(dropna=True)),
            "top_value": (
                str(values.value_counts(dropna=True).index[0])
                if values.notna().any()
                else None
            ),
            "top_value_count": (
                int(values.value_counts(dropna=True).iloc[0])
                if values.notna().any()
                else 0
            ),
        }
    )

categorical_inventory = pd.DataFrame(
    categorical_inventory_rows
)

actions = actions.copy()

actions["benefit_positive"] = (
    actions["relevant_discovery_delta"] > 0.0
)

actions["no_irrelevant_increase"] = (
    actions["irrelevant_exposure_delta"] <= 0.0
)

actions["no_false_warmup_increase"] = (
    actions["false_warmup_delta"] <= 0.0
)

actions["nonnegative_worst_utility"] = (
    actions["worst_scenario_utility_delta"] >= 0.0
)

actions["nonnegative_p10_utility"] = (
    actions[
        "p10_scenario_replication_utility_delta"
    ] >= 0.0
)

actions["strict_safe_improving"] = (
    actions["benefit_positive"]
    & actions["no_irrelevant_increase"]
    & actions["no_false_warmup_increase"]
    & actions["nonnegative_worst_utility"]
    & actions["nonnegative_p10_utility"]
)

actions["core_safe_improving"] = (
    actions["benefit_positive"]
    & actions["no_irrelevant_increase"]
    & actions["no_false_warmup_increase"]
    & actions["nonnegative_worst_utility"]
)

actions["risk_event_any"] = (
    ~actions["no_irrelevant_increase"]
    | ~actions["no_false_warmup_increase"]
    | ~actions["nonnegative_worst_utility"]
    | ~actions["nonnegative_p10_utility"]
)

by_action = (
    actions.groupby("action", as_index=False)
    .agg(
        action_count=("action", "size"),
        relevant_discovery_mean=(
            "relevant_discovery_delta",
            "mean",
        ),
        relevant_discovery_positive_rate=(
            "benefit_positive",
            "mean",
        ),
        irrelevant_exposure_mean=(
            "irrelevant_exposure_delta",
            "mean",
        ),
        irrelevant_increase_rate=(
            "no_irrelevant_increase",
            lambda value: float(1.0 - value.mean()),
        ),
        false_warmup_mean=(
            "false_warmup_delta",
            "mean",
        ),
        false_warmup_increase_rate=(
            "no_false_warmup_increase",
            lambda value: float(1.0 - value.mean()),
        ),
        mean_utility_mean=(
            "mean_scenario_replication_utility_delta",
            "mean",
        ),
        worst_utility_mean=(
            "worst_scenario_utility_delta",
            "mean",
        ),
        p10_utility_mean=(
            "p10_scenario_replication_utility_delta",
            "mean",
        ),
        strict_safe_improving_count=(
            "strict_safe_improving",
            "sum",
        ),
        core_safe_improving_count=(
            "core_safe_improving",
            "sum",
        ),
        any_risk_event_count=(
            "risk_event_any",
            "sum",
        ),
    )
)

by_seed = (
    actions.groupby("seed", as_index=False)
    .agg(
        action_count=("action", "size"),
        context_count=("proposal_index", "nunique"),
        strict_safe_improving_count=(
            "strict_safe_improving",
            "sum",
        ),
        core_safe_improving_count=(
            "core_safe_improving",
            "sum",
        ),
        benefit_positive_count=(
            "benefit_positive",
            "sum",
        ),
        risk_event_count=("risk_event_any", "sum"),
        mean_utility_mean=(
            "mean_scenario_replication_utility_delta",
            "mean",
        ),
        p10_utility_mean=(
            "p10_scenario_replication_utility_delta",
            "mean",
        ),
    )
)

context_best = (
    actions.sort_values(
        [
            "seed",
            "proposal_index",
            "mean_scenario_replication_utility_delta",
            "requested_rank",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    .groupby(
        ["seed", "proposal_index"],
        as_index=False,
    )
    .first()
)

best_action_distribution = (
    context_best.groupby(
        ["action", "requested_rank"],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "best_action_context_count"})
)

context_safety = (
    actions.groupby(
        ["seed", "proposal_index"],
        as_index=False,
    )
    .agg(
        strict_safe_action_count=(
            "strict_safe_improving",
            "sum",
        ),
        core_safe_action_count=(
            "core_safe_improving",
            "sum",
        ),
        positive_benefit_action_count=(
            "benefit_positive",
            "sum",
        ),
        any_risk_action_count=("risk_event_any", "sum"),
        action_utility_unique_count=(
            "mean_scenario_replication_utility_delta",
            "nunique",
        ),
    )
)

label_feasibility = {
    "context_count": int(len(context_safety)),
    "action_label_count": int(len(actions)),
    "seed_count": int(actions["seed"].nunique()),
    "strict_safe_improving_action_count": int(
        actions["strict_safe_improving"].sum()
    ),
    "strict_safe_improving_context_count": int(
        context_safety[
            "strict_safe_action_count"
        ].gt(0).sum()
    ),
    "core_safe_improving_action_count": int(
        actions["core_safe_improving"].sum()
    ),
    "core_safe_improving_context_count": int(
        context_safety[
            "core_safe_action_count"
        ].gt(0).sum()
    ),
    "contexts_with_any_positive_benefit_action": int(
        context_safety[
            "positive_benefit_action_count"
        ].gt(0).sum()
    ),
    "contexts_with_action_utility_variation": int(
        context_safety[
            "action_utility_unique_count"
        ].gt(1).sum()
    ),
    "contexts_with_no_risk_free_action": int(
        context_safety[
            "strict_safe_action_count"
        ].eq(0).sum()
    ),
}

modeling_recommendation = {
    "fit_single_multiclass_action_classifier_now": False,
    "fit_continuous_utility_regressor_after_audit": True,
    "fit_separate_harm_models_after_audit": True,
    "future_validation_group": "seed",
    "future_validation_protocol": (
        "Use seed-disjoint grouped folds. Do not randomly "
        "split rows from the same seed across training and test."
    ),
    "future_serving_contract": (
        "NO_PROMOTION remains the default. Choose a placement "
        "only after benefit and harm models both pass "
        "pre-registered development thresholds."
    ),
}

next_gate = (
    "Freeze the V5-E runtime feature contract and label "
    "definitions. Then run a seed-disjoint development-only "
    "multi-head model viability audit: utility regression plus "
    "separate irrelevant-exposure, false-warmup, worst-utility, "
    "and P10-risk models. Do not fit any final artifact, choose "
    "a serving threshold, or execute calibration/confirmation seeds."
)

decision_out = {
    "status": (
        "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_"
        "AUDIT_COMPLETE"
    ),
    "baseline_commit": expected_head,
    "v5d_decision_path": str(decision_path),
    "v5d_decision_sha256": sha256(decision_path),
    "completion_contract_path": str(completion_path),
    "completion_contract_sha256": sha256(completion_path),
    "action_effects_path": str(action_effects_path),
    "action_effects_sha256": sha256(action_effects_path),
    "proposal_manifest_path": str(proposal_manifest_path),
    "proposal_manifest_sha256": sha256(
        proposal_manifest_path
    ),
    "daily_path": str(daily_path),
    "daily_sha256": sha256(daily_path),
    "summary_path": str(summary_path),
    "summary_sha256": sha256(summary_path),
    "runtime_feature_columns_numeric": (
        numeric_proposal_columns
    ),
    "runtime_feature_columns_categorical": (
        categorical_proposal_columns
    ),
    "forbidden_proposal_columns_present": forbidden_present,
    "label_feasibility": label_feasibility,
    "modeling_recommendation": modeling_recommendation,
    "source_modified": False,
    "config_modified": False,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
    "next_gate": next_gate,
}

output_root.mkdir(parents=True, exist_ok=True)

feature_inventory.to_csv(
    output_root / "v5e_numeric_feature_inventory.csv",
    index=False,
)

categorical_inventory.to_csv(
    output_root / "v5e_categorical_feature_inventory.csv",
    index=False,
)

by_action.to_csv(
    output_root / "v5e_label_feasibility_by_action.csv",
    index=False,
)

by_seed.to_csv(
    output_root / "v5e_label_feasibility_by_seed.csv",
    index=False,
)

context_safety.to_csv(
    output_root / "v5e_context_safety_coverage.csv",
    index=False,
)

best_action_distribution.to_csv(
    output_root / "v5e_best_mean_utility_action_distribution.csv",
    index=False,
)

write_json(
    output_root / "v5e_label_feasibility_and_feature_provenance_decision.json",
    decision_out,
)

print("===== V5-E LABEL FEASIBILITY + FEATURE PROVENANCE AUDIT =====")
print(
    json.dumps(
        {
            "decision": decision_out,
            "label_feasibility_by_action": by_action.to_dict(
                orient="records"
            ),
            "best_mean_utility_action_distribution": (
                best_action_distribution.to_dict(
                    orient="records"
                )
            ),
        },
        indent=2,
        sort_keys=True,
        default=native,
    )
)