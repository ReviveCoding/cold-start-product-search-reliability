from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


v5d_decision_path = Path(sys.argv[1]).resolve()
v5e_decision_path = Path(sys.argv[2]).resolve()
v5g1_decision_path = Path(sys.argv[3]).resolve()
v5h_decision_path = Path(sys.argv[4]).resolve()
output_root = Path(sys.argv[5]).resolve()
expected_head = sys.argv[6]

ACTION_KEYS = (
    "seed",
    "proposal_index",
    "action",
)

CONTEXT_KEYS = (
    "seed",
    "proposal_index",
)

EXPECTED_UTILITY_COLUMN = (
    "mean_scenario_replication_utility_delta"
)

ACTION_ORDER = (
    "PLACE_AT_1",
    "PLACE_AT_2",
    "PLACE_AT_3",
    "PLACE_AT_5",
    "PLACE_AT_10",
)

SCENARIO_COLUMN = "scenario"
REPLICATION_COLUMN = "replication"
DAY_COLUMN = "day"
POLICY_COLUMN = "policy"
DAILY_UTILITY_COLUMN = "utility"

TOLERANCE = 1e-8


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
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def validate_decision(
    decision: dict[str, Any],
    *,
    label: str,
    expected_status: str,
) -> None:
    if decision.get("status") != expected_status:
        raise RuntimeError(
            f"{label}: unexpected status {decision.get('status')}"
        )

    if decision.get("baseline_commit") != expected_head:
        raise RuntimeError(
            f"{label}: baseline commit mismatch."
        )

    for key in (
        "calibration_seeds_executed",
        "confirmation_seeds_executed",
    ):
        value = decision.get(key)

        if value not in (None, [], False):
            raise RuntimeError(
                f"{label}: {key} indicates execution."
            )


def normalize_integer(
    series: pd.Series,
    *,
    label: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: non-finite values.")

    rounded = np.rint(values)

    if not np.allclose(values, rounded):
        raise RuntimeError(f"{label}: non-integral values.")

    return pd.Series(
        rounded.astype(np.int64),
        index=series.index,
    )


def action_level_error(
    expected: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_name: str,
) -> dict[str, Any]:
    joined = expected.merge(
        candidate,
        on=list(ACTION_KEYS),
        how="left",
        validate="one_to_one",
    )

    if joined["candidate_value"].isna().any():
        return {
            "candidate_name": candidate_name,
            "complete_action_join": False,
            "mean_abs_error": math.inf,
            "max_abs_error": math.inf,
            "exact_reconciliation": False,
        }

    error = np.abs(
        joined["candidate_value"].to_numpy(dtype=float)
        - joined[
            EXPECTED_UTILITY_COLUMN
        ].to_numpy(dtype=float)
    )

    return {
        "candidate_name": candidate_name,
        "complete_action_join": True,
        "mean_abs_error": float(error.mean()),
        "max_abs_error": float(error.max()),
        "exact_reconciliation": bool(
            error.max() <= TOLERANCE
        ),
    }


def make_policy_pair_delta(
    daily: pd.DataFrame,
    *,
    reference_policy: str,
    treatment_policy: str,
) -> pd.DataFrame:
    pairing_keys = [
        *ACTION_KEYS,
        SCENARIO_COLUMN,
        REPLICATION_COLUMN,
        DAY_COLUMN,
    ]

    reference = daily.loc[
        daily[POLICY_COLUMN].eq(reference_policy),
        [
            *pairing_keys,
            DAILY_UTILITY_COLUMN,
        ],
    ].rename(
        columns={
            DAILY_UTILITY_COLUMN: "reference_utility",
        }
    )

    treatment = daily.loc[
        daily[POLICY_COLUMN].eq(treatment_policy),
        [
            *pairing_keys,
            DAILY_UTILITY_COLUMN,
        ],
    ].rename(
        columns={
            DAILY_UTILITY_COLUMN: "treatment_utility",
        }
    )

    if reference.duplicated(pairing_keys).any():
        raise RuntimeError(
            f"Reference policy={reference_policy} does not have "
            "unique daily pairing rows."
        )

    if treatment.duplicated(pairing_keys).any():
        raise RuntimeError(
            f"Treatment policy={treatment_policy} does not have "
            "unique daily pairing rows."
        )

    paired = treatment.merge(
        reference,
        on=pairing_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not paired["_merge"].eq("both").all():
        summary = paired["_merge"].value_counts().to_dict()

        raise RuntimeError(
            "Policy-paired daily utility has unmatched rows for "
            f"reference={reference_policy}, treatment={treatment_policy}: "
            f"{summary}"
        )

    paired["policy_utility_delta"] = (
        paired["treatment_utility"]
        - paired["reference_utility"]
    )

    if not np.isfinite(
        paired["policy_utility_delta"].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Policy-paired utility delta is non-finite."
        )

    return paired


def candidate_aggregates(
    paired: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    daily_mean = (
        paired.groupby(
            list(ACTION_KEYS),
            as_index=False,
        )["policy_utility_delta"]
        .mean()
        .rename(
            columns={
                "policy_utility_delta": "candidate_value",
            }
        )
    )

    block = (
        paired.groupby(
            [
                *ACTION_KEYS,
                SCENARIO_COLUMN,
                REPLICATION_COLUMN,
            ],
            as_index=False,
        )
        .agg(
            block_daily_count=(
                "policy_utility_delta",
                "size",
            ),
            block_sum_policy_utility_delta=(
                "policy_utility_delta",
                "sum",
            ),
            block_mean_policy_utility_delta=(
                "policy_utility_delta",
                "mean",
            ),
        )
    )

    if not block["block_daily_count"].eq(5).all():
        raise RuntimeError(
            "A scenario-replication policy-delta block does not "
            "contain exactly five daily observations."
        )

    replication_mean_of_daily_mean = (
        block.groupby(
            list(ACTION_KEYS),
            as_index=False,
        )["block_mean_policy_utility_delta"]
        .mean()
        .rename(
            columns={
                "block_mean_policy_utility_delta": (
                    "candidate_value"
                ),
            }
        )
    )

    replication_mean_of_daily_sum = (
        block.groupby(
            list(ACTION_KEYS),
            as_index=False,
        )["block_sum_policy_utility_delta"]
        .mean()
        .rename(
            columns={
                "block_sum_policy_utility_delta": (
                    "candidate_value"
                ),
            }
        )
    )

    return {
        "mean_paired_daily_policy_utility_delta": (
            daily_mean
        ),
        "mean_scenario_replication_daily_mean_policy_delta": (
            replication_mean_of_daily_mean
        ),
        "mean_scenario_replication_five_day_sum_policy_delta": (
            replication_mean_of_daily_sum
        ),
    }


v5d = read_json(v5d_decision_path)
v5e = read_json(v5e_decision_path)
v5g1 = read_json(v5g1_decision_path)
v5h = read_json(v5h_decision_path)

validate_decision(
    v5d,
    label="V5-D",
    expected_status=(
        "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
    ),
)

validate_decision(
    v5e,
    label="V5-E",
    expected_status=(
        "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_AUDIT_COMPLETE"
    ),
)

validate_decision(
    v5g1,
    label="V5-G1",
    expected_status=(
        "V5G1_PREACTION_RUNTIME_STATE_RECOVERED_AND_CONTRACT_READY"
    ),
)

validate_decision(
    v5h,
    label="V5-H",
    expected_status=(
        "V5H_RICH_PREACTION_RUNTIME_SIGNAL_INSUFFICIENT"
    ),
)

if int(v5d.get("action_count", -1)) != 720:
    raise RuntimeError("V5-D action count mismatch.")

if int(v5d.get("proposal_count", -1)) != 144:
    raise RuntimeError("V5-D proposal count mismatch.")

if int(v5g1.get("context_count", -1)) != 144:
    raise RuntimeError("V5-G1 context count mismatch.")

if int(v5g1.get("state_action_row_count", -1)) != 720:
    raise RuntimeError("V5-G1 state-action count mismatch.")

if v5h.get("final_serving_model_trained") is not False:
    raise RuntimeError(
        "V5-H unexpectedly contains a final serving model."
    )

if v5h.get("threshold_selected") is not False:
    raise RuntimeError(
        "V5-H unexpectedly selected a serving threshold."
    )

effects_path = Path(
    v5d["action_effects_path"]
).resolve()

daily_path = Path(
    v5d["daily_path"]
).resolve()

if not effects_path.is_file() or not daily_path.is_file():
    raise RuntimeError(
        "V5-D action-effect/daily evidence is missing."
    )

effects = pd.read_csv(effects_path)
daily = pd.read_csv(daily_path)

required_daily = {
    *ACTION_KEYS,
    SCENARIO_COLUMN,
    REPLICATION_COLUMN,
    DAY_COLUMN,
    POLICY_COLUMN,
    DAILY_UTILITY_COLUMN,
}

required_effects = {
    *ACTION_KEYS,
    EXPECTED_UTILITY_COLUMN,
}

missing_daily = sorted(
    required_daily - set(daily.columns)
)

missing_effects = sorted(
    required_effects - set(effects.columns)
)

if missing_daily:
    raise RuntimeError(
        f"Daily table missing required policy-delta columns: {missing_daily}"
    )

if missing_effects:
    raise RuntimeError(
        f"Action-effects table missing target columns: {missing_effects}"
    )

for column in ("seed", "proposal_index"):
    daily[column] = normalize_integer(
        daily[column],
        label=f"daily {column}",
    )
    effects[column] = normalize_integer(
        effects[column],
        label=f"effects {column}",
    )

daily[DAILY_UTILITY_COLUMN] = pd.to_numeric(
    daily[DAILY_UTILITY_COLUMN],
    errors="raise",
)

effects[EXPECTED_UTILITY_COLUMN] = pd.to_numeric(
    effects[EXPECTED_UTILITY_COLUMN],
    errors="raise",
)

if not np.isfinite(
    daily[DAILY_UTILITY_COLUMN].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Daily raw utility is non-finite."
    )

if not np.isfinite(
    effects[EXPECTED_UTILITY_COLUMN].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Action-effect utility target is non-finite."
    )

if len(effects) != 720:
    raise RuntimeError(
        f"Expected 720 action effects, found {len(effects)}."
    )

if effects.duplicated(list(ACTION_KEYS)).any():
    raise RuntimeError(
        "Action-effect table has duplicate action keys."
    )

if len(daily) != 108_000:
    raise RuntimeError(
        f"Expected 108000 daily rows, found {len(daily)}."
    )

daily_counts = (
    daily.groupby(
        list(ACTION_KEYS),
        as_index=False,
    )
    .size()
    .rename(columns={"size": "daily_row_count"})
)

if len(daily_counts) != 720:
    raise RuntimeError(
        "Daily table does not cover each action effect."
    )

if not daily_counts["daily_row_count"].eq(150).all():
    raise RuntimeError(
        "Daily table must have exactly 150 rows per action."
    )

policy_values = sorted(
    str(value)
    for value in daily[POLICY_COLUMN].dropna().unique()
)

if len(policy_values) != 2:
    raise RuntimeError(
        "V5-I.3 requires exactly two daily policy levels; found "
        f"{policy_values}."
    )

if daily[POLICY_COLUMN].isna().any():
    raise RuntimeError(
        "Daily policy column contains missing values."
    )

daily[POLICY_COLUMN] = daily[POLICY_COLUMN].astype(str)

grain = [
    *ACTION_KEYS,
    SCENARIO_COLUMN,
    REPLICATION_COLUMN,
    DAY_COLUMN,
    POLICY_COLUMN,
]

duplicate_grain = daily.loc[
    daily.duplicated(grain, keep=False),
    grain,
]

if not duplicate_grain.empty:
    examples = duplicate_grain.head(12).to_dict(
        orient="records"
    )

    raise RuntimeError(
        "Policy-augmented daily grain is still non-unique. "
        f"Examples: {examples}"
    )

cell_counts = (
    daily.groupby(
        grain,
        as_index=False,
    )
    .size()
    .rename(columns={"size": "raw_row_count"})
)

if not cell_counts["raw_row_count"].eq(1).all():
    raise RuntimeError(
        "Policy-augmented daily grain is unexpectedly not unique."
    )

expected = effects.loc[
    :,
    [
        *ACTION_KEYS,
        EXPECTED_UTILITY_COLUMN,
    ],
].copy()

reconciliation_rows: list[dict[str, Any]] = []
paired_coverage_rows: list[dict[str, Any]] = []
candidate_matrices: dict[str, pd.DataFrame] = {}

for reference_policy, treatment_policy in (
    (policy_values[0], policy_values[1]),
    (policy_values[1], policy_values[0]),
):
    paired = make_policy_pair_delta(
        daily,
        reference_policy=reference_policy,
        treatment_policy=treatment_policy,
    )

    paired_coverage_rows.append(
        {
            "reference_policy": reference_policy,
            "treatment_policy": treatment_policy,
            "paired_daily_row_count": int(len(paired)),
            "expected_paired_daily_row_count": 54_000,
            "unique_context_action_count": int(
                paired.loc[
                    :,
                    list(ACTION_KEYS),
                ].drop_duplicates().shape[0]
            ),
        }
    )

    aggregates = candidate_aggregates(paired)

    for aggregation_name, candidate in aggregates.items():
        candidate_name = (
            f"{treatment_policy}_minus_{reference_policy}"
            f"__{aggregation_name}"
        )

        candidate_matrices[candidate_name] = candidate

        record = action_level_error(
            expected,
            candidate,
            candidate_name=candidate_name,
        )

        record["reference_policy"] = reference_policy
        record["treatment_policy"] = treatment_policy
        record["aggregation_name"] = aggregation_name

        reconciliation_rows.append(record)

reconciliation = pd.DataFrame(reconciliation_rows).sort_values(
    [
        "exact_reconciliation",
        "max_abs_error",
        "mean_abs_error",
        "candidate_name",
    ],
    ascending=[False, True, True, True],
    kind="mergesort",
).reset_index(drop=True)

exact = reconciliation.loc[
    reconciliation["exact_reconciliation"]
].copy()

if len(exact) == 1:
    status = "V5I3_POLICY_DELTA_UTILITY_RECONCILED"
    selected = exact.iloc[0]
elif len(exact) > 1:
    status = "V5I3_POLICY_DELTA_UTILITY_AMBIGUOUS"
    selected = None
else:
    status = "V5I3_POLICY_DELTA_UTILITY_UNRESOLVED"
    selected = None

if selected is not None:
    selected_name = str(selected["candidate_name"])
    selected_reference_policy = str(
        selected["reference_policy"]
    )
    selected_treatment_policy = str(
        selected["treatment_policy"]
    )
    selected_aggregation = str(
        selected["aggregation_name"]
    )
    selected_max_abs_error = float(
        selected["max_abs_error"]
    )
    selected_mean_abs_error = float(
        selected["mean_abs_error"]
    )

    selected_matrix = candidate_matrices[selected_name].copy()

    if len(selected_matrix) != 720:
        raise RuntimeError(
            "Selected policy-delta reconstruction does not "
            "cover all 720 action labels."
        )
else:
    selected_name = None
    selected_reference_policy = None
    selected_treatment_policy = None
    selected_aggregation = None
    selected_max_abs_error = None
    selected_mean_abs_error = None
    selected_matrix = pd.DataFrame(
        columns=[
            *ACTION_KEYS,
            "candidate_value",
        ]
    )

output_root.mkdir(parents=True, exist_ok=True)

policy_levels_path = (
    output_root
    / "v5i3_daily_policy_levels.csv"
)

grain_path = (
    output_root
    / "v5i3_policy_augmented_daily_grain_counts.csv"
)

coverage_path = (
    output_root
    / "v5i3_policy_pairing_coverage.csv"
)

reconciliation_path = (
    output_root
    / "v5i3_policy_delta_reconciliation.csv"
)

selected_matrix_path = (
    output_root
    / "v5i3_selected_policy_delta_action_matrix.csv"
)

pd.DataFrame(
    {
        "policy": policy_values,
    }
).to_csv(
    policy_levels_path,
    index=False,
)

cell_counts.to_csv(
    grain_path,
    index=False,
)

pd.DataFrame(paired_coverage_rows).to_csv(
    coverage_path,
    index=False,
)

reconciliation.to_csv(
    reconciliation_path,
    index=False,
)

selected_matrix.to_csv(
    selected_matrix_path,
    index=False,
)

decision = {
    "status": status,
    "baseline_commit": expected_head,
    "v5d_decision_path": str(v5d_decision_path),
    "v5d_decision_sha256": sha256(v5d_decision_path),
    "v5e_decision_path": str(v5e_decision_path),
    "v5e_decision_sha256": sha256(v5e_decision_path),
    "v5g1_decision_path": str(v5g1_decision_path),
    "v5g1_decision_sha256": sha256(v5g1_decision_path),
    "v5h_decision_path": str(v5h_decision_path),
    "v5h_decision_sha256": sha256(v5h_decision_path),
    "action_effects_path": str(effects_path),
    "action_effects_sha256": sha256(effects_path),
    "daily_path": str(daily_path),
    "daily_sha256": sha256(daily_path),
    "action_effect_row_count": int(len(effects)),
    "daily_row_count": int(len(daily)),
    "daily_rows_per_action": {
        "min": int(daily_counts["daily_row_count"].min()),
        "max": int(daily_counts["daily_row_count"].max()),
    },
    "frozen_action_effect_target": EXPECTED_UTILITY_COLUMN,
    "daily_policy_delta_contract": {
        "policy_column": POLICY_COLUMN,
        "policy_values": policy_values,
        "scenario_column": SCENARIO_COLUMN,
        "replication_column": REPLICATION_COLUMN,
        "day_column": DAY_COLUMN,
        "daily_utility_column": DAILY_UTILITY_COLUMN,
        "policy_augmented_pair_grain": grain,
        "policy_augmented_grain_unique": True,
        "policy_pairing_rows_per_direction": 54_000,
        "candidate_aggregations": [
            "mean_paired_daily_policy_utility_delta",
            "mean_scenario_replication_daily_mean_policy_delta",
            "mean_scenario_replication_five_day_sum_policy_delta",
        ],
        "reconciliation_tolerance": TOLERANCE,
    },
    "exact_reconciliation_candidate_count": int(len(exact)),
    "selected_reconstruction": {
        "candidate_name": selected_name,
        "reference_policy": selected_reference_policy,
        "treatment_policy": selected_treatment_policy,
        "aggregation_name": selected_aggregation,
        "max_abs_error": selected_max_abs_error,
        "mean_abs_error": selected_mean_abs_error,
        "selected_action_matrix_path": (
            str(selected_matrix_path)
            if selected_name is not None
            else None
        ),
        "selected_action_matrix_sha256": (
            sha256(selected_matrix_path)
            if selected_name is not None
            else None
        ),
    },
    "top_reconciliation_candidates": reconciliation.head(
        10
    ).to_dict(orient="records"),
    "artifacts": {
        "policy_levels_path": str(policy_levels_path),
        "policy_augmented_daily_grain_counts_path": (
            str(grain_path)
        ),
        "policy_pairing_coverage_path": str(coverage_path),
        "policy_delta_reconciliation_path": (
            str(reconciliation_path)
        ),
        "selected_policy_delta_action_matrix_path": (
            str(selected_matrix_path)
        ),
    },
    "source_modified": False,
    "config_modified": False,
    "v5d_corpus_rerun": False,
    "model_trained": False,
    "final_serving_model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "dynamic_replay_executed_at_release_scale": False,
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Patch V5-I only to use the uniquely reconciled policy "
        "difference and selected scenario-replication aggregation. "
        "Do not fit a predictive model, choose a serving threshold, "
        "or execute calibration/confirmation in this audit."
        if status == "V5I3_POLICY_DELTA_UTILITY_RECONCILED"
        else (
            "Inspect the exact policy-delta candidates before "
            "choosing a semantic contract. Do not rerun V5-I "
            "automatically."
            if status == "V5I3_POLICY_DELTA_UTILITY_AMBIGUOUS"
            else "Do not rerun V5-I. The raw daily utility policy "
            "difference does not reconstruct the frozen action "
            "utility label under the pre-registered daily or "
            "scenario-replication aggregation candidates."
        )
    ),
}

decision_path = (
    output_root
    / "v5i3_policy_delta_utility_reconstruction_decision.json"
)

write_json(decision_path, decision)

print("===== V5-I.3 POLICY-DELTA UTILITY RECONSTRUCTION =====")
print(
    json.dumps(
        {
            "decision": decision,
            "reconciliation": reconciliation.to_dict(
                orient="records"
            ),
        },
        indent=2,
        sort_keys=True,
    )
)

print("===== V5-I.3 DECISION =====")
print(
    json.dumps(
        {
            "status": status,
            "policy_values": policy_values,
            "exact_reconciliation_candidate_count": int(
                len(exact)
            ),
            "selected_reconstruction": (
                decision["selected_reconstruction"]
            ),
            "top_reconciliation_candidates": (
                reconciliation.head(10).to_dict(
                    orient="records"
                )
            ),
        },
        indent=2,
        sort_keys=True,
    )
)