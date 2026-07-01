from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


decision_path = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
expected_head = sys.argv[3]

EXPECTED_TRAINING_SEEDS = (
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
)

EXPECTED_ACTION_TO_RANK = {
    "PLACE_AT_1": 1,
    "PLACE_AT_2": 2,
    "PLACE_AT_3": 3,
    "PLACE_AT_5": 5,
    "PLACE_AT_10": 10,
}

METRIC_COLUMNS = (
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


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def strict_false(value: Any, *, label: str) -> None:
    if type(value) is not bool or value is not False:
        raise RuntimeError(
            f"{label} must be JSON boolean false, got {value!r}"
        )


def false_series(series: pd.Series) -> bool:
    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return bool(values.eq("false").all())


if not decision_path.is_file():
    raise RuntimeError(
        f"Missing V5-D decision: {decision_path}"
    )

decision = json.loads(
    decision_path.read_text(encoding="utf-8-sig")
)

if (
    decision.get("status")
    != "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
):
    raise RuntimeError(
        f"Unexpected V5-D status: {decision.get('status')!r}"
    )

if decision.get("baseline_commit") != expected_head:
    raise RuntimeError(
        "V5-D baseline commit mismatch."
    )

if int(decision.get("action_count", -1)) != 720:
    raise RuntimeError(
        f"Expected 720 action labels, got {decision.get('action_count')}"
    )

if int(decision.get("proposal_count", -1)) != 144:
    raise RuntimeError(
        f"Expected 144 proposals, got {decision.get('proposal_count')}"
    )

training_seeds = tuple(
    int(seed)
    for seed in decision.get("training_seeds", [])
)

if training_seeds != EXPECTED_TRAINING_SEEDS:
    raise RuntimeError(
        "Final training seed registry mismatch."
    )

calibration_executed = decision.get(
    "calibration_seeds_executed"
)

if not isinstance(calibration_executed, list):
    raise RuntimeError(
        "calibration_seeds_executed must be a JSON list."
    )

if calibration_executed:
    raise RuntimeError(
        f"Calibration seeds were executed: {calibration_executed}"
    )

strict_false(
    decision.get("confirmation_seeds_executed"),
    label="confirmation_seeds_executed",
)

for key in (
    "source_modified",
    "config_modified",
    "model_trained",
    "threshold_selected",
    "commit_created",
    "push_performed",
):
    strict_false(
        decision.get(key),
        label=key,
    )

required_paths = {
    "action_effects_path": Path(
        decision["action_effects_path"]
    ).resolve(),
    "proposal_manifest_path": Path(
        decision["proposal_manifest_path"]
    ).resolve(),
    "daily_path": Path(
        decision["daily_path"]
    ).resolve(),
    "summary_by_action_path": Path(
        decision["summary_by_action_path"]
    ).resolve(),
    "seed_runs_path": Path(
        decision["seed_runs_path"]
    ).resolve(),
}

for label, path in required_paths.items():
    if not path.is_file():
        raise RuntimeError(
            f"Missing V5-D output [{label}]: {path}"
        )

actions = pd.read_csv(
    required_paths["action_effects_path"]
)

proposals = pd.read_csv(
    required_paths["proposal_manifest_path"]
)

daily = pd.read_csv(
    required_paths["daily_path"]
)

seed_runs = json.loads(
    required_paths["seed_runs_path"].read_text(
        encoding="utf-8"
    )
)

require_columns(
    actions,
    {
        "seed",
        "proposal_index",
        "query_id",
        "product_id",
        "action",
        "requested_rank",
        "achieved_rank",
        "base_score",
        "target_score",
        "qrsbt_boost",
        *METRIC_COLUMNS,
    },
    label="V5-D action labels",
)

require_columns(
    proposals,
    {
        "seed",
        "proposal_index",
        "query_id",
        "product_id",
        "teacher_or_oracle_columns_used",
    },
    label="V5-D proposal manifest",
)

require_columns(
    daily,
    {
        "seed",
        "proposal_index",
        "query_id",
        "product_id",
        "action",
    },
    label="V5-D daily outcomes",
)

if len(actions) != 720:
    raise RuntimeError(
        f"Action-label file has {len(actions)} rows, expected 720."
    )

if len(proposals) != 144:
    raise RuntimeError(
        f"Proposal manifest has {len(proposals)} rows, expected 144."
    )

observed_action_seeds = tuple(
    sorted(
        int(seed)
        for seed in actions["seed"].unique()
    )
)

if observed_action_seeds != EXPECTED_TRAINING_SEEDS:
    raise RuntimeError(
        "Action-label seed coverage mismatch."
    )

observed_proposal_seeds = tuple(
    sorted(
        int(seed)
        for seed in proposals["seed"].unique()
    )
)

if observed_proposal_seeds != EXPECTED_TRAINING_SEEDS:
    raise RuntimeError(
        "Proposal-manifest seed coverage mismatch."
    )

if not false_series(
    proposals["teacher_or_oracle_columns_used"]
):
    raise RuntimeError(
        "Proposal manifest indicates oracle-derived inputs."
    )

forbidden_proposal_columns = {
    "teacher_selected",
    "relevance",
    "relevance_for_diagnostic_only",
    "clicked",
    "purchased",
    "cold_gain",
    "warm_gain",
    "required_boost",
    "target_score",
    "final_score",
    "final_rank",
    "qrsbt_boost",
}

unexpected_forbidden_columns = sorted(
    forbidden_proposal_columns
    & set(proposals.columns)
)

if unexpected_forbidden_columns:
    raise RuntimeError(
        "Proposal manifest contains forbidden fields: "
        f"{unexpected_forbidden_columns}"
    )

proposal_counts = (
    proposals.groupby("seed", sort=True)
    .size()
)

if not proposal_counts.eq(8).all():
    raise RuntimeError(
        "Every training seed must contribute exactly 8 contexts."
    )

action_counts = (
    actions.groupby("seed", sort=True)
    .size()
)

if not action_counts.eq(40).all():
    raise RuntimeError(
        "Every training seed must contribute 8 contexts × 5 actions."
    )

if actions.duplicated(
    ["seed", "proposal_index", "action"]
).any():
    raise RuntimeError(
        "Duplicate seed/proposal/action labels found."
    )

action_set = set(actions["action"].unique())

if action_set != set(EXPECTED_ACTION_TO_RANK):
    raise RuntimeError(
        f"Unexpected action space: {sorted(action_set)}"
    )

expected_ranks = actions["action"].map(
    EXPECTED_ACTION_TO_RANK
)

if not actions["requested_rank"].eq(
    expected_ranks
).all():
    raise RuntimeError(
        "Action-to-requested-rank mapping mismatch."
    )

if not actions["achieved_rank"].eq(
    actions["requested_rank"]
).all():
    raise RuntimeError(
        "At least one action violates exact canonical placement."
    )

numeric_columns = (
    "base_score",
    "target_score",
    "qrsbt_boost",
    *METRIC_COLUMNS,
)

for column in numeric_columns:
    actions[column] = pd.to_numeric(
        actions[column],
        errors="raise",
    )

if not np.isfinite(
    actions.loc[:, numeric_columns].to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Non-finite numeric action value detected."
    )

if not actions["target_score"].gt(
    actions["base_score"]
).all():
    raise RuntimeError(
        "At least one intervention target score is not above base score."
    )

if not actions["qrsbt_boost"].gt(0.0).all():
    raise RuntimeError(
        "At least one intervention has a non-positive boost."
    )

proposal_join = actions.merge(
    proposals.loc[
        :,
        [
            "seed",
            "proposal_index",
            "query_id",
            "product_id",
        ],
    ],
    on=[
        "seed",
        "proposal_index",
        "query_id",
        "product_id",
    ],
    how="inner",
    validate="many_to_one",
)

if len(proposal_join) != len(actions):
    raise RuntimeError(
        "Action labels do not map one-to-one to proposal contexts."
    )

daily_counts = (
    daily.groupby(
        ["seed", "proposal_index", "action"],
        sort=True,
    )
    .size()
    .rename("daily_rows")
    .reset_index()
)

daily_join = actions.loc[
    :,
    ["seed", "proposal_index", "action"],
].merge(
    daily_counts,
    on=["seed", "proposal_index", "action"],
    how="left",
    validate="one_to_one",
)

if daily_join["daily_rows"].isna().any():
    raise RuntimeError(
        "At least one action lacks daily dynamic outcomes."
    )

if not daily_join["daily_rows"].gt(0).all():
    raise RuntimeError(
        "At least one action has empty daily dynamic outcomes."
    )

if len(seed_runs) != 18:
    raise RuntimeError(
        f"Expected 18 seed-run records, got {len(seed_runs)}."
    )

seed_run_ids = tuple(
    int(row["seed"])
    for row in seed_runs
)

if seed_run_ids != EXPECTED_TRAINING_SEEDS:
    raise RuntimeError(
        "Seed-run manifest registry mismatch."
    )

for row in seed_runs:
    selected = row.get("selected_pilot_queries")

    if not isinstance(selected, list) or len(selected) != 8:
        raise RuntimeError(
            f"seed={row.get('seed')}: expected 8 selected contexts."
        )

metadata_notes = []

if "remaining 15" in str(
    decision.get("next_gate", "")
).lower():
    metadata_notes.append(
        "The inherited next_gate text refers to expansion "
        "to remaining 15 seeds even though the completed V5-D "
        "corpus already contains all 18 training seeds."
    )

completion = {
    "status": (
        "V5D_FINAL_TRAINING_CORPUS_VERIFIED_"
        "WITH_BOOLEAN_CONFIRMATION_CONTRACT"
    ),
    "baseline_commit": expected_head,
    "original_decision_path": str(decision_path),
    "original_decision_sha256": sha256(decision_path),
    "original_decision_status": decision["status"],
    "training_seed_count": len(EXPECTED_TRAINING_SEEDS),
    "training_seeds": list(EXPECTED_TRAINING_SEEDS),
    "proposal_count": int(len(proposals)),
    "action_count": int(len(actions)),
    "calibration_seeds_executed": calibration_executed,
    "confirmation_seeds_executed": (
        decision["confirmation_seeds_executed"]
    ),
    "confirmation_contract_interpretation": (
        "The V5-D runner emits confirmation_seeds_executed "
        "as JSON boolean false, not as an empty list. "
        "The prior PowerShell .Count-based verifier incorrectly "
        "treated this scalar as a non-empty collection."
    ),
    "source_modified": decision["source_modified"],
    "config_modified": decision["config_modified"],
    "model_trained": decision["model_trained"],
    "threshold_selected": decision["threshold_selected"],
    "completion_integrity": {
        "all_18_training_seeds_covered": True,
        "eight_contexts_per_seed": True,
        "five_actions_per_context": True,
        "all_achieved_ranks_exact": True,
        "all_numeric_outputs_finite": True,
        "all_actions_have_daily_output": True,
        "runtime_only_proposal_contract": True,
        "daily_rows_total": int(len(daily)),
        "daily_rows_per_action_min": int(
            daily_join["daily_rows"].min()
        ),
        "daily_rows_per_action_max": int(
            daily_join["daily_rows"].max()
        ),
    },
    "output_hashes": {
        name: sha256(path)
        for name, path in required_paths.items()
    },
    "metadata_notes": metadata_notes,
    "source_modified_by_audit": False,
    "config_modified_by_audit": False,
    "model_trained_by_audit": False,
    "threshold_selected_by_audit": False,
    "dynamic_replay_executed_at_release_scale": False,
    "confirmation_seeds_executed_by_audit": False,
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Freeze the V5-D corpus schema and conduct a "
        "development-only label-feasibility and feature-provenance "
        "audit before fitting any direct-utility or harm model. "
        "Do not execute calibration or confirmation seeds."
    ),
}

output_root.mkdir(parents=True, exist_ok=True)

coverage = (
    actions.groupby(
        ["seed", "action"],
        sort=True,
    )
    .size()
    .rename("label_count")
    .reset_index()
)

coverage.to_csv(
    output_root / "v5d_seed_action_coverage.csv",
    index=False,
)

daily_counts.to_csv(
    output_root / "v5d_daily_output_coverage.csv",
    index=False,
)

completion_path = (
    output_root
    / "v5d_completion_contract_resolution.json"
)

write_json(completion_path, completion)

report_path = (
    output_root
    / "v5d_completion_and_integrity_report.md"
)

report_path.write_text(
    "\n".join(
        [
            "# V5-D Completion and Corpus Integrity Report",
            "",
            f"- **Status:** `{completion['status']}`",
            f"- **Baseline commit:** `{expected_head}`",
            f"- **Training seeds:** {completion['training_seed_count']}",
            f"- **Contexts:** {completion['proposal_count']}",
            f"- **Action-effect labels:** {completion['action_count']}",
            f"- **Calibration seeds executed:** `{completion['calibration_seeds_executed']}`",
            f"- **Confirmation seeds executed:** `{completion['confirmation_seeds_executed']}`",
            f"- **Daily dynamic rows:** {completion['completion_integrity']['daily_rows_total']}",
            "",
            "## Verifier Resolution",
            "",
            "The prior PowerShell wrapper failed after corpus generation because it applied `.Count` to a Boolean `false` confirmation field.",
            "The corpus decision itself uses `false` as the explicit no-confirmation-execution contract.",
            "",
            "## Metadata Note",
            "",
            *[
                f"- {note}"
                for note in metadata_notes
            ] or ["- None."],
            "",
            "## Next Gate",
            "",
            completion["next_gate"],
            "",
        ]
    ),
    encoding="utf-8",
)

print("===== V5-D COMPLETION + CORPUS INTEGRITY AUDIT =====")
print(
    json.dumps(
        {
            "completion": completion,
            "completion_path": str(completion_path),
            "completion_sha256": sha256(completion_path),
            "report_path": str(report_path),
            "report_sha256": sha256(report_path),
        },
        indent=2,
        sort_keys=True,
    )
)