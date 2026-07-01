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
v5i3_decision_path = Path(sys.argv[5]).resolve()
output_root = Path(sys.argv[6]).resolve()
expected_head = sys.argv[7]

RANDOM_STATE = 17
BOOTSTRAP_REPS = 10_000
PAIR_BASELINE_ACTION = "PLACE_AT_1"
ACTION_ORDER = (
    "PLACE_AT_1",
    "PLACE_AT_2",
    "PLACE_AT_3",
    "PLACE_AT_5",
    "PLACE_AT_10",
)
ALTERNATIVE_ACTIONS = tuple(
    action
    for action in ACTION_ORDER
    if action != PAIR_BASELINE_ACTION
)
ACTION_TO_TARGET_RANK = {
    "PLACE_AT_1": 1,
    "PLACE_AT_2": 2,
    "PLACE_AT_3": 3,
    "PLACE_AT_5": 5,
    "PLACE_AT_10": 10,
}

ACTION_KEYS = (
    "seed",
    "proposal_index",
    "action",
)
CONTEXT_KEYS = (
    "seed",
    "proposal_index",
)

TARGET_UTILITY_COLUMN = (
    "mean_scenario_replication_utility_delta"
)

SAFETY_COLUMNS = {
    "relevant_discovery": "relevant_discovery_delta",
    "irrelevant_exposure": "irrelevant_exposure_delta",
    "false_warmup": "false_warmup_delta",
    "worst_utility": "worst_scenario_utility_delta",
    "p10_utility": "p10_scenario_replication_utility_delta",
}

EXPECTED_REFERENCE_POLICY = "base"
EXPECTED_TREATMENT_POLICY = "qrsbt_gate"
EXPECTED_AGGREGATION = (
    "mean_scenario_replication_five_day_sum_policy_delta"
)

TOLERANCE = 1e-8
MIN_STABLE_CONTEXTS = 12
MIN_STABLE_SEEDS = 6
MIN_STABLE_ORACLE_GAP_SHARE = 0.50


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


def as_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


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


def require_unique_rows(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    label: str,
) -> None:
    duplicates = frame.loc[
        frame.duplicated(keys, keep=False),
        keys,
    ]

    if not duplicates.empty:
        examples = duplicates.head(10).to_dict(
            orient="records"
        )

        raise RuntimeError(
            f"{label} has non-unique keys: {examples}"
        )


def policy_delta_block_table(
    daily: pd.DataFrame,
    *,
    policy_column: str,
    reference_policy: str,
    treatment_policy: str,
    scenario_column: str,
    replication_column: str,
    day_column: str,
    utility_column: str,
) -> pd.DataFrame:
    daily_keys = [
        *ACTION_KEYS,
        scenario_column,
        replication_column,
        day_column,
    ]

    reference = daily.loc[
        daily[policy_column].eq(reference_policy),
        [
            *daily_keys,
            utility_column,
        ],
    ].rename(
        columns={
            utility_column: "base_policy_utility",
        }
    )

    treatment = daily.loc[
        daily[policy_column].eq(treatment_policy),
        [
            *daily_keys,
            utility_column,
        ],
    ].rename(
        columns={
            utility_column: "qrsbt_gate_policy_utility",
        }
    )

    require_unique_rows(
        reference,
        keys=daily_keys,
        label="reference-policy daily frame",
    )

    require_unique_rows(
        treatment,
        keys=daily_keys,
        label="treatment-policy daily frame",
    )

    paired = treatment.merge(
        reference,
        on=daily_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()

        raise RuntimeError(
            "Policy daily pairing is incomplete: "
            f"{counts}"
        )

    paired["policy_utility_delta"] = (
        paired["qrsbt_gate_policy_utility"]
        - paired["base_policy_utility"]
    )

    if not np.isfinite(
        paired["policy_utility_delta"].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Policy-paired daily utility delta is non-finite."
        )

    block_keys = [
        *ACTION_KEYS,
        scenario_column,
        replication_column,
    ]

    blocks = (
        paired.groupby(
            block_keys,
            as_index=False,
            dropna=False,
        )
        .agg(
            block_day_count=(
                "policy_utility_delta",
                "size",
            ),
            scenario_replication_five_day_policy_delta=(
                "policy_utility_delta",
                "sum",
            ),
        )
    )

    if not blocks["block_day_count"].eq(5).all():
        bad = blocks.loc[
            ~blocks["block_day_count"].eq(5),
            [*block_keys, "block_day_count"],
        ].head(10)

        raise RuntimeError(
            "Every scenario-replication block must have five "
            "paired daily observations. Examples: "
            f"{bad.to_dict(orient='records')}"
        )

    expected_block_counts = (
        blocks.groupby(
            list(ACTION_KEYS),
            as_index=False,
        )
        .agg(
            block_count=(
                "scenario_replication_five_day_policy_delta",
                "size",
            ),
            scenario_count=(scenario_column, "nunique"),
            replication_count=(replication_column, "nunique"),
        )
    )

    if not expected_block_counts["block_count"].eq(15).all():
        raise RuntimeError(
            "Each action must have exactly 15 scenario-replication "
            "blocks."
        )

    if not expected_block_counts["scenario_count"].eq(3).all():
        raise RuntimeError(
            "Each action must have exactly three scenarios."
        )

    if not expected_block_counts["replication_count"].eq(5).all():
        raise RuntimeError(
            "Each action must have exactly five replication labels."
        )

    return blocks


def action_means_from_blocks(
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    return (
        blocks.groupby(
            list(ACTION_KEYS),
            as_index=False,
        )[
            "scenario_replication_five_day_policy_delta"
        ]
        .mean()
        .rename(
            columns={
                "scenario_replication_five_day_policy_delta": (
                    "reconstructed_mean_utility_delta"
                )
            }
        )
    )


def assert_exact_action_reconstruction(
    effects: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> None:
    expected = effects.loc[
        :,
        [
            *ACTION_KEYS,
            TARGET_UTILITY_COLUMN,
        ],
    ]

    joined = expected.merge(
        reconstructed,
        on=list(ACTION_KEYS),
        how="left",
        validate="one_to_one",
    )

    if joined[
        "reconstructed_mean_utility_delta"
    ].isna().any():
        raise RuntimeError(
            "Reconstructed daily blocks do not cover all actions."
        )

    error = np.abs(
        joined[
            TARGET_UTILITY_COLUMN
        ].to_numpy(dtype=float)
        - joined[
            "reconstructed_mean_utility_delta"
        ].to_numpy(dtype=float)
    )

    if error.max() > TOLERANCE:
        raise RuntimeError(
            "Policy-delta block aggregation does not exactly "
            "reconstruct V5-D utility labels. "
            f"max_error={error.max()}"
        )


def context_action_contrast_blocks(
    blocks: pd.DataFrame,
    *,
    alternative_action: str,
    scenario_column: str,
    replication_column: str,
) -> pd.DataFrame:
    pair_keys = [
        *CONTEXT_KEYS,
        scenario_column,
        replication_column,
    ]

    baseline = blocks.loc[
        blocks["action"].eq(PAIR_BASELINE_ACTION),
        [
            *pair_keys,
            "scenario_replication_five_day_policy_delta",
        ],
    ].rename(
        columns={
            "scenario_replication_five_day_policy_delta": (
                "baseline_block_utility_delta"
            )
        }
    )

    alternative = blocks.loc[
        blocks["action"].eq(alternative_action),
        [
            *pair_keys,
            "scenario_replication_five_day_policy_delta",
        ],
    ].rename(
        columns={
            "scenario_replication_five_day_policy_delta": (
                "alternative_block_utility_delta"
            )
        }
    )

    require_unique_rows(
        baseline,
        keys=pair_keys,
        label=f"{PAIR_BASELINE_ACTION} utility blocks",
    )

    require_unique_rows(
        alternative,
        keys=pair_keys,
        label=f"{alternative_action} utility blocks",
    )

    paired = alternative.merge(
        baseline,
        on=pair_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()

        raise RuntimeError(
            "Action-contrast block pairing is incomplete for "
            f"{alternative_action}: {counts}"
        )

    paired["paired_action_utility_contrast"] = (
        paired["alternative_block_utility_delta"]
        - paired["baseline_block_utility_delta"]
    )

    if not np.isfinite(
        paired[
            "paired_action_utility_contrast"
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Paired action utility contrast is non-finite."
        )

    return paired


def stratified_bootstrap(
    contrast_blocks: pd.DataFrame,
    *,
    scenario_column: str,
    replication_column: str,
    seed: int,
    proposal_index: int,
    action: str,
) -> dict[str, Any]:
    scenario_values = sorted(
        contrast_blocks[scenario_column]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(scenario_values) != 3:
        raise RuntimeError(
            "Expected exactly three scenarios in paired contrast."
        )

    target_rank = ACTION_TO_TARGET_RANK[action]
    seed_sequence = np.random.SeedSequence(
        [
            RANDOM_STATE,
            int(seed),
            int(proposal_index),
            int(target_rank),
        ]
    )

    rng = np.random.default_rng(seed_sequence)
    bootstrap_scenario_means: list[np.ndarray] = []
    observed_scenario_means: list[float] = []
    block_counts: list[int] = []

    for scenario in scenario_values:
        values = contrast_blocks.loc[
            contrast_blocks[scenario_column]
            .astype(str)
            .eq(scenario),
            "paired_action_utility_contrast",
        ].to_numpy(dtype=float)

        replication_values = contrast_blocks.loc[
            contrast_blocks[scenario_column]
            .astype(str)
            .eq(scenario),
            replication_column,
        ]

        if len(values) != 5:
            raise RuntimeError(
                "Each scenario must have exactly five paired "
                f"replication blocks, found {len(values)}."
            )

        if replication_values.nunique() != 5:
            raise RuntimeError(
                "Paired scenario blocks do not have five unique "
                "replication labels."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Paired scenario block values are non-finite."
            )

        indices = rng.integers(
            low=0,
            high=len(values),
            size=(BOOTSTRAP_REPS, len(values)),
        )

        bootstrap_scenario_means.append(
            values[indices].mean(axis=1)
        )

        observed_scenario_means.append(
            float(values.mean())
        )
        block_counts.append(len(values))

    bootstrap_distribution = np.vstack(
        bootstrap_scenario_means
    ).mean(axis=0)

    observed_mean = float(
        np.mean(observed_scenario_means)
    )

    flat_values = contrast_blocks[
        "paired_action_utility_contrast"
    ].to_numpy(dtype=float)

    return {
        "scenario_count": int(len(scenario_values)),
        "replications_per_scenario": int(
            min(block_counts)
        ),
        "paired_block_count": int(len(flat_values)),
        "paired_mean_utility_contrast": observed_mean,
        "paired_std_utility_contrast": float(
            flat_values.std(ddof=1)
        ),
        "paired_ci_low_95": float(
            np.quantile(bootstrap_distribution, 0.025)
        ),
        "paired_ci_high_95": float(
            np.quantile(bootstrap_distribution, 0.975)
        ),
        "bootstrap_probability_positive": float(
            (bootstrap_distribution > 0.0).mean()
        ),
        "bootstrap_probability_negative": float(
            (bootstrap_distribution < 0.0).mean()
        ),
    }


def safety_contrast(
    effect_context: pd.DataFrame,
    *,
    alternative_action: str,
) -> dict[str, Any]:
    baseline = effect_context.loc[
        effect_context["action"].eq(PAIR_BASELINE_ACTION)
    ].iloc[0]

    alternative = effect_context.loc[
        effect_context["action"].eq(alternative_action)
    ].iloc[0]

    relevant = float(
        alternative["relevant_discovery"]
        - baseline["relevant_discovery"]
    )

    irrelevant = float(
        alternative["irrelevant_exposure"]
        - baseline["irrelevant_exposure"]
    )

    false_warmup = float(
        alternative["false_warmup"]
        - baseline["false_warmup"]
    )

    worst = float(
        alternative["worst_utility"]
        - baseline["worst_utility"]
    )

    p10 = float(
        alternative["p10_utility"]
        - baseline["p10_utility"]
    )

    nonworsening = bool(
        irrelevant <= TOLERANCE
        and false_warmup <= TOLERANCE
        and worst >= -TOLERANCE
        and p10 >= -TOLERANCE
    )

    return {
        "relevant_discovery_contrast": relevant,
        "irrelevant_exposure_contrast": irrelevant,
        "false_warmup_contrast": false_warmup,
        "worst_utility_contrast": worst,
        "p10_utility_contrast": p10,
        "aggregate_safety_nonworsening": nonworsening,
    }


def require_complete_action_contexts(
    effects: pd.DataFrame,
) -> None:
    counts = (
        effects.groupby(
            list(CONTEXT_KEYS),
            as_index=False,
        )
        .agg(
            action_count=("action", "size"),
            unique_action_count=("action", "nunique"),
        )
    )

    if len(counts) != 144:
        raise RuntimeError(
            f"Expected 144 contexts, found {len(counts)}."
        )

    if not counts["action_count"].eq(5).all():
        raise RuntimeError(
            "At least one V5-D context lacks action rows."
        )

    if not counts["unique_action_count"].eq(5).all():
        raise RuntimeError(
            "At least one V5-D context has duplicated actions."
        )


v5d = read_json(v5d_decision_path)
v5e = read_json(v5e_decision_path)
v5g1 = read_json(v5g1_decision_path)
v5h = read_json(v5h_decision_path)
v5i3 = read_json(v5i3_decision_path)

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

validate_decision(
    v5i3,
    label="V5-I.3",
    expected_status=(
        "V5I3_POLICY_DELTA_UTILITY_RECONCILED"
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

if v5g1.get("forbidden_outcome_or_oracle_inputs") is not True:
    raise RuntimeError(
        "V5-G1 pre-action leakage-exclusion contract failed."
    )

if v5h.get("final_serving_model_trained") is not False:
    raise RuntimeError(
        "V5-H unexpectedly includes a final serving model."
    )

if v5h.get("threshold_selected") is not False:
    raise RuntimeError(
        "V5-H unexpectedly selected a serving threshold."
    )

selected_reconstruction = v5i3.get(
    "selected_reconstruction",
    {}
)

if (
    selected_reconstruction.get("reference_policy")
    != EXPECTED_REFERENCE_POLICY
    or selected_reconstruction.get("treatment_policy")
    != EXPECTED_TREATMENT_POLICY
    or selected_reconstruction.get("aggregation_name")
    != EXPECTED_AGGREGATION
):
    raise RuntimeError(
        "V5-I.3 selected an unexpected policy-delta contract."
    )

if int(v5i3.get(
    "exact_reconciliation_candidate_count",
    -1,
)) != 1:
    raise RuntimeError(
        "V5-I.3 did not identify a unique reconstruction."
    )

daily_contract = v5i3.get(
    "daily_policy_delta_contract",
    {}
)

policy_column = str(
    daily_contract.get("policy_column")
)
scenario_column = str(
    daily_contract.get("scenario_column")
)
replication_column = str(
    daily_contract.get("replication_column")
)
day_column = str(
    daily_contract.get("day_column")
)
utility_column = str(
    daily_contract.get("daily_utility_column")
)

if (
    policy_column != "policy"
    or scenario_column != "scenario"
    or replication_column != "replication"
    or day_column != "day"
    or utility_column != "utility"
):
    raise RuntimeError(
        "V5-I.3 daily column contract is unexpected."
    )

effects_path = Path(
    v5d["action_effects_path"]
).resolve()

daily_path = Path(
    v5d["daily_path"]
).resolve()

for path in (effects_path, daily_path):
    if not path.is_file():
        raise RuntimeError(
            f"Required input artifact missing: {path}"
        )

effects = pd.read_csv(effects_path)
daily = pd.read_csv(daily_path)

required_effects = {
    *ACTION_KEYS,
    TARGET_UTILITY_COLUMN,
    *SAFETY_COLUMNS.values(),
    "requested_rank",
    "achieved_rank",
}

required_daily = {
    *ACTION_KEYS,
    policy_column,
    scenario_column,
    replication_column,
    day_column,
    utility_column,
}

missing_effects = sorted(
    required_effects - set(effects.columns)
)
missing_daily = sorted(
    required_daily - set(daily.columns)
)

if missing_effects:
    raise RuntimeError(
        f"Action effects missing required columns: {missing_effects}"
    )

if missing_daily:
    raise RuntimeError(
        f"Daily table missing required columns: {missing_daily}"
    )

for column in ("seed", "proposal_index"):
    effects[column] = normalize_integer(
        effects[column],
        label=f"effects {column}",
    )
    daily[column] = normalize_integer(
        daily[column],
        label=f"daily {column}",
    )

effects[TARGET_UTILITY_COLUMN] = pd.to_numeric(
    effects[TARGET_UTILITY_COLUMN],
    errors="raise",
)

daily[utility_column] = pd.to_numeric(
    daily[utility_column],
    errors="raise",
)

for column in SAFETY_COLUMNS.values():
    effects[column] = pd.to_numeric(
        effects[column],
        errors="raise",
    )

if not np.isfinite(
    effects.loc[
        :,
        [
            TARGET_UTILITY_COLUMN,
            *SAFETY_COLUMNS.values(),
        ],
    ].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Action-effect outcome labels are non-finite."
    )

if not np.isfinite(
    daily[utility_column].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Daily utility values are non-finite."
    )

if len(effects) != 720:
    raise RuntimeError(
        f"Expected 720 action labels, got {len(effects)}."
    )

if len(daily) != 108_000:
    raise RuntimeError(
        f"Expected 108000 daily rows, got {len(daily)}."
    )

require_unique_rows(
    effects,
    keys=list(ACTION_KEYS),
    label="V5-D action-effects",
)

if not effects["action"].isin(ACTION_ORDER).all():
    raise RuntimeError(
        "Action-effect action space is unexpected."
    )

require_complete_action_contexts(effects)

effects["requested_rank"] = normalize_integer(
    effects["requested_rank"],
    label="effects requested_rank",
)
effects["achieved_rank"] = normalize_integer(
    effects["achieved_rank"],
    label="effects achieved_rank",
)

expected_requested_rank = effects["action"].map(
    ACTION_TO_TARGET_RANK
).astype(np.int64)

if not effects["requested_rank"].eq(
    expected_requested_rank
).all():
    raise RuntimeError(
        "Action effects requested rank mismatch."
    )

if not effects["achieved_rank"].eq(
    effects["requested_rank"]
).all():
    raise RuntimeError(
        "Action effects did not achieve requested rank."
    )

policy_values = sorted(
    daily[policy_column]
    .astype(str)
    .dropna()
    .unique()
    .tolist()
)

if policy_values != [
    EXPECTED_REFERENCE_POLICY,
    EXPECTED_TREATMENT_POLICY,
]:
    raise RuntimeError(
        "Daily policy values do not match V5-I.3 contract: "
        f"{policy_values}"
    )

daily[policy_column] = daily[policy_column].astype(str)

policy_grain = [
    *ACTION_KEYS,
    scenario_column,
    replication_column,
    day_column,
    policy_column,
]

require_unique_rows(
    daily,
    keys=policy_grain,
    label="policy-augmented daily table",
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
        "Daily table does not cover all 720 action labels."
    )

if not daily_counts["daily_row_count"].eq(150).all():
    raise RuntimeError(
        "Daily table does not have 150 rows per action."
    )

blocks = policy_delta_block_table(
    daily,
    policy_column=policy_column,
    reference_policy=EXPECTED_REFERENCE_POLICY,
    treatment_policy=EXPECTED_TREATMENT_POLICY,
    scenario_column=scenario_column,
    replication_column=replication_column,
    day_column=day_column,
    utility_column=utility_column,
)

reconstructed = action_means_from_blocks(blocks)
assert_exact_action_reconstruction(
    effects,
    reconstructed,
)

effect_metrics = effects.loc[
    :,
    [
        *ACTION_KEYS,
        TARGET_UTILITY_COLUMN,
        *SAFETY_COLUMNS.values(),
    ],
].rename(
    columns={
        TARGET_UTILITY_COLUMN: "mean_utility",
        SAFETY_COLUMNS[
            "relevant_discovery"
        ]: "relevant_discovery",
        SAFETY_COLUMNS[
            "irrelevant_exposure"
        ]: "irrelevant_exposure",
        SAFETY_COLUMNS[
            "false_warmup"
        ]: "false_warmup",
        SAFETY_COLUMNS[
            "worst_utility"
        ]: "worst_utility",
        SAFETY_COLUMNS[
            "p10_utility"
        ]: "p10_utility",
    }
)

contrast_rows: list[dict[str, Any]] = []
block_coverage_rows: list[dict[str, Any]] = []

for alternative_action in ALTERNATIVE_ACTIONS:
    paired_blocks = context_action_contrast_blocks(
        blocks,
        alternative_action=alternative_action,
        scenario_column=scenario_column,
        replication_column=replication_column,
    )

    coverage = (
        paired_blocks.groupby(
            list(CONTEXT_KEYS),
            as_index=False,
        )
        .agg(
            paired_block_count=(
                "paired_action_utility_contrast",
                "size",
            ),
            scenario_count=(scenario_column, "nunique"),
            replication_count=(replication_column, "nunique"),
        )
    )

    coverage.insert(
        2,
        "alternative_action",
        alternative_action,
    )

    if not coverage["paired_block_count"].eq(15).all():
        raise RuntimeError(
            "Every context-action contrast must have 15 paired "
            "scenario-replication blocks."
        )

    if not coverage["scenario_count"].eq(3).all():
        raise RuntimeError(
            "Every context-action contrast must have 3 scenarios."
        )

    if not coverage["replication_count"].eq(5).all():
        raise RuntimeError(
            "Every context-action contrast must have 5 replication labels."
        )

    block_coverage_rows.extend(
        coverage.to_dict(orient="records")
    )

    for (seed, proposal_index), local_blocks in paired_blocks.groupby(
        list(CONTEXT_KEYS),
        sort=True,
    ):
        bootstrap = stratified_bootstrap(
            local_blocks,
            scenario_column=scenario_column,
            replication_column=replication_column,
            seed=int(seed),
            proposal_index=int(proposal_index),
            action=alternative_action,
        )

        effect_context = effect_metrics.loc[
            (effect_metrics["seed"].eq(seed))
            & (
                effect_metrics["proposal_index"]
                .eq(proposal_index)
            ),
            :,
        ].copy()

        if len(effect_context) != 5:
            raise RuntimeError(
                "Action-effect context is incomplete."
            )

        safety = safety_contrast(
            effect_context,
            alternative_action=alternative_action,
        )

        expected_mean = float(
            effect_context.loc[
                effect_context["action"].eq(
                    alternative_action
                ),
                "mean_utility",
            ].iloc[0]
            - effect_context.loc[
                effect_context["action"].eq(
                    PAIR_BASELINE_ACTION
                ),
                "mean_utility",
            ].iloc[0]
        )

        if not math.isclose(
            bootstrap["paired_mean_utility_contrast"],
            expected_mean,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Paired block contrast does not reconstruct its "
                "action-effect contrast for "
                f"seed={seed}, proposal_index={proposal_index}, "
                f"action={alternative_action}."
            )

        stable_positive = bool(
            bootstrap["paired_ci_low_95"] > 0.0
        )
        stable_negative = bool(
            bootstrap["paired_ci_high_95"] < 0.0
        )
        uncertain = bool(
            not stable_positive and not stable_negative
        )
        strict_safe_positive = bool(
            stable_positive
            and safety["aggregate_safety_nonworsening"]
        )

        contrast_rows.append(
            {
                "seed": int(seed),
                "proposal_index": int(proposal_index),
                "baseline_action": PAIR_BASELINE_ACTION,
                "alternative_action": alternative_action,
                "baseline_target_rank": ACTION_TO_TARGET_RANK[
                    PAIR_BASELINE_ACTION
                ],
                "alternative_target_rank": ACTION_TO_TARGET_RANK[
                    alternative_action
                ],
                **bootstrap,
                **safety,
                "strict_paired_utility_advantage": stable_positive,
                "strict_paired_utility_disadvantage": stable_negative,
                "paired_utility_direction_uncertain": uncertain,
                "strict_safe_paired_advantage": strict_safe_positive,
            }
        )

contrasts = pd.DataFrame(contrast_rows)
block_coverage = pd.DataFrame(block_coverage_rows)

if len(contrasts) != 576:
    raise RuntimeError(
        "Expected 576 context-action paired contrasts, found "
        f"{len(contrasts)}."
    )

if len(block_coverage) != 576:
    raise RuntimeError(
        "Expected 576 paired-block coverage rows."
    )

if not block_coverage["paired_block_count"].eq(15).all():
    raise RuntimeError(
        "Paired block coverage is incomplete."
    )

utility_pivot = (
    effect_metrics.pivot(
        index=list(CONTEXT_KEYS),
        columns="action",
        values="mean_utility",
    )
    .reindex(columns=list(ACTION_ORDER))
    .reset_index()
)

if utility_pivot.isna().any().any():
    raise RuntimeError(
        "Action utility pivot is incomplete."
    )

oracle_long = utility_pivot.melt(
    id_vars=list(CONTEXT_KEYS),
    value_vars=list(ACTION_ORDER),
    var_name="action",
    value_name="mean_utility",
)

oracle_long["target_rank"] = oracle_long["action"].map(
    ACTION_TO_TARGET_RANK
)

oracle = (
    oracle_long.sort_values(
        [
            "seed",
            "proposal_index",
            "mean_utility",
            "target_rank",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    .groupby(
        list(CONTEXT_KEYS),
        as_index=False,
        sort=False,
    )
    .first()
    .rename(
        columns={
            "action": "oracle_action",
            "target_rank": "oracle_target_rank",
            "mean_utility": "oracle_mean_utility",
        }
    )
)

baseline_utility = utility_pivot.loc[
    :,
    [
        *CONTEXT_KEYS,
        PAIR_BASELINE_ACTION,
    ],
].rename(
    columns={
        PAIR_BASELINE_ACTION: "place_at_1_mean_utility",
    }
)

oracle = oracle.merge(
    baseline_utility,
    on=list(CONTEXT_KEYS),
    how="inner",
    validate="one_to_one",
)

oracle["oracle_gap_vs_place_at_1"] = np.maximum(
    0.0,
    (
        oracle["oracle_mean_utility"]
        - oracle["place_at_1_mean_utility"]
    ).to_numpy(dtype=float),
)

oracle["oracle_differs_from_place_at_1"] = (
    oracle["oracle_action"].ne(PAIR_BASELINE_ACTION)
)

oracle_contrast = contrasts.loc[
    :,
    [
        *CONTEXT_KEYS,
        "alternative_action",
        "paired_ci_low_95",
        "paired_ci_high_95",
        "strict_paired_utility_advantage",
        "strict_safe_paired_advantage",
    ],
].rename(
    columns={
        "alternative_action": "oracle_action",
        "paired_ci_low_95": (
            "oracle_action_paired_ci_low_95"
        ),
        "paired_ci_high_95": (
            "oracle_action_paired_ci_high_95"
        ),
        "strict_paired_utility_advantage": (
            "oracle_action_strict_paired_advantage"
        ),
        "strict_safe_paired_advantage": (
            "oracle_action_strict_safe_paired_advantage"
        ),
    }
)

oracle = oracle.merge(
    oracle_contrast,
    on=[*CONTEXT_KEYS, "oracle_action"],
    how="left",
    validate="one_to_one",
)

baseline_oracle_mask = oracle[
    "oracle_action"
].eq(PAIR_BASELINE_ACTION)

oracle.loc[
    baseline_oracle_mask,
    [
        "oracle_action_paired_ci_low_95",
        "oracle_action_paired_ci_high_95",
    ],
] = 0.0

oracle.loc[
    baseline_oracle_mask,
    [
        "oracle_action_strict_paired_advantage",
        "oracle_action_strict_safe_paired_advantage",
    ],
] = False

if oracle[
    [
        "oracle_action_strict_paired_advantage",
        "oracle_action_strict_safe_paired_advantage",
    ]
].isna().any().any():
    raise RuntimeError(
        "Non-baseline oracle action did not join to its paired "
        "contrast stability record."
    )

oracle[
    "oracle_action_strict_paired_advantage"
] = oracle[
    "oracle_action_strict_paired_advantage"
].astype(bool)

oracle[
    "oracle_action_strict_safe_paired_advantage"
] = oracle[
    "oracle_action_strict_safe_paired_advantage"
].astype(bool)

oracle["stable_oracle_gap"] = np.where(
    (
        oracle["oracle_differs_from_place_at_1"]
        & oracle[
            "oracle_action_strict_paired_advantage"
        ]
    ),
    oracle["oracle_gap_vs_place_at_1"],
    0.0,
)

oracle["strict_safe_stable_oracle_gap"] = np.where(
    (
        oracle["oracle_differs_from_place_at_1"]
        & oracle[
            "oracle_action_strict_safe_paired_advantage"
        ]
    ),
    oracle["oracle_gap_vs_place_at_1"],
    0.0,
)

oracle["unstable_oracle_gap"] = (
    oracle["oracle_gap_vs_place_at_1"]
    - oracle["stable_oracle_gap"]
)

total_oracle_gap = float(
    oracle["oracle_gap_vs_place_at_1"].sum()
)
stable_oracle_gap = float(
    oracle["stable_oracle_gap"].sum()
)
strict_safe_stable_oracle_gap = float(
    oracle["strict_safe_stable_oracle_gap"].sum()
)

stable_oracle_gap_share = (
    stable_oracle_gap / total_oracle_gap
    if total_oracle_gap > 0.0
    else 0.0
)

strict_safe_stable_oracle_gap_share = (
    strict_safe_stable_oracle_gap / total_oracle_gap
    if total_oracle_gap > 0.0
    else 0.0
)

by_action = (
    contrasts.groupby(
        "alternative_action",
        as_index=False,
    )
    .agg(
        contrast_count=("proposal_index", "size"),
        mean_action_effect_utility_contrast=(
            "paired_mean_utility_contrast",
            "mean",
        ),
        median_action_effect_utility_contrast=(
            "paired_mean_utility_contrast",
            "median",
        ),
        strict_paired_utility_advantage_count=(
            "strict_paired_utility_advantage",
            "sum",
        ),
        strict_paired_utility_disadvantage_count=(
            "strict_paired_utility_disadvantage",
            "sum",
        ),
        uncertain_contrast_count=(
            "paired_utility_direction_uncertain",
            "sum",
        ),
        strict_safe_paired_advantage_count=(
            "strict_safe_paired_advantage",
            "sum",
        ),
        mean_bootstrap_probability_positive=(
            "bootstrap_probability_positive",
            "mean",
        ),
        mean_irrelevant_exposure_contrast=(
            "irrelevant_exposure_contrast",
            "mean",
        ),
        mean_false_warmup_contrast=(
            "false_warmup_contrast",
            "mean",
        ),
        mean_worst_utility_contrast=(
            "worst_utility_contrast",
            "mean",
        ),
        mean_p10_utility_contrast=(
            "p10_utility_contrast",
            "mean",
        ),
    )
    .sort_values(
        "alternative_action",
        kind="mergesort",
    )
)

by_seed = (
    contrasts.groupby("seed", as_index=False)
    .agg(
        paired_contrast_count=("proposal_index", "size"),
        strict_paired_utility_advantage_count=(
            "strict_paired_utility_advantage",
            "sum",
        ),
        strict_paired_utility_disadvantage_count=(
            "strict_paired_utility_disadvantage",
            "sum",
        ),
        strict_safe_paired_advantage_count=(
            "strict_safe_paired_advantage",
            "sum",
        ),
        uncertain_contrast_count=(
            "paired_utility_direction_uncertain",
            "sum",
        ),
    )
)

utility_stable_contexts = (
    contrasts.loc[
        contrasts["strict_paired_utility_advantage"],
        list(CONTEXT_KEYS),
    ]
    .drop_duplicates()
)

safe_stable_contexts = (
    contrasts.loc[
        contrasts["strict_safe_paired_advantage"],
        list(CONTEXT_KEYS),
    ]
    .drop_duplicates()
)

utility_stable_context_count = int(
    len(utility_stable_contexts)
)
utility_stable_seed_count = int(
    utility_stable_contexts["seed"].nunique()
)

safe_stable_context_count = int(
    len(safe_stable_contexts)
)
safe_stable_seed_count = int(
    safe_stable_contexts["seed"].nunique()
)

utility_label_signal = bool(
    utility_stable_context_count >= MIN_STABLE_CONTEXTS
    and utility_stable_seed_count >= MIN_STABLE_SEEDS
    and stable_oracle_gap_share >= MIN_STABLE_ORACLE_GAP_SHARE
)

safe_label_signal = bool(
    safe_stable_context_count >= MIN_STABLE_CONTEXTS
    and safe_stable_seed_count >= MIN_STABLE_SEEDS
    and strict_safe_stable_oracle_gap_share
    >= MIN_STABLE_ORACLE_GAP_SHARE
)

if safe_label_signal:
    status = (
        "V5I4_PAIRED_CONTRAST_STABLE_SAFE_LABEL_SIGNAL_PRESENT"
    )
elif utility_label_signal:
    status = (
        "V5I4_PAIRED_CONTRAST_UTILITY_STABILITY_ONLY"
    )
else:
    status = (
        "V5I4_PAIRED_CONTRAST_STABILITY_INSUFFICIENT"
    )

output_root.mkdir(parents=True, exist_ok=True)

blocks_path = (
    output_root
    / "v5i4_policy_delta_scenario_replication_blocks.csv"
)
reconstructed_path = (
    output_root
    / "v5i4_reconstructed_action_utility_from_blocks.csv"
)
block_coverage_path = (
    output_root
    / "v5i4_action_contrast_block_coverage.csv"
)
contrasts_path = (
    output_root
    / "v5i4_paired_context_action_contrasts.csv"
)
by_action_path = (
    output_root
    / "v5i4_paired_contrast_summary_by_action.csv"
)
by_seed_path = (
    output_root
    / "v5i4_paired_contrast_summary_by_seed.csv"
)
oracle_path = (
    output_root
    / "v5i4_oracle_gap_stability_decomposition.csv"
)
daily_counts_path = (
    output_root
    / "v5i4_daily_rows_per_action.csv"
)

blocks.to_csv(blocks_path, index=False)
reconstructed.to_csv(reconstructed_path, index=False)
block_coverage.to_csv(block_coverage_path, index=False)
contrasts.to_csv(contrasts_path, index=False)
by_action.to_csv(by_action_path, index=False)
by_seed.to_csv(by_seed_path, index=False)
oracle.to_csv(oracle_path, index=False)
daily_counts.to_csv(daily_counts_path, index=False)

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
    "v5i3_decision_path": str(v5i3_decision_path),
    "v5i3_decision_sha256": sha256(v5i3_decision_path),
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
    "policy_delta_label_contract": {
        "reference_policy": EXPECTED_REFERENCE_POLICY,
        "treatment_policy": EXPECTED_TREATMENT_POLICY,
        "daily_utility_column": utility_column,
        "scenario_column": scenario_column,
        "replication_column": replication_column,
        "day_column": day_column,
        "daily_policy_pair_grain": policy_grain,
        "action_label_aggregation": (
            "mean over 15 scenario-replication blocks of "
            "the 5-day qrsbt_gate-minus-base utility sum"
        ),
        "utility_reconstruction_max_abs_error": float(
            np.abs(
                effects[
                    TARGET_UTILITY_COLUMN
                ].to_numpy(dtype=float)
                - reconstructed[
                    "reconstructed_mean_utility_delta"
                ].to_numpy(dtype=float)
            ).max()
        ),
    },
    "paired_contrast_bootstrap_contract": {
        "baseline_action": PAIR_BASELINE_ACTION,
        "alternative_actions": list(ALTERNATIVE_ACTIONS),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "resampling_unit": (
            "paired replication blocks resampled with replacement "
            "within each of three scenarios, preserving paired "
            "action differences and equal scenario weighting"
        ),
        "paired_block_count_per_context_action": 15,
        "scenario_count_per_context_action": 3,
        "replications_per_scenario": 5,
        "stable_positive_rule": (
            "scenario-stratified paired bootstrap 95% lower bound > 0"
        ),
        "stable_negative_rule": (
            "scenario-stratified paired bootstrap 95% upper bound < 0"
        ),
        "strict_safe_rule": (
            "stable positive utility contrast plus aggregate "
            "nonworsening irrelevant exposure, false warmup, "
            "worst utility, and P10 utility versus PLACE_AT_1"
        ),
    },
    "label_stability_gate": {
        "minimum_stable_contexts": MIN_STABLE_CONTEXTS,
        "minimum_stable_seeds": MIN_STABLE_SEEDS,
        "minimum_stable_oracle_gap_share": (
            MIN_STABLE_ORACLE_GAP_SHARE
        ),
        "utility_stable_context_count": (
            utility_stable_context_count
        ),
        "utility_stable_seed_count": (
            utility_stable_seed_count
        ),
        "safe_stable_context_count": (
            safe_stable_context_count
        ),
        "safe_stable_seed_count": (
            safe_stable_seed_count
        ),
        "utility_label_signal": utility_label_signal,
        "safe_label_signal": safe_label_signal,
    },
    "oracle_gap_decomposition": {
        "oracle_non_place_at_1_context_count": int(
            oracle["oracle_differs_from_place_at_1"].sum()
        ),
        "total_oracle_gap_vs_place_at_1": total_oracle_gap,
        "stable_oracle_gap": stable_oracle_gap,
        "unstable_oracle_gap": float(
            oracle["unstable_oracle_gap"].sum()
        ),
        "stable_oracle_gap_share": stable_oracle_gap_share,
        "strict_safe_stable_oracle_gap": (
            strict_safe_stable_oracle_gap
        ),
        "strict_safe_stable_oracle_gap_share": (
            strict_safe_stable_oracle_gap_share
        ),
    },
    "artifacts": {
        "policy_delta_blocks_path": str(blocks_path),
        "reconstructed_action_utility_path": (
            str(reconstructed_path)
        ),
        "action_contrast_block_coverage_path": (
            str(block_coverage_path)
        ),
        "paired_context_action_contrasts_path": (
            str(contrasts_path)
        ),
        "paired_contrast_summary_by_action_path": (
            str(by_action_path)
        ),
        "paired_contrast_summary_by_seed_path": (
            str(by_seed_path)
        ),
        "oracle_gap_stability_decomposition_path": (
            str(oracle_path)
        ),
        "daily_rows_per_action_path": str(daily_counts_path),
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
        "A paired-contrast model viability audit may be considered "
        "only if the stable safe-label gate passes. Do not fit a "
        "final artifact or execute calibration/confirmation seeds "
        "from this read-only audit."
        if status
        == "V5I4_PAIRED_CONTRAST_STABLE_SAFE_LABEL_SIGNAL_PRESENT"
        else (
            "Utility contrast is stable without meeting the strict "
            "relative-safety gate. Do not fit a serving policy or "
            "execute calibration/confirmation."
            if status
            == "V5I4_PAIRED_CONTRAST_UTILITY_STABILITY_ONLY"
            else "Close V5 selective-promotion optimization with "
            "the source baseline retained. The fully observed "
            "simulator does not provide enough stable, safe "
            "paired action contrast to justify another predictive "
            "model search."
        )
    ),
}

decision_path = (
    output_root
    / "v5i4_policy_delta_paired_contrast_stability_decision.json"
)

write_json(decision_path, decision)

print("===== V5-I.4 POLICY-DELTA PAIRED CONTRAST STABILITY AUDIT =====")
print(
    json.dumps(
        {
            "decision": decision,
            "contrast_summary_by_action": by_action.to_dict(
                orient="records"
            ),
        },
        indent=2,
        sort_keys=True,
        default=as_native,
    )
)

print("===== V5-I.4 DECISION =====")
print(
    json.dumps(
        {
            "status": status,
            "label_stability_gate": (
                decision["label_stability_gate"]
            ),
            "oracle_gap_decomposition": (
                decision["oracle_gap_decomposition"]
            ),
        },
        indent=2,
        sort_keys=True,
    )
)