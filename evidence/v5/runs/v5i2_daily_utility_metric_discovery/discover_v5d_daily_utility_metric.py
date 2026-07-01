from __future__ import annotations

import hashlib
import json
import math
import re
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

EXPECTED_UTILITY_COLUMN = (
    "mean_scenario_replication_utility_delta"
)

DIRECT_RECONCILIATION_TOLERANCE = 1e-8
TOP_CANDIDATE_COUNT = 30

UTILITY_NAME_HINTS = (
    "utility",
    "reward",
    "value",
    "benefit",
    "gain",
    "objective",
    "replication",
)

KEY_OR_INDEX_NAME_TOKENS = (
    "seed",
    "proposal",
    "action",
    "scenario",
    "replication",
    "replicate",
    "day",
    "index",
    "id",
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


def candidate_numeric_series(
    series: pd.Series,
) -> tuple[pd.Series, float, bool]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = numeric.to_numpy(dtype=float)

    finite_rate = float(np.isfinite(values).mean())

    return (
        numeric,
        finite_rate,
        finite_rate == 1.0,
    )


def named_hint(column: str) -> bool:
    lower = column.lower()

    return any(
        hint in lower
        for hint in UTILITY_NAME_HINTS
    )


def key_or_index_like(column: str) -> bool:
    lower = column.lower()

    return (
        lower in ACTION_KEYS
        or any(
            token == lower
            or lower.endswith(f"_{token}")
            for token in KEY_OR_INDEX_NAME_TOKENS
        )
    )


def correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    if (
        len(left) < 2
        or np.std(left) <= 0.0
        or np.std(right) <= 0.0
    ):
        return None

    return float(np.corrcoef(left, right)[0, 1])


def aggregate_candidate(
    daily: pd.DataFrame,
    *,
    column: str,
) -> pd.DataFrame:
    return (
        daily.groupby(
            list(ACTION_KEYS),
            as_index=False,
            dropna=False,
        )[column]
        .mean()
        .rename(
            columns={
                column: "candidate_action_mean",
            }
        )
    )


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
        "V5-H unexpectedly includes a final serving model."
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
        "Required V5-D action-effect/daily artifact is missing."
    )

effects = pd.read_csv(effects_path)
daily = pd.read_csv(daily_path)

for frame, label in (
    (effects, "V5-D action effects"),
    (daily, "V5-D daily table"),
):
    missing = sorted(
        set(ACTION_KEYS) - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} missing action keys: {missing}"
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

if effects.duplicated(list(ACTION_KEYS)).any():
    raise RuntimeError(
        "Action-effects table has duplicate action keys."
    )

if len(effects) != 720:
    raise RuntimeError(
        f"Expected 720 action-effect rows, found {len(effects)}."
    )

if not effects["action"].notna().all():
    raise RuntimeError(
        "Action-effects table has a missing action label."
    )

if EXPECTED_UTILITY_COLUMN not in effects.columns:
    raise RuntimeError(
        "Action-effects table is missing frozen mean-utility label."
    )

effects[EXPECTED_UTILITY_COLUMN] = pd.to_numeric(
    effects[EXPECTED_UTILITY_COLUMN],
    errors="raise",
)

if not np.isfinite(
    effects[EXPECTED_UTILITY_COLUMN].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Frozen action-effect mean-utility label is non-finite."
    )

expected = effects.loc[
    :,
    [
        *ACTION_KEYS,
        EXPECTED_UTILITY_COLUMN,
    ],
].copy()

daily_action_counts = (
    daily.groupby(
        list(ACTION_KEYS),
        as_index=False,
        dropna=False,
    )
    .size()
    .rename(columns={"size": "daily_row_count"})
)

if len(daily_action_counts) != 720:
    raise RuntimeError(
        "Daily table does not cover all 720 V5-D actions."
    )

if not daily_action_counts["daily_row_count"].eq(150).all():
    raise RuntimeError(
        "Daily table violates the 150 rows/action cardinality."
    )

schema_rows: list[dict[str, Any]] = []
candidate_rows: list[dict[str, Any]] = []

for column in daily.columns:
    numeric, finite_rate, fully_numeric = candidate_numeric_series(
        daily[column]
    )

    source_dtype = str(daily[column].dtype)
    non_null_count = int(daily[column].notna().sum())
    unique_count = int(daily[column].nunique(dropna=False))
    is_key = column in ACTION_KEYS
    is_index_like = key_or_index_like(column)
    hint = named_hint(column)

    schema_rows.append(
        {
            "column": column,
            "source_dtype": source_dtype,
            "row_count": int(len(daily)),
            "non_null_count": non_null_count,
            "missing_rate": float(daily[column].isna().mean()),
            "unique_count_including_missing": unique_count,
            "numeric_finite_rate_after_to_numeric": finite_rate,
            "fully_numeric_after_to_numeric": fully_numeric,
            "is_action_key": is_key,
            "is_key_or_index_like_name": is_index_like,
            "has_utility_semantic_hint": hint,
        }
    )

    if is_key or not fully_numeric:
        continue

    temporary = daily.loc[
        :,
        list(ACTION_KEYS),
    ].copy()

    temporary[column] = numeric

    aggregate = aggregate_candidate(
        temporary,
        column=column,
    )

    joined = expected.merge(
        aggregate,
        on=list(ACTION_KEYS),
        how="left",
        validate="one_to_one",
    )

    complete_join = bool(
        joined["candidate_action_mean"].notna().all()
    )

    if not complete_join:
        candidate_rows.append(
            {
                "column": column,
                "source_dtype": source_dtype,
                "has_utility_semantic_hint": hint,
                "is_key_or_index_like_name": is_index_like,
                "complete_action_join": False,
                "mean_abs_error_identity": math.inf,
                "max_abs_error_identity": math.inf,
                "mean_abs_error_negative_identity": math.inf,
                "max_abs_error_negative_identity": math.inf,
                "pearson_correlation_identity": None,
                "direct_identity_exact": False,
                "negative_identity_exact": False,
            }
        )
        continue

    observed = joined[
        "candidate_action_mean"
    ].to_numpy(dtype=float)

    target = joined[
        EXPECTED_UTILITY_COLUMN
    ].to_numpy(dtype=float)

    identity_error = np.abs(observed - target)
    negative_error = np.abs(-observed - target)

    identity_mean_error = float(identity_error.mean())
    identity_max_error = float(identity_error.max())
    negative_mean_error = float(negative_error.mean())
    negative_max_error = float(negative_error.max())

    candidate_rows.append(
        {
            "column": column,
            "source_dtype": source_dtype,
            "has_utility_semantic_hint": hint,
            "is_key_or_index_like_name": is_index_like,
            "complete_action_join": True,
            "mean_abs_error_identity": identity_mean_error,
            "max_abs_error_identity": identity_max_error,
            "mean_abs_error_negative_identity": negative_mean_error,
            "max_abs_error_negative_identity": negative_max_error,
            "pearson_correlation_identity": correlation(
                observed,
                target,
            ),
            "direct_identity_exact": bool(
                identity_max_error
                <= DIRECT_RECONCILIATION_TOLERANCE
            ),
            "negative_identity_exact": bool(
                negative_max_error
                <= DIRECT_RECONCILIATION_TOLERANCE
            ),
        }
    )

schema = pd.DataFrame(schema_rows).sort_values(
    "column",
    kind="mergesort",
)

candidates = pd.DataFrame(candidate_rows)

if candidates.empty:
    raise RuntimeError(
        "Daily table contains no fully numeric non-key columns "
        "after numeric coercion."
    )

candidates["best_direct_transform"] = np.where(
    candidates["max_abs_error_identity"]
    <= candidates["max_abs_error_negative_identity"],
    "identity",
    "negative_identity",
)

candidates["best_direct_max_abs_error"] = np.minimum(
    candidates["max_abs_error_identity"],
    candidates["max_abs_error_negative_identity"],
)

candidates["best_direct_mean_abs_error"] = np.minimum(
    candidates["mean_abs_error_identity"],
    candidates["mean_abs_error_negative_identity"],
)

candidates["direct_reconciliation_exact"] = (
    candidates["best_direct_max_abs_error"]
    <= DIRECT_RECONCILIATION_TOLERANCE
)

candidates = candidates.sort_values(
    [
        "direct_reconciliation_exact",
        "best_direct_max_abs_error",
        "best_direct_mean_abs_error",
        "has_utility_semantic_hint",
        "is_key_or_index_like_name",
        "column",
    ],
    ascending=[False, True, True, False, True, True],
    kind="mergesort",
).reset_index(drop=True)

exact = candidates.loc[
    candidates["direct_reconciliation_exact"]
].copy()

if len(exact) == 1:
    status = "V5I2_DAILY_UTILITY_IDENTITY_RECOVERED"
    selected = exact.iloc[0]
elif len(exact) > 1:
    status = "V5I2_DAILY_UTILITY_IDENTITY_AMBIGUOUS"
    selected = None
else:
    status = "V5I2_DAILY_UTILITY_IDENTITY_UNRESOLVED"
    selected = None

if selected is not None:
    selected_column = str(selected["column"])
    selected_transform = str(
        selected["best_direct_transform"]
    )
    selected_max_abs_error = float(
        selected["best_direct_max_abs_error"]
    )
    selected_mean_abs_error = float(
        selected["best_direct_mean_abs_error"]
    )
else:
    selected_column = None
    selected_transform = None
    selected_max_abs_error = None
    selected_mean_abs_error = None

output_root.mkdir(parents=True, exist_ok=True)

schema_path = (
    output_root
    / "v5i2_daily_schema_inventory.csv"
)

candidate_path = (
    output_root
    / "v5i2_daily_numeric_metric_reconciliation.csv"
)

top_candidate_path = (
    output_root
    / "v5i2_top_daily_metric_candidates.csv"
)

action_count_path = (
    output_root
    / "v5i2_daily_rows_per_action.csv"
)

schema.to_csv(schema_path, index=False)
candidates.to_csv(candidate_path, index=False)
candidates.head(TOP_CANDIDATE_COUNT).to_csv(
    top_candidate_path,
    index=False,
)
daily_action_counts.to_csv(
    action_count_path,
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
    "daily_row_count": int(len(daily)),
    "action_effect_row_count": int(len(effects)),
    "daily_rows_per_action": {
        "min": int(daily_action_counts["daily_row_count"].min()),
        "max": int(daily_action_counts["daily_row_count"].max()),
    },
    "frozen_target_column": EXPECTED_UTILITY_COLUMN,
    "reconciliation_method": (
        "Evaluate every fully numeric, non-action-key daily "
        "column by its mean within each seed/proposal/action. "
        "Compare both identity and negative-identity transforms "
        "against the frozen V5-D mean utility label. No outcome "
        "model is fitted and no source artifact is modified."
    ),
    "direct_reconciliation_tolerance": (
        DIRECT_RECONCILIATION_TOLERANCE
    ),
    "exact_reconciliation_candidate_count": int(len(exact)),
    "selected_daily_utility_column": selected_column,
    "selected_daily_utility_transform": selected_transform,
    "selected_max_abs_error": selected_max_abs_error,
    "selected_mean_abs_error": selected_mean_abs_error,
    "top_candidate_records": candidates.head(
        TOP_CANDIDATE_COUNT
    ).to_dict(orient="records"),
    "artifacts": {
        "daily_schema_inventory_path": str(schema_path),
        "numeric_metric_reconciliation_path": str(
            candidate_path
        ),
        "top_metric_candidates_path": str(
            top_candidate_path
        ),
        "daily_rows_per_action_path": str(
            action_count_path
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
        "Patch V5-I only to use the uniquely reconciled daily "
        "utility metric and its recorded transform. Do not "
        "execute calibration/confirmation or fit a predictive "
        "model in this discovery audit."
        if status
        == "V5I2_DAILY_UTILITY_IDENTITY_RECOVERED"
        else (
            "Inspect the exact-reconciliation candidates before "
            "choosing a scientifically defined utility metric. "
            "Do not rerun V5-I automatically."
            if status
            == "V5I2_DAILY_UTILITY_IDENTITY_AMBIGUOUS"
            else "Inspect the top numeric reconciliation report "
            "and the immutable daily schema. The daily table "
            "does not expose a direct mean-utility identity "
            "under the current aggregation contract."
        )
    ),
}

decision_path = (
    output_root
    / "v5i2_daily_utility_metric_discovery_decision.json"
)

write_json(decision_path, decision)

print("===== V5-I.2 DAILY UTILITY-METRIC DISCOVERY =====")
print(
    json.dumps(
        {
            "decision": decision,
            "top_daily_metric_candidates": (
                candidates.head(TOP_CANDIDATE_COUNT).to_dict(
                    orient="records"
                )
            ),
            "daily_schema": schema.to_dict(
                orient="records"
            ),
        },
        indent=2,
        sort_keys=True,
        default=str,
    )
)

print("===== V5-I.2 DECISION =====")
print(
    json.dumps(
        {
            "status": status,
            "exact_reconciliation_candidate_count": int(
                len(exact)
            ),
            "selected_daily_utility_column": selected_column,
            "selected_daily_utility_transform": (
                selected_transform
            ),
            "selected_max_abs_error": selected_max_abs_error,
            "selected_mean_abs_error": selected_mean_abs_error,
            "top_daily_metric_candidates": (
                candidates.head(10).to_dict(
                    orient="records"
                )
            ),
        },
        indent=2,
        sort_keys=True,
        default=str,
    )
)