from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


v5d_decision_path = Path(sys.argv[1]).resolve()
v5e_decision_path = Path(sys.argv[2]).resolve()
v5f_decision_path = Path(sys.argv[3]).resolve()
corpus_root = Path(sys.argv[4]).resolve()
candidate_worktree = Path(sys.argv[5]).resolve()
output_root = Path(sys.argv[6]).resolve()
expected_head = sys.argv[7]

KEY_COLUMNS = ("seed", "query_id", "product_id")
CURRENT_FEATURES = {
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
    "base_rank",
    "proposal_selection_rule",
}

FORBIDDEN_TOKENS = (
    "teacher",
    "oracle",
    "future",
    "outcome",
    "label",
    "utility_delta",
    "discovery_delta",
    "exposure_delta",
    "warmup_delta",
    "achieved_rank",
    "requested_rank",
    "final_score",
    "qrsbt_boost",
)

RUNTIME_PRIORITY_TOKENS = (
    "score",
    "rank",
    "margin",
    "gap",
    "dispersion",
    "entropy",
    "support",
    "confidence",
    "semantic",
    "dense",
    "bm25",
    "retrieval",
    "compatibility",
    "irrelevant",
    "category",
    "intent",
    "relation",
    "source",
    "price",
    "quality",
    "history",
    "cold",
    "candidate",
    "query",
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
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_frame(path: Path, rows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=rows)
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.head(rows) if rows is not None else frame
    raise ValueError(f"Unsupported tabular suffix: {path.suffix}")


def normalise_key_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")
    return series.astype("string").str.strip()


def candidate_roots() -> list[Path]:
    roots = [corpus_root]
    validation_root = candidate_worktree / "artifacts" / "_validation"
    if validation_root.is_dir():
        for child in validation_root.iterdir():
            if child.is_dir() and "v5" in child.name.lower():
                roots.append(child)
    return roots


def tabular_files() -> list[Path]:
    files: set[Path] = set()
    for root in candidate_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            name = path.name.lower()
            if any(token in name for token in ("action_effect", "daily", "summary_by_placement")):
                continue
            files.add(path.resolve())
    return sorted(files)


def forbidden_columns(columns: list[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    )


def safe_runtime_columns(frame: pd.DataFrame) -> list[str]:
    safe = []
    for column in frame.columns:
        lower = column.lower()
        if column in KEY_COLUMNS or column in {"proposal_index", "action"}:
            continue
        if any(token in lower for token in FORBIDDEN_TOKENS):
            continue
        if any(token in lower for token in RUNTIME_PRIORITY_TOKENS):
            safe.append(column)
    return sorted(set(safe))


def make_action_geometry(proposals: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = proposals.loc[:, ["seed", "proposal_index", "query_id", "product_id", "base_rank"]].copy()
    out["base_rank"] = pd.to_numeric(out["base_rank"], errors="coerce")

    for target_rank in (1, 2, 3, 5, 10):
        out[f"move_distance_to_{target_rank}"] = out["base_rank"] - target_rank
        out[f"absolute_move_distance_to_{target_rank}"] = (out["base_rank"] - target_rank).abs()

    candidate_columns = [
        column
        for column in frame.columns
        if column.lower() in {
            "base_score", "score", "retrieval_score", "semantic_rank_score", "dense_score", "bm25_score"
        }
    ]

    if not candidate_columns:
        return out

    raw = frame.copy()
    for key in KEY_COLUMNS:
        raw[key] = normalise_key_series(raw[key])

    proposal_keys = out.copy()
    for key in KEY_COLUMNS:
        proposal_keys[key] = normalise_key_series(proposal_keys[key])

    candidate_score = candidate_columns[0]
    raw[candidate_score] = pd.to_numeric(raw[candidate_score], errors="coerce")

    raw = raw.dropna(subset=[candidate_score])
    raw = raw.sort_values(["seed", "query_id", candidate_score, "product_id"], ascending=[True, True, False, True], kind="mergesort")
    raw["runtime_rank"] = raw.groupby(["seed", "query_id"]).cumcount() + 1

    boundary = raw.loc[raw["runtime_rank"].isin([1, 2, 3, 5, 10]), ["seed", "query_id", "runtime_rank", candidate_score]].copy()
    boundary = boundary.pivot_table(index=["seed", "query_id"], columns="runtime_rank", values=candidate_score, aggfunc="first")
    boundary.columns = [f"runtime_boundary_score_{int(column)}" for column in boundary.columns]
    boundary = boundary.reset_index()

    proposal_scores = proposal_keys.merge(
        raw.loc[:, ["seed", "query_id", "product_id", candidate_score]],
        on=["seed", "query_id", "product_id"],
        how="left",
        validate="one_to_one",
    )

    geometry = out.copy()
    for key in KEY_COLUMNS:
        geometry[key] = normalise_key_series(geometry[key])

    geometry = geometry.merge(boundary, on=["seed", "query_id"], how="left", validate="many_to_one")
    geometry = geometry.merge(
        proposal_scores.loc[:, ["seed", "query_id", "product_id", candidate_score]],
        on=["seed", "query_id", "product_id"],
        how="left",
        validate="one_to_one",
    )

    geometry = geometry.rename(columns={candidate_score: "runtime_candidate_score"})

    for target_rank in (1, 2, 3, 5, 10):
        boundary_column = f"runtime_boundary_score_{target_rank}"
        if boundary_column in geometry.columns:
            geometry[f"runtime_score_gap_to_{target_rank}"] = (
                geometry[boundary_column] - geometry["runtime_candidate_score"]
            )

    return geometry


v5d = read_json(v5d_decision_path)
v5e = read_json(v5e_decision_path)
v5f = read_json(v5f_decision_path)

for decision, status in (
    (v5d, "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"),
    (v5e, "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_AUDIT_COMPLETE"),
    (v5f, "V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT"),
):
    if decision.get("status") != status:
        raise RuntimeError(f"Unexpected prior V5 status: {decision.get('status')}")
    if decision.get("baseline_commit") != expected_head:
        raise RuntimeError("Prior V5 decision baseline commit mismatch.")

proposal_path = Path(v5d["proposal_manifest_path"]).resolve()
action_path = Path(v5d["action_effects_path"]).resolve()
if not proposal_path.is_file() or not action_path.is_file():
    raise RuntimeError("V5-D corpus artifacts are missing.")

proposals = pd.read_csv(proposal_path)
actions = pd.read_csv(action_path, nrows=5)

required_proposal_columns = {"seed", "proposal_index", "query_id", "product_id", "base_rank"}
missing = sorted(required_proposal_columns - set(proposals.columns))
if missing:
    raise RuntimeError(f"Proposal manifest missing required keys: {missing}")

if len(proposals) != 144:
    raise RuntimeError(f"Expected 144 V5-D contexts, found {len(proposals)}.")

if proposals.duplicated(["seed", "proposal_index"]).any():
    raise RuntimeError("Proposal manifest has duplicate seed/proposal keys.")

inventory_rows: list[dict[str, Any]] = []
recoverable: list[tuple[Path, pd.DataFrame, list[str], float]] = []

for path in tabular_files():
    try:
        header = read_frame(path, rows=3)
    except Exception as exc:
        inventory_rows.append({
            "path": str(path),
            "readable": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "row_count": None,
            "columns": None,
            "key_coverage": 0.0,
            "safe_runtime_column_count": 0,
            "forbidden_columns": None,
        })
        continue

    columns = [str(column) for column in header.columns]
    has_keys = set(KEY_COLUMNS).issubset(columns)
    forbidden = forbidden_columns(columns)
    safe_columns = safe_runtime_columns(header)

    row = {
        "path": str(path),
        "readable": True,
        "reason": "",
        "row_count": None,
        "columns": "|".join(columns),
        "has_seed_query_product_keys": has_keys,
        "key_coverage": 0.0,
        "safe_runtime_column_count": len(safe_columns),
        "safe_runtime_columns": "|".join(safe_columns),
        "forbidden_columns": "|".join(forbidden),
    }

    if not has_keys or forbidden:
        inventory_rows.append(row)
        continue

    try:
        full = read_frame(path)
    except Exception as exc:
        row["readable"] = False
        row["reason"] = f"{type(exc).__name__}: {exc}"
        inventory_rows.append(row)
        continue

    for key in KEY_COLUMNS:
        full[key] = normalise_key_series(full[key])

    proposal_key_frame = proposals.loc[:, list(KEY_COLUMNS)].copy()
    for key in KEY_COLUMNS:
        proposal_key_frame[key] = normalise_key_series(proposal_key_frame[key])

    deduplicated = full.drop_duplicates(list(KEY_COLUMNS), keep=False)
    if len(deduplicated) != len(full):
        row["reason"] = "duplicate seed/query/product keys"
        inventory_rows.append(row)
        continue

    joined = proposal_key_frame.merge(
        full.loc[:, list(KEY_COLUMNS)],
        on=list(KEY_COLUMNS),
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    coverage = float((joined["_merge"] == "both").mean())
    row["row_count"] = int(len(full))
    row["key_coverage"] = coverage
    inventory_rows.append(row)

    if coverage == 1.0 and len(safe_columns) > 0:
        recoverable.append((path, full, safe_columns, coverage))

inventory = pd.DataFrame(inventory_rows).sort_values(
    ["key_coverage", "safe_runtime_column_count", "path"],
    ascending=[False, False, True],
    kind="mergesort",
)

if recoverable:
    recoverable.sort(key=lambda item: (-len(item[2]), str(item[0])))
    best_path, best_frame, best_safe_columns, _ = recoverable[0]

    for key in KEY_COLUMNS:
        best_frame[key] = normalise_key_series(best_frame[key])

    selected = proposals.loc[:, ["seed", "proposal_index", "query_id", "product_id"]].copy()
    for key in KEY_COLUMNS:
        selected[key] = normalise_key_series(selected[key])

    raw_columns = list(KEY_COLUMNS) + best_safe_columns
    joined_runtime = selected.merge(
        best_frame.loc[:, raw_columns],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )

    geometry = make_action_geometry(proposals, best_frame)
    for key in KEY_COLUMNS:
        geometry[key] = normalise_key_series(geometry[key])
    for key in KEY_COLUMNS:
        joined_runtime[key] = normalise_key_series(joined_runtime[key])

    joined_runtime = joined_runtime.merge(
        geometry,
        on=["seed", "proposal_index", "query_id", "product_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_geometry"),
    )

    candidate_feature_columns = [
        column
        for column in joined_runtime.columns
        if column not in {"seed", "proposal_index", "query_id", "product_id"}
        and column not in CURRENT_FEATURES
    ]

    feature_rows = []
    for column in candidate_feature_columns:
        series = joined_runtime[column]
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna()
        finite = finite[np.isfinite(finite.to_numpy(dtype=float))]
        is_numeric = len(finite) > 0 or pd.api.types.is_numeric_dtype(series)
        feature_rows.append({
            "feature": column,
            "dtype": str(series.dtype),
            "numeric": bool(is_numeric),
            "missing_count": int(series.isna().sum()),
            "coverage_rate": float(series.notna().mean()),
            "finite_count": int(len(finite)),
            "unique_count": int(series.nunique(dropna=True)),
            "constant": bool(series.nunique(dropna=True) <= 1),
        })
    feature_report = pd.DataFrame(feature_rows).sort_values(
        ["coverage_rate", "constant", "feature"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    eligible_expanded_features = feature_report.loc[
        (feature_report["coverage_rate"] >= 0.99)
        & (~feature_report["constant"])
        & (feature_report["feature"].str.lower().map(lambda value: not any(token in value for token in FORBIDDEN_TOKENS))),
        "feature",
    ].tolist()

    geometry_features = [
        column
        for column in eligible_expanded_features
        if column.startswith("move_distance_")
        or column.startswith("absolute_move_distance_")
        or column.startswith("runtime_score_gap_")
        or column.startswith("runtime_boundary_score_")
        or column == "runtime_candidate_score"
    ]

    status = (
        "V5G_RUNTIME_STATE_RECOVERABILITY_READY"
        if len(eligible_expanded_features) >= 6 and len(geometry_features) >= 4
        else "V5G_RUNTIME_STATE_REINSTRUMENTATION_REQUIRED"
    )

    selected_frame_path = output_root / "v5g_recoverable_runtime_context_snapshot.csv"
    joined_runtime.to_csv(selected_frame_path, index=False)
else:
    best_path = None
    best_safe_columns = []
    feature_report = pd.DataFrame(
        columns=["feature", "dtype", "numeric", "missing_count", "coverage_rate", "finite_count", "unique_count", "constant"]
    )
    eligible_expanded_features = []
    geometry_features = []
    status = "V5G_RUNTIME_STATE_REINSTRUMENTATION_REQUIRED"
    selected_frame_path = None

output_root.mkdir(parents=True, exist_ok=True)
inventory.to_csv(output_root / "v5g_runtime_frame_inventory.csv", index=False)
feature_report.to_csv(output_root / "v5g_candidate_feature_report.csv", index=False)

next_gate = (
    "Freeze the expanded runtime feature contract and run a second seed-disjoint action-value viability audit."
    if status == "V5G_RUNTIME_STATE_RECOVERABILITY_READY"
    else "Do not fit another model. Add a pre-action ranked-frame snapshot to the V5 runner, then regenerate only training-seed contexts under the same fixed sampling rule."
)

decision = {
    "status": status,
    "baseline_commit": expected_head,
    "v5d_decision_path": str(v5d_decision_path),
    "v5d_decision_sha256": sha256(v5d_decision_path),
    "v5e_decision_path": str(v5e_decision_path),
    "v5e_decision_sha256": sha256(v5e_decision_path),
    "v5f_decision_path": str(v5f_decision_path),
    "v5f_decision_sha256": sha256(v5f_decision_path),
    "proposal_manifest_path": str(proposal_path),
    "proposal_manifest_sha256": sha256(proposal_path),
    "action_effects_path": str(action_path),
    "action_effects_sha256": sha256(action_path),
    "runtime_frame_inventory_path": str(output_root / "v5g_runtime_frame_inventory.csv"),
    "candidate_feature_report_path": str(output_root / "v5g_candidate_feature_report.csv"),
    "recoverable_frame_path": str(best_path) if best_path else None,
    "recoverable_frame_sha256": sha256(best_path) if best_path else None,
    "recoverable_runtime_snapshot_path": str(selected_frame_path) if selected_frame_path else None,
    "eligible_expanded_feature_count": len(eligible_expanded_features),
    "eligible_expanded_features": eligible_expanded_features,
    "action_geometry_feature_count": len(geometry_features),
    "action_geometry_features": geometry_features,
    "current_runtime_features": sorted(CURRENT_FEATURES),
    "forbidden_outcome_or_oracle_inputs": True,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "source_modified": False,
    "config_modified": False,
    "commit_created": False,
    "push_performed": False,
    "next_gate": next_gate,
}

write_json(output_root / "v5g_runtime_state_recoverability_decision.json", decision)

print("===== V5-G RUNTIME-STATE RECOVERABILITY AUDIT =====")
print(json.dumps({"decision": decision, "top_frame_inventory": inventory.head(25).to_dict(orient="records")}, indent=2, sort_keys=True, default=str))