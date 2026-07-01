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
v5f_decision_path = Path(sys.argv[3]).resolve()
v5g_decision_path = Path(sys.argv[4]).resolve()
candidate_worktree = Path(sys.argv[5]).resolve()
output_root = Path(sys.argv[6]).resolve()
expected_head = sys.argv[7]

TARGET_RANKS = (1, 2, 3, 5, 10)
EXPECTED_ACTIONS = {
    "PLACE_AT_1": 1,
    "PLACE_AT_2": 2,
    "PLACE_AT_3": 3,
    "PLACE_AT_5": 5,
    "PLACE_AT_10": 10,
}

REQUIRED_MATCH_COLUMNS = (
    "base_rank",
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
)

NUMERIC_WHITELIST = (
    "base_score",
    "base_rank",
    "bm25_score",
    "dense_score",
    "retrieval_score",
    "semantic_rank_score",
    "semantic_component",
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
    "qrsbt_dispersion",
    "qrsbt_support",
    "qrsbt_compatibility",
    "qrsbt_irrelevant_probability",
    "price",
    "price_log",
    "quality",
    "attribute_compatible",
    "zero_history",
    "sparse_history",
)

CATEGORICAL_WHITELIST = (
    "candidate_source",
    "category",
    "query_category",
    "query_intent",
    "relation",
    "qrsbt_transfer_source",
)

FORBIDDEN_EXACT_COLUMNS = {
    "clicked",
    "purchased",
    "relevance",
    "judged",
    "logging_propensity",
    "position",
    "time_block",
    "user_id",
    "behavior_velocity",
    "first_observed_age",
    "final_score",
    "qrsbt_boost",
    "gate_action",
    "gate_reason",
    "requested_rank",
    "achieved_rank",
}

FORBIDDEN_PREFIXES = (
    "prior_",
    "smoothed_",
)

FORBIDDEN_SUBSTRINGS = (
    "teacher",
    "oracle",
    "future",
    "outcome",
    "label",
    "utility_delta",
    "discovery_delta",
    "exposure_delta",
    "warmup_delta",
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
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_integer_key(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite key encountered.")

    rounded = np.rint(values.to_numpy(dtype=float))

    if not np.allclose(values.to_numpy(dtype=float), rounded):
        raise RuntimeError("Non-integral key encountered.")

    return pd.Series(rounded.astype(np.int64), index=series.index)


def seed_from_ranked_path(path: Path) -> int | None:
    for parent in (path.parent, *path.parents):
        match = re.fullmatch(r"seed-(\d+)", parent.name)

        if match:
            return int(match.group(1))

    return None


def validation_experiment_root(path: Path) -> Path | None:
    parts = list(path.parts)

    try:
        index = [
            value.lower()
            for value in parts
        ].index("_validation")
    except ValueError:
        return None

    if index + 1 >= len(parts):
        return None

    return Path(*parts[: index + 2])


def find_complete_ranked_frame_root(
    validation_root: Path,
    expected_seeds: set[int],
) -> tuple[Path, dict[int, Path], list[dict[str, Any]]]:
    candidates: dict[Path, dict[int, list[Path]]] = {}
    inventory_rows: list[dict[str, Any]] = []

    for path in sorted(validation_root.rglob("ranked_test.csv")):
        if not path.is_file():
            continue

        seed = seed_from_ranked_path(path)
        root = validation_experiment_root(path)

        if seed is None or root is None:
            inventory_rows.append(
                {
                    "path": str(path),
                    "seed_from_path": seed,
                    "experiment_root": str(root) if root else None,
                    "usable_for_v5d_recovery": False,
                    "reason": "missing path-derived seed or validation root",
                }
            )
            continue

        candidates.setdefault(root, {}).setdefault(seed, []).append(path)

    complete: list[tuple[Path, dict[int, Path]]] = []

    for root, by_seed in candidates.items():
        selected: dict[int, Path] = {}
        valid = True

        for seed in expected_seeds:
            paths = by_seed.get(seed, [])

            if len(paths) != 1:
                valid = False
                continue

            selected[seed] = paths[0]

        if valid and set(selected) == expected_seeds:
            complete.append((root, selected))

    for root, by_seed in candidates.items():
        available = sorted(by_seed)
        expected_present = sorted(
            expected_seeds & set(available)
        )
        duplicate_seed_count = sum(
            len(paths) != 1
            for paths in by_seed.values()
        )

        inventory_rows.append(
            {
                "path": str(root),
                "seed_from_path": None,
                "experiment_root": str(root),
                "usable_for_v5d_recovery": (
                    len(expected_present) == len(expected_seeds)
                    and duplicate_seed_count == 0
                ),
                "reason": (
                    f"expected_seed_coverage={len(expected_present)}/"
                    f"{len(expected_seeds)}; "
                    f"duplicate_seed_path_count={duplicate_seed_count}"
                ),
            }
        )

    if len(complete) != 1:
        raise RuntimeError(
            "Expected exactly one V5-D ranked-frame root with "
            f"one frame for each training seed; found {len(complete)}."
        )

    return complete[0][0], complete[0][1], inventory_rows


def safe_columns_present(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        column
        for column in NUMERIC_WHITELIST
        if column in frame.columns
    ]

    categorical = [
        column
        for column in CATEGORICAL_WHITELIST
        if column in frame.columns
    ]

    return numeric, categorical


def is_forbidden_column(column: str) -> bool:
    lower = column.lower()

    if lower in FORBIDDEN_EXACT_COLUMNS:
        return True

    if any(lower.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True

    return any(
        token in lower
        for token in FORBIDDEN_SUBSTRINGS
    )


def assert_saved_snapshot_has_no_forbidden(
    frame: pd.DataFrame,
) -> None:
    forbidden = [
        column
        for column in frame.columns
        if is_forbidden_column(column)
    ]

    if forbidden:
        raise RuntimeError(
            "Recovered snapshot contains forbidden columns: "
            f"{sorted(forbidden)}"
        )


def validate_frame_schema(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> None:
    required = {
        "query_id",
        "product_id",
        "base_score",
        "base_rank",
        *REQUIRED_MATCH_COLUMNS,
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise RuntimeError(
            f"seed={seed}: ranked_test missing required columns: {missing}"
        )

    if frame.duplicated(["query_id", "product_id"]).any():
        raise RuntimeError(
            f"seed={seed}: ranked_test has duplicate query-product rows."
        )

    for column in (
        "query_id",
        "product_id",
        "base_rank",
    ):
        frame[column] = normalize_integer_key(frame[column])

    for column in (
        "base_score",
        "qrsbt_relevance_probability",
        "qrsbt_confidence",
        "qrsbt_support_score",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if not np.isfinite(
        frame.loc[
            :,
            [
                "base_score",
                "qrsbt_relevance_probability",
                "qrsbt_confidence",
                "qrsbt_support_score",
            ],
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"seed={seed}: non-finite required runtime value."
        )


def query_geometry(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "seed",
        "query_id",
        "product_id",
        "base_score",
        "base_rank",
        "qrsbt_confidence",
        "qrsbt_relevance_probability",
        "qrsbt_irrelevant_probability",
        "qrsbt_support_score",
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise RuntimeError(
            "Ranked frame missing geometry inputs: "
            f"{missing}"
        )

    work = frame.loc[
        :,
        [
            "seed",
            "query_id",
            "product_id",
            "base_score",
            "base_rank",
            "qrsbt_confidence",
            "qrsbt_relevance_probability",
            "qrsbt_irrelevant_probability",
            "qrsbt_support_score",
        ],
    ].copy()

    work = work.sort_values(
        ["seed", "query_id", "base_rank"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    for (seed, query_id), group in work.groupby(
        ["seed", "query_id"],
        sort=False,
    ):
        expected = np.arange(
            1,
            len(group) + 1,
            dtype=np.int64,
        )

        observed = group["base_rank"].to_numpy(
            dtype=np.int64,
        )

        if not np.array_equal(observed, expected):
            raise RuntimeError(
                "Source-native base_rank is not a contiguous "
                f"one-based permutation for seed={seed}, "
                f"query_id={query_id}."
            )

        scores = group["base_score"].to_numpy(
            dtype=float,
        )

        if not np.all(np.diff(scores) <= 1e-12):
            raise RuntimeError(
                "base_score is not non-increasing in source-native "
                f"base_rank order for seed={seed}, query_id={query_id}."
            )

    work["source_rank"] = work["base_rank"]

    query_size = (
        work.groupby(["seed", "query_id"], as_index=False)
        .agg(
            query_candidate_count=(
                "product_id",
                "size",
            )
        )
    )

    base_summary = (
        work.groupby(["seed", "query_id"], as_index=False)
        .agg(
            query_base_score_max=("base_score", "max"),
            query_base_score_min=("base_score", "min"),
            query_base_score_mean=("base_score", "mean"),
            query_base_score_std=("base_score", "std"),
        )
    )

    for column in (
        "qrsbt_confidence",
        "qrsbt_relevance_probability",
        "qrsbt_irrelevant_probability",
        "qrsbt_support_score",
    ):
        summary = (
            work.groupby(["seed", "query_id"], as_index=False)
            .agg(
                **{
                    f"query_{column}_mean": (
                        column,
                        "mean",
                    ),
                    f"query_{column}_std": (
                        column,
                        "std",
                    ),
                    f"query_{column}_max": (
                        column,
                        "max",
                    ),
                }
            )
        )

        base_summary = base_summary.merge(
            summary,
            on=["seed", "query_id"],
            how="left",
            validate="one_to_one",
        )

    boundaries = (
        work.loc[
            work["source_rank"].isin(TARGET_RANKS),
            [
                "seed",
                "query_id",
                "source_rank",
                "base_score",
            ],
        ]
        .pivot(
            index=["seed", "query_id"],
            columns="source_rank",
            values="base_score",
        )
        .reindex(columns=list(TARGET_RANKS))
        .rename(
            columns={
                rank: f"base_boundary_score_at_{rank}"
                for rank in TARGET_RANKS
            }
        )
        .reset_index()
    )

    top_scores = (
        work.loc[
            work["source_rank"].isin([1, 2]),
            [
                "seed",
                "query_id",
                "source_rank",
                "base_score",
            ],
        ]
        .pivot(
            index=["seed", "query_id"],
            columns="source_rank",
            values="base_score",
        )
        .reindex(columns=[1, 2])
        .rename(
            columns={
                1: "query_top1_base_score",
                2: "query_top2_base_score",
            }
        )
        .reset_index()
    )

    geometry = work.loc[
        :,
        [
            "seed",
            "query_id",
            "product_id",
            "base_score",
            "base_rank",
            "source_rank",
        ],
    ].copy()

    for summary in (
        query_size,
        base_summary,
        boundaries,
        top_scores,
    ):
        geometry = geometry.merge(
            summary,
            on=["seed", "query_id"],
            how="left",
            validate="many_to_one",
        )

    above = work.loc[
        :,
        [
            "seed",
            "query_id",
            "source_rank",
            "base_score",
        ],
    ].copy()

    above["source_rank"] += 1
    above = above.rename(
        columns={
            "base_score": "score_immediately_above",
        }
    )

    below = work.loc[
        :,
        [
            "seed",
            "query_id",
            "source_rank",
            "base_score",
        ],
    ].copy()

    below["source_rank"] -= 1
    below = below.rename(
        columns={
            "base_score": "score_immediately_below",
        }
    )

    geometry = geometry.merge(
        above,
        on=["seed", "query_id", "source_rank"],
        how="left",
        validate="one_to_one",
    ).merge(
        below,
        on=["seed", "query_id", "source_rank"],
        how="left",
        validate="one_to_one",
    )

    geometry["query_base_score_range"] = (
        geometry["query_base_score_max"]
        - geometry["query_base_score_min"]
    )

    geometry["query_top1_to_top2_gap"] = (
        geometry["query_top1_base_score"]
        - geometry["query_top2_base_score"]
    )

    geometry["candidate_score_minus_query_mean"] = (
        geometry["base_score"]
        - geometry["query_base_score_mean"]
    )

    geometry["candidate_score_z"] = np.where(
        geometry["query_base_score_std"].gt(0.0),
        (
            geometry["candidate_score_minus_query_mean"]
            / geometry["query_base_score_std"]
        ),
        0.0,
    )

    geometry["candidate_rank_fraction"] = (
        geometry["source_rank"]
        / geometry["query_candidate_count"]
    )

    geometry["candidate_gap_to_above"] = (
        geometry["score_immediately_above"]
        - geometry["base_score"]
    )

    geometry["candidate_gap_to_below"] = (
        geometry["base_score"]
        - geometry["score_immediately_below"]
    )

    for rank in TARGET_RANKS:
        boundary = f"base_boundary_score_at_{rank}"

        geometry[f"action_score_gap_to_{rank}"] = (
            geometry[boundary]
            - geometry["base_score"]
        )

        geometry[f"action_required_lift_to_{rank}"] = (
            np.nextafter(
                geometry[boundary].to_numpy(dtype=float),
                np.inf,
            )
            - geometry["base_score"].to_numpy(dtype=float)
        )

        geometry[f"action_move_distance_to_{rank}"] = (
            geometry["base_rank"] - rank
        )

    return geometry



def assert_exact_proposal_match(
    proposals: pd.DataFrame,
    recovered: pd.DataFrame,
) -> None:
    merge_columns = [
        "seed",
        "query_id",
        "product_id",
    ]

    source_columns = (
        merge_columns
        + list(REQUIRED_MATCH_COLUMNS)
    )

    source = recovered.loc[:, source_columns].copy()

    joined = proposals.merge(
        source,
        on=merge_columns,
        how="left",
        validate="one_to_one",
        suffixes=("_proposal", "_ranked"),
        indicator=True,
    )

    if not joined["_merge"].eq("both").all():
        missing = joined.loc[
            ~joined["_merge"].eq("both"),
            merge_columns,
        ].to_dict(orient="records")

        raise RuntimeError(
            "Proposal contexts did not all join to ranked frames: "
            f"{missing[:8]}"
        )

    for column in REQUIRED_MATCH_COLUMNS:
        left = pd.to_numeric(
            joined[f"{column}_proposal"],
            errors="raise",
        )
        right = pd.to_numeric(
            joined[f"{column}_ranked"],
            errors="raise",
        )

        if not np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            atol=1e-12,
            rtol=1e-12,
            equal_nan=False,
        ):
            mismatch = joined.loc[
                ~np.isclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    atol=1e-12,
                    rtol=1e-12,
                    equal_nan=False,
                ),
                merge_columns
                + [
                    f"{column}_proposal",
                    f"{column}_ranked",
                ],
            ].head(8)

            raise RuntimeError(
                "Recovered ranked frame does not exactly match "
                f"the V5-D proposal contract for {column}: "
                f"{mismatch.to_dict(orient='records')}"
            )


def feature_inventory(
    snapshot: pd.DataFrame,
    *,
    protected: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in snapshot.columns:
        if column in protected:
            continue

        values = snapshot[column]
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[
            np.isfinite(numeric.to_numpy(dtype=float))
        ]

        rows.append(
            {
                "feature": column,
                "dtype": str(values.dtype),
                "numeric": bool(
                    pd.api.types.is_numeric_dtype(values)
                    or len(finite) > 0
                ),
                "coverage_rate": float(values.notna().mean()),
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "constant": bool(
                    values.nunique(dropna=True) <= 1
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "coverage_rate",
            "constant",
            "feature",
        ],
        ascending=[False, True, True],
        kind="mergesort",
    )


v5d = read_json(v5d_decision_path)
v5e = read_json(v5e_decision_path)
v5f = read_json(v5f_decision_path)
v5g = read_json(v5g_decision_path)

expected_statuses = {
    "v5d": "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE",
    "v5e": "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_AUDIT_COMPLETE",
    "v5f": "V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT",
    "v5g": "V5G_RUNTIME_STATE_REINSTRUMENTATION_REQUIRED",
}

for name, decision in {
    "v5d": v5d,
    "v5e": v5e,
    "v5f": v5f,
    "v5g": v5g,
}.items():
    if decision.get("status") != expected_statuses[name]:
        raise RuntimeError(
            f"Unexpected {name} status: {decision.get('status')}"
        )

    if decision.get("baseline_commit") != expected_head:
        raise RuntimeError(
            f"{name} baseline commit mismatch."
        )

proposal_path = Path(v5d["proposal_manifest_path"]).resolve()
action_path = Path(v5d["action_effects_path"]).resolve()

if not proposal_path.is_file() or not action_path.is_file():
    raise RuntimeError(
        "V5-D proposal/action artifacts are missing."
    )

proposals = pd.read_csv(proposal_path)

required_proposals = {
    "seed",
    "proposal_index",
    "query_id",
    "product_id",
    *REQUIRED_MATCH_COLUMNS,
}

missing_proposals = sorted(
    required_proposals - set(proposals.columns)
)

if missing_proposals:
    raise RuntimeError(
        "Proposal manifest missing required columns: "
        f"{missing_proposals}"
    )

if len(proposals) != 144:
    raise RuntimeError(
        f"Expected 144 contexts, found {len(proposals)}."
    )

if proposals.duplicated(
    ["seed", "proposal_index"]
).any():
    raise RuntimeError(
        "Proposal manifest has duplicate seed/proposal keys."
    )

for column in ("seed", "query_id", "product_id"):
    proposals[column] = normalize_integer_key(proposals[column])

expected_seeds = {
    int(value)
    for value in proposals["seed"].unique()
}

if len(expected_seeds) != 18:
    raise RuntimeError(
        f"Expected 18 training seeds, found {len(expected_seeds)}."
    )

validation_root = (
    candidate_worktree
    / "artifacts"
    / "_validation"
)

if not validation_root.is_dir():
    raise RuntimeError(
        f"Validation root missing: {validation_root}"
    )

ranked_root, ranked_paths, root_inventory_rows = (
    find_complete_ranked_frame_root(
        validation_root,
        expected_seeds,
    )
)

frames: list[pd.DataFrame] = []
frame_inventory_rows: list[dict[str, Any]] = []

for seed in sorted(expected_seeds):
    path = ranked_paths[seed]
    frame = pd.read_csv(path)
    validate_frame_schema(frame, seed=seed)

    numeric_present, categorical_present = safe_columns_present(frame)

    frame = frame.copy()
    frame.insert(0, "seed", int(seed))

    frames.append(frame)

    frame_inventory_rows.append(
        {
            "seed": int(seed),
            "path": str(path),
            "sha256": sha256(path),
            "row_count": int(len(frame)),
            "query_count": int(frame["query_id"].nunique()),
            "numeric_whitelist_available": "|".join(numeric_present),
            "categorical_whitelist_available": "|".join(categorical_present),
            "forbidden_columns_present_but_excluded": "|".join(
                sorted(
                    column
                    for column in frame.columns
                    if is_forbidden_column(column)
                )
            ),
        }
    )

all_ranked = pd.concat(
    frames,
    ignore_index=True,
)

assert_exact_proposal_match(
    proposals,
    all_ranked,
)

available_numeric = [
    column
    for column in NUMERIC_WHITELIST
    if column in all_ranked.columns
]

available_categorical = [
    column
    for column in CATEGORICAL_WHITELIST
    if column in all_ranked.columns
]

candidate_columns = [
    "seed",
    "query_id",
    "product_id",
    *available_numeric,
    *available_categorical,
]

candidate_state = proposals.loc[
    :,
    [
        "seed",
        "proposal_index",
        "query_id",
        "product_id",
    ],
].merge(
    all_ranked.loc[:, candidate_columns],
    on=["seed", "query_id", "product_id"],
    how="left",
    validate="one_to_one",
)

if len(candidate_state) != len(proposals):
    raise RuntimeError(
        "Candidate state row count changed during recovery."
    )

geometry = query_geometry(all_ranked)

candidate_state = candidate_state.merge(
    geometry,
    on=[
        "seed",
        "query_id",
        "product_id",
        "base_score",
        "base_rank",
    ],
    how="left",
    validate="one_to_one",
)

if candidate_state.isna().all(axis=1).any():
    raise RuntimeError(
        "Recovered candidate state contains an empty row."
    )

required_selected_geometry_columns = [
    "query_candidate_count",
    "query_base_score_max",
    "query_base_score_min",
    "query_base_score_mean",
    "query_top1_base_score",
    "query_top2_base_score",
    "candidate_rank_fraction",
    "candidate_gap_to_above",
    "candidate_gap_to_below",
    *[
        f"base_boundary_score_at_{rank}"
        for rank in TARGET_RANKS
    ],
    *[
        f"action_score_gap_to_{rank}"
        for rank in TARGET_RANKS
    ],
    *[
        f"action_required_lift_to_{rank}"
        for rank in TARGET_RANKS
    ],
]

missing_selected_geometry = [
    column
    for column in required_selected_geometry_columns
    if (
        column not in candidate_state.columns
        or candidate_state[column].isna().any()
    )
]

if missing_selected_geometry:
    raise RuntimeError(
        "A selected V5-D context lacks a required source-native "
        "pre-action geometry feature: "
        f"{missing_selected_geometry}"
    )

for action_name, target_rank in EXPECTED_ACTIONS.items():
    action_frame = candidate_state.copy()
    action_frame.insert(4, "action", action_name)
    action_frame.insert(5, "target_rank", int(target_rank))
    action_frame["action_move_distance"] = (
        action_frame["base_rank"] - int(target_rank)
    )
    action_frame["action_boundary_score"] = (
        action_frame[
            f"base_boundary_score_at_{target_rank}"
        ]
    )
    action_frame["action_score_gap"] = (
        action_frame[
            f"action_score_gap_to_{target_rank}"
        ]
    )
    action_frame["action_required_lift"] = (
        action_frame[
            f"action_required_lift_to_{target_rank}"
        ]
    )

    if action_name == EXPECTED_ACTIONS.keys().__iter__().__next__():
        state_action = action_frame
    else:
        state_action = pd.concat(
            [state_action, action_frame],
            ignore_index=True,
        )

if len(state_action) != 720:
    raise RuntimeError(
        f"Expected 720 state-action rows, found {len(state_action)}."
    )

if state_action.duplicated(
    ["seed", "proposal_index", "action"]
).any():
    raise RuntimeError(
        "Recovered state-action matrix has duplicate keys."
    )

assert_saved_snapshot_has_no_forbidden(candidate_state)
assert_saved_snapshot_has_no_forbidden(state_action)

protected = {
    "seed",
    "proposal_index",
    "query_id",
    "product_id",
}

candidate_feature_report = feature_inventory(
    candidate_state,
    protected=protected,
)

state_action_feature_report = feature_inventory(
    state_action,
    protected=protected | {"action"},
)

eligible_candidate_numeric = candidate_feature_report.loc[
    candidate_feature_report["numeric"]
    & candidate_feature_report["coverage_rate"].ge(0.99)
    & ~candidate_feature_report["constant"],
    "feature",
].tolist()

eligible_candidate_categorical = candidate_feature_report.loc[
    ~candidate_feature_report["numeric"]
    & candidate_feature_report["coverage_rate"].ge(0.99)
    & ~candidate_feature_report["constant"],
    "feature",
].tolist()

eligible_action_geometry = [
    feature
    for feature in state_action_feature_report.loc[
        state_action_feature_report["numeric"]
        & state_action_feature_report["coverage_rate"].ge(0.99)
        & ~state_action_feature_report["constant"],
        "feature",
    ].tolist()
    if (
        feature.startswith("action_")
        or feature.startswith("base_boundary_score_at_")
        or feature.startswith("query_")
        or feature.startswith("candidate_")
    )
]

status = (
    "V5G1_PREACTION_RUNTIME_STATE_RECOVERED_AND_CONTRACT_READY"
    if (
        len(eligible_candidate_numeric) >= 10
        and len(eligible_action_geometry) >= 10
        and len(eligible_candidate_categorical) >= 1
    )
    else "V5G1_PREACTION_RUNTIME_STATE_RECOVERY_INSUFFICIENT"
)

output_root.mkdir(parents=True, exist_ok=True)

candidate_state_path = (
    output_root
    / "v5g1_recovered_preaction_context_state.csv"
)

state_action_path = (
    output_root
    / "v5g1_recovered_preaction_state_action_matrix.csv"
)

candidate_report_path = (
    output_root
    / "v5g1_candidate_feature_inventory.csv"
)

state_action_report_path = (
    output_root
    / "v5g1_state_action_feature_inventory.csv"
)

frame_inventory_path = (
    output_root
    / "v5g1_ranked_frame_inventory.csv"
)

root_inventory_path = (
    output_root
    / "v5g1_ranked_frame_root_inventory.csv"
)

candidate_state.to_csv(
    candidate_state_path,
    index=False,
)

state_action.to_csv(
    state_action_path,
    index=False,
)

candidate_feature_report.to_csv(
    candidate_report_path,
    index=False,
)

state_action_feature_report.to_csv(
    state_action_report_path,
    index=False,
)

pd.DataFrame(frame_inventory_rows).to_csv(
    frame_inventory_path,
    index=False,
)

pd.DataFrame(root_inventory_rows).to_csv(
    root_inventory_path,
    index=False,
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
    "supersedes_v5g_false_negative": {
        "v5g_decision_path": str(v5g_decision_path),
        "v5g_decision_sha256": sha256(v5g_decision_path),
        "reason": (
            "V5-G required a literal seed column and rejected "
            "whole ranked frames when they contained post-gate "
            "columns. The V5-D ranked frames encode seed in their "
            "seed-<id> directory paths and retain a pre-action "
            "whitelist. V5-G1 derives seed from the immutable "
            "path, validates exact V5-D proposal matches, and "
            "selects only explicitly whitelisted pre-action columns."
        ),
    },
    "ranked_frame_root": str(ranked_root),
    "ranked_frame_root_sha256_note": (
        "Directory roots do not have a single SHA256; each seed "
        "frame hash is recorded in v5g1_ranked_frame_inventory.csv."
    ),
    "exact_context_match_count": int(len(candidate_state)),
    "exact_context_match_rate": 1.0,
    "training_seed_count": int(len(expected_seeds)),
    "context_count": int(len(candidate_state)),
    "state_action_row_count": int(len(state_action)),
    "available_numeric_whitelist": available_numeric,
    "available_categorical_whitelist": available_categorical,
    "eligible_candidate_numeric_feature_count": int(
        len(eligible_candidate_numeric)
    ),
    "eligible_candidate_numeric_features": (
        eligible_candidate_numeric
    ),
    "eligible_candidate_categorical_feature_count": int(
        len(eligible_candidate_categorical)
    ),
    "eligible_candidate_categorical_features": (
        eligible_candidate_categorical
    ),
    "eligible_action_geometry_feature_count": int(
        len(eligible_action_geometry)
    ),
    "eligible_action_geometry_features": (
        eligible_action_geometry
    ),
    "candidate_state_path": str(candidate_state_path),
    "candidate_state_sha256": sha256(candidate_state_path),
    "state_action_path": str(state_action_path),
    "state_action_sha256": sha256(state_action_path),
    "candidate_feature_report_path": str(candidate_report_path),
    "state_action_feature_report_path": str(
        state_action_report_path
    ),
    "frame_inventory_path": str(frame_inventory_path),
    "root_inventory_path": str(root_inventory_path),
    "forbidden_outcome_or_oracle_inputs": True,
    "source_modified": False,
    "config_modified": False,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Freeze the recovered pre-action feature contract, then "
        "run a second seed-disjoint development-only viability "
        "audit against the same V5-D action labels. Do not "
        "rerun V5-D, fit a final artifact, select a serving "
        "threshold, or execute calibration/confirmation seeds."
        if status
        == "V5G1_PREACTION_RUNTIME_STATE_RECOVERED_AND_CONTRACT_READY"
        else "Inspect the whitelisted feature inventory before "
        "considering any new instrumentation. Do not rerun V5-D "
        "or fit another model yet."
    ),
}

decision_path = (
    output_root
    / "v5g1_preaction_runtime_state_recovery_decision.json"
)

write_json(decision_path, decision)

print("===== V5-G1 PATH-KEYED PRE-ACTION STATE RECOVERY =====")
print(
    json.dumps(
        {
            "decision": decision,
            "frame_inventory": frame_inventory_rows,
        },
        indent=2,
        sort_keys=True,
    )
)