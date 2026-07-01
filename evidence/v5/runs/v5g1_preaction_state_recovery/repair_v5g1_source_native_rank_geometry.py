from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path


recovery_path = Path(sys.argv[1]).resolve()
repair_decision_path = Path(sys.argv[2]).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def line_offsets(source: str) -> list[int]:
    offsets = [0]
    total = 0

    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)

    return offsets


def node_span(
    source: str,
    node: ast.AST,
) -> tuple[int, int]:
    if not (
        hasattr(node, "lineno")
        and hasattr(node, "col_offset")
        and hasattr(node, "end_lineno")
        and hasattr(node, "end_col_offset")
    ):
        raise RuntimeError(
            f"AST node lacks a complete span: {type(node).__name__}"
        )

    offsets = line_offsets(source)

    return (
        offsets[node.lineno - 1] + node.col_offset,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def replace_function(
    source: str,
    function_name: str,
    replacement: str,
) -> str:
    tree = ast.parse(source, filename=str(recovery_path))

    matches = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.name == function_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one function named {function_name}; "
            f"found {len(matches)}."
        )

    start, end = node_span(source, matches[0])

    return source[:start] + replacement + source[end:]


source = recovery_path.read_text(encoding="utf-8")

if (
    "base_rank does not agree with canonical base-score ranking."
    not in source
):
    raise RuntimeError(
        "The expected pre-repair canonical-rank guard is absent."
    )

if (
    "candidate_state = candidate_state.merge("
    not in source
    or '"seed",' not in source
):
    raise RuntimeError(
        "Expected seed-aware candidate-state merge is absent."
    )

replacement_function = '''def query_geometry(
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
'''

repaired = replace_function(
    source,
    "query_geometry",
    replacement_function,
)

old_guard = '''if candidate_state.isna().all(axis=1).any():
    raise RuntimeError(
        "Recovered candidate state contains an empty row."
    )
'''

new_guard = '''if candidate_state.isna().all(axis=1).any():
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
'''

if old_guard not in repaired:
    raise RuntimeError(
        "Could not locate the selected-context geometry guard."
    )

repaired = repaired.replace(
    old_guard,
    new_guard,
    1,
)

for required_fragment in (
    "work[\"source_rank\"] = work[\"base_rank\"]",
    "Source-native base_rank is not a contiguous",
    "base_score is not non-increasing in source-native",
    "required_selected_geometry_columns",
):
    if required_fragment not in repaired:
        raise RuntimeError(
            f"Required repaired fragment missing: {required_fragment}"
        )

for forbidden_fragment in (
    "base_rank does not agree with canonical base-score ranking.",
    "work[\"canonical_rank\"]",
):
    if forbidden_fragment in repaired:
        raise RuntimeError(
            f"Forbidden pre-repair fragment remains: {forbidden_fragment}"
        )

ast.parse(
    repaired,
    filename=str(recovery_path),
)

recovery_path.write_text(
    repaired,
    encoding="utf-8",
)

decision = {
    "status": "V5G1_SOURCE_NATIVE_RANK_GEOMETRY_REPAIRED",
    "recovery_path": str(recovery_path),
    "recovery_sha256_after": sha256(recovery_path),
    "repair_scope": [
        "replace reconstructed canonical rank with source-native base_rank",
        "validate each source query has contiguous one-based ranks",
        "validate base_score is non-increasing by source-native rank",
        "construct target-rank boundaries and local gaps from source rank",
        "require complete geometry for every selected V5-D context",
    ],
    "source_modified": False,
    "config_modified": False,
    "v5d_corpus_rerun": False,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
}

repair_decision_path.write_text(
    json.dumps(
        decision,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print(json.dumps(decision, indent=2, sort_keys=True))