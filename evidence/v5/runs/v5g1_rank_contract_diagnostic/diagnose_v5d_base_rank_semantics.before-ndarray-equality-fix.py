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
candidate_worktree = Path(sys.argv[2]).resolve()
output_root = Path(sys.argv[3]).resolve()
expected_head = sys.argv[4]

TARGET_RANKS = (1, 2, 3, 5, 10)
PROPOSAL_MATCH_COLUMNS = (
    "base_rank",
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
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


def normalize_integer(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy(dtype=float)

    if not np.isfinite(array).all():
        raise RuntimeError(f"{label}: non-finite values.")

    rounded = np.rint(array)

    if not np.allclose(array, rounded):
        raise RuntimeError(f"{label}: non-integral values.")

    return pd.Series(
        rounded.astype(np.int64),
        index=values.index,
    )


def seed_from_path(path: Path) -> int | None:
    for parent in (path.parent, *path.parents):
        match = re.fullmatch(r"seed-(\d+)", parent.name)

        if match:
            return int(match.group(1))

    return None


def validation_experiment_root(path: Path) -> Path | None:
    parts = list(path.parts)
    lower = [part.lower() for part in parts]

    if "_validation" not in lower:
        return None

    index = lower.index("_validation")

    if index + 1 >= len(parts):
        return None

    return Path(*parts[: index + 2])


def discover_complete_ranked_root(
    validation_root: Path,
    expected_seeds: set[int],
) -> tuple[Path, dict[int, Path], pd.DataFrame]:
    by_root: dict[Path, dict[int, list[Path]]] = {}

    for path in sorted(validation_root.rglob("ranked_test.csv")):
        seed = seed_from_path(path)
        root = validation_experiment_root(path)

        if seed is None or root is None:
            continue

        by_root.setdefault(root, {}).setdefault(seed, []).append(path)

    inventory: list[dict[str, Any]] = []
    complete: list[tuple[Path, dict[int, Path]]] = []

    for root, seeds in sorted(
        by_root.items(),
        key=lambda item: str(item[0]),
    ):
        selected: dict[int, Path] = {}
        duplicate_count = 0

        for seed in expected_seeds:
            options = seeds.get(seed, [])

            if len(options) == 1:
                selected[seed] = options[0]
            elif len(options) > 1:
                duplicate_count += 1

        coverage = len(selected)

        inventory.append(
            {
                "experiment_root": str(root),
                "training_seed_coverage": coverage,
                "expected_training_seed_count": len(expected_seeds),
                "duplicate_seed_path_count": duplicate_count,
                "is_complete_unique_root": (
                    coverage == len(expected_seeds)
                    and duplicate_count == 0
                ),
            }
        )

        if (
            coverage == len(expected_seeds)
            and duplicate_count == 0
        ):
            complete.append((root, selected))

    if len(complete) != 1:
        raise RuntimeError(
            "Expected exactly one complete V5-D ranked-frame root; "
            f"found {len(complete)}."
        )

    return (
        complete[0][0],
        complete[0][1],
        pd.DataFrame(inventory),
    )


def ordinal_rank(
    frame: pd.DataFrame,
    *,
    order_columns: list[str],
    ascending: list[bool],
) -> pd.Series:
    ranked = frame.sort_values(
        order_columns,
        ascending=ascending,
        kind="mergesort",
    )

    output = pd.Series(
        np.arange(1, len(ranked) + 1, dtype=np.int64),
        index=ranked.index,
    )

    return output.reindex(frame.index)


def rank_contract_for_query(
    query: pd.DataFrame,
) -> dict[str, Any]:
    observed = query["base_rank"].to_numpy(dtype=np.int64)
    count = int(len(query))

    expected_one = np.arange(1, count + 1, dtype=np.int64)
    expected_zero = np.arange(0, count, dtype=np.int64)
    sorted_observed = np.sort(observed)

    one_based = bool(np.array_equal(sorted_observed, expected_one))
    zero_based = bool(np.array_equal(sorted_observed, expected_zero))
    duplicate_rank_count = int(
        pd.Series(observed).duplicated().sum()
    )
    gap_count = int(
        max(
            0,
            count - len(np.unique(observed)),
        )
    )

    strategies = {
        "base_score_desc_stable": ordinal_rank(
            query,
            order_columns=["base_score"],
            ascending=[False],
        ),
        "base_score_desc_product_id_asc": ordinal_rank(
            query,
            order_columns=["base_score", "product_id"],
            ascending=[False, True],
        ),
        "base_score_desc_product_id_desc": ordinal_rank(
            query,
            order_columns=["base_score", "product_id"],
            ascending=[False, False],
        ),
        "base_score_asc_stable": ordinal_rank(
            query,
            order_columns=["base_score"],
            ascending=[True],
        ),
    }

    if "retrieval_score" in query.columns:
        strategies["retrieval_score_desc_stable"] = ordinal_rank(
            query,
            order_columns=["retrieval_score"],
            ascending=[False],
        )

    if "semantic_rank_score" in query.columns:
        strategies["semantic_rank_score_desc_stable"] = ordinal_rank(
            query,
            order_columns=["semantic_rank_score"],
            ascending=[False],
        )

    if "final_score" in query.columns:
        strategies["final_score_desc_stable_diagnostic_only"] = (
            ordinal_rank(
                query,
                order_columns=["final_score"],
                ascending=[False],
            )
        )

    result: dict[str, Any] = {
        "candidate_count": count,
        "base_rank_one_based_permutation": one_based,
        "base_rank_zero_based_permutation": zero_based,
        "duplicate_rank_count": duplicate_rank_count,
        "gap_count": gap_count,
    }

    score_by_source_rank = (
        query.sort_values(
            "base_rank",
            kind="mergesort",
        )["base_score"]
        .to_numpy(dtype=float)
    )

    result["base_score_nonincreasing_by_base_rank"] = bool(
        np.all(np.diff(score_by_source_rank) <= 1e-12)
    )

    for name, predicted in strategies.items():
        result[f"{name}_agreement_rate"] = float(
            predicted.to_numpy(dtype=np.int64).eq(
                query["base_rank"]
            ).mean()
        )
        result[f"{name}_exact"] = bool(
            predicted.to_numpy(dtype=np.int64).eq(
                query["base_rank"]
            ).all()
        )

    return result


def proposal_source_contract(
    proposals: pd.DataFrame,
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_columns = [
        "seed",
        "query_id",
        "product_id",
        *PROPOSAL_MATCH_COLUMNS,
    ]

    source = ranked.loc[:, source_columns].copy()

    merged = proposals.merge(
        source,
        on=["seed", "query_id", "product_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_proposal", "_ranked"),
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        missing = merged.loc[
            ~merged["_merge"].eq("both"),
            ["seed", "query_id", "product_id"],
        ]

        raise RuntimeError(
            "V5-D proposal keys not found in source ranked frames: "
            f"{missing.head(10).to_dict(orient='records')}"
        )

    for column in PROPOSAL_MATCH_COLUMNS:
        left = pd.to_numeric(
            merged[f"{column}_proposal"],
            errors="raise",
        )
        right = pd.to_numeric(
            merged[f"{column}_ranked"],
            errors="raise",
        )

        if not np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ):
            mismatch = merged.loc[
                ~np.isclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=False,
                ),
                [
                    "seed",
                    "query_id",
                    "product_id",
                    f"{column}_proposal",
                    f"{column}_ranked",
                ],
            ].head(10)

            raise RuntimeError(
                f"V5-D proposal provenance mismatch for {column}: "
                f"{mismatch.to_dict(orient='records')}"
            )

    context_rank_coverage = (
        ranked.loc[
            ranked.set_index(
                ["seed", "query_id"]
            ).index.isin(
                proposals.set_index(
                    ["seed", "query_id"]
                ).index
            ),
            [
                "seed",
                "query_id",
                "base_rank",
            ],
        ]
        .groupby(
            ["seed", "query_id"],
            as_index=False,
        )
        .agg(
            min_base_rank=("base_rank", "min"),
            max_base_rank=("base_rank", "max"),
            unique_base_rank_count=("base_rank", "nunique"),
        )
    )

    proposal_contexts = proposals.loc[
        :,
        ["seed", "query_id", "proposal_index"],
    ].drop_duplicates()

    context_rank_coverage = proposal_contexts.merge(
        context_rank_coverage,
        on=["seed", "query_id"],
        how="left",
        validate="one_to_one",
    )

    for rank in TARGET_RANKS:
        presence = (
            ranked.loc[
                ranked["base_rank"].eq(rank),
                ["seed", "query_id"],
            ]
            .drop_duplicates()
            .assign(**{f"has_rank_{rank}": True})
        )

        context_rank_coverage = context_rank_coverage.merge(
            presence,
            on=["seed", "query_id"],
            how="left",
            validate="one_to_one",
        )

        context_rank_coverage[f"has_rank_{rank}"] = (
            context_rank_coverage[f"has_rank_{rank}"]
            .fillna(False)
            .astype(bool)
        )

    required_boundary_columns = [
        f"has_rank_{rank}"
        for rank in TARGET_RANKS
    ]

    context_rank_coverage["all_target_boundaries_present"] = (
        context_rank_coverage.loc[
            :,
            required_boundary_columns,
        ].all(axis=1)
    )

    summary = {
        "exact_proposal_key_match_count": int(len(merged)),
        "exact_proposal_runtime_contract_match_count": int(
            len(merged)
        ),
        "proposal_context_count": int(
            len(proposal_contexts)
        ),
        "proposal_contexts_with_all_target_boundaries": int(
            context_rank_coverage[
                "all_target_boundaries_present"
            ].sum()
        ),
    }

    return context_rank_coverage, summary


v5d = json.loads(
    v5d_decision_path.read_text(encoding="utf-8-sig")
)

if (
    v5d.get("status")
    != "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
):
    raise RuntimeError("Unexpected V5-D status.")

if v5d.get("baseline_commit") != expected_head:
    raise RuntimeError("V5-D baseline mismatch.")

proposal_path = Path(
    v5d["proposal_manifest_path"]
).resolve()

if not proposal_path.is_file():
    raise RuntimeError(f"Proposal manifest missing: {proposal_path}")

proposals = pd.read_csv(proposal_path)

required_proposal_columns = {
    "seed",
    "proposal_index",
    "query_id",
    "product_id",
    *PROPOSAL_MATCH_COLUMNS,
}

missing = sorted(
    required_proposal_columns - set(proposals.columns)
)

if missing:
    raise RuntimeError(
        f"Proposal manifest missing columns: {missing}"
    )

for column in ("seed", "query_id", "product_id"):
    proposals[column] = normalize_integer(
        proposals[column],
        label=f"proposal {column}",
    )

if len(proposals) != 144:
    raise RuntimeError(
        f"Expected 144 proposals, found {len(proposals)}."
    )

expected_seeds = {
    int(seed)
    for seed in proposals["seed"].unique()
}

if len(expected_seeds) != 18:
    raise RuntimeError("Expected 18 V5-D training seeds.")

validation_root = (
    candidate_worktree
    / "artifacts"
    / "_validation"
)

if not validation_root.is_dir():
    raise RuntimeError(
        f"Validation root missing: {validation_root}"
    )

ranked_root, ranked_paths, root_inventory = (
    discover_complete_ranked_root(
        validation_root,
        expected_seeds,
    )
)

frames: list[pd.DataFrame] = []
per_seed_rows: list[dict[str, Any]] = []
per_query_rows: list[dict[str, Any]] = []

for seed in sorted(expected_seeds):
    path = ranked_paths[seed]
    frame = pd.read_csv(path)

    required_ranked = {
        "query_id",
        "product_id",
        "base_rank",
        "base_score",
        *PROPOSAL_MATCH_COLUMNS,
    }

    missing_ranked = sorted(
        required_ranked - set(frame.columns)
    )

    if missing_ranked:
        raise RuntimeError(
            f"seed={seed}: ranked frame missing {missing_ranked}"
        )

    frame = frame.copy()
    frame.insert(0, "seed", int(seed))
    frame["query_id"] = normalize_integer(
        frame["query_id"],
        label=f"seed={seed} query_id",
    )
    frame["product_id"] = normalize_integer(
        frame["product_id"],
        label=f"seed={seed} product_id",
    )
    frame["base_rank"] = normalize_integer(
        frame["base_rank"],
        label=f"seed={seed} base_rank",
    )

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

    if frame.duplicated(
        ["query_id", "product_id"]
    ).any():
        raise RuntimeError(
            f"seed={seed}: duplicate query-product rows."
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

    seed_query_rows = []

    for query_id, query in frame.groupby(
        "query_id",
        sort=False,
    ):
        local = query.copy()
        local["source_row_order"] = np.arange(
            len(local),
            dtype=np.int64,
        )

        row = rank_contract_for_query(local)
        row["seed"] = int(seed)
        row["query_id"] = int(query_id)
        seed_query_rows.append(row)

    per_query_rows.extend(seed_query_rows)

    query_frame = pd.DataFrame(seed_query_rows)

    per_seed_rows.append(
        {
            "seed": int(seed),
            "ranked_path": str(path),
            "ranked_sha256": sha256(path),
            "row_count": int(len(frame)),
            "query_count": int(frame["query_id"].nunique()),
            "one_based_permutation_query_count": int(
                query_frame[
                    "base_rank_one_based_permutation"
                ].sum()
            ),
            "zero_based_permutation_query_count": int(
                query_frame[
                    "base_rank_zero_based_permutation"
                ].sum()
            ),
            "base_score_monotone_query_count": int(
                query_frame[
                    "base_score_nonincreasing_by_base_rank"
                ].sum()
            ),
            "duplicate_rank_query_count": int(
                query_frame[
                    "duplicate_rank_count"
                ].gt(0).sum()
            ),
            "gap_rank_query_count": int(
                query_frame["gap_count"].gt(0).sum()
            ),
            **{
                f"{strategy}_mean_agreement": float(
                    query_frame[
                        f"{strategy}_agreement_rate"
                    ].mean()
                )
                for strategy in (
                    "base_score_desc_stable",
                    "base_score_desc_product_id_asc",
                    "base_score_desc_product_id_desc",
                    "base_score_asc_stable",
                    "retrieval_score_desc_stable",
                    "semantic_rank_score_desc_stable",
                    "final_score_desc_stable_diagnostic_only",
                )
                if f"{strategy}_agreement_rate"
                in query_frame.columns
            },
        }
    )

    frames.append(frame)

ranked = pd.concat(
    frames,
    ignore_index=True,
)

per_query = pd.DataFrame(per_query_rows)
per_seed = pd.DataFrame(per_seed_rows)

proposal_context_contract, proposal_summary = (
    proposal_source_contract(
        proposals,
        ranked,
    )
)

strategies = [
    strategy
    for strategy in (
        "base_score_desc_stable",
        "base_score_desc_product_id_asc",
        "base_score_desc_product_id_desc",
        "base_score_asc_stable",
        "retrieval_score_desc_stable",
        "semantic_rank_score_desc_stable",
        "final_score_desc_stable_diagnostic_only",
    )
    if f"{strategy}_agreement_rate" in per_query.columns
]

strategy_summary_rows = []

for strategy in strategies:
    agreement = per_query[
        f"{strategy}_agreement_rate"
    ]

    strategy_summary_rows.append(
        {
            "strategy": strategy,
            "mean_query_row_agreement": float(
                agreement.mean()
            ),
            "minimum_query_row_agreement": float(
                agreement.min()
            ),
            "exact_query_count": int(
                per_query[
                    f"{strategy}_exact"
                ].sum()
            ),
            "total_query_count": int(len(per_query)),
        }
    )

strategy_summary = pd.DataFrame(
    strategy_summary_rows
).sort_values(
    [
        "mean_query_row_agreement",
        "exact_query_count",
        "strategy",
    ],
    ascending=[False, False, True],
    kind="mergesort",
)

all_one_based = bool(
    per_query["base_rank_one_based_permutation"].all()
)

all_score_monotone = bool(
    per_query[
        "base_score_nonincreasing_by_base_rank"
    ].all()
)

all_boundaries = bool(
    proposal_context_contract[
        "all_target_boundaries_present"
    ].all()
)

recoverable_native_rank_contract = bool(
    all_one_based
    and all_score_monotone
    and all_boundaries
    and proposal_summary[
        "exact_proposal_runtime_contract_match_count"
    ] == 144
)

status = (
    "V5G1_1_NATIVE_BASE_RANK_CONTRACT_RECOVERABLE"
    if recoverable_native_rank_contract
    else "V5G1_1_BASE_RANK_CONTRACT_REQUIRES_TARGETED_REPAIR"
)

output_root.mkdir(parents=True, exist_ok=True)

per_query_path = (
    output_root
    / "v5g1_1_rank_contract_by_query.csv"
)

per_seed_path = (
    output_root
    / "v5g1_1_rank_contract_by_seed.csv"
)

strategy_path = (
    output_root
    / "v5g1_1_rank_strategy_comparison.csv"
)

proposal_context_path = (
    output_root
    / "v5g1_1_proposal_context_rank_coverage.csv"
)

root_inventory_path = (
    output_root
    / "v5g1_1_ranked_frame_root_inventory.csv"
)

per_query.to_csv(per_query_path, index=False)
per_seed.to_csv(per_seed_path, index=False)
strategy_summary.to_csv(strategy_path, index=False)
proposal_context_contract.to_csv(
    proposal_context_path,
    index=False,
)
root_inventory.to_csv(root_inventory_path, index=False)

decision = {
    "status": status,
    "baseline_commit": expected_head,
    "v5d_decision_path": str(v5d_decision_path),
    "v5d_decision_sha256": sha256(v5d_decision_path),
    "proposal_manifest_path": str(proposal_path),
    "proposal_manifest_sha256": sha256(proposal_path),
    "ranked_frame_root": str(ranked_root),
    "training_seed_count": int(len(expected_seeds)),
    "ranked_row_count": int(len(ranked)),
    "query_count": int(len(per_query)),
    "proposal_source_contract": proposal_summary,
    "rank_semantics": {
        "all_queries_one_based_rank_permutation": all_one_based,
        "all_queries_base_score_nonincreasing_by_source_rank": (
            all_score_monotone
        ),
        "all_144_proposal_contexts_have_rank_1_2_3_5_10": (
            all_boundaries
        ),
        "best_external_sort_candidate": (
            strategy_summary.iloc[0].to_dict()
            if len(strategy_summary)
            else None
        ),
        "interpretation": (
            "This audit distinguishes source-native base_rank "
            "semantics from the recovery script's prior "
            "product_id tie-break assumption. It does not "
            "modify a ranked frame or fit a model."
        ),
    },
    "rank_contract_by_query_path": str(per_query_path),
    "rank_contract_by_seed_path": str(per_seed_path),
    "rank_strategy_comparison_path": str(strategy_path),
    "proposal_context_rank_coverage_path": str(
        proposal_context_path
    ),
    "ranked_frame_root_inventory_path": str(
        root_inventory_path
    ),
    "source_modified": False,
    "config_modified": False,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Repair the V5-G1 recovery runner to treat the proven "
        "source-native base_rank as authoritative, build query "
        "geometry from source rank boundaries, and rerun only "
        "the read-only recovery."
        if status
        == "V5G1_1_NATIVE_BASE_RANK_CONTRACT_RECOVERABLE"
        else "Use the per-query rank contract report to isolate "
        "whether the mismatch is tied ordering, non-contiguous "
        "source rank, or a non-base-score rank definition before "
        "changing recovery logic."
    ),
}

decision_path = (
    output_root
    / "v5g1_1_base_rank_semantics_decision.json"
)

write_json(decision_path, decision)

print("===== V5-G1.1 BASE-RANK SEMANTICS DIAGNOSTIC =====")
print(
    json.dumps(
        {
            "decision": decision,
            "strategy_summary": strategy_summary.to_dict(
                orient="records"
            ),
            "per_seed": per_seed.to_dict(
                orient="records"
            ),
        },
        indent=2,
        sort_keys=True,
    )
)