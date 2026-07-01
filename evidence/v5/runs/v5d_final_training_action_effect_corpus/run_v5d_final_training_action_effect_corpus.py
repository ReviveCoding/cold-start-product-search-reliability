from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


repo = Path(sys.argv[1]).resolve()
smoke_config_path = Path(sys.argv[2]).resolve()
charter_path = Path(sys.argv[3]).resolve()
output_root = Path(sys.argv[4]).resolve()
repository_output_root = Path(sys.argv[5]).resolve()
expected_head = sys.argv[6]

sys.path.insert(0, str(repo / "src"))

from product_search.config import load_config
from product_search.pipeline import _gate_config, run_model_stage
from product_search.policy.gate import apply_coverage_overreach_gate
from product_search.simulation.dynamic import run_dynamic_simulation


FINAL_TRAINING_SEEDS = (
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
PLACEMENT_ACTIONS = {
    "PLACE_AT_10": 10,
    "PLACE_AT_5": 5,
    "PLACE_AT_3": 3,
    "PLACE_AT_2": 2,
    "PLACE_AT_1": 1,
}

PILOT_DAYS = 5
PILOT_TRAFFIC_PER_DAY = 80
PILOT_REPLICATIONS = 5
QUERIES_PER_SEED = 8

RUNTIME_PROPOSAL_COLUMNS = [
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
    "base_rank",
    "product_id",
]


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


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def jsonable_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): json_scalar(value)
        for key, value in payload.items()
    }


def current_head() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    return completed.stdout.strip()


def dynamic_required_columns() -> list[str]:
    path = repo / "src/product_search/simulation/dynamic.py"
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id == "required"
            for target in node.targets
        ):
            continue

        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue

        if isinstance(value, set) and all(
            isinstance(item, str)
            for item in value
        ):
            return sorted(value)

    raise RuntimeError(
        "Could not recover dynamic simulation required schema."
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported table suffix: {path}")


def discover_ranked_frame(
    root: Path,
    *,
    required_columns: set[str],
) -> tuple[Path, pd.DataFrame]:
    records: list[tuple[int, Path, pd.DataFrame]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue

        try:
            frame = read_table(path)
        except Exception:
            continue

        columns = set(frame.columns)

        if required_columns.issubset(columns):
            records.append(
                (
                    len(columns),
                    path,
                    frame,
                )
            )

    if not records:
        raise RuntimeError(
            "No ranked frame found with required dynamic schema: "
            f"{sorted(required_columns)}"
        )

    records.sort(
        key=lambda item: (
            -item[0],
            str(item[1]),
        )
    )

    _, path, frame = records[0]

    return path, frame.copy()


def canonical_rank(
    frame: pd.DataFrame,
    *,
    query_id: int,
    product_id: int,
) -> int:
    local = frame.loc[
        frame["query_id"].eq(query_id)
    ].copy()

    ranked = local.sort_values(
        ["final_score", "product_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    matching = np.flatnonzero(
        ranked["product_id"].to_numpy(dtype=int)
        == int(product_id)
    )

    if len(matching) != 1:
        raise RuntimeError(
            "Could not recover candidate rank after action."
        )

    return int(matching[0] + 1)


def candidate_target_score(
    baseline_frame: pd.DataFrame,
    *,
    query_id: int,
    product_id: int,
    target_rank: int,
) -> tuple[float, int]:
    local = baseline_frame.loc[
        baseline_frame["query_id"].eq(query_id)
    ].copy()

    local = local.sort_values(
        ["base_score", "product_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(local) < target_rank:
        raise RuntimeError(
            f"query={query_id} has fewer than {target_rank} candidates."
        )

    current_rank = int(
        np.flatnonzero(
            local["product_id"].to_numpy(dtype=int)
            == int(product_id)
        )[0]
        + 1
    )

    boundary_score = float(
        local.iloc[target_rank - 1]["base_score"]
    )

    return (
        float(np.nextafter(boundary_score, np.inf)),
        current_rank,
    )


def make_no_promotion_frame(
    ranked_frame: pd.DataFrame,
) -> pd.DataFrame:
    result = ranked_frame.copy()

    result["base_score"] = pd.to_numeric(
        result["base_score"],
        errors="raise",
    ).astype(float)

    result["final_score"] = result["base_score"]
    result["qrsbt_boost"] = 0.0
    result["gate_action"] = "NO_PROMOTION"
    result["gate_reason"] = "v5_no_promotion"
    result["boundary_entry_rejected"] = False

    return result


def inject_action(
    baseline_frame: pd.DataFrame,
    *,
    query_id: int,
    product_id: int,
    action_name: str,
    target_rank: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = baseline_frame.copy()

    candidate_mask = (
        result["query_id"].eq(query_id)
        & result["product_id"].eq(product_id)
    )

    if int(candidate_mask.sum()) != 1:
        raise RuntimeError(
            "Action candidate is not uniquely identified."
        )

    target_score, original_rank = candidate_target_score(
        baseline_frame,
        query_id=query_id,
        product_id=product_id,
        target_rank=target_rank,
    )

    base_score = float(
        result.loc[candidate_mask, "base_score"].iloc[0]
    )

    boost = float(target_score - base_score)

    if boost <= 0.0:
        raise RuntimeError(
            f"{action_name}: non-positive intervention boost."
        )

    result.loc[candidate_mask, "qrsbt_boost"] = boost
    result.loc[candidate_mask, "final_score"] = target_score
    result.loc[candidate_mask, "gate_action"] = "V5_ACTION"
    result.loc[candidate_mask, "gate_reason"] = action_name
    result.loc[
        candidate_mask,
        "boundary_entry_rejected",
    ] = False

    achieved_rank = canonical_rank(
        result,
        query_id=query_id,
        product_id=product_id,
    )

    if achieved_rank != target_rank:
        raise RuntimeError(
            f"{action_name}: requested rank={target_rank}, "
            f"achieved rank={achieved_rank}. "
            "Tie-safe exact placement is unavailable."
        )

    metadata = {
        "action": action_name,
        "requested_rank": int(target_rank),
        "achieved_rank": int(achieved_rank),
        "original_base_rank": int(original_rank),
        "target_score": target_score,
        "base_score": base_score,
        "qrsbt_boost": boost,
    }

    return result, metadata


def placement_feasible(
    baseline_frame: pd.DataFrame,
    *,
    query_id: int,
    product_id: int,
) -> bool:
    try:
        for action_name, target_rank in PLACEMENT_ACTIONS.items():
            _, metadata = inject_action(
                baseline_frame,
                query_id=query_id,
                product_id=product_id,
                action_name=action_name,
                target_rank=target_rank,
            )

            if metadata["achieved_rank"] != target_rank:
                return False

        return True
    except Exception:
        return False


def deterministic_query_sample(
    proposals: pd.DataFrame,
) -> pd.DataFrame:
    ordered = proposals.sort_values(
        [
            "qrsbt_relevance_probability",
            "qrsbt_confidence",
            "qrsbt_support_score",
            "base_rank",
            "product_id",
            "query_id",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(ordered) < QUERIES_PER_SEED:
        raise RuntimeError(
            "Fewer than the pre-registered number of "
            "placement-feasible proposal queries."
        )

    positions = np.rint(
        np.linspace(
            0,
            len(ordered) - 1,
            num=QUERIES_PER_SEED,
        )
    ).astype(int)

    if len(set(int(value) for value in positions)) != (
        QUERIES_PER_SEED
    ):
        raise RuntimeError(
            "Deterministic confidence-stratified selection "
            "duplicated a proposal index."
        )

    selected = ordered.iloc[positions].copy()

    if selected["query_id"].duplicated().any():
        raise RuntimeError(
            "Confidence-stratified sample contains duplicate queries."
        )

    return selected.sort_values(
        ["query_id"],
        kind="mergesort",
    ).reset_index(drop=True)



def extract_dynamic_metrics(
    summary: dict[str, Any],
) -> dict[str, Any]:
    output = jsonable_mapping(summary)

    pair_definitions = {
        "relevant_discovery_delta": (
            "qrsbt_relevant_discovery",
            "base_relevant_discovery",
        ),
        "irrelevant_exposure_delta": (
            "qrsbt_irrelevant_exposure",
            "base_irrelevant_exposure",
        ),
        "false_warmup_delta": (
            "qrsbt_false_warmup",
            "base_false_warmup",
        ),
    }

    for output_name, (numerator, denominator) in pair_definitions.items():
        if (
            numerator in output
            and denominator in output
            and output[numerator] is not None
            and output[denominator] is not None
        ):
            output[output_name] = float(
                output[numerator]
                - output[denominator]
            )

    required_utility_keys = {
        "mean_scenario_replication_utility_delta",
        "worst_scenario_utility_delta",
        "p10_scenario_replication_utility_delta",
    }

    missing = sorted(
        required_utility_keys - set(output)
    )

    if missing:
        raise RuntimeError(
            "Dynamic summary missing direct-utility fields: "
            f"{missing}"
        )

    return output


def ensure_runtime_only_proposal_columns(
    frame: pd.DataFrame,
) -> None:
    required = {
        "query_id",
        "product_id",
        "base_rank",
        *RUNTIME_PROPOSAL_COLUMNS,
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise RuntimeError(
            "Candidate frame lacks runtime-only proposal columns: "
            f"{missing}"
        )


if current_head() != expected_head:
    raise RuntimeError(
        f"Unexpected repository HEAD: {current_head()}"
    )

if not smoke_config_path.is_file():
    raise RuntimeError(
        f"Missing smoke config: {smoke_config_path}"
    )

charter = json.loads(
    charter_path.read_text(encoding="utf-8-sig")
)

if charter.get("baseline_commit") != expected_head:
    raise RuntimeError(
        "V5 charter baseline mismatch."
    )

registered_model_training_seeds = tuple(
    int(seed)
    for seed in charter["seed_registry"]["model_training"]
)

if registered_model_training_seeds != FINAL_TRAINING_SEEDS:
    raise RuntimeError(
        "V5-D final training registry mismatch."
    )

if set(FINAL_TRAINING_SEEDS) & set(
    charter["seed_registry"]["threshold_calibration"]
):
    raise RuntimeError(
        "Final training overlaps threshold calibration."
    )

if set(FINAL_TRAINING_SEEDS) & set(
    charter["seed_registry"]["confirmation"]
):
    raise RuntimeError(
        "Final training overlaps confirmation."
    )


if any(
    seed in charter["seed_registry"]["threshold_calibration"]
    for seed in FINAL_TRAINING_SEEDS
):
    raise RuntimeError(
        "Pilot overlaps threshold calibration."
    )

if any(
    seed in charter["seed_registry"]["confirmation"]
    for seed in FINAL_TRAINING_SEEDS
):
    raise RuntimeError(
        "Pilot overlaps confirmation."
    )

raw_smoke = yaml.safe_load(
    smoke_config_path.read_text(encoding="utf-8")
)

if not isinstance(raw_smoke, dict):
    raise RuntimeError(
        "Smoke config must parse to a mapping."
    )

required_dynamic_columns = set(
    dynamic_required_columns()
)

required_ranked_columns = {
    *required_dynamic_columns,
    "base_score",
    "base_rank",
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
}

output_root.mkdir(parents=True, exist_ok=True)
repository_output_root.mkdir(parents=True, exist_ok=True)

action_rows: list[dict[str, Any]] = []
daily_frames: list[pd.DataFrame] = []
proposal_rows: list[dict[str, Any]] = []
seed_runs: list[dict[str, Any]] = []

for seed in FINAL_TRAINING_SEEDS:
    model_output = (
        repository_output_root
        / f"seed-{seed}"
        / "smoke"
    )

    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_config = copy.deepcopy(raw_smoke)
    raw_config["seed"] = int(seed)
    raw_config["output_dir"] = str(model_output)

    execution_config_path = (
        output_root
        / "execution_configs"
        / f"smoke.seed{seed}.v5b.yaml"
    )

    execution_config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_config_path.write_text(
        yaml.safe_dump(
            raw_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = load_config(execution_config_path)

    run_model_stage(config)

    ranked_path, ranked_frame = discover_ranked_frame(
        model_output,
        required_columns=required_ranked_columns,
    )

    if ranked_frame.duplicated(
        ["query_id", "product_id"]
    ).any():
        raise RuntimeError(
            f"seed={seed}: ranked frame has duplicate query-product rows."
        )

    ensure_runtime_only_proposal_columns(ranked_frame)

    baseline_frame = make_no_promotion_frame(
        ranked_frame
    )

    gate_config = _gate_config(raw_config)

    eligibility_probe = apply_coverage_overreach_gate(
        ranked_frame.copy(),
        replace(
            gate_config,
            max_boost=1_000_000.0,
        ),
    )

    eligible = eligibility_probe.loc[
        eligibility_probe["gate_action"].eq("BOOST")
        & eligibility_probe["base_rank"].between(
            11,
            15,
            inclusive="both",
        )
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            f"seed={seed}: no runtime-safe actionable candidates."
        )

    eligible["placement_feasible"] = [
        placement_feasible(
            baseline_frame,
            query_id=int(row.query_id),
            product_id=int(row.product_id),
        )
        for row in eligible.itertuples(index=False)
    ]

    eligible = eligible.loc[
        eligible["placement_feasible"].eq(True)
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            f"seed={seed}: no placement-feasible safe candidates."
        )

    eligible = eligible.sort_values(
        [
            "query_id",
            "qrsbt_relevance_probability",
            "qrsbt_confidence",
            "qrsbt_support_score",
            "base_rank",
            "product_id",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    proposals = (
        eligible.groupby(
            "query_id",
            sort=False,
        )
        .head(1)
        .copy()
    )

    selected_proposals = deterministic_query_sample(
        proposals
    )

    seed_runs.append(
        {
            "seed": int(seed),
            "model_output_dir": str(model_output),
            "ranked_frame_path": str(ranked_path),
            "ranked_frame_rows": int(len(ranked_frame)),
            "dynamic_required_columns": sorted(
                required_dynamic_columns
            ),
            "runtime_safe_actionable_rows": int(
                len(eligible)
            ),
            "runtime_safe_actionable_queries": int(
                proposals["query_id"].nunique()
            ),
            "selected_pilot_queries": [
                int(value)
                for value in selected_proposals[
                    "query_id"
                ].tolist()
            ],
        }
    )

    for proposal_index, proposal in enumerate(
        selected_proposals.itertuples(index=False),
        start=1,
    ):
        query_id = int(proposal.query_id)
        product_id = int(proposal.product_id)

        proposal_rows.append(
            {
                "seed": int(seed),
                "proposal_index": int(proposal_index),
                "query_id": query_id,
                "product_id": product_id,
                "proposal_selection_rule": (
                    "query-local runtime-safe candidate with "
                    "highest qrsbt_relevance_probability, then "
                    "qrsbt_confidence, qrsbt_support_score, "
                    "base_rank, product_id"
                ),
                "qrsbt_relevance_probability": float(
                    proposal.qrsbt_relevance_probability
                ),
                "qrsbt_confidence": float(
                    proposal.qrsbt_confidence
                ),
                "qrsbt_support_score": float(
                    proposal.qrsbt_support_score
                ),
                "base_rank": int(proposal.base_rank),
                "teacher_or_oracle_columns_used": False,
            }
        )

        for action_name, target_rank in PLACEMENT_ACTIONS.items():
            action_frame, action_metadata = inject_action(
                baseline_frame,
                query_id=query_id,
                product_id=product_id,
                action_name=action_name,
                target_rank=target_rank,
            )

            result = run_dynamic_simulation(
                action_frame,
                days=PILOT_DAYS,
                traffic_per_day=PILOT_TRAFFIC_PER_DAY,
                seed=int(seed),
                replications=PILOT_REPLICATIONS,
            )

            metrics = extract_dynamic_metrics(
                result.summary
            )

            row = {
                "seed": int(seed),
                "proposal_index": int(proposal_index),
                "query_id": query_id,
                "product_id": product_id,
                **action_metadata,
                "days": PILOT_DAYS,
                "traffic_per_day": PILOT_TRAFFIC_PER_DAY,
                "replications": PILOT_REPLICATIONS,
                **metrics,
            }

            action_rows.append(row)

            daily = result.daily.copy()
            daily.insert(0, "seed", int(seed))
            daily.insert(1, "proposal_index", int(proposal_index))
            daily.insert(2, "query_id", query_id)
            daily.insert(3, "product_id", product_id)
            daily.insert(4, "action", action_name)

            daily_frames.append(daily)

action_effects = pd.DataFrame(action_rows).sort_values(
    [
        "seed",
        "proposal_index",
        "requested_rank",
    ],
    kind="mergesort",
).reset_index(drop=True)

proposal_manifest = pd.DataFrame(proposal_rows).sort_values(
    ["seed", "proposal_index"],
    kind="mergesort",
).reset_index(drop=True)

expected_action_count = (
    len(FINAL_TRAINING_SEEDS)
    * QUERIES_PER_SEED
    * len(PLACEMENT_ACTIONS)
)

if len(action_effects) != expected_action_count:
    raise RuntimeError(
        "Unexpected action-label count: "
        f"{len(action_effects)} != {expected_action_count}"
    )

if not action_effects["achieved_rank"].eq(
    action_effects["requested_rank"]
).all():
    raise RuntimeError(
        "At least one action violated exact placement contract."
    )

if proposal_manifest["teacher_or_oracle_columns_used"].any():
    raise RuntimeError(
        "V5-D proposal selection used forbidden labels."
    )

if daily_frames:
    daily_output = pd.concat(
        daily_frames,
        ignore_index=True,
    )
else:
    daily_output = pd.DataFrame()

action_effects_path = (
    output_root
    / "v5d_dynamic_action_effect_labels.csv"
)

proposal_manifest_path = (
    output_root
    / "v5d_runtime_safe_proposal_manifest.csv"
)

daily_path = (
    output_root
    / "v5d_dynamic_action_effect_daily.csv"
)

seed_runs_path = (
    output_root
    / "v5d_seed_execution_manifest.json"
)

action_effects.to_csv(
    action_effects_path,
    index=False,
)

proposal_manifest.to_csv(
    proposal_manifest_path,
    index=False,
)

daily_output.to_csv(
    daily_path,
    index=False,
)

write_json(seed_runs_path, seed_runs)

primary_metrics = [
    column
    for column in [
        "relevant_discovery_delta",
        "irrelevant_exposure_delta",
        "false_warmup_delta",
        "mean_scenario_replication_utility_delta",
        "worst_scenario_utility_delta",
        "p10_scenario_replication_utility_delta",
    ]
    if column in action_effects.columns
]

summary_by_action = (
    action_effects.groupby(
        "action",
        as_index=False,
    )[primary_metrics]
    .agg(["mean", "min", "max"])
)

summary_by_action.columns = [
    (
        column[0]
        if column[1] == ""
        else f"{column[0]}_{column[1]}"
    )
    for column in summary_by_action.columns
]

summary_by_action_path = (
    output_root
    / "v5d_action_effect_summary_by_placement.csv"
)

summary_by_action.to_csv(
    summary_by_action_path,
    index=False,
)

decision = {
    "status": (
        "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
    ),
    "baseline_commit": expected_head,
    "training_seeds": list(FINAL_TRAINING_SEEDS),
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "proposal_count": int(len(proposal_manifest)),
    "action_count": int(len(action_effects)),
    "expected_action_count": int(expected_action_count),
    "action_space": [
        "NO_PROMOTION",
        *list(PLACEMENT_ACTIONS),
    ],
    "counterfactual_comparator": "NO_PROMOTION",
    "proposal_runtime_columns": RUNTIME_PROPOSAL_COLUMNS,
    "teacher_or_oracle_columns_used_for_proposal": False,
    "placement_contract": (
        "Set candidate final_score to nextafter(base-score "
        "at requested rank boundary, +infinity); retain only "
        "counterfactuals that achieve the requested canonical "
        "rank under final_score descending and product_id "
        "ascending ordering."
    ),
    "dynamic_pilot_budget": {
        "days": PILOT_DAYS,
        "traffic_per_day": PILOT_TRAFFIC_PER_DAY,
        "replications": PILOT_REPLICATIONS,
        "scenarios": [
            "conservative",
            "neutral",
            "exploratory",
        ],
    },
    "outcome_label_boundary": (
        "Synthetic dynamic simulator outcomes only. "
        "Not a production causal-effect estimate."
    ),
    "available_dynamic_metrics": primary_metrics,
    "seed_runs_path": str(seed_runs_path),
    "proposal_manifest_path": str(proposal_manifest_path),
    "action_effects_path": str(action_effects_path),
    "daily_path": str(daily_path),
    "summary_by_action_path": str(summary_by_action_path),
    "source_modified": False,
    "config_modified": False,
    "model_trained": False,
    "threshold_selected": False,
    "dynamic_replay_executed_at_release_scale": False,
    "confirmation_seeds_executed": False,
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Audit action-label integrity, common outcome coverage, "
        "and placement monotonicity. Only then expand V5 direct "
        "action-effect enumeration to the remaining independent "
        "model-training seeds."
    ),
}

decision_path = (
    output_root
    / "v5d_direct_action_effect_decision.json"
)

write_json(decision_path, decision)

print("===== V5-D DIRECT ACTION-EFFECT PILOT =====")
print(
    json.dumps(
        {
            "decision": decision,
            "seed_runs": seed_runs,
            "action_summary_by_placement": (
                summary_by_action.to_dict(
                    orient="records"
                )
            ),
        },
        indent=2,
        sort_keys=True,
    )
)