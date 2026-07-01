from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path


runner_path = Path(sys.argv[1]).resolve()
decision_path = Path(sys.argv[2]).resolve()

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def replace_once(
    source: str,
    old: str,
    new: str,
    *,
    label: str,
) -> str:
    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected one exact replacement, found {count}."
        )

    return source.replace(old, new, 1)


source = runner_path.read_text(encoding="utf-8")

if (
    "from product_search.config import load_config"
    not in source
):
    raise RuntimeError(
        "Copied V5-B runner lacks repaired load_config contract."
    )

if "validate_config(raw_config)" in source:
    raise RuntimeError(
        "Copied V5-B runner still uses invalid validate_config return."
    )

source = source.replace(
    "PILOT_SEEDS",
    "FINAL_TRAINING_SEEDS",
)

source = replace_once(
    source,
    "FINAL_TRAINING_SEEDS = (263, 269, 271)",
    (
        "FINAL_TRAINING_SEEDS = (\n"
        + "".join(
            f"    {seed},\n"
            for seed in FINAL_TRAINING_SEEDS
        )
        + ")"
    ),
    label="training-seed constant",
)

source = replace_once(
    source,
    "QUERIES_PER_SEED = 3",
    "QUERIES_PER_SEED = 8",
    label="query quota",
)

registry_pattern = re.compile(
    r'''
if\ tuple\(
    \s*int\(seed\)
    \s*for\ seed\ in\ charter\["v5b_pilot_scope"\]\[
    \s*"allowed_model_training_pilot_seeds"
    \s*\]
\s*\)\ !=\ FINAL_TRAINING_SEEDS:
    \s*raise\ RuntimeError\(
    \s*"V5-B\ pilot\ registry\ mismatch\."
    \s*\)
''',
    flags=re.VERBOSE,
)

registry_replacement = '''
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
'''

source, registry_count = registry_pattern.subn(
    registry_replacement,
    source,
    count=1,
)

if registry_count != 1:
    raise RuntimeError(
        "Could not replace V5-B pilot-only seed validation."
    )

sample_pattern = re.compile(
    r'''
def\ deterministic_query_sample\(
    proposals:\ pd\.DataFrame,
\)\ ->\ pd\.DataFrame:
.*?
(?=
def\ extract_dynamic_metrics)
''',
    flags=re.DOTALL | re.VERBOSE,
)

sample_replacement = '''
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


'''

source, sample_count = sample_pattern.subn(
    sample_replacement,
    source,
    count=1,
)

if sample_count != 1:
    raise RuntimeError(
        "Could not replace deterministic query sampler."
    )

source = source.replace(
    "V5B_DIRECT_DYNAMIC_ACTION_EFFECT_PILOT_COMPLETE",
    "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE",
)

source = source.replace(
    "V5-B",
    "V5-D",
)

source = source.replace(
    "v5b_",
    "v5d_",
)

source = source.replace(
    '"pilot_seeds": list(FINAL_TRAINING_SEEDS),',
    '"training_seeds": list(FINAL_TRAINING_SEEDS),',
)

source = source.replace(
    '"Pilot overlaps threshold calibration."',
    '"Final training overlaps threshold calibration."',
)

source = source.replace(
    '"Pilot overlaps confirmation."',
    '"Final training overlaps confirmation."',
)

if "v5b_pilot_scope" in source:
    raise RuntimeError(
        "V5-D runner still depends on pilot-only charter scope."
    )

if "PILOT_SEEDS" in source:
    raise RuntimeError(
        "V5-D runner still contains PILOT_SEEDS."
    )

if "QUERIES_PER_SEED = 8" not in source:
    raise RuntimeError(
        "V5-D runner query quota replacement failed."
    )

if source.count("FINAL_TRAINING_SEEDS") < 5:
    raise RuntimeError(
        "V5-D runner seed replacement appears incomplete."
    )

ast.parse(
    source,
    filename=str(runner_path),
)

runner_path.write_text(
    source,
    encoding="utf-8",
)

decision = {
    "status": "V5D_FINAL_TRAINING_CORPUS_RUNNER_PATCHED",
    "runner_path": str(runner_path),
    "runner_sha256_after": sha256(runner_path),
    "final_training_seed_count": len(
        FINAL_TRAINING_SEEDS
    ),
    "final_training_seeds": list(
        FINAL_TRAINING_SEEDS
    ),
    "contexts_per_seed": 8,
    "action_count_expected": (
        len(FINAL_TRAINING_SEEDS) * 8 * 5
    ),
    "proposal_count_expected": (
        len(FINAL_TRAINING_SEEDS) * 8
    ),
    "selection_rule": (
        "Within each seed, choose one runtime-safe and "
        "placement-feasible candidate per query, order those "
        "query-level proposals by runtime QRSBT confidence "
        "spectrum, and select eight evenly spaced strata. "
        "The rule uses no teacher, relevance, outcome, or "
        "oracle-derived fields."
    ),
    "source_modified": False,
    "config_modified": False,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
}

decision_path.write_text(
    json.dumps(
        decision,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        decision,
        indent=2,
        sort_keys=True,
    )
)