from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path


runner_path = Path(sys.argv[1]).resolve()
decision_path = Path(sys.argv[2]).resolve()
expected_runner_sha256 = sys.argv[3].lower()

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
            f"{label}: expected one replacement, found {count}."
        )

    return source.replace(old, new, 1)


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
    if (
        not hasattr(node, "lineno")
        or not hasattr(node, "col_offset")
        or not hasattr(node, "end_lineno")
        or not hasattr(node, "end_col_offset")
    ):
        raise RuntimeError(
            f"AST node lacks source span: {type(node).__name__}"
        )

    offsets = line_offsets(source)

    start = (
        offsets[node.lineno - 1]
        + node.col_offset
    )

    end = (
        offsets[node.end_lineno - 1]
        + node.end_col_offset
    )

    return start, end


def replace_nodes(
    source: str,
    replacements: list[tuple[ast.AST, str]],
) -> str:
    resolved = []

    for node, replacement in replacements:
        start, end = node_span(source, node)
        resolved.append((start, end, replacement))

    resolved.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for start, end, replacement in resolved:
        source = (
            source[:start]
            + replacement
            + source[end:]
        )

    return source


if sha256(runner_path).lower() != expected_runner_sha256:
    raise RuntimeError(
        "Runner hash is not the expected untouched copied V5-B runner."
    )

source = runner_path.read_text(encoding="utf-8")

if (
    "from product_search.config import load_config"
    not in source
):
    raise RuntimeError(
        "Runner lacks repaired load_config contract."
    )

if "validate_config(raw_config)" in source:
    raise RuntimeError(
        "Runner still contains invalid validate_config usage."
    )

source = source.replace(
    "PILOT_SEEDS",
    "FINAL_TRAINING_SEEDS",
)

final_seed_literal = (
    "FINAL_TRAINING_SEEDS = (\n"
    + "".join(
        f"    {seed},\n"
        for seed in FINAL_TRAINING_SEEDS
    )
    + ")"
)

source = replace_once(
    source,
    "FINAL_TRAINING_SEEDS = (263, 269, 271)",
    final_seed_literal,
    label="final-training seed constant",
)

source = replace_once(
    source,
    "QUERIES_PER_SEED = 3",
    "QUERIES_PER_SEED = 8",
    label="context quota",
)

tree = ast.parse(
    source,
    filename=str(runner_path),
)

sampler_nodes = [
    node
    for node in tree.body
    if (
        isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.name == "deterministic_query_sample"
    )
]

if len(sampler_nodes) != 1:
    raise RuntimeError(
        "Could not identify exactly one deterministic sampler function."
    )

registry_nodes = []

for node in tree.body:
    if not isinstance(node, ast.If):
        continue

    segment = ast.get_source_segment(
        source,
        node,
    ) or ""

    if "v5b_pilot_scope" in segment:
        registry_nodes.append(node)

if len(registry_nodes) != 1:
    raise RuntimeError(
        "Could not identify exactly one V5-B pilot registry guard."
    )

sampler_replacement = '''def deterministic_query_sample(
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

registry_replacement = '''registered_model_training_seeds = tuple(
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

source = replace_nodes(
    source,
    [
        (sampler_nodes[0], sampler_replacement),
        (registry_nodes[0], registry_replacement),
    ],
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

required_fragments = (
    "FINAL_TRAINING_SEEDS = (",
    "QUERIES_PER_SEED = 8",
    "registered_model_training_seeds",
    "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE",
    "from product_search.config import load_config",
    "config = load_config(execution_config_path)",
    "teacher_or_oracle_columns_used",
)

for fragment in required_fragments:
    if fragment not in source:
        raise RuntimeError(
            f"Required V5-D fragment missing: {fragment}"
        )

for forbidden in (
    "PILOT_SEEDS",
    "v5b_pilot_scope",
    "validate_config(raw_config)",
):
    if forbidden in source:
        raise RuntimeError(
            f"Forbidden V5-B fragment remains: {forbidden}"
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
    "status": "V5D_FINAL_TRAINING_CORPUS_RUNNER_AST_PATCHED",
    "runner_path": str(runner_path),
    "runner_sha256_after": sha256(runner_path),
    "final_training_seed_count": len(
        FINAL_TRAINING_SEEDS
    ),
    "final_training_seeds": list(
        FINAL_TRAINING_SEEDS
    ),
    "contexts_per_seed": 8,
    "proposal_count_expected": (
        len(FINAL_TRAINING_SEEDS) * 8
    ),
    "action_count_expected": (
        len(FINAL_TRAINING_SEEDS) * 8 * 5
    ),
    "sampling_rule": (
        "Per seed, choose one runtime-safe and placement-feasible "
        "candidate per query. Sort query-level candidates by the "
        "runtime QRSBT confidence spectrum and take eight evenly "
        "spaced strata. No teacher, relevance, action outcome, "
        "or oracle-only field is used."
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