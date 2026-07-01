from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


audit_path = Path(sys.argv[1]).resolve()
repair_decision_path = Path(sys.argv[2]).resolve()

EXPECTED_CONSTANTS = (
    "qrsbt_support_score",
    "qrsbt_support",
    "attribute_compatible",
    "zero_history",
    "sparse_history",
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
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


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
            f"Node does not have a complete source span: {type(node).__name__}"
        )

    offsets = line_offsets(source)

    return (
        offsets[node.lineno - 1] + node.col_offset,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def find_assignment(
    source: str,
    name: str,
) -> ast.Assign:
    tree = ast.parse(source, filename=str(audit_path))

    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one assignment for {name}; found {len(matches)}."
        )

    return matches[0]


source = audit_path.read_text(encoding="utf-8")

assignment = find_assignment(
    source,
    "CONTEXT_NUMERIC_FEATURES",
)

start, end = node_span(source, assignment)
assignment_text = source[start:end]

assignment_value = assignment.value
frozen_before = tuple(ast.literal_eval(assignment_value))

if not set(EXPECTED_CONSTANTS).issubset(frozen_before):
    raise RuntimeError(
        "The expected constant features are absent from the "
        "pre-repair V5-H context feature contract."
    )

repaired_assignment = assignment_text

for feature in EXPECTED_CONSTANTS:
    pattern = re.compile(
        rf'(?m)^[ \t]*"{re.escape(feature)}",[ \t]*\r?\n'
    )

    repaired_assignment, replacement_count = pattern.subn(
        "",
        repaired_assignment,
        count=1,
    )

    if replacement_count != 1:
        raise RuntimeError(
            f"Expected to remove exactly one contract line for "
            f"{feature}; removed {replacement_count}."
        )

repaired_source = (
    source[:start]
    + repaired_assignment
    + source[end:]
)

repaired_assignment_node = find_assignment(
    repaired_source,
    "CONTEXT_NUMERIC_FEATURES",
)

frozen_after = tuple(
    ast.literal_eval(repaired_assignment_node.value)
)

unexpected_remaining = sorted(
    set(EXPECTED_CONSTANTS) & set(frozen_after)
)

if unexpected_remaining:
    raise RuntimeError(
        "Constant feature(s) remain after repair: "
        f"{unexpected_remaining}"
    )

removed = [
    feature
    for feature in frozen_before
    if feature not in frozen_after
]

if tuple(removed) != EXPECTED_CONSTANTS:
    raise RuntimeError(
        "Repair removed an unexpected feature set: "
        f"{removed}"
    )

for forbidden in (
    "teacher",
    "oracle",
    "future",
    "outcome",
    "label",
    "utility_delta",
    "discovery_delta",
    "exposure_delta",
    "warmup_delta",
    "final_score",
    "qrsbt_boost",
    "gate_action",
    "gate_reason",
):
    if any(forbidden in value.lower() for value in frozen_after):
        raise RuntimeError(
            "The repaired context contract contains an unexpected "
            f"forbidden feature token: {forbidden}"
        )

ast.parse(repaired_source, filename=str(audit_path))

audit_path.write_text(
    repaired_source,
    encoding="utf-8",
)

decision = {
    "status": "V5H_CONSTANT_FEATURE_CONTRACT_REPAIRED",
    "audit_path": str(audit_path),
    "audit_sha256_after": sha256(audit_path),
    "removed_globally_constant_preaction_features": list(
        EXPECTED_CONSTANTS
    ),
    "context_numeric_count_before": len(frozen_before),
    "context_numeric_count_after": len(frozen_after),
    "repair_rule": (
        "Remove only columns proven invariant across all 720 "
        "recovered pre-action state-action rows. This is an "
        "unsupervised X-only contract cleanup; no outcome, teacher, "
        "oracle, calibration, or confirmation information is used."
    ),
    "source_modified": False,
    "config_modified": False,
    "v5d_corpus_rerun": False,
    "final_serving_model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
}

write_json(repair_decision_path, decision)

print(json.dumps(decision, indent=2, sort_keys=True))