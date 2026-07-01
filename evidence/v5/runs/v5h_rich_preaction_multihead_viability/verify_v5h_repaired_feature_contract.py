from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


audit_path = Path(sys.argv[1]).resolve()
recovery_decision_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()

REMOVED_CONSTANTS = {
    "qrsbt_support_score",
    "qrsbt_support",
    "attribute_compatible",
    "zero_history",
    "sparse_history",
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
    "clicked",
    "purchased",
    "judged",
    "logging_propensity",
    "position",
    "time_block",
    "user_id",
    "prior_",
    "smoothed_",
    "behavior_velocity",
    "first_observed_age",
    "final_score",
    "qrsbt_boost",
    "gate_action",
    "gate_reason",
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


def read_assignment(
    source: str,
    name: str,
) -> tuple[str, ...]:
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
            f"Expected exactly one assignment for {name}."
        )

    return tuple(ast.literal_eval(matches[0].value))


source = audit_path.read_text(encoding="utf-8")

context_numeric = read_assignment(
    source,
    "CONTEXT_NUMERIC_FEATURES",
)

action_numeric = read_assignment(
    source,
    "ACTION_NUMERIC_FEATURES",
)

categorical = read_assignment(
    source,
    "CATEGORICAL_FEATURES",
)

selected_features = (
    *context_numeric,
    *action_numeric,
    *categorical,
)

if REMOVED_CONSTANTS & set(selected_features):
    raise RuntimeError(
        "A removed globally constant feature remains in the "
        "repaired V5-H feature contract."
    )

for feature in selected_features:
    if any(
        token in feature.lower()
        for token in FORBIDDEN_TOKENS
    ):
        raise RuntimeError(
            "V5-H feature contract contains a forbidden token: "
            f"{feature}"
        )

recovery = json.loads(
    recovery_decision_path.read_text(encoding="utf-8-sig")
)

if (
    recovery.get("status")
    != "V5G1_PREACTION_RUNTIME_STATE_RECOVERED_AND_CONTRACT_READY"
):
    raise RuntimeError(
        "Unexpected V5-G1 recovery status."
    )

state_action_path = Path(
    recovery["state_action_path"]
).resolve()

frame = pd.read_csv(state_action_path)

if len(frame) != 720:
    raise RuntimeError(
        f"Expected 720 recovered state-action rows, found {len(frame)}."
    )

if frame["seed"].nunique() != 18:
    raise RuntimeError("Expected 18 recovered seed groups.")

missing = sorted(set(selected_features) - set(frame.columns))

if missing:
    raise RuntimeError(
        f"Repaired feature contract columns missing: {missing}"
    )

constant_features = [
    feature
    for feature in selected_features
    if frame[feature].nunique(dropna=True) <= 1
]

if constant_features:
    raise RuntimeError(
        "Repaired V5-H feature contract still has global constants: "
        f"{constant_features}"
    )

payload: dict[str, Any] = {
    "status": "V5H_REPAIRED_FEATURE_CONTRACT_PREFLIGHT_PASS",
    "audit_path": str(audit_path),
    "audit_sha256": sha256(audit_path),
    "state_action_path": str(state_action_path),
    "state_action_sha256": sha256(state_action_path),
    "state_action_row_count": int(len(frame)),
    "seed_count": int(frame["seed"].nunique()),
    "context_numeric_count": len(context_numeric),
    "action_numeric_count": len(action_numeric),
    "categorical_count": len(categorical),
    "total_feature_count": len(selected_features),
    "removed_global_constants_confirmed_absent": sorted(
        REMOVED_CONSTANTS
    ),
    "remaining_global_constant_feature_count": 0,
    "source_modified": False,
    "config_modified": False,
    "model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
}

output_path.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print(json.dumps(payload, indent=2, sort_keys=True))