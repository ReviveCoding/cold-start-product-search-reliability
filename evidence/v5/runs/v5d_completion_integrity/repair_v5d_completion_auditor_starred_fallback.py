from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


auditor_path = Path(sys.argv[1]).resolve()
repair_decision_path = Path(sys.argv[2]).resolve()
interface_path = Path(sys.argv[3]).resolve()


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


def find_matching_list_open(
    source: str,
    close_index: int,
) -> int:
    depth = 0

    for index in range(close_index, -1, -1):
        character = source[index]

        if character == "]":
            depth += 1
            continue

        if character == "[":
            depth -= 1

            if depth == 0:
                return index

    raise RuntimeError(
        "Could not locate matching list opening bracket."
    )


def extract_argv_bindings(
    source: str,
) -> list[dict[str, Any]]:
    tree = ast.parse(
        source,
        filename=str(auditor_path),
    )

    bindings: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if not isinstance(target, ast.Name):
            continue

        argv_index: int | None = None

        for child in ast.walk(node.value):
            if not isinstance(child, ast.Subscript):
                continue

            if not (
                isinstance(child.value, ast.Attribute)
                and isinstance(child.value.value, ast.Name)
                and child.value.value.id == "sys"
                and child.value.attr == "argv"
            ):
                continue

            slice_node = child.slice

            if (
                isinstance(slice_node, ast.Constant)
                and isinstance(slice_node.value, int)
            ):
                argv_index = int(slice_node.value)
                break

        if argv_index is not None:
            bindings.append(
                {
                    "argv_index": argv_index,
                    "target_name": target.id,
                    "line": int(node.lineno),
                    "source": (
                        ast.get_source_segment(
                            source,
                            node,
                        )
                        or ""
                    )[:400],
                }
            )

    return sorted(
        bindings,
        key=lambda item: (
            item["argv_index"],
            item["line"],
        ),
    )


source = auditor_path.read_text(encoding="utf-8")

fallback_pattern = re.compile(
    r"""\]\s+or\s+\[\s*["']-\s*None\.\s*["']\s*\]\s*,""",
)

matches = list(fallback_pattern.finditer(source))

if len(matches) != 1:
    raise RuntimeError(
        "Expected exactly one malformed list-fallback pattern; "
        f"found {len(matches)}."
    )

match = matches[0]
close_bracket_index = match.start()
open_bracket_index = find_matching_list_open(
    source,
    close_bracket_index,
)

star_index = open_bracket_index - 1

if star_index < 0 or source[star_index] != "*":
    raise RuntimeError(
        "Malformed fallback is not preceded by a starred list expression."
    )

fragment = source[
    open_bracket_index : match.end()
]

fragment_without_comma = fragment.rstrip()

if not fragment_without_comma.endswith(","):
    raise RuntimeError(
        "Malformed fallback fragment does not end with a comma."
    )

expression = fragment_without_comma[:-1].rstrip()

replacement = f"*({expression}),"

repaired = (
    source[:star_index]
    + replacement
    + source[match.end():]
)

if repaired.count('or ["- None."]') != 1:
    raise RuntimeError(
        "Unexpected fallback expression count after repair."
    )

ast.parse(
    repaired,
    filename=str(auditor_path),
)

auditor_path.write_text(
    repaired,
    encoding="utf-8",
)

bindings = extract_argv_bindings(repaired)

if not bindings:
    raise RuntimeError(
        "No sys.argv bindings were discovered after repair."
    )

repair_decision = {
    "status": "V5D_COMPLETION_AUDITOR_STARRED_FALLBACK_REPAIRED",
    "auditor_path": str(auditor_path),
    "auditor_sha256_after": sha256(auditor_path),
    "repair_scope": (
        "Wrapped exactly one starred list fallback expression "
        "in parentheses so the fallback boolean expression is "
        "valid before iterable unpacking."
    ),
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

write_json(repair_decision_path, repair_decision)

write_json(
    interface_path,
    {
        "status": "V5D_COMPLETION_AUDITOR_INTERFACE_DISCOVERED",
        "auditor_path": str(auditor_path),
        "argv_bindings": bindings,
    },
)

print("===== V5-D COMPLETION AUDITOR REPAIR =====")
print(
    json.dumps(
        repair_decision,
        indent=2,
        sort_keys=True,
    )
)

print("===== V5-D COMPLETION AUDITOR INTERFACE =====")
print(
    json.dumps(
        {
            "argv_bindings": bindings,
        },
        indent=2,
        sort_keys=True,
    )
)