from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path


diagnostic_path = Path(sys.argv[1]).resolve()
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


source = diagnostic_path.read_text(encoding="utf-8")

mean_pattern = re.compile(
    r'''
    predicted\.to_numpy\(dtype=np\.int64\)\.eq\(
        \s*query\["base_rank"\]\s*
    \)\.mean\(\)
    ''',
    flags=re.VERBOSE,
)

all_pattern = re.compile(
    r'''
    predicted\.to_numpy\(dtype=np\.int64\)\.eq\(
        \s*query\["base_rank"\]\s*
    \)\.all\(\)
    ''',
    flags=re.VERBOSE,
)

mean_replacement = '''np.equal(
            predicted.to_numpy(dtype=np.int64),
            query["base_rank"].to_numpy(dtype=np.int64),
        ).mean()'''

all_replacement = '''np.equal(
            predicted.to_numpy(dtype=np.int64),
            query["base_rank"].to_numpy(dtype=np.int64),
        ).all()'''

repaired, mean_count = mean_pattern.subn(
    mean_replacement,
    source,
    count=1,
)

repaired, all_count = all_pattern.subn(
    all_replacement,
    repaired,
    count=1,
)

if mean_count != 1 or all_count != 1:
    raise RuntimeError(
        "Expected one ndarray equality mean and one all expression; "
        f"found mean={mean_count}, all={all_count}."
    )

if ".to_numpy(dtype=np.int64).eq(" in repaired:
    raise RuntimeError(
        "A pandas-only .eq call remains on a NumPy array."
    )

ast.parse(
    repaired,
    filename=str(diagnostic_path),
)

diagnostic_path.write_text(
    repaired,
    encoding="utf-8",
)

decision = {
    "status": "V5G1_1_NDARRAY_EQUALITY_CONTRACT_REPAIRED",
    "diagnostic_path": str(diagnostic_path),
    "diagnostic_sha256_after": sha256(diagnostic_path),
    "repair_scope": [
        "replace ndarray .eq(...).mean() with np.equal(...).mean()",
        "replace ndarray .eq(...).all() with np.equal(...).all()",
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