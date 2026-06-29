from __future__ import annotations

from _bootstrap import bootstrap_src

root = bootstrap_src()

import argparse
import json
from pathlib import Path

from product_search.candidate_handoff import validate_handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", default="release_candidate_handoff.json")
    parser.add_argument(
        "--allow-missing-build-artifacts",
        action="store_true",
        help=(
            "Validate source-only handoff metadata before dist/ artifacts are built. "
            "Dependency manifests and command evidence remain strict; wheel/sdist records "
            "are reported as missing instead of failing. Omit this flag for final release validation."
        ),
    )
    args = parser.parse_args()
    path = Path(args.handoff)
    if not path.is_absolute():
        path = root / path
    print(
        json.dumps(
            validate_handoff(
                root,
                path,
                allow_missing_build_artifacts=args.allow_missing_build_artifacts,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
