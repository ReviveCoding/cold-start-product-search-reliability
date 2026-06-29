from __future__ import annotations

from _bootstrap import bootstrap_src

root = bootstrap_src()

import argparse
import json
from pathlib import Path

from product_search.candidate_handoff import build_handoff, validate_handoff


def _artifact(value: str) -> tuple[str, Path]:
    try:
        kind, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("artifact must be KIND=PATH") from exc
    return kind, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="release_candidate_handoff.json")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--baseline-hashes", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--artifact", action="append", type=_artifact, default=[])
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    artifacts = [
        (kind, path if path.is_absolute() else root / path) for kind, path in args.artifact
    ]
    build_handoff(
        root=root,
        output=output,
        candidate_version=args.candidate_version,
        source_commit=args.source_commit,
        baseline_hashes_path=Path(args.baseline_hashes),
        evidence_dir=Path(args.evidence_dir),
        release_manifest_path=Path(args.release_manifest),
        build_artifacts=artifacts,
    )
    print(json.dumps(validate_handoff(root, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
