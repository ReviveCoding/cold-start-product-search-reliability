from __future__ import annotations

from _bootstrap import bootstrap_src

root = bootstrap_src()

import argparse
import json
from pathlib import Path

import pandas as pd

from product_search.config import load_config
from product_search.evaluation.sensitivity import (
    evaluate_policy_sensitivity,
    parse_policy_values,
    sensitivity_markdown,
)
from product_search.provenance import atomic_write_csv, atomic_write_json, atomic_write_text


from product_search.evaluation.sensitivity import DEFAULT_BOOSTS, DEFAULT_DYNAMIC_FINALISTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--input")
    parser.add_argument(
        "--boosts",
        default=",".join(str(value) for value in DEFAULT_BOOSTS),
    )
    parser.add_argument(
        "--dynamic-finalists",
        default=",".join(str(value) for value in DEFAULT_DYNAMIC_FINALISTS),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = config.output_dir
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir
    input_path = Path(args.input) if args.input else artifact_dir / "ranked_test.csv"
    output_dir = Path(args.output_dir) if args.output_dir else artifact_dir / "policy_sensitivity"
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not input_path.exists():
        raise FileNotFoundError(f"Ranked candidate input does not exist: {input_path}")

    result, summary = evaluate_policy_sensitivity(
        pd.read_csv(input_path),
        config.raw,
        boosts=parse_policy_values(args.boosts, label="Boosts"),
        dynamic_finalists=parse_policy_values(args.dynamic_finalists, label="Dynamic finalists"),
        bootstrap_samples=args.bootstrap_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "policy_sensitivity.csv", result)
    atomic_write_json(output_dir / "policy_sensitivity.json", summary)
    atomic_write_text(
        output_dir / "policy_sensitivity.md",
        sensitivity_markdown(result, float(summary["selected_max_boost"])),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
