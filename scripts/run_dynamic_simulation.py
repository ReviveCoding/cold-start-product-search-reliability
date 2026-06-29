from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import json
from pathlib import Path

import pandas as pd

from product_search.provenance import atomic_write_csv, atomic_write_json
from product_search.simulation.dynamic import run_dynamic_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--traffic-per-day", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replications", type=int, default=1)
    args = parser.parse_args()
    input_path = Path(args.input)
    output = Path(args.output_dir)
    metadata_path = output / "model_stage_metadata.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Ranked candidate input does not exist: {input_path}")
    if not metadata_path.exists():
        raise RuntimeError("Model stage metadata is missing; refusing to run simulation out of order")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("stage_status") != "complete":
        raise RuntimeError("Model stage is not marked complete")

    frame = pd.read_csv(input_path)
    result = run_dynamic_simulation(
        frame,
        days=args.days,
        traffic_per_day=args.traffic_per_day,
        seed=args.seed,
        replications=args.replications,
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "dynamic_daily.csv", result.daily)
    atomic_write_json(output / "dynamic_summary.json", result.summary)
    atomic_write_json(
        output / "dynamic_stage_metadata.json",
        {
            "stage_status": "complete",
            "input_rows": int(len(frame)),
            "days": int(args.days),
            "traffic_per_day": int(args.traffic_per_day),
            "replications": int(args.replications),
            "scenarios": sorted(result.daily.scenario.unique().tolist()),
        },
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
