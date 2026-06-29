from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from product_search.data.synthetic import generate_synthetic_bundle


def test_canonical_bundle_runs_model_stage_without_oracle_target(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = generate_synthetic_bundle(
        seed=19,
        n_products=96,
        n_queries=16,
        n_users=48,
        n_time_blocks=6,
        impressions_per_query_time=10,
    )
    bundle.queries = bundle.queries.drop(columns=["target_product_id"])
    canonical_dir = tmp_path / "canonical"
    bundle.write(canonical_dir)

    config = yaml.safe_load((root / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    output_dir = tmp_path / "artifacts"
    config["data"] = {"source": "canonical", "canonical_dir": str(canonical_dir)}
    config["output_dir"] = str(output_dir)
    config["retrieval"]["candidate_k"] = 32
    config["ranking"]["xgb_estimators"] = 8
    config["qrsbt"]["neighbors"] = 4
    config["qrsbt"]["min_support"] = 2
    config_path = tmp_path / "canonical.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [sys.executable, str(root / "scripts" / "run_model_stage.py"), "--config", str(config_path)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    metadata = json.loads((output_dir / "model_stage_metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage_status"] == "complete"
    assert metadata["data_source"] == "canonical"
    assert metadata["n_products"] == 96
    assert metadata["n_interactions"] == len(bundle.interactions)
