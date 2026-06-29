"""Cross-platform independent replay against an existing validated smoke run."""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DETERMINISTIC_FILES = (
    "metrics.json",
    "data/products.csv",
    "data/queries.csv",
    "data/relevance.csv",
    "data/interactions.csv",
    "ranked_test.csv",
    "release_catalog.csv",
    "behavior_future_audit_features.csv",
    "product_behavior_snapshot.csv",
    "behavior_features.json",
    "lambdamart_metadata.json",
    "bm25.joblib",
    "dense.joblib",
    "lambdamart_model.json",
    "behavior_model.joblib",
    "qrsbt.joblib",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_model_stage(root: Path, config: Path, timeout_seconds: float) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PRODUCT_SEARCH_HARD_STAGE_EXIT"] = "1"
    env.setdefault("PYTHONHASHSEED", "0")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_model_stage.py"), "--config", str(config)],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
        start_new_session=(os.name != "nt"),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Independent reproducibility stage failed for {config.name}:\n"
            + completed.stderr[-4000:]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--stage-timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    supplied = Path(args.config)
    config_path = (
        supplied.resolve() if supplied.is_absolute() else (Path.cwd() / supplied).resolve()
    )
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise RuntimeError("Reproducibility configuration root must be a mapping")
    configured_output = Path(str(base["output_dir"]))
    reference = (
        configured_output.resolve()
        if configured_output.is_absolute()
        else (root / configured_output).resolve()
    )
    metadata_path = reference / "model_stage_metadata.json"
    manifest_path = reference / "manifest.json"
    if not metadata_path.exists() or not manifest_path.exists():
        raise RuntimeError(
            "A completed reference run is required; run the smoke or integration pipeline first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("stage_status") != "complete" or manifest.get("stage_status") != "complete":
        raise RuntimeError("Configured reference artifact is not marked complete")
    if manifest.get("config_file_sha256") != _sha256(config_path):
        raise RuntimeError("Configured reference artifact was produced from a different config file")
    missing_reference = [relative for relative in DETERMINISTIC_FILES if not (reference / relative).is_file()]
    if missing_reference:
        raise RuntimeError(f"Reference artifact is incomplete: {missing_reference}")

    replay = root / "artifacts" / "repro_check"
    replay_config = root / "artifacts" / "repro_check.yaml"
    shutil.rmtree(replay, ignore_errors=True)
    run_config = dict(base)
    run_config["output_dir"] = str(replay.relative_to(root))
    replay_config.parent.mkdir(parents=True, exist_ok=True)
    replay_config.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    try:
        _run_model_stage(root, replay_config, args.stage_timeout_seconds)
        failures = [
            relative
            for relative in DETERMINISTIC_FILES
            if not filecmp.cmp(reference / relative, replay / relative, shallow=False)
        ]
        result = {
            "status": "PASS" if not failures else "FAIL",
            "model_runs": 2,
            "reference_run": str(reference),
            "independent_replay_runs": 1,
            "compared_files": len(DETERMINISTIC_FILES),
            "dynamic_and_ope_determinism": "covered_by_unit_tests",
            "failures": failures,
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if not failures else 1
    finally:
        if not args.keep:
            shutil.rmtree(replay, ignore_errors=True)
            replay_config.unlink(missing_ok=True)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
