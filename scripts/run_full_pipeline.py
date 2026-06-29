"""Cross-platform, process-isolated end-to-end pipeline orchestrator."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_stage(
    command: list[str],
    *,
    root: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"Stage failed with exit code {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _run_release_stage(
    command: list[str],
    *,
    root: Path,
    env: dict[str, str],
    output: Path,
    log_path: Path,
    timeout: float,
) -> None:
    marker = output / "release_stage_metadata.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    release_env = env.copy()
    release_env["PRODUCT_SEARCH_HARD_EXIT"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=release_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        deadline = time.monotonic() + timeout
        complete = False
        while time.monotonic() < deadline:
            if marker.exists():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    complete = payload.get("stage_status") == "complete"
                except (OSError, json.JSONDecodeError):
                    complete = False
                if complete:
                    break
            code = process.poll()
            if code is not None:
                break
            time.sleep(0.1)
        if complete:
            _terminate_process_tree(process)
        else:
            code = process.poll()
            _terminate_process_tree(process)
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            if code is None:
                raise TimeoutError(f"Release stage timed out before completion\n{tail}")
            raise RuntimeError(f"Release stage exited with code {code} before completion\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--stage-timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    supplied = Path(args.config)
    config_path = (
        supplied.resolve() if supplied.is_absolute() else (Path.cwd() / supplied).resolve()
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (root / output).resolve()
    env = _environment(root)
    python = sys.executable
    started = time.perf_counter()

    _run_stage(
        [python, str(root / "scripts" / "prepare_output.py"), "--config", str(config_path)],
        root=root,
        env=env,
        log_path=output.parent / "prepare_output.log",
        timeout=60,
    )
    logs = output / "logs"
    _run_stage(
        [python, str(root / "scripts" / "run_model_stage.py"), "--config", str(config_path)],
        root=root,
        env=env,
        log_path=logs / "model_stage.log",
        timeout=args.stage_timeout_seconds,
    )
    print("[pipeline] model stage complete", file=sys.stderr, flush=True)

    simulation = config["simulation"]
    _run_stage(
        [
            python,
            str(root / "scripts" / "run_dynamic_simulation.py"),
            "--input",
            str(output / "ranked_test.csv"),
            "--output-dir",
            str(output),
            "--days",
            str(simulation["days"]),
            "--traffic-per-day",
            str(simulation["traffic_per_day"]),
            "--replications",
            str(simulation["replications"]),
            "--seed",
            str(config["seed"]),
        ],
        root=root,
        env=env,
        log_path=logs / "dynamic_stage.log",
        timeout=args.stage_timeout_seconds,
    )
    print("[pipeline] dynamic stage complete", file=sys.stderr, flush=True)

    _run_release_stage(
        [
            python,
            str(root / "scripts" / "run_ope_validation.py"),
            "--seed",
            str(config["seed"]),
            "--rows",
            "5000",
            "--output",
            str(output / "ope_metrics.json"),
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--orchestrator-runtime",
            str(time.perf_counter() - started),
        ],
        root=root,
        env=env,
        output=output,
        log_path=logs / "release_stage.log",
        timeout=args.stage_timeout_seconds,
    )

    release = json.loads((output / "release_decision.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "status": release["status"],
        "gates_passed": sum(bool(value) for value in release["gates"].values()),
        "gates_total": len(release["gates"]),
        "cold_ndcg_lift": metrics["cold_ndcg_at_10_final"]
        - metrics["cold_ndcg_at_10_base"],
        "output_dir": str(output),
        "runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
