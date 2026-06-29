"""Run every test module in an isolated process and combine coverage evidence.

Native ML runtimes can retain OpenMP/BLAS state across many XGBoost-heavy tests.  A single
pytest interpreter therefore has a larger failure surface than the application itself.  This
runner keeps each test module independent, enforces a per-shard timeout, and combines the
resulting coverage files before applying the repository-wide coverage gate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _environment(root: Path, coverage_file: Path, marker: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PRODUCT_SEARCH_PYTEST_HARD_EXIT"] = "1"
    env["PRODUCT_SEARCH_PYTEST_COMPLETION_MARKER"] = str(marker)
    env["PYTHONHASHSEED"] = "0"
    env["COVERAGE_FILE"] = str(coverage_file)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root / "scripts"),
            str(root / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def _terminate(process: subprocess.Popen[str]) -> None:
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


def _run_shard(
    test_path: Path,
    *,
    root: Path,
    work_dir: Path,
    timeout_seconds: float,
    extra_args: list[str],
) -> tuple[int, float, str]:
    name = test_path.stem
    marker = work_dir / f"{name}.complete.json"
    coverage_file = work_dir / f".coverage.{name}"
    log_path = work_dir / f"{name}.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "pytest_cov",
        "-p",
        "pytest_completion_plugin",
        "--cov=product_search",
        "--cov-report=",
        "--cov-fail-under=0",
        str(test_path),
        *extra_args,
    ]
    started = time.perf_counter()
    env = _environment(root, coverage_file, marker)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        deadline = time.monotonic() + timeout_seconds
        code: int | None = None
        marker_status: int | None = None
        while time.monotonic() < deadline:
            if marker.exists():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    marker_status = int(payload.get("exitstatus", -1))
                except (OSError, ValueError, json.JSONDecodeError):
                    marker_status = None
                if marker_status is not None and marker_status >= 0:
                    code = marker_status
                    _terminate(process)
                    break
            polled = process.poll()
            if polled is not None:
                code = int(polled)
                break
            time.sleep(0.05)
        if code is None:
            _terminate(process)
            code = 124
    elapsed = time.perf_counter() - started
    output = log_path.read_text(encoding="utf-8", errors="replace")
    if code == 0 and marker_status is None:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            marker_status = int(payload.get("exitstatus", -1))
            if marker_status != 0:
                code = marker_status if marker_status >= 0 else 3
        except (OSError, ValueError, json.JSONDecodeError):
            code = 3
            output += "\nMissing or invalid pytest completion marker.\n"
    return code, elapsed, output


def _discover(root: Path, requested: list[str]) -> list[Path]:
    if requested:
        paths = [Path(item) for item in requested]
        return [path if path.is_absolute() else root / path for path in paths]
    return sorted((root / "tests").glob("test_*.py"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", help="Optional test modules to execute")
    parser.add_argument("--shard-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--verbose-shards", action="store_true")
    args, extra_args = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    tests = _discover(root, args.tests)
    if not tests:
        raise SystemExit("No test modules were discovered")
    missing = [str(path) for path in tests if not path.is_file()]
    if missing:
        raise SystemExit(f"Test modules do not exist: {missing}")

    work_dir = Path(tempfile.mkdtemp(prefix="product-search-test-shards-"))
    failures: list[tuple[Path, int, str]] = []
    completed = 0
    started = time.perf_counter()
    try:
        for index, test_path in enumerate(tests, start=1):
            code, elapsed, output = _run_shard(
                test_path,
                root=root,
                work_dir=work_dir,
                timeout_seconds=args.shard_timeout_seconds,
                extra_args=extra_args,
            )
            label = "PASS" if code == 0 else "FAIL"
            print(
                f"[{index:02d}/{len(tests):02d}] {label} {test_path.name} "
                f"({elapsed:.2f}s)",
                flush=True,
            )
            if args.verbose_shards or code != 0:
                print(output.rstrip(), flush=True)
            if code == 0:
                completed += 1
            else:
                failures.append((test_path, code, output))
                break

        if failures:
            path, code, output = failures[0]
            print(f"\nFirst failed shard: {path} (exit {code})", file=sys.stderr)
            if not args.verbose_shards:
                print(output[-5000:], file=sys.stderr)
            raise SystemExit(code)

        coverage_files = sorted(work_dir.glob(".coverage.*"))
        if len(coverage_files) != len(tests):
            raise SystemExit(
                f"Expected {len(tests)} coverage shards, found {len(coverage_files)}"
            )
        final_coverage = root / ".coverage"
        final_coverage.unlink(missing_ok=True)
        combine_env = os.environ.copy()
        combine_env["COVERAGE_FILE"] = str(final_coverage)
        subprocess.run(
            [sys.executable, "-m", "coverage", "combine", *map(str, coverage_files)],
            cwd=root,
            env=combine_env,
            check=True,
        )
        report = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--show-missing",
                "--skip-covered",
                "--fail-under=70",
            ],
            cwd=root,
            env=combine_env,
            text=True,
            check=False,
        )
        if report.returncode != 0:
            raise SystemExit(report.returncode)
        print(
            f"\nAll {completed} test modules passed in "
            f"{time.perf_counter() - started:.2f}s.",
            flush=True,
        )
    finally:
        if args.keep_work_dir:
            print(f"Shard logs retained at {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
