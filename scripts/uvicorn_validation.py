"""Launch a real Uvicorn process and validate readiness, concurrency, and shutdown."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _get_json(url: str, *, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, *, timeout: float = 10.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return int(exc.code), body


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def _cleanup_uvicorn_log_dir(path: Path, *, attempts: int = 8) -> None:
    """Remove validator logs, retrying transient Windows file-handle locks."""
    last_error: OSError | None = None

    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(0.05 * (2**attempt), 0.75))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to clean Uvicorn log directory: {path}")


def validate_uvicorn(
    artifact_dir: Path,
    *,
    requests: int = 16,
    concurrency: int = 4,
    workers: int = 1,
    max_concurrency: int | None = None,
    admission_timeout_ms: float = 1000.0,
    expect_overload: bool = False,
    test_admission_hold_ms: float | None = None,
    startup_timeout: float = 30.0,
) -> dict:
    if requests < 1 or concurrency < 1 or concurrency > requests or workers < 1:
        raise ValueError("requests, concurrency, or workers are outside valid bounds")
    if max_concurrency is None:
        max_concurrency = max(concurrency, 1)
    if max_concurrency < 1 or admission_timeout_ms < 0:
        raise ValueError("admission settings are outside valid bounds")

    if expect_overload:
        if admission_timeout_ms == 1000.0:
            admission_timeout_ms = 25.0
        if test_admission_hold_ms is None:
            test_admission_hold_ms = 200.0

    if test_admission_hold_ms is None:
        test_admission_hold_ms = 0.0
    if test_admission_hold_ms < 0:
        raise ValueError("test_admission_hold_ms must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    artifact_dir = artifact_dir.resolve(strict=True)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PRODUCT_SEARCH_ARTIFACT_DIR"] = str(artifact_dir)
    env["PRODUCT_SEARCH_VERIFY_ARTIFACTS"] = "1"
    env["PRODUCT_SEARCH_STRICT_ENV"] = "1"
    env["PRODUCT_SEARCH_MAX_CONCURRENCY"] = str(max_concurrency)
    env["PRODUCT_SEARCH_ADMISSION_TIMEOUT_MS"] = str(admission_timeout_ms)
    env["PRODUCT_SEARCH_TEST_ADMISSION_HOLD_MS"] = str(test_admission_hold_ms)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "product_search.serving.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        "warning",
    ]
    log_dir = Path(tempfile.mkdtemp(prefix="product-search-uvicorn-"))
    log_path = log_dir / "uvicorn.log"
    log_handle = log_path.open("w+", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=(os.name != "nt"),
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    started = time.monotonic()
    try:
        ready: dict | None = None
        while time.monotonic() - started < startup_timeout:
            if process.poll() is not None:
                log_handle.flush()
                output = (
                    log_path.read_text(encoding="utf-8", errors="replace")
                    if log_path.exists()
                    else ""
                )
                raise RuntimeError(f"Uvicorn exited before readiness:\n{output[-4000:]}")
            try:
                ready = _get_json(f"{base_url}/ready", timeout=1.0)
                break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(0.1)
        if ready is None:
            raise TimeoutError("Uvicorn did not become ready before the deadline")

        queries = [
            "wireless headphones",
            "trail running shoes",
            "business laptop",
            "portable charger",
        ]
        payloads = [{"query": queries[index % len(queries)], "k": 5} for index in range(requests)]
        request_started = time.perf_counter()
        barrier = threading.Barrier(concurrency) if expect_overload else None

        def send(body: dict) -> tuple[int, dict]:
            if barrier is not None:
                barrier.wait(timeout=10)
            return _post_json(f"{base_url}/search", body)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            responses = list(executor.map(send, payloads))
        elapsed = time.perf_counter() - request_started
        statuses = [status for status, _ in responses]
        accepted = [(status, body) for status, body in responses if status == 200]
        rejected = [(status, body) for status, body in responses if status == 503]
        unexpected = [status for status in statuses if status not in {200, 503}]
        if unexpected:
            raise RuntimeError(
                f"Concurrent Uvicorn requests returned unexpected statuses: {statuses}"
            )
        if expect_overload:
            if not accepted or not rejected:
                raise RuntimeError(
                    f"Overload validation requires both accepted and rejected requests: {statuses}"
                )
        elif rejected or len(accepted) != requests:
            raise RuntimeError(f"Concurrent Uvicorn requests failed: {statuses}")
        accepted_bodies = [body for _, body in accepted]
        if any(len(body.get("results", [])) != 5 for body in accepted_bodies):
            raise RuntimeError("Uvicorn response contract returned an unexpected result count")
        versions = {str(body.get("model_version")) for body in accepted_bodies}
        if versions != {str(ready.get("model_version"))}:
            raise RuntimeError("Concurrent requests observed inconsistent model versions")
        if any(bool(body.get("fallback_used")) for body in accepted_bodies):
            raise RuntimeError("Concurrent Uvicorn validation unexpectedly used a fallback")
        metrics = _get_json(f"{base_url}/metrics")
        # Uvicorn metrics are process-local. Exact request totals are asserted only for one worker.
        if workers == 1:
            if int(metrics.get("requests_total", -1)) != len(accepted):
                raise RuntimeError("Uvicorn metrics did not record every accepted request")
            if int(metrics.get("overload_rejections_total", -1)) != len(rejected):
                raise RuntimeError("Uvicorn metrics did not record every overload rejection")
        return {
            "status": "PASS",
            "requests": requests,
            "accepted": len(accepted),
            "concurrency": concurrency,
            "workers": workers,
            "max_concurrency_per_worker": max_concurrency,
            "model_version": ready.get("model_version"),
            "release_generation": ready.get("release_generation"),
            "errors": 0,
            "fallbacks": 0,
            "overload_rejections": len(rejected),
            "admission_timeout_ms": admission_timeout_ms,
            "test_admission_hold_ms": test_admission_hold_ms,
            "elapsed_seconds": elapsed,
            "throughput_requests_per_second": requests / max(elapsed, 1e-9),
        }
    finally:
        _terminate(process)
        log_handle.close()
        try:
            _cleanup_uvicorn_log_dir(log_dir)
        except OSError as cleanup_error:
            if sys.exc_info()[0] is None:
                raise
            print(
                f"Uvicorn validator cleanup warning: {cleanup_error}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/smoke")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--admission-timeout-ms", type=float, default=1000.0)
    parser.add_argument("--expect-overload", action="store_true")
    parser.add_argument("--test-admission-hold-ms", type=float)
    args = parser.parse_args()
    result = validate_uvicorn(
        Path(args.artifact_dir),
        requests=args.requests,
        concurrency=args.concurrency,
        workers=args.workers,
        max_concurrency=args.max_concurrency,
        admission_timeout_ms=args.admission_timeout_ms,
        expect_overload=args.expect_overload,
        test_admission_hold_ms=args.test_admission_hold_ms,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
