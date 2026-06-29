from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from product_search.release_packaging import DEFAULT_SOURCE_DATE_EPOCH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sdist(source: Path, destination: Path, *, source_date_epoch: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tar_buffer = io.BytesIO()
    with tarfile.open(source, "r:gz") as incoming, tarfile.open(
        fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT
    ) as outgoing:
        for member in sorted(incoming.getmembers(), key=lambda item: item.name):
            if member.issym() or member.islnk():
                raise RuntimeError(f"Release sdist must not contain links: {member.name}")
            normalized = tarfile.TarInfo(member.name)
            normalized.type = member.type
            normalized.mtime = int(source_date_epoch)
            normalized.uid = 0
            normalized.gid = 0
            normalized.uname = ""
            normalized.gname = ""
            normalized.pax_headers = {}
            if member.isdir():
                normalized.mode = 0o755
                normalized.size = 0
                outgoing.addfile(normalized)
            elif member.isfile():
                extracted = incoming.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not read sdist member: {member.name}")
                data = extracted.read()
                executable = bool(member.mode & stat.S_IXUSR)
                normalized.mode = 0o755 if executable else 0o644
                normalized.size = len(data)
                outgoing.addfile(normalized, io.BytesIO(data))
            else:
                raise RuntimeError(f"Unsupported sdist member type: {member.name}")
    temp = destination.with_name(f".{destination.name}.tmp")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                fileobj=raw,
                mode="wb",
                compresslevel=9,
                mtime=int(source_date_epoch),
            ) as compressed:
                compressed.write(tar_buffer.getvalue())
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _clean_build_state(root: Path) -> None:
    shutil.rmtree(root / "build", ignore_errors=True)
    for path in (root / "src").glob("*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)


def build_release_distributions(
    root: Path,
    output_dir: Path,
    *,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(int(source_date_epoch))
    env.setdefault("PYTHONHASHSEED", "0")
    with tempfile.TemporaryDirectory(prefix="product-search-build-") as temp_dir:
        raw_dir = Path(temp_dir) / "raw"
        raw_dir.mkdir()
        _clean_build_state(root)
        completed = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(raw_dir)],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Distribution build failed:\n" + completed.stdout[-4000:]
            )
        wheels = list(raw_dir.glob("*.whl"))
        sdists = list(raw_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError("Expected exactly one wheel and one sdist")
        wheel_out = output_dir / wheels[0].name
        sdist_out = output_dir / sdists[0].name
        shutil.copyfile(wheels[0], wheel_out)
        normalize_sdist(sdists[0], sdist_out, source_date_epoch=source_date_epoch)
    _clean_build_state(root)
    return {
        "wheel": {"path": str(wheel_out), "sha256": _sha256(wheel_out)},
        "sdist": {"path": str(sdist_out), "sha256": _sha256(sdist_out)},
        "source_date_epoch": int(source_date_epoch),
    }


def verify_reproducible_distributions(
    root: Path,
    *,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="product-search-build-verify-") as temp_dir:
        first_dir = Path(temp_dir) / "first"
        second_dir = Path(temp_dir) / "second"
        first = build_release_distributions(
            root, first_dir, source_date_epoch=source_date_epoch
        )
        second = build_release_distributions(
            root, second_dir, source_date_epoch=source_date_epoch
        )
        wheel_equal = first["wheel"]["sha256"] == second["wheel"]["sha256"]
        sdist_equal = first["sdist"]["sha256"] == second["sdist"]["sha256"]
        return {
            "status": "PASS" if wheel_equal and sdist_equal else "FAIL",
            "wheel_byte_identical": wheel_equal,
            "sdist_byte_identical": sdist_equal,
            "wheel_sha256": first["wheel"]["sha256"],
            "sdist_sha256": first["sdist"]["sha256"],
            "source_date_epoch": int(source_date_epoch),
        }
