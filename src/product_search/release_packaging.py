from __future__ import annotations

import os
import stat
import time
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".coverage",
    ".mypy_cache",
    ".tox",
    ".hypothesis",
    "htmlcov",
}
DEFAULT_SOURCE_DATE_EPOCH = 1_577_836_800  # 2020-01-01T00:00:00Z


def _include_path(relative: Path, *, include_generated_artifacts: bool) -> bool:
    if any(
        part in EXCLUDED_PARTS
        or part.endswith(".egg-info")
        or part.endswith(".pyc")
        for part in relative.parts
    ):
        return False
    if relative.parts[:2] == ("data", "external"):
        return False
    if relative.parts and relative.parts[0] == "artifacts":
        return include_generated_artifacts or (
            len(relative.parts) >= 2 and relative.parts[1] == "sample"
        )
    return True


def build_deterministic_zip(
    root: Path,
    output: Path,
    *,
    include_generated_artifacts: bool = False,
    root_name: str | None = None,
    source_date_epoch: int | None = None,
) -> Path:
    root = root.resolve(strict=True)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    epoch = int(
        source_date_epoch
        if source_date_epoch is not None
        else os.getenv("SOURCE_DATE_EPOCH", DEFAULT_SOURCE_DATE_EPOCH)
    )
    date_time = time.gmtime(max(epoch, 315_532_800))[:6]  # ZIP cannot represent pre-1980.
    archive_root = root_name or root.name
    if not archive_root or Path(archive_root).name != archive_root:
        raise ValueError("root_name must be one safe path component")

    entries: list[tuple[Path, Path]] = []
    for path in root.rglob("*"):
        if path == output or path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if _include_path(relative, include_generated_artifacts=include_generated_artifacts):
            entries.append((relative, path))

    temp = output.with_name(f".{output.name}.tmp")
    temp.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temp,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for relative, path in sorted(entries, key=lambda item: item[0].as_posix()):
                arcname = (Path(archive_root) / relative).as_posix()
                info = zipfile.ZipInfo(arcname, date_time=date_time)
                info.create_system = 3
                executable = path.suffix in {".sh", ".ps1"}
                mode = stat.S_IFREG | (0o755 if executable else 0o644)
                info.external_attr = mode << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return output
