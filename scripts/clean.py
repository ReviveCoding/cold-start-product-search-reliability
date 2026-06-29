from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixed = ["artifacts", ".pytest_cache", ".ruff_cache", "build", "dist"]
    for relative in fixed:
        shutil.rmtree(root / relative, ignore_errors=True)
    for pattern in ("__pycache__", "*.egg-info"):
        for path in root.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    print("Removed generated artifacts, caches, and build outputs")


if __name__ == "__main__":
    main()
