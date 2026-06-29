"""Allow repository scripts to run from a source checkout before editable installation."""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root
