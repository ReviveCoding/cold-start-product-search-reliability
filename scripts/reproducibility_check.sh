#!/usr/bin/env bash
set -euo pipefail
PYTHON_EXE="${PRODUCT_SEARCH_PYTHON:-$(command -v python)}"
exec "$PYTHON_EXE" "$(dirname "$0")/reproducibility_check.py" --config "${1:-configs/smoke.yaml}" ${2:+"$2"}
