#!/usr/bin/env bash
set -euo pipefail
exec "${PRODUCT_SEARCH_PYTHON:-python}" scripts/run_full_pipeline.py --config "${1:-configs/smoke.yaml}"
