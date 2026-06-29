from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import json
import os
import sys

from product_search.config import load_config
from product_search.pipeline import run_model_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument(
        "--normal-exit",
        action="store_true",
        help="Use normal interpreter shutdown instead of the isolated native-runtime stage exit.",
    )
    args = parser.parse_args()
    result = run_model_stage(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    # Some XGBoost/OpenMP combinations can stall during interpreter finalization after every
    # artifact has already been atomically persisted. The command-line stage is intentionally
    # process-isolated, so a hard exit avoids that native shutdown deadlock without affecting the
    # importable pipeline function or losing buffered output.
    if not args.normal_exit and os.getenv("PRODUCT_SEARCH_HARD_STAGE_EXIT", "1") != "0":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
