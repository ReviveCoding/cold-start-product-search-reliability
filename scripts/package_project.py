from __future__ import annotations

from _bootstrap import bootstrap_src

root = bootstrap_src()

import argparse
from pathlib import Path

from product_search.release_packaging import build_deterministic_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="dist/cold-start-product-search-reliability.zip"
    )
    parser.add_argument("--include-generated-artifacts", action="store_true")
    parser.add_argument("--root-name")
    parser.add_argument("--source-date-epoch", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    result = build_deterministic_zip(
        root,
        output,
        include_generated_artifacts=args.include_generated_artifacts,
        root_name=args.root_name,
        source_date_epoch=args.source_date_epoch,
    )
    print(result)


if __name__ == "__main__":
    main()
