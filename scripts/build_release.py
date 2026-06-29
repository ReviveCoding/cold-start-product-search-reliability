from __future__ import annotations

from _bootstrap import bootstrap_src

root = bootstrap_src()

import argparse
import json
from pathlib import Path

from product_search.release_build import build_release_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--source-date-epoch", type=int, default=1_577_836_800)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    result = build_release_distributions(
        root, output, source_date_epoch=args.source_date_epoch
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
