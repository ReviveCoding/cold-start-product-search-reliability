from __future__ import annotations

from _bootstrap import bootstrap_src

root = bootstrap_src()

import argparse
import json

from product_search.release_build import verify_reproducible_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-date-epoch", type=int, default=1_577_836_800)
    args = parser.parse_args()
    result = verify_reproducible_distributions(
        root, source_date_epoch=args.source_date_epoch
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
