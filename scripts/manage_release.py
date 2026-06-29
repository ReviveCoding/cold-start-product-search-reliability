"""Publish, inspect, unlock, or roll back an immutable product-search release store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from product_search.release_store import (  # noqa: E402
    force_unlock,
    publish_release,
    read_pointer,
    resolve_current_release,
    rollback_release,
)
from product_search.serving.app import SearchService  # noqa: E402


def _runtime_validator(path: Path) -> None:
    SearchService(path, verify_hashes=True, allow_nonlaunch=False, strict_environment=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--source", required=True)
    publish.add_argument("--release-root", required=True)
    publish.add_argument("--generation")
    publish.add_argument("--skip-runtime-load", action="store_true")

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--release-root", required=True)
    rollback.add_argument("--generation")
    rollback.add_argument("--skip-runtime-load", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--release-root", required=True)

    unlock = subparsers.add_parser("force-unlock")
    unlock.add_argument("--release-root", required=True)

    args = parser.parse_args()
    validator = None if getattr(args, "skip_runtime_load", False) else _runtime_validator
    if args.command == "publish":
        pointer = publish_release(
            Path(args.source),
            Path(args.release_root),
            generation=args.generation,
            runtime_validator=validator,
        )
        print(json.dumps(pointer.as_dict(), indent=2, sort_keys=True))
    elif args.command == "rollback":
        pointer = rollback_release(
            Path(args.release_root),
            generation=args.generation,
            runtime_validator=validator,
        )
        print(json.dumps(pointer.as_dict(), indent=2, sort_keys=True))
    elif args.command == "status":
        root = Path(args.release_root)
        pointer = read_pointer(root)
        payload = pointer.as_dict()
        payload["artifact_dir"] = str(resolve_current_release(root))
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({"unlocked": force_unlock(Path(args.release_root))}))


if __name__ == "__main__":
    main()
