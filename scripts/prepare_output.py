from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import shutil
from pathlib import Path

from product_search.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely reset a generated artifact directory")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    output = (root / config.output_dir).resolve() if not config.output_dir.is_absolute() else config.output_dir.resolve()
    artifact_root = (root / "artifacts").resolve()
    try:
        output.relative_to(artifact_root)
    except ValueError as exc:
        raise SystemExit(
            f"Refusing to delete output outside repository artifacts directory: {output}"
        ) from exc
    if output == artifact_root:
        raise SystemExit("Refusing to delete the artifact root itself")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".product_search_artifacts").write_text("generated\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
