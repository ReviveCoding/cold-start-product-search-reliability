from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download user-selected public dataset snapshots")
    parser.add_argument("dataset", choices=["kuaisearch", "esci"])
    parser.add_argument("--output", default="data/external")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "kuaisearch":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub first") from exc
        snapshot_download(
            repo_id="benchen4395/KuaiSearch",
            repo_type="dataset",
            local_dir=output / "KuaiSearch",
        )
    else:
        raise SystemExit(
            "ESCI is distributed through the amazon-science/esci-data repository. "
            "Follow its official data-download instructions and record the snapshot in data/manifests/."
        )


if __name__ == "__main__":
    main()
