from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "v5"
EVIDENCE_MANIFEST = EVIDENCE_ROOT / "EVIDENCE_MANIFEST.json"
PACKAGE_MANIFEST = EVIDENCE_ROOT / "FILE_MANIFEST.csv"
CLEANUP_RECEIPT = EVIDENCE_ROOT / "CLEANUP_RECEIPT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (EVIDENCE_ROOT, EVIDENCE_MANIFEST, PACKAGE_MANIFEST, CLEANUP_RECEIPT):
        if not path.exists():
            raise RuntimeError(f"Required evidence path is missing: {path}")

    evidence = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8-sig"))
    receipt = json.loads(CLEANUP_RECEIPT.read_text(encoding="utf-8-sig"))

    if evidence.get("closeout_status") != "V5_BASELINE_RETAINED":
        raise RuntimeError("Unexpected V5 evidence closeout status.")

    final_decision = receipt.get("final_decision", {})
    required = {
        "v5j_paired_contrast_predictive_model_viability": "NOT_JUSTIFIED",
        "source_baseline_retained": True,
        "final_serving_model_trained": False,
        "threshold_selected": False,
    }
    for key, expected in required.items():
        if final_decision.get(key) != expected:
            raise RuntimeError(f"Unexpected final-decision field: {key}")

    if final_decision.get("calibration_seeds_executed") not in (None, [], False):
        raise RuntimeError("Unexpected calibration execution.")
    if final_decision.get("confirmation_seeds_executed") not in (None, [], False):
        raise RuntimeError("Unexpected confirmation execution.")

    with PACKAGE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Package manifest is empty.")

    for row in rows:
        relative = row["package_relative_path"]
        if relative == "FILE_MANIFEST.csv":
            raise RuntimeError("FILE_MANIFEST.csv must not hash itself.")

        artifact = EVIDENCE_ROOT / Path(relative)
        if not artifact.is_file():
            raise RuntimeError(f"Manifest artifact missing: {artifact}")

        if artifact.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"Byte-count mismatch: {artifact}")

        if sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {artifact}")

    print(
        json.dumps(
            {
                "status": "V5_CLOSEOUT_EVIDENCE_VERIFIED",
                "evidence_root": str(EVIDENCE_ROOT),
                "package_file_count_excluding_self": len(rows),
                "closeout_status": evidence["closeout_status"],
                "v5j_paired_contrast_predictive_model_viability": (
                    final_decision[
                        "v5j_paired_contrast_predictive_model_viability"
                    ]
                ),
                "source_baseline_retained": final_decision[
                    "source_baseline_retained"
                ],
                "final_serving_model_trained": final_decision[
                    "final_serving_model_trained"
                ],
                "threshold_selected": final_decision["threshold_selected"],
                "calibration_seeds_executed": final_decision[
                    "calibration_seeds_executed"
                ],
                "confirmation_seeds_executed": final_decision[
                    "confirmation_seeds_executed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
