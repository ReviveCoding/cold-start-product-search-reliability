"""Hosted CI35 promotion-family ablation with baseline-replay enforcement.

Runs on a forensic-only branch whose changes are restricted to this driver and
its manual workflow. It downloads and evaluates the canonical artifact from a
specified GitHub Actions run in the matching Ubuntu/Python 3.11 environment.

The control variant preserves every artifact `final_score` exactly. An ablation
changes only rows belonging to a disabled promoted product family:
    qrsbt_boost = 0
    final_score = base_score

Results are emitted only after `current_policy` dynamically matches the
artifact's dynamic_summary.json within the requested tolerance.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ALLOWED_FORENSIC_PATHS = {
    "scripts/forensics/ci35_promotion_family_ablation_hosted.py",
    ".github/workflows/ci35-promotion-family-ablation.yml",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"Missing {label}: {path}")


def git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def verify_forensic_branch(repo_root: Path, artifact_commit: str) -> list[str]:
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                artifact_commit,
                "HEAD",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        fail(
            "Artifact commit must be an ancestor of the forensic branch HEAD. "
            f"artifact={artifact_commit}; stderr={exc.stderr.strip()}"
        )

    changed = [
        line
        for line in git(repo_root, "diff", "--name-only", f"{artifact_commit}..HEAD").splitlines()
        if line
    ]
    unexpected = sorted(set(changed) - ALLOWED_FORENSIC_PATHS)
    if unexpected:
        fail(
            "Forensic branch differs from artifact commit outside the allowed "
            f"forensic files: {unexpected}"
        )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    requested_output_root = Path(args.output_root).expanduser().resolve()
    ranked_path = artifact_root / "ranked_test.csv"
    manifest_path = artifact_root / "manifest.json"
    dynamic_path = artifact_root / "dynamic_summary.json"
    config_path = repo_root / "configs" / "smoke.yaml"

    for path, label in (
        (ranked_path, "ranked_test.csv"),
        (manifest_path, "manifest.json"),
        (dynamic_path, "dynamic_summary.json"),
        (config_path, "smoke config"),
    ):
        require_file(path, label)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_commit = str(manifest.get("git_commit", ""))
    if not artifact_commit:
        fail("Artifact manifest has no git_commit.")
    changed_forensic_paths = verify_forensic_branch(repo_root, artifact_commit)
    repo_head = git(repo_root, "rev-parse", "HEAD")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(raw["seed"])
    simulation = raw["simulation"]
    release = raw["release"]

    sys.path.insert(0, str(repo_root / "src"))
    from product_search.evaluation.metrics import ranking_report
    from product_search.simulation.dynamic import run_dynamic_simulation

    ranked = pd.read_csv(ranked_path).copy()
    required = {
        "query_id",
        "product_id",
        "base_score",
        "final_score",
        "qrsbt_boost",
        "relevance",
        "zero_history",
        "quality",
    }
    missing = sorted(required - set(ranked.columns))
    if missing:
        fail(f"ranked_test.csv missing required columns: {missing}")

    for column in ("base_score", "final_score", "qrsbt_boost"):
        ranked[column] = pd.to_numeric(ranked[column], errors="raise")

    promoted = ranked.loc[ranked["qrsbt_boost"].gt(0.0)].copy()
    families = sorted(promoted["product_id"].astype(int).unique().tolist())
    if not families:
        fail("Artifact has no positive Q-RSBT boosts to ablate.")

    artifact_dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
    dynamic_keys = (
        "base_relevant_discovery",
        "qrsbt_relevant_discovery",
        "base_irrelevant_exposure",
        "qrsbt_irrelevant_exposure",
        "base_false_warmup",
        "qrsbt_false_warmup",
        "base_cold_to_warm",
        "qrsbt_cold_to_warm",
        "base_utility",
        "qrsbt_utility",
        "worst_scenario_relevant_delta",
        "worst_scenario_irrelevant_delta",
        "worst_scenario_utility_delta",
        "p10_scenario_replication_utility_delta",
        "mean_scenario_replication_utility_delta",
    )
    control_dynamic = run_dynamic_simulation(
        ranked,
        days=int(simulation["days"]),
        traffic_per_day=int(simulation["traffic_per_day"]),
        seed=seed,
        replications=int(simulation["replications"]),
    ).summary
    mismatches: dict[str, dict[str, float]] = {}
    for key in dynamic_keys:
        actual = float(control_dynamic[key])
        expected = float(artifact_dynamic[key])
        if abs(actual - expected) > args.tolerance:
            mismatches[key] = {
                "replayed": actual,
                "artifact": expected,
                "difference": actual - expected,
            }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = requested_output_root / f"ci35_promotion_family_ablation_{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    baseline = {
        "exact_within_tolerance": not mismatches,
        "tolerance": args.tolerance,
        "mismatches": mismatches,
    }
    receipt_base = {
        "artifact_commit": artifact_commit,
        "repo_head": repo_head,
        "forensic_branch_changed_paths": changed_forensic_paths,
        "promoted_product_families": families,
        "baseline_dynamic_match": baseline,
        "output_root": str(output_root),
    }
    if mismatches:
        receipt = {"status": "FAIL_BASELINE_REPLAY", **receipt_base}
        (output_root / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(2)

    variants: list[tuple[str, tuple[int, ...]]] = [("current_policy", tuple())]
    for count in range(1, len(families) + 1):
        for disabled in itertools.combinations(families, count):
            variants.append(
                (
                    "disable_" + "_".join(f"p{product_id}" for product_id in disabled),
                    disabled,
                )
            )

    results: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    for variant, disabled in variants:
        candidate = ranked.copy()
        mask = (
            candidate["product_id"].astype(int).isin(disabled)
            & candidate["qrsbt_boost"].gt(0.0)
        )
        candidate.loc[mask, "qrsbt_boost"] = 0.0
        candidate.loc[mask, "final_score"] = candidate.loc[mask, "base_score"]

        static = ranking_report(candidate, bootstrap_samples=300, seed=seed)
        dynamic = run_dynamic_simulation(
            candidate,
            days=int(simulation["days"]),
            traffic_per_day=int(simulation["traffic_per_day"]),
            seed=seed,
            replications=int(simulation["replications"]),
        ).summary

        active = candidate.loc[candidate["qrsbt_boost"].gt(0.0)]
        for row in active.itertuples(index=False):
            active_rows.append(
                {
                    "variant": variant,
                    "disabled_families": "|".join(map(str, disabled)),
                    "query_id": int(row.query_id),
                    "product_id": int(row.product_id),
                    "base_rank": int(row.base_rank),
                    "qrsbt_boost": float(row.qrsbt_boost),
                    "relevance": float(row.relevance),
                }
            )

        results.append(
            {
                "variant": variant,
                "disabled_families": "|".join(map(str, disabled)),
                "disabled_static_promotions": int(mask.sum()),
                "remaining_static_promotions": int(len(active)),
                "remaining_total_boost": float(active["qrsbt_boost"].sum()),
                "overall_ndcg_delta": float(
                    static["final_ndcg_at_10"] - static["base_ndcg_at_10"]
                ),
                "cold_ndcg_lift": float(
                    static["cold_ndcg_at_10_final"] - static["cold_ndcg_at_10_base"]
                ),
                "cold_ndcg_ci_low": float(static["cold_ndcg_lift_ci_low"]),
                "warm_ndcg_delta": float(
                    static["warm_ndcg_at_10_final"] - static["warm_ndcg_at_10_base"]
                ),
                "static_irrelevant_exposure_delta": float(
                    static["irrelevant_exposure_final"]
                    - static["irrelevant_exposure_base"]
                ),
                "dynamic_irrelevant_exposure_delta": float(
                    dynamic["qrsbt_irrelevant_exposure"]
                    - dynamic["base_irrelevant_exposure"]
                ),
                "dynamic_mean_utility_delta": float(
                    dynamic["mean_scenario_replication_utility_delta"]
                ),
                "dynamic_worst_utility_delta": float(
                    dynamic["worst_scenario_utility_delta"]
                ),
                "dynamic_p10_utility_delta": float(
                    dynamic["p10_scenario_replication_utility_delta"]
                ),
                "dynamic_relevant_discovery_delta": float(
                    dynamic["qrsbt_relevant_discovery"]
                    - dynamic["base_relevant_discovery"]
                ),
            }
        )

    result = pd.DataFrame(results)
    result["static_cold_lift_pass"] = result["cold_ndcg_lift"].ge(
        float(release["min_cold_ndcg_lift"])
    )
    result["dynamic_irrelevant_exposure_pass"] = result[
        "dynamic_irrelevant_exposure_delta"
    ].le(float(release["max_dynamic_irrelevant_exposure_increase"]))
    result["dynamic_p10_pass"] = result["dynamic_p10_utility_delta"].ge(
        float(release["min_p10_scenario_replication_utility_delta"])
    )
    result["dynamic_worst_pass"] = result["dynamic_worst_utility_delta"].ge(
        float(release["min_worst_scenario_utility_delta"])
    )
    result["screen_pass"] = (
        result["static_cold_lift_pass"]
        & result["dynamic_irrelevant_exposure_pass"]
        & result["dynamic_p10_pass"]
        & result["dynamic_worst_pass"]
    )
    result = result.sort_values(
        [
            "screen_pass",
            "dynamic_irrelevant_exposure_delta",
            "cold_ndcg_lift",
            "dynamic_p10_utility_delta",
        ],
        ascending=[False, True, False, False],
        kind="mergesort",
    )
    result.to_csv(output_root / "ablation_screen_qualification.csv", index=False)
    pd.DataFrame(active_rows).sort_values(
        ["variant", "query_id", "product_id"],
        kind="mergesort",
    ).to_csv(output_root / "variant_active_promotions.csv", index=False)

    receipt = {
        "status": "PASS",
        **receipt_base,
        "variant_count": len(variants),
        "note": (
            "Hosted read-only counterfactual screen. No policy implementation "
            "or release qualification claim follows from this result alone."
        ),
    }
    (output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
