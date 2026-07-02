"""CI37 hosted row-level promotion-ablation forensic screen.

This read-only forensic evaluates every subset of the canonical artifact's
positive Q-RSBT promotion rows. It preserves the artifact's serialized
final_score for all active rows and reverts only disabled promotion rows to
base_score. Static screening is exhaustive; dynamic simulation is run only
for static-qualified candidates.

The script never modifies runtime source, release thresholds, configuration, or
the downloaded canonical artifact. It writes forensic outputs only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXPECTED_ARTIFACT_COMMIT = "1f3e54d78060eb9d745c9480c88c22ba19d1ced2"
ALLOWED_FORENSIC_PATHS = {
    ".github/workflows/ci37-row-level-promotion-ablation.yml",
    "scripts/forensics/ci37_row_level_promotion_ablation_hosted.py",
}
EXPECTED_PROMOTION_ROW_COUNT = 5
BASELINE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PromotionUnit:
    unit_id: str
    query_id: int
    product_id: int
    row_index: int
    base_rank: int
    qrsbt_boost: float
    relevance: float


def fail(message: str) -> None:
    raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "Git command failed: "
            + " ".join(args)
            + f"; stdout={completed.stdout.strip()!r}; "
            + f"stderr={completed.stderr.strip()!r}"
        )
    return completed.stdout.strip()


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        fail(f"Missing canonical-artifact file: {path}")
    return path


def verify_forensic_branch(repo_root: Path, artifact_commit: str) -> list[str]:
    if artifact_commit != EXPECTED_ARTIFACT_COMMIT:
        fail(
            "Unexpected artifact commit. "
            f"expected={EXPECTED_ARTIFACT_COMMIT}; actual={artifact_commit}"
        )

    ancestry = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", artifact_commit, "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestry.returncode != 0:
        fail(
            "Artifact commit is not an ancestor of forensic branch HEAD. "
            f"artifact={artifact_commit}; stderr={ancestry.stderr.strip()!r}"
        )

    changed = [
        item.strip()
        for item in git(
            repo_root,
            "diff",
            "--name-only",
            f"{artifact_commit}..HEAD",
        ).splitlines()
        if item.strip()
    ]
    unexpected = sorted(set(changed) - ALLOWED_FORENSIC_PATHS)
    missing = sorted(ALLOWED_FORENSIC_PATHS - set(changed))
    if unexpected:
        fail(f"Unexpected source changes on CI37 branch: {unexpected}")
    if missing:
        fail(f"Missing CI37 forensic files: {missing}")
    return changed


def validate_ranked_frame(ranked: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id",
        "product_id",
        "base_rank",
        "base_score",
        "final_score",
        "qrsbt_boost",
        "relevance",
        "zero_history",
        "quality",
        "gate_action",
    }
    missing = sorted(required - set(ranked.columns))
    if missing:
        fail(f"ranked_test.csv missing required columns: {missing}")

    ranked = ranked.copy()
    for column in (
        "query_id",
        "product_id",
        "base_rank",
        "zero_history",
    ):
        ranked[column] = pd.to_numeric(ranked[column], errors="raise").astype(int)
    for column in (
        "base_score",
        "final_score",
        "qrsbt_boost",
        "relevance",
        "quality",
    ):
        ranked[column] = pd.to_numeric(ranked[column], errors="raise").astype(float)

    if ranked.duplicated(["query_id", "product_id"]).any():
        fail("ranked_test.csv contains duplicate query-product rows.")
    return ranked.sort_values(["query_id", "product_id"], kind="mergesort").reset_index(drop=True)


def promotion_units(ranked: pd.DataFrame) -> list[PromotionUnit]:
    positive = ranked.loc[ranked["qrsbt_boost"].gt(0.0)].copy()
    if len(positive) != EXPECTED_PROMOTION_ROW_COUNT:
        fail(
            "Expected exactly "
            f"{EXPECTED_PROMOTION_ROW_COUNT} positive promotion rows; found {len(positive)}."
        )

    units: list[PromotionUnit] = []
    for row in positive.itertuples():
        unit_id = f"q{int(row.query_id)}_p{int(row.product_id)}"
        units.append(
            PromotionUnit(
                unit_id=unit_id,
                query_id=int(row.query_id),
                product_id=int(row.product_id),
                row_index=int(row.Index),
                base_rank=int(row.base_rank),
                qrsbt_boost=float(row.qrsbt_boost),
                relevance=float(row.relevance),
            )
        )

    if len({unit.unit_id for unit in units}) != len(units):
        fail("Promotion unit IDs are not unique.")
    return sorted(units, key=lambda unit: (unit.query_id, unit.product_id))


def candidate_from_disabled(
    ranked: pd.DataFrame,
    units: list[PromotionUnit],
    disabled_ids: set[str],
) -> pd.DataFrame:
    candidate = ranked.copy()
    for unit in units:
        if unit.unit_id not in disabled_ids:
            continue
        candidate.loc[unit.row_index, "qrsbt_boost"] = 0.0
        candidate.loc[unit.row_index, "final_score"] = candidate.loc[
            unit.row_index,
            "base_score",
        ]
    return candidate


def static_metrics(
    report: dict[str, float],
    release: dict[str, Any],
) -> dict[str, Any]:
    overall_delta = float(report["final_ndcg_at_10"] - report["base_ndcg_at_10"])
    cold_delta = float(
        report["cold_ndcg_at_10_final"] - report["cold_ndcg_at_10_base"]
    )
    warm_delta = float(
        report["warm_ndcg_at_10_final"] - report["warm_ndcg_at_10_base"]
    )
    irrelevant_delta = float(
        report["irrelevant_exposure_final"] - report["irrelevant_exposure_base"]
    )
    coverage = min(
        float(report["judgment_coverage_base"]),
        float(report["judgment_coverage_final"]),
    )

    checks = {
        "static_overall_non_inferiority": overall_delta
        >= -float(release["max_overall_ndcg_drop"]),
        "static_overall_ci_non_inferiority": float(report["overall_ndcg_delta_ci_low"])
        >= -float(release["max_overall_ndcg_drop_ci"]),
        "static_cold_point_improvement": cold_delta
        >= float(release["min_cold_ndcg_lift"]),
        "static_cold_ci_non_inferiority": float(report["cold_ndcg_lift_ci_low"])
        >= float(release["min_cold_ndcg_lift_ci_low"]),
        "static_warm_non_inferiority": warm_delta
        >= -float(release["max_warm_ndcg_drop"]),
        "static_warm_ci_non_inferiority": float(report["warm_ndcg_delta_ci_low"])
        >= -float(release["max_warm_ndcg_drop_ci"]),
        "static_irrelevant_guardrail": irrelevant_delta
        <= float(release["max_irrelevant_exposure_increase"]),
        "static_irrelevant_ci_guardrail": float(
            report["irrelevant_exposure_delta_ci_high"]
        )
        <= float(release["max_irrelevant_exposure_ci_high"]),
        "static_judgment_coverage": coverage
        >= float(release["min_judgment_coverage_at_10"]),
    }

    return {
        "overall_ndcg_delta": overall_delta,
        "cold_ndcg_lift": cold_delta,
        "cold_ndcg_ci_low": float(report["cold_ndcg_lift_ci_low"]),
        "cold_ndcg_ci_high": float(report["cold_ndcg_lift_ci_high"]),
        "warm_ndcg_delta": warm_delta,
        "warm_ndcg_ci_low": float(report["warm_ndcg_delta_ci_low"]),
        "warm_ndcg_ci_high": float(report["warm_ndcg_delta_ci_high"]),
        "irrelevant_exposure_delta": irrelevant_delta,
        "irrelevant_exposure_ci_high": float(
            report["irrelevant_exposure_delta_ci_high"]
        ),
        "judgment_coverage_at_10": coverage,
        **checks,
        "static_screen_pass": bool(all(checks.values())),
    }


def dynamic_metrics(
    summary: dict[str, Any],
    release: dict[str, Any],
) -> dict[str, Any]:
    base_relevant = float(summary["base_relevant_discovery"])
    gate_relevant = float(summary["qrsbt_relevant_discovery"])
    relevant_drop_rate = max(
        0.0,
        (base_relevant - gate_relevant) / max(base_relevant, 1.0),
    )
    false_warmup_delta = float(
        summary["qrsbt_false_warmup"] - summary["base_false_warmup"]
    )
    irrelevant_delta = float(
        summary["qrsbt_irrelevant_exposure"]
        - summary["base_irrelevant_exposure"]
    )
    checks = {
        "dynamic_relevant_discovery_non_inferiority": relevant_drop_rate
        <= float(release["max_dynamic_relevant_discovery_drop_rate"]),
        "dynamic_false_warmup_guardrail": false_warmup_delta
        <= float(release["max_dynamic_false_warmup_increase"]),
        "dynamic_irrelevant_exposure_guardrail": irrelevant_delta
        <= float(release["max_dynamic_irrelevant_exposure_increase"]),
        "dynamic_worst_utility_guardrail": float(
            summary["worst_scenario_utility_delta"]
        )
        >= float(release["min_worst_scenario_utility_delta"]),
        "dynamic_p10_utility_guardrail": float(
            summary["p10_scenario_replication_utility_delta"]
        )
        >= float(release["min_p10_scenario_replication_utility_delta"]),
    }
    return {
        "dynamic_relevant_discovery_delta": gate_relevant - base_relevant,
        "dynamic_relevant_discovery_drop_rate": relevant_drop_rate,
        "dynamic_false_warmup_delta": false_warmup_delta,
        "dynamic_irrelevant_exposure_delta": irrelevant_delta,
        "dynamic_mean_utility_delta": float(
            summary["mean_scenario_replication_utility_delta"]
        ),
        "dynamic_worst_utility_delta": float(
            summary["worst_scenario_utility_delta"]
        ),
        "dynamic_p10_utility_delta": float(
            summary["p10_scenario_replication_utility_delta"]
        ),
        **checks,
        "dynamic_screen_pass": bool(all(checks.values())),
    }


def compare_dynamic_daily(
    artifact_daily: pd.DataFrame,
    replay_daily: pd.DataFrame,
) -> dict[str, Any]:
    columns = [
        "replication",
        "scenario",
        "day",
        "policy",
        "relevant_discovery",
        "irrelevant_exposure",
        "false_warmup",
        "cold_to_warm",
        "clicks",
        "utility",
    ]
    keys = ["replication", "scenario", "day", "policy"]
    missing_artifact = sorted(set(columns) - set(artifact_daily.columns))
    missing_replay = sorted(set(columns) - set(replay_daily.columns))
    if missing_artifact or missing_replay:
        return {
            "exact": False,
            "reason": "missing_columns",
            "artifact_missing": missing_artifact,
            "replay_missing": missing_replay,
        }

    left = artifact_daily.loc[:, columns].sort_values(
        keys,
        kind="mergesort",
    ).reset_index(drop=True)
    right = replay_daily.loc[:, columns].sort_values(
        keys,
        kind="mergesort",
    ).reset_index(drop=True)

    if len(left) != len(right):
        return {
            "exact": False,
            "reason": "row_count",
            "artifact_rows": int(len(left)),
            "replay_rows": int(len(right)),
        }

    mismatches: dict[str, Any] = {}
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            delta = np.abs(
                left[column].to_numpy(dtype=float)
                - right[column].to_numpy(dtype=float)
            )
            count = int((delta > BASELINE_TOLERANCE).sum())
            if count:
                mismatches[column] = {
                    "count": count,
                    "max_abs_difference": float(delta.max()),
                }
        else:
            count = int((left[column].astype(str) != right[column].astype(str)).sum())
            if count:
                mismatches[column] = {"count": count}

    return {
        "exact": not mismatches,
        "rows": int(len(left)),
        "mismatches": mismatches,
    }


def compare_dynamic_summary(
    artifact_summary: dict[str, Any],
    replay_summary: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = (
        "base_cold_to_warm",
        "base_false_warmup",
        "base_irrelevant_exposure",
        "base_relevant_discovery",
        "base_utility",
        "mean_scenario_replication_utility_delta",
        "p10_scenario_replication_utility_delta",
        "qrsbt_cold_to_warm",
        "qrsbt_false_warmup",
        "qrsbt_irrelevant_exposure",
        "qrsbt_relevant_discovery",
        "qrsbt_utility",
        "replications",
        "scenario_count",
        "worst_scenario_irrelevant_delta",
        "worst_scenario_relevant_delta",
        "worst_scenario_utility_delta",
    )
    missing = sorted(
        {
            key
            for key in expected_keys
            if key not in artifact_summary or key not in replay_summary
        }
    )
    if missing:
        return {
            "exact": False,
            "reason": "missing_keys",
            "missing": missing,
        }

    mismatches: dict[str, Any] = {}
    for key in expected_keys:
        artifact_value = float(artifact_summary[key])
        replay_value = float(replay_summary[key])
        delta = abs(artifact_value - replay_value)
        if delta > BASELINE_TOLERANCE:
            mismatches[key] = {
                "artifact": artifact_value,
                "replayed": replay_value,
                "abs_difference": delta,
            }
    return {"exact": not mismatches, "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root.exists():
        fail(f"Output root already exists: {output_root}")

    manifest_path = require_file(artifact_root, "manifest.json")
    ranked_path = require_file(artifact_root, "ranked_test.csv")
    daily_path = require_file(artifact_root, "dynamic_daily.csv")
    dynamic_summary_path = require_file(artifact_root, "dynamic_summary.json")
    metadata_path = require_file(artifact_root, "dynamic_stage_metadata.json")
    release_path = require_file(artifact_root, "release_decision.json")
    config_path = repo_root / "configs" / "smoke.yaml"
    if not config_path.is_file():
        fail(f"Missing repository smoke config: {config_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_commit = str(manifest.get("git_commit", ""))
    changed_forensic_paths = verify_forensic_branch(repo_root, artifact_commit)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    simulation = config["simulation"]
    release = config["release"]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("days", "traffic_per_day", "replications"):
        if int(metadata[key]) != int(simulation[key]):
            fail(
                "Artifact dynamic metadata conflicts with frozen smoke config "
                f"for {key}: artifact={metadata[key]} config={simulation[key]}"
            )

    ranked = validate_ranked_frame(pd.read_csv(ranked_path))
    units = promotion_units(ranked)
    artifact_daily = pd.read_csv(daily_path)
    artifact_summary = json.loads(dynamic_summary_path.read_text(encoding="utf-8"))

    sys.path.insert(0, str(repo_root / "src"))
    from product_search.evaluation.metrics import ranking_report
    from product_search.simulation.dynamic import run_dynamic_simulation

    baseline = run_dynamic_simulation(
        ranked,
        days=int(simulation["days"]),
        traffic_per_day=int(simulation["traffic_per_day"]),
        seed=seed,
        replications=int(simulation["replications"]),
    )
    baseline_daily = compare_dynamic_daily(artifact_daily, baseline.daily)
    baseline_summary = compare_dynamic_summary(artifact_summary, baseline.summary)

    output_root.mkdir(parents=True, exist_ok=False)

    baseline_receipt = {
        "artifact_commit": artifact_commit,
        "repo_head": git(repo_root, "rev-parse", "HEAD"),
        "baseline_artifact_replay": {
            "exact": bool(baseline_daily["exact"] and baseline_summary["exact"]),
            "daily": baseline_daily,
            "summary": baseline_summary,
        },
    }
    if not baseline_receipt["baseline_artifact_replay"]["exact"]:
        write_json(
            output_root / "receipt.json",
            {"status": "FAIL_BASELINE_REPLAY", **baseline_receipt},
        )
        raise SystemExit("Exact canonical dynamic replay failed; no row variants evaluated.")

    unit_rows = [
        {
            "unit_id": unit.unit_id,
            "query_id": unit.query_id,
            "product_id": unit.product_id,
            "base_rank": unit.base_rank,
            "qrsbt_boost": unit.qrsbt_boost,
            "relevance": unit.relevance,
        }
        for unit in units
    ]
    write_csv(output_root / "promotion_unit_catalog.csv", pd.DataFrame(unit_rows))

    static_rows: list[dict[str, Any]] = []
    candidate_cache: dict[str, pd.DataFrame] = {}
    all_unit_ids = [unit.unit_id for unit in units]

    for disabled_size in range(len(units) + 1):
        for disabled_tuple in itertools.combinations(all_unit_ids, disabled_size):
            disabled_ids = set(disabled_tuple)
            variant = (
                "current_policy"
                if not disabled_tuple
                else "disable_" + "_".join(disabled_tuple)
            )
            candidate = candidate_from_disabled(ranked, units, disabled_ids)
            report = ranking_report(
                candidate,
                bootstrap_samples=300,
                seed=seed,
            )
            metrics = static_metrics(report, release)
            static_rows.append(
                {
                    "variant": variant,
                    "disabled_unit_ids": "|".join(disabled_tuple),
                    "disabled_promotion_rows": len(disabled_tuple),
                    "active_promotion_rows": len(units) - len(disabled_tuple),
                    "active_total_boost": float(
                        candidate.loc[candidate["qrsbt_boost"].gt(0.0), "qrsbt_boost"].sum()
                    ),
                    **metrics,
                }
            )
            candidate_cache[variant] = candidate

    static_frame = pd.DataFrame(static_rows).sort_values(
        ["static_screen_pass", "cold_ndcg_lift", "active_promotion_rows"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    write_csv(output_root / "static_row_lattice.csv", static_frame)

    dynamic_rows: list[dict[str, Any]] = []
    static_candidates = static_frame.loc[
        static_frame["static_screen_pass"].astype(bool)
    ].copy()

    for row in static_candidates.itertuples(index=False):
        variant = str(row.variant)
        if variant == "current_policy":
            candidate_summary = baseline.summary
        else:
            candidate_summary = run_dynamic_simulation(
                candidate_cache[variant],
                days=int(simulation["days"]),
                traffic_per_day=int(simulation["traffic_per_day"]),
                seed=seed,
                replications=int(simulation["replications"]),
            ).summary
        metrics = dynamic_metrics(candidate_summary, release)
        static_dict = {
            key: getattr(row, key)
            for key in (
                "variant",
                "disabled_unit_ids",
                "disabled_promotion_rows",
                "active_promotion_rows",
                "active_total_boost",
                "cold_ndcg_lift",
                "cold_ndcg_ci_low",
                "warm_ndcg_delta",
                "irrelevant_exposure_delta",
                "static_screen_pass",
            )
        }
        dynamic_rows.append({**static_dict, **metrics})

    dynamic_frame = pd.DataFrame(dynamic_rows)
    if dynamic_frame.empty:
        fail("Static screen produced zero candidates, including current policy.")

    dynamic_frame["screen_pass"] = (
        dynamic_frame["static_screen_pass"].astype(bool)
        & dynamic_frame["dynamic_screen_pass"].astype(bool)
    )
    dynamic_frame = dynamic_frame.sort_values(
        [
            "screen_pass",
            "dynamic_irrelevant_exposure_delta",
            "cold_ndcg_lift",
            "dynamic_p10_utility_delta",
        ],
        ascending=[False, True, False, False],
        kind="mergesort",
    )
    write_csv(output_root / "dynamic_row_lattice.csv", dynamic_frame)

    qualifying = dynamic_frame.loc[dynamic_frame["screen_pass"].astype(bool)].copy()
    selected_variant = None
    if not qualifying.empty:
        selected_variant = str(qualifying.iloc[0]["variant"])

    receipt = {
        "status": "PASS",
        "decision": (
            "ROW_LEVEL_CANDIDATE_FOUND"
            if selected_variant is not None
            else "NO_ROW_LEVEL_CANDIDATE"
        ),
        "artifact_commit": artifact_commit,
        "repo_head": git(repo_root, "rev-parse", "HEAD"),
        "changed_forensic_paths": changed_forensic_paths,
        "artifact_hashes": {
            "manifest.json": sha256(manifest_path),
            "ranked_test.csv": sha256(ranked_path),
            "dynamic_daily.csv": sha256(daily_path),
            "dynamic_summary.json": sha256(dynamic_summary_path),
            "dynamic_stage_metadata.json": sha256(metadata_path),
            "release_decision.json": sha256(release_path),
        },
        "dynamic_parameters": {
            "days": int(simulation["days"]),
            "traffic_per_day": int(simulation["traffic_per_day"]),
            "replications": int(simulation["replications"]),
            "seed": seed,
        },
        **baseline_receipt,
        "promotion_row_count": len(units),
        "total_lattice_variants": int(2 ** len(units)),
        "static_screen_candidate_count": int(len(static_candidates)),
        "dynamic_screen_candidate_count": int(len(dynamic_frame)),
        "qualifying_variants": qualifying["variant"].astype(str).tolist(),
        "selected_variant": selected_variant,
        "decision_boundary": (
            "This is a read-only counterfactual screen. Any qualifying "
            "row-level pattern requires a general serving-safe policy "
            "implementation and hosted canonical CI before release claims."
        ),
    }
    write_json(output_root / "receipt.json", receipt)

    summary_lines = [
        "# CI37 row-level promotion-ablation",
        "",
        f"- Artifact commit: `{artifact_commit}`",
        f"- Exact canonical dynamic replay: `{receipt['baseline_artifact_replay']['exact']}`",
        f"- Positive promotion rows: `{len(units)}`",
        f"- Complete row lattice: `{2 ** len(units)}` variants",
        f"- Static-qualified variants: `{len(static_candidates)}`",
        f"- Dynamic-qualified variants: `{len(qualifying)}`",
        f"- Decision: `{receipt['decision']}`",
        "",
        "This is an evidence screen only. It does not change serving policy, runtime source, release thresholds, or the canonical artifact.",
        "",
    ]
    (output_root / "README.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": receipt["decision"],
                "output_root": str(output_root),
                "artifact_commit": artifact_commit,
                "promotion_row_count": len(units),
                "total_lattice_variants": int(2 ** len(units)),
                "static_screen_candidate_count": int(len(static_candidates)),
                "dynamic_screen_candidate_count": int(len(dynamic_frame)),
                "qualifying_variants": receipt["qualifying_variants"],
                "selected_variant": selected_variant,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
