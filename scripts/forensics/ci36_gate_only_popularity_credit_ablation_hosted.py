from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
    ".github/workflows/ci36-gate-only-popularity-credit-ablation.yml",
    "scripts/forensics/ci36_gate_only_popularity_credit_ablation_hosted.py",
}
DEFAULT_CREDITS = (0.0, 0.25, 0.50, 0.75, 1.0)
TOL = 1e-12


@dataclass(frozen=True)
class QueryArrays:
    state_indices: np.ndarray
    product_ids: np.ndarray
    base_score: np.ndarray
    final_score: np.ndarray
    relevance: np.ndarray
    quality: np.ndarray
    zero_history: np.ndarray


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        fail(
            "Git command failed: "
            + " ".join(args)
            + f"; stdout={completed.stdout.strip()!r}; stderr={completed.stderr.strip()!r}"
        )
    return completed.stdout.strip()


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        fail(f"Missing canonical-artifact file: {path}")
    return path


def verify_branch(repo_root: Path, artifact_commit: str) -> list[str]:
    if artifact_commit != EXPECTED_ARTIFACT_COMMIT:
        fail(
            f"Unexpected artifact commit. expected={EXPECTED_ARTIFACT_COMMIT} actual={artifact_commit}"
        )
    git(repo_root, "merge-base", "--is-ancestor", artifact_commit, "HEAD")
    changed = [
        item.strip()
        for item in git(repo_root, "diff", "--name-only", f"{artifact_commit}..HEAD").splitlines()
        if item.strip()
    ]
    unexpected = sorted(set(changed) - ALLOWED_FORENSIC_PATHS)
    missing = sorted(ALLOWED_FORENSIC_PATHS - set(changed))
    if unexpected:
        fail(f"Unexpected source changes on CI36 branch: {unexpected}")
    if missing:
        fail(f"Missing CI36 forensic files: {missing}")
    return changed


def parse_credits(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for text in raw.split(","):
        item = text.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            fail(f"Invalid gate-only credit: {value}")
        values.append(value)
    if not values or len(set(values)) != len(values):
        fail(f"Credits must be nonempty and unique: {values}")
    if 1.0 not in values:
        fail("Credits must include 1.0 for the exact no-intervention replay.")
    return tuple(sorted(values))


def recursive_seed_candidates(value: Any, source: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() == "seed" and isinstance(nested, (int, float)) and int(nested) == nested:
                results.append({"source": source, "value": int(nested)})
            results.extend(recursive_seed_candidates(nested, source))
    elif isinstance(value, list):
        for nested in value:
            results.extend(recursive_seed_candidates(nested, source))
    return results


def resolve_seed(artifact_root: Path, manifest: dict[str, Any], explicit: int | None) -> tuple[int, list[dict[str, Any]]]:
    candidates = recursive_seed_candidates(manifest, "manifest.json")
    command_seed = re.compile(r"(?:^|\s)--seed\s+(\d+)(?:\s|$)")
    for path in sorted(artifact_root.rglob("*.json")):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.extend(
            recursive_seed_candidates(parsed, str(path.relative_to(artifact_root)).replace("\\", "/"))
        )
        for match in command_seed.finditer(path.read_text(encoding="utf-8")):
            candidates.append(
                {
                    "source": str(path.relative_to(artifact_root)).replace("\\", "/"),
                    "value": int(match.group(1)),
                }
            )
    for path in sorted(artifact_root.rglob("*.y*ml")):
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.extend(
            recursive_seed_candidates(parsed, str(path.relative_to(artifact_root)).replace("\\", "/"))
        )

    if explicit is not None:
        if explicit < 0:
            fail("--seed must be nonnegative")
        return explicit, candidates
    values = sorted({entry["value"] for entry in candidates})
    if len(values) != 1:
        fail(
            "Unable to derive one unique dynamic seed. "
            f"Candidates={candidates}. Re-run with explicit --seed."
        )
    return values[0], candidates


def top_indices(score: np.ndarray, product_ids: np.ndarray, k: int = 10) -> np.ndarray:
    count = min(k, len(score))
    if count == 0:
        return np.empty(0, dtype=int)
    return np.lexsort((product_ids.astype(int), -score))[:count]


def build_arrays(ranked: pd.DataFrame) -> tuple[np.ndarray, dict[int, QueryArrays]]:
    required = {
        "query_id",
        "product_id",
        "base_score",
        "final_score",
        "relevance",
        "quality",
        "zero_history",
    }
    missing = sorted(required - set(ranked.columns))
    if missing:
        fail(f"ranked_test.csv missing columns: {missing}")
    if ranked.duplicated(["query_id", "product_id"]).any():
        fail("ranked_test.csv contains duplicate query-product rows")

    ranked = ranked.sort_values(["query_id", "product_id"], kind="mergesort").reset_index(drop=True)
    products = np.sort(ranked.product_id.astype(int).unique())
    position = {int(product_id): index for index, product_id in enumerate(products)}
    arrays: dict[int, QueryArrays] = {}
    for query_id, group in ranked.groupby("query_id", sort=True):
        product_ids = group.product_id.to_numpy(dtype=int)
        arrays[int(query_id)] = QueryArrays(
            state_indices=np.asarray([position[int(item)] for item in product_ids], dtype=int),
            product_ids=product_ids,
            base_score=group.base_score.to_numpy(dtype=float),
            final_score=group.final_score.to_numpy(dtype=float),
            relevance=group.relevance.to_numpy(dtype=float),
            quality=group.quality.to_numpy(dtype=float),
            zero_history=group.zero_history.to_numpy(dtype=int),
        )
    return products, arrays


def click_outcomes(query: QueryArrays, top: np.ndarray, scenario: Any, uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relevance = query.relevance[top]
    quality = query.quality[top]
    cold = query.zero_history[top]
    positions = np.arange(1, len(top) + 1, dtype=float)
    exam = 1.0 / np.log2(positions + 1.5)
    logits = (
        scenario.intercept
        + scenario.relevance_weight * relevance
        + scenario.quality_weight * quality
        + np.log(exam)
        + scenario.novelty_bonus * cold
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    clicked = uniforms[: len(top)] < probability
    return clicked, relevance, cold


def add_metrics(
    *,
    relevance: np.ndarray,
    cold: np.ndarray,
    clicked: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[int, int, int, int, int]:
    relevant_discovery = int(((cold == 1) & (relevance >= 2) & clicked).sum())
    irrelevant_exposure = int((relevance == 0).sum())
    false_warmup = int(
        ((cold == 1) & (relevance == 0) & (before < 3) & (after >= 3)).sum()
    )
    cold_to_warm = int(
        ((cold == 1) & (relevance >= 2) & (before < 3) & (after >= 3)).sum()
    )
    return relevant_discovery, irrelevant_exposure, false_warmup, cold_to_warm, int(clicked.sum())


def daily_row(
    *,
    replication: int,
    scenario: str,
    day: int,
    policy: str,
    relevant: int,
    irrelevant: int,
    false_warmup: int,
    cold_to_warm: int,
    clicks: int,
    gate_only_entries: int = 0,
    gate_only_clicked: int = 0,
    gate_only_credit: float = 0.0,
    total_credit: float = 0.0,
) -> dict[str, Any]:
    return {
        "replication": replication,
        "scenario": scenario,
        "day": day + 1,
        "policy": policy,
        "relevant_discovery": relevant,
        "irrelevant_exposure": irrelevant,
        "false_warmup": false_warmup,
        "cold_to_warm": cold_to_warm,
        "clicks": clicks,
        "utility": relevant - 0.2 * irrelevant - 2.0 * false_warmup,
        "gate_only_top10_entries": gate_only_entries,
        "gate_only_clicked_entries": gate_only_clicked,
        "gate_only_popularity_credit": gate_only_credit,
        "total_popularity_credit": total_credit,
    }


def run_variant(
    *,
    ranked: pd.DataFrame,
    days: int,
    traffic_per_day: int,
    seed: int,
    replications: int,
    credit: float,
) -> pd.DataFrame:
    from product_search.simulation.dynamic import DEFAULT_SCENARIOS

    product_ids, arrays = build_arrays(ranked)
    query_ids = np.asarray(sorted(arrays), dtype=int)
    rows: list[dict[str, Any]] = []

    for replication in range(replications):
        rng = np.random.default_rng(seed + 1009 * replication)
        traffic = rng.choice(
            query_ids,
            size=(len(DEFAULT_SCENARIOS), days, traffic_per_day),
            replace=True,
        )
        uniforms = rng.random((len(DEFAULT_SCENARIOS), days, traffic_per_day, 10))

        for scenario_index, scenario in enumerate(DEFAULT_SCENARIOS):
            base_state = np.zeros(len(product_ids), dtype=float)
            gate_state = np.zeros(len(product_ids), dtype=float)
            for day in range(days):
                base_sets: list[set[int]] = []
                base_totals = [0, 0, 0, 0, 0]
                base_total_credit = 0.0
                for event_index, query_id in enumerate(traffic[scenario_index, day]):
                    query = arrays[int(query_id)]
                    score = query.base_score + scenario.popularity_weight * np.log1p(
                        base_state[query.state_indices]
                    )
                    top = top_indices(score, query.product_ids)
                    if top.size == 0:
                        base_sets.append(set())
                        continue
                    state_indices = query.state_indices[top]
                    base_sets.append({int(item) for item in query.product_ids[top]})
                    clicked, relevance, cold = click_outcomes(
                        query,
                        top,
                        scenario,
                        uniforms[scenario_index, day, event_index],
                    )
                    before = base_state[state_indices].copy()
                    increments = clicked.astype(float)
                    base_state[state_indices] += increments
                    after = base_state[state_indices]
                    metrics = add_metrics(
                        relevance=relevance,
                        cold=cold,
                        clicked=clicked,
                        before=before,
                        after=after,
                    )
                    base_totals = [left + right for left, right in zip(base_totals, metrics)]
                    base_total_credit += float(increments.sum())

                rows.append(
                    daily_row(
                        replication=replication,
                        scenario=scenario.name,
                        day=day,
                        policy="base",
                        relevant=base_totals[0],
                        irrelevant=base_totals[1],
                        false_warmup=base_totals[2],
                        cold_to_warm=base_totals[3],
                        clicks=base_totals[4],
                        total_credit=base_total_credit,
                    )
                )

                gate_totals = [0, 0, 0, 0, 0]
                gate_only_entries = 0
                gate_only_clicked = 0
                gate_only_credit = 0.0
                gate_total_credit = 0.0
                for event_index, query_id in enumerate(traffic[scenario_index, day]):
                    query = arrays[int(query_id)]
                    score = query.final_score + scenario.popularity_weight * np.log1p(
                        gate_state[query.state_indices]
                    )
                    top = top_indices(score, query.product_ids)
                    if top.size == 0:
                        continue
                    state_indices = query.state_indices[top]
                    product_view = query.product_ids[top]
                    base_products = base_sets[event_index]
                    is_gate_only = np.asarray(
                        [int(item) not in base_products for item in product_view],
                        dtype=bool,
                    )
                    clicked, relevance, cold = click_outcomes(
                        query,
                        top,
                        scenario,
                        uniforms[scenario_index, day, event_index],
                    )
                    before = gate_state[state_indices].copy()
                    increments = clicked.astype(float)
                    increments[is_gate_only] *= credit
                    gate_state[state_indices] += increments
                    after = gate_state[state_indices]
                    metrics = add_metrics(
                        relevance=relevance,
                        cold=cold,
                        clicked=clicked,
                        before=before,
                        after=after,
                    )
                    gate_totals = [left + right for left, right in zip(gate_totals, metrics)]
                    gate_only_entries += int(is_gate_only.sum())
                    gate_only_clicked += int((is_gate_only & clicked).sum())
                    gate_only_credit += float(increments[is_gate_only].sum())
                    gate_total_credit += float(increments.sum())

                rows.append(
                    daily_row(
                        replication=replication,
                        scenario=scenario.name,
                        day=day,
                        policy="qrsbt_gate",
                        relevant=gate_totals[0],
                        irrelevant=gate_totals[1],
                        false_warmup=gate_totals[2],
                        cold_to_warm=gate_totals[3],
                        clicks=gate_totals[4],
                        gate_only_entries=gate_only_entries,
                        gate_only_clicked=gate_only_clicked,
                        gate_only_credit=gate_only_credit,
                        total_credit=gate_total_credit,
                    )
                )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    aggregates = daily.groupby(["replication", "policy"], as_index=False).sum(numeric_only=True)
    means = aggregates.groupby("policy").mean(numeric_only=True)
    scenario = daily.groupby(["replication", "scenario", "policy"], as_index=False).sum(numeric_only=True)
    deltas: list[dict[str, Any]] = []
    for (replication, scenario_name), group in scenario.groupby(["replication", "scenario"], sort=False):
        view = group.set_index("policy")
        deltas.append(
            {
                "replication": int(replication),
                "scenario": str(scenario_name),
                "relevant_delta": float(
                    view.loc["qrsbt_gate", "relevant_discovery"] - view.loc["base", "relevant_discovery"]
                ),
                "irrelevant_delta": float(
                    view.loc["qrsbt_gate", "irrelevant_exposure"] - view.loc["base", "irrelevant_exposure"]
                ),
                "false_warmup_delta": float(
                    view.loc["qrsbt_gate", "false_warmup"] - view.loc["base", "false_warmup"]
                ),
                "cold_to_warm_delta": float(
                    view.loc["qrsbt_gate", "cold_to_warm"] - view.loc["base", "cold_to_warm"]
                ),
                "clicks_delta": float(view.loc["qrsbt_gate", "clicks"] - view.loc["base", "clicks"]),
                "utility_delta": float(view.loc["qrsbt_gate", "utility"] - view.loc["base", "utility"]),
            }
        )
    delta_frame = pd.DataFrame(deltas)
    scenario_means = delta_frame.groupby("scenario").mean(numeric_only=True)
    return {
        "base_relevant_discovery": float(means.loc["base", "relevant_discovery"]),
        "qrsbt_relevant_discovery": float(means.loc["qrsbt_gate", "relevant_discovery"]),
        "base_irrelevant_exposure": float(means.loc["base", "irrelevant_exposure"]),
        "qrsbt_irrelevant_exposure": float(means.loc["qrsbt_gate", "irrelevant_exposure"]),
        "base_false_warmup": float(means.loc["base", "false_warmup"]),
        "qrsbt_false_warmup": float(means.loc["qrsbt_gate", "false_warmup"]),
        "base_cold_to_warm": float(means.loc["base", "cold_to_warm"]),
        "qrsbt_cold_to_warm": float(means.loc["qrsbt_gate", "cold_to_warm"]),
        "base_utility": float(means.loc["base", "utility"]),
        "qrsbt_utility": float(means.loc["qrsbt_gate", "utility"]),
        "worst_scenario_relevant_delta": float(scenario_means.relevant_delta.min()),
        "worst_scenario_irrelevant_delta": float(scenario_means.irrelevant_delta.max()),
        "worst_scenario_utility_delta": float(scenario_means.utility_delta.min()),
        "p10_scenario_replication_utility_delta": float(delta_frame.utility_delta.quantile(0.10)),
        "mean_scenario_replication_utility_delta": float(delta_frame.utility_delta.mean()),
        "scenario_count": float(delta_frame.scenario.nunique()),
        "replications": float(delta_frame.replication.nunique()),
        "dynamic_relevant_discovery_delta": float(delta_frame.relevant_delta.mean()),
        "dynamic_irrelevant_exposure_delta": float(delta_frame.irrelevant_delta.mean()),
        "dynamic_false_warmup_delta": float(delta_frame.false_warmup_delta.mean()),
        "dynamic_cold_to_warm_delta": float(delta_frame.cold_to_warm_delta.mean()),
        "dynamic_clicks_delta": float(delta_frame.clicks_delta.mean()),
        "dynamic_mean_utility_delta": float(delta_frame.utility_delta.mean()),
    }


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    keys = ["replication", "scenario", "day", "policy"]
    left = left[columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = right[columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(left) != len(right):
        return {"exact": False, "reason": "row_count", "left_rows": len(left), "right_rows": len(right)}
    differences: dict[str, Any] = {}
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            error = np.abs(left[column].to_numpy(dtype=float) - right[column].to_numpy(dtype=float))
            count = int((error > TOL).sum())
            if count:
                differences[column] = {"count": count, "max_abs_difference": float(error.max())}
        else:
            count = int((left[column].astype(str) != right[column].astype(str)).sum())
            if count:
                differences[column] = {"count": count}
    return {"exact": not differences, "mismatches": differences, "rows": int(len(left))}


def compare_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if set(left) != set(right):
        return {
            "exact": False,
            "reason": "keys",
            "left_only": sorted(set(left) - set(right)),
            "right_only": sorted(set(right) - set(left)),
        }
    differences: dict[str, Any] = {}
    for key in sorted(left):
        if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float)):
            error = abs(float(left[key]) - float(right[key]))
            if error > TOL:
                differences[key] = {"artifact": left[key], "replayed": right[key], "abs_difference": error}
        elif left[key] != right[key]:
            differences[key] = {"artifact": left[key], "replayed": right[key]}
    return {"exact": not differences, "mismatches": differences}


def screen(summary: dict[str, float], static_ok: bool) -> dict[str, bool]:
    return {
        "static_contract_preserved": static_ok,
        "relevant_discovery_nonnegative": summary["dynamic_relevant_discovery_delta"] >= -TOL,
        "irrelevant_exposure_nonpositive": summary["dynamic_irrelevant_exposure_delta"] <= TOL,
        "false_warmup_nonpositive": summary["dynamic_false_warmup_delta"] <= TOL,
        "worst_utility_nonnegative": summary["worst_scenario_utility_delta"] >= -TOL,
        "p10_utility_nonnegative": summary["p10_scenario_replication_utility_delta"] >= -TOL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--credits", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    output_root = Path(args.output_root).resolve()
    credits = parse_credits(args.credits)
    if output_root.exists():
        fail(f"Output root already exists: {output_root}")

    manifest_path = require_file(artifact_root, "manifest.json")
    ranked_path = require_file(artifact_root, "ranked_test.csv")
    daily_path = require_file(artifact_root, "dynamic_daily.csv")
    summary_path = require_file(artifact_root, "dynamic_summary.json")
    metadata_path = require_file(artifact_root, "dynamic_stage_metadata.json")
    release_path = require_file(artifact_root, "release_decision.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_commit = str(manifest.get("git_commit", ""))
    changed_paths = verify_branch(repo_root, artifact_commit)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("stage_status") != "complete":
        fail("Canonical dynamic stage is not complete")
    days = int(metadata["days"])
    traffic_per_day = int(metadata["traffic_per_day"])
    replications = int(metadata["replications"])
    if min(days, traffic_per_day, replications) < 1:
        fail(f"Invalid dynamic parameters: {metadata}")

    seed, seed_candidates = resolve_seed(artifact_root, manifest, args.seed)
    ranked = pd.read_csv(ranked_path)
    artifact_daily = pd.read_csv(daily_path)
    artifact_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))

    sys.path.insert(0, str(repo_root / "src"))
    from product_search.simulation.dynamic import run_dynamic_simulation

    reference = run_dynamic_simulation(
        ranked,
        days=days,
        traffic_per_day=traffic_per_day,
        seed=seed,
        replications=replications,
    )
    standard_columns = [
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
    baseline_daily = compare_frames(artifact_daily, reference.daily, standard_columns)
    baseline_summary = compare_summary(artifact_summary, reference.summary)
    baseline_exact = bool(baseline_daily["exact"] and baseline_summary["exact"])
    if not baseline_exact:
        output_root.mkdir(parents=True, exist_ok=False)
        write_json(
            output_root / "baseline_replay_failure.json",
            {
                "status": "FAIL_BASELINE_REPLAY",
                "artifact_commit": artifact_commit,
                "seed": seed,
                "parameters": {
                    "days": days,
                    "traffic_per_day": traffic_per_day,
                    "replications": replications,
                },
                "daily": baseline_daily,
                "summary": baseline_summary,
            },
        )
        fail("Canonical dynamic replay did not match exactly; credit variants were not inspected.")

    failed = [str(item) for item in release.get("failed_gates", [])]
    dynamic_gate_names = {
        "dynamic_irrelevant_exposure_guardrail",
        "dynamic_false_warmup_guardrail",
        "dynamic_relevant_discovery_guardrail",
        "dynamic_worst_scenario_utility",
        "dynamic_p10_replication_utility",
        "policy_sensitivity",
    }
    static_failed = sorted(set(failed) - dynamic_gate_names)
    static_ok = not static_failed
    static_cold_lift = release.get("diagnostics", {}).get("cold_ndcg_lift")

    output_root.mkdir(parents=True, exist_ok=False)
    outputs: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    credit_one_replay: dict[str, Any] | None = None
    for credit in credits:
        daily = run_variant(
            ranked=ranked,
            days=days,
            traffic_per_day=traffic_per_day,
            seed=seed,
            replications=replications,
            credit=credit,
        )
        if credit == 1.0:
            credit_one_replay = compare_frames(reference.daily, daily, standard_columns)
            if not credit_one_replay["exact"]:
                write_json(
                    output_root / "credit_one_replay_failure.json",
                    {
                        "status": "FAIL_CREDIT_ONE_REPLAY",
                        "artifact_commit": artifact_commit,
                        "credit_one_replay": credit_one_replay,
                    },
                )
                fail("Credit=1.0 did not exactly reproduce baseline dynamics.")
        metrics = summarize(daily)
        gate_rows = daily[daily.policy.eq("qrsbt_gate")]
        checks = screen(metrics, static_ok)
        rows.append(
            {
                "gate_only_credit": credit,
                "static_cold_ndcg_lift": static_cold_lift,
                **metrics,
                **checks,
                "credit_screen_pass": bool(all(checks.values())),
                "gate_only_top10_entries": int(gate_rows.gate_only_top10_entries.sum()),
                "gate_only_clicked_entries": int(gate_rows.gate_only_clicked_entries.sum()),
                "gate_only_popularity_credit": float(gate_rows.gate_only_popularity_credit.sum()),
                "total_popularity_credit": float(gate_rows.total_popularity_credit.sum()),
            }
        )
        daily.insert(0, "gate_only_credit", credit)
        outputs.append(daily)

    if credit_one_replay is None:
        fail("Credit=1.0 validation did not run")

    summary_frame = pd.DataFrame(rows).sort_values("gate_only_credit", kind="mergesort")
    daily_frame = pd.concat(outputs, ignore_index=True)
    qualifying = summary_frame.loc[
        summary_frame.credit_screen_pass.astype(bool),
        "gate_only_credit",
    ].astype(float).tolist()
    selected = max(qualifying) if qualifying else None
    decision = "CANDIDATE_FOUND_FOR_POLICY_SENSITIVITY" if selected is not None else "NO_CREDIT_CANDIDATE"

    write_csv(output_root / "credit_ablation_summary.csv", summary_frame)
    write_csv(output_root / "credit_ablation_daily.csv", daily_frame)
    write_json(
        output_root / "credit_ablation_summary.json",
        {
            "artifact_commit": artifact_commit,
            "credits": list(credits),
            "qualifying_credits": qualifying,
            "selected_gate_only_credit": selected,
            "decision": decision,
            "rows": rows,
        },
    )
    receipt = {
        "status": "PASS",
        "decision": decision,
        "artifact_commit": artifact_commit,
        "repo_head": git(repo_root, "rev-parse", "HEAD"),
        "changed_forensic_paths": changed_paths,
        "artifact_hashes": {
            "manifest.json": sha256(manifest_path),
            "ranked_test.csv": sha256(ranked_path),
            "dynamic_daily.csv": sha256(daily_path),
            "dynamic_summary.json": sha256(summary_path),
            "dynamic_stage_metadata.json": sha256(metadata_path),
            "release_decision.json": sha256(release_path),
        },
        "dynamic_parameters": {
            "days": days,
            "traffic_per_day": traffic_per_day,
            "replications": replications,
            "seed": seed,
            "seed_candidates": seed_candidates,
        },
        "baseline_artifact_replay": {
            "exact": baseline_exact,
            "daily": baseline_daily,
            "summary": baseline_summary,
        },
        "credit_one_replay": credit_one_replay,
        "static_contract": {
            "ranked_test_sha256": sha256(ranked_path),
            "cold_ndcg_lift": static_cold_lift,
            "original_failed_gates": failed,
            "static_failed_gates": static_failed,
            "static_contract_preserved": static_ok,
        },
        "credits": list(credits),
        "qualifying_credits": qualifying,
        "selected_gate_only_credit": selected,
        "decision_boundary": (
            "A qualifying credit requires a separate policy-sensitivity replay; "
            "CI36 neither changes source runtime nor makes a release decision."
        ),
    }
    write_json(output_root / "receipt.json", receipt)

    md = [
        "# CI36 gate-only popularity-credit ablation",
        "",
        f"- Artifact commit: `{artifact_commit}`",
        f"- Exact canonical dynamic replay: `{baseline_exact}`",
        f"- Exact custom credit=1.0 replay: `{credit_one_replay['exact']}`",
        f"- Decision: `{decision}`",
        "",
        "| credit | irrelevant delta | p10 utility delta | worst utility delta | relevant delta | false-warm delta | screen |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['gate_only_credit']:.2f} | "
            f"{row['dynamic_irrelevant_exposure_delta']:.6f} | "
            f"{row['p10_scenario_replication_utility_delta']:.6f} | "
            f"{row['worst_scenario_utility_delta']:.6f} | "
            f"{row['dynamic_relevant_discovery_delta']:.6f} | "
            f"{row['dynamic_false_warmup_delta']:.6f} | "
            f"{'PASS' if row['credit_screen_pass'] else 'FAIL'} |"
        )
    md.extend(
        [
            "",
            "PASS means only that the configuration should enter a separate policy-sensitivity screen.",
            "This forensic does not modify static promotion, runtime source, release thresholds, or canonical artifacts.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "output_root": str(output_root),
                "artifact_commit": artifact_commit,
                "qualifying_credits": qualifying,
                "selected_gate_only_credit": selected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
