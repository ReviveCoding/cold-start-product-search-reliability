
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold


v5e_decision_path = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
expected_head = sys.argv[3]

EXPECTED_ACTIONS = ("PLACE_AT_1", "PLACE_AT_2", "PLACE_AT_3", "PLACE_AT_5", "PLACE_AT_10")
NUMERIC_FEATURES = (
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_support_score",
    "base_rank",
)
CATEGORICAL_FEATURES = ("proposal_selection_rule",)
OUTER_SPLITS = 6
INNER_SPLITS = 5
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LOGISTIC_CS = (0.05, 0.2, 1.0, 5.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=native),
        encoding="utf-8",
    )
    temp.replace(path)


def bool_is_false(value: Any) -> bool:
    return value is False or (isinstance(value, np.bool_) and not bool(value))


class ActionConditionedDesign:
    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.category_levels_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "ActionConditionedDesign":
        numeric = frame.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        self.medians_ = numeric.median(axis=0).fillna(0.0)
        numeric = numeric.fillna(self.medians_)
        self.means_ = numeric.mean(axis=0)
        stds = numeric.std(axis=0, ddof=0)
        self.stds_ = stds.where(stds > 1e-12, 1.0)

        self.category_levels_ = {}
        for column in CATEGORICAL_FEATURES:
            values = frame[column].astype("string").fillna("__MISSING__")
            levels = sorted(str(value) for value in values.unique())
            if len(levels) > 1:
                self.category_levels_[column] = levels[1:]

        names: list[str] = []
        names.extend([f"num::{column}" for column in NUMERIC_FEATURES])

        nonbaseline_actions = EXPECTED_ACTIONS[1:]
        names.extend([f"action::{action}" for action in nonbaseline_actions])

        for column in NUMERIC_FEATURES:
            names.extend(
                [
                    f"interaction::{column}__{action}"
                    for action in nonbaseline_actions
                ]
            )

        for column, levels in self.category_levels_.items():
            names.extend([f"categorical::{column}={level}" for level in levels])

        self.feature_names_ = names
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise RuntimeError("Design must be fitted before transform.")

        numeric = frame.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.medians_)
        scaled = (numeric - self.means_) / self.stds_

        columns: list[np.ndarray] = []
        for column in NUMERIC_FEATURES:
            columns.append(scaled[column].to_numpy(dtype=float))

        nonbaseline_actions = EXPECTED_ACTIONS[1:]
        action_indicators: dict[str, np.ndarray] = {}
        for action in nonbaseline_actions:
            indicator = (frame["action"].astype(str) == action).to_numpy(dtype=float)
            action_indicators[action] = indicator
            columns.append(indicator)

        for column in NUMERIC_FEATURES:
            base = scaled[column].to_numpy(dtype=float)
            for action in nonbaseline_actions:
                columns.append(base * action_indicators[action])

        for column, levels in self.category_levels_.items():
            values = frame[column].astype("string").fillna("__MISSING__").astype(str)
            for level in levels:
                columns.append((values == level).to_numpy(dtype=float))

        matrix = np.column_stack(columns).astype(np.float64, copy=False)
        if matrix.shape[1] != len(self.feature_names_):
            raise RuntimeError("Design matrix feature width mismatch.")
        if not np.isfinite(matrix).all():
            raise RuntimeError("Design matrix contains non-finite values.")
        return np.ascontiguousarray(matrix, dtype=np.float64)


def action_mean_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> np.ndarray:
    means = train.groupby("action")[target].mean()
    overall = float(train[target].mean())
    return (
        test["action"]
        .map(means)
        .fillna(overall)
        .to_numpy(dtype=float)
    )


def inner_group_splits(frame: pd.DataFrame, random_state: int):
    groups = frame["seed"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(INNER_SPLITS, len(unique_groups))
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(frame, groups=groups))


def choose_ridge_alpha(
    train: pd.DataFrame,
    target: str,
    random_state: int,
) -> tuple[float, list[dict[str, Any]]]:
    splits = inner_group_splits(train, random_state)
    if not splits:
        return float(RIDGE_ALPHAS[0]), []

    rows: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        fold_losses: list[float] = []
        for inner_fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            val = train.iloc[val_idx].reset_index(drop=True)
            design = ActionConditionedDesign().fit(fit)
            model = Ridge(alpha=float(alpha))
            model.fit(design.transform(fit), fit[target].to_numpy(dtype=float))
            prediction = model.predict(design.transform(val))
            fold_losses.append(
                float(
                    mean_absolute_error(
                        val[target].to_numpy(dtype=float),
                        prediction,
                    )
                )
            )
        rows.append(
            {
                "candidate": float(alpha),
                "inner_metric": "mae",
                "mean_inner_loss": float(np.mean(fold_losses)),
                "fold_count": int(len(fold_losses)),
            }
        )
    rows.sort(key=lambda row: (row["mean_inner_loss"], row["candidate"]))
    return float(rows[0]["candidate"]), rows


def predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    alpha: float,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    design = ActionConditionedDesign().fit(train)
    model = Ridge(alpha=float(alpha))
    model.fit(design.transform(train), train[target].to_numpy(dtype=float))
    prediction = model.predict(design.transform(test))
    return (
        np.asarray(prediction, dtype=float),
        list(design.feature_names_),
        np.asarray(model.coef_, dtype=float),
    )


def choose_logistic_c(
    train: pd.DataFrame,
    target: str,
    random_state: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    y_all = train[target].to_numpy(dtype=int)
    if np.unique(y_all).size < 2:
        return None, []

    splits = inner_group_splits(train, random_state)
    if not splits:
        return float(LOGISTIC_CS[0]), []

    rows: list[dict[str, Any]] = []
    for c_value in LOGISTIC_CS:
        fold_losses: list[float] = []
        for inner_fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            val = train.iloc[val_idx].reset_index(drop=True)
            fit_y = fit[target].to_numpy(dtype=int)
            val_y = val[target].to_numpy(dtype=int)
            if np.unique(fit_y).size < 2:
                prediction = np.full(
                    len(val),
                    float(np.mean(fit_y)),
                    dtype=float,
                )
            else:
                design = ActionConditionedDesign().fit(fit)
                model = LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=random_state,
                )
                model.fit(design.transform(fit), fit_y)
                prediction = model.predict_proba(design.transform(val))[:, 1]
            fold_losses.append(float(brier_score_loss(val_y, prediction)))
        rows.append(
            {
                "candidate": float(c_value),
                "inner_metric": "brier",
                "mean_inner_loss": float(np.mean(fold_losses)),
                "fold_count": int(len(fold_losses)),
            }
        )
    rows.sort(key=lambda row: (row["mean_inner_loss"], row["candidate"]))
    return float(rows[0]["candidate"]), rows


def predict_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    c_value: float | None,
    random_state: int,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    train_y = train[target].to_numpy(dtype=int)
    if c_value is None or np.unique(train_y).size < 2:
        return (
            np.full(len(test), float(np.mean(train_y)), dtype=float),
            [],
            None,
        )

    design = ActionConditionedDesign().fit(train)
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(design.transform(train), train_y)
    prediction = model.predict_proba(design.transform(test))[:, 1]
    return (
        np.asarray(prediction, dtype=float),
        list(design.feature_names_),
        np.asarray(model.coef_.ravel(), dtype=float),
    )


def select_action_per_context(
    frame: pd.DataFrame,
    score_column: str,
    *,
    ascending: bool,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["seed", "proposal_index", score_column, "requested_rank"],
        ascending=[True, True, ascending, True],
        kind="mergesort",
    )
    return ordered.groupby(["seed", "proposal_index"], as_index=False).first()


def auc_or_none(y_true: np.ndarray, prediction: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, prediction))


v5e = json.loads(v5e_decision_path.read_text(encoding="utf-8-sig"))
if v5e.get("status") != "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_AUDIT_COMPLETE":
    raise RuntimeError("Unexpected V5-E decision status.")
if v5e.get("baseline_commit") != expected_head:
    raise RuntimeError("V5-E baseline commit mismatch.")
if v5e.get("model_trained") is not False or v5e.get("threshold_selected") is not False:
    raise RuntimeError("V5-E must not have trained a model or selected a threshold.")
if v5e.get("calibration_seeds_executed") not in ([], None):
    raise RuntimeError("V5-E indicates calibration seed execution.")
if v5e.get("confirmation_seeds_executed") not in ([], None):
    raise RuntimeError("V5-E indicates confirmation seed execution.")

action_path = Path(v5e["action_effects_path"]).resolve()
proposal_path = Path(v5e["proposal_manifest_path"]).resolve()
if not action_path.is_file() or not proposal_path.is_file():
    raise RuntimeError("V5-D input artifacts are missing.")

actions = pd.read_csv(action_path)
proposals = pd.read_csv(proposal_path)

keys = ["seed", "proposal_index", "query_id", "product_id"]
frame = actions.merge(proposals, on=keys, how="inner", validate="many_to_one")
if len(frame) != len(actions):
    raise RuntimeError("Action labels do not join exactly to proposal rows.")

required = {
    *keys,
    "action",
    "requested_rank",
    "achieved_rank",
    "mean_scenario_replication_utility_delta",
    "relevant_discovery_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
}
missing = sorted(required - set(frame.columns))
if missing:
    raise RuntimeError(f"V5-F required columns missing: {missing}")

if len(frame) != 720:
    raise RuntimeError(f"Expected 720 action rows, found {len(frame)}.")
if frame["seed"].nunique() != 18:
    raise RuntimeError("Expected exactly 18 model-training seeds.")
if not frame["achieved_rank"].eq(frame["requested_rank"]).all():
    raise RuntimeError("Rank-placement contract violated.")

action_set = tuple(sorted(str(v) for v in frame["action"].unique()))
if action_set != tuple(sorted(EXPECTED_ACTIONS)):
    raise RuntimeError("Unexpected action space.")

for column in (
    "mean_scenario_replication_utility_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
):
    frame[column] = pd.to_numeric(frame[column], errors="raise")
if not np.isfinite(
    frame.loc[
        :,
        [
            "mean_scenario_replication_utility_delta",
            "irrelevant_exposure_delta",
            "false_warmup_delta",
            "worst_scenario_utility_delta",
            "p10_scenario_replication_utility_delta",
        ],
    ].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite outcome label detected.")

frame["risk_irrelevant_increase"] = (frame["irrelevant_exposure_delta"] > 0.0).astype(int)
frame["risk_false_warmup_increase"] = (frame["false_warmup_delta"] > 0.0).astype(int)
frame["risk_worst_utility_negative"] = (frame["worst_scenario_utility_delta"] < 0.0).astype(int)
frame["risk_p10_utility_negative"] = (frame["p10_scenario_replication_utility_delta"] < 0.0).astype(int)
frame["strict_safe_improving"] = (
    (frame["relevant_discovery_delta"] > 0.0)
    & (frame["risk_irrelevant_increase"] == 0)
    & (frame["risk_false_warmup_increase"] == 0)
    & (frame["risk_worst_utility_negative"] == 0)
    & (frame["risk_p10_utility_negative"] == 0)
).astype(int)

task_specs = {
    "utility": {
        "target": "mean_scenario_replication_utility_delta",
        "kind": "regression",
    },
    "irrelevant": {
        "target": "risk_irrelevant_increase",
        "kind": "binary",
    },
    "false_warmup": {
        "target": "risk_false_warmup_increase",
        "kind": "binary",
    },
    "worst_utility": {
        "target": "risk_worst_utility_negative",
        "kind": "binary",
    },
    "p10_utility": {
        "target": "risk_p10_utility_negative",
        "kind": "binary",
    },
}

oof = frame.loc[:, keys + ["action", "requested_rank", "seed"]].copy()
for task_name in task_specs:
    oof[f"{task_name}_baseline"] = np.nan
    oof[f"{task_name}_model"] = np.nan

outer_rows: list[dict[str, Any]] = []
selection_rows: list[dict[str, Any]] = []
coefficient_rows: list[dict[str, Any]] = []
groups = frame["seed"].to_numpy()
outer = GroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=17)

for outer_fold, (train_idx, test_idx) in enumerate(
    outer.split(frame, groups=groups),
    start=1,
):
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    train_seeds = sorted(int(value) for value in train["seed"].unique())
    test_seeds = sorted(int(value) for value in test["seed"].unique())
    if set(train_seeds) & set(test_seeds):
        raise RuntimeError("Outer seed leakage detected.")

    outer_rows.append(
        {
            "outer_fold": int(outer_fold),
            "train_seed_count": int(len(train_seeds)),
            "test_seed_count": int(len(test_seeds)),
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
    )

    for task_offset, (task_name, spec) in enumerate(task_specs.items()):
        target = str(spec["target"])
        kind = str(spec["kind"])

        baseline_prediction = action_mean_baseline(train, test, target)

        if kind == "regression":
            selected_value, selection = choose_ridge_alpha(
                train,
                target,
                random_state=10_000 + 100 * outer_fold + task_offset,
            )
            model_prediction, feature_names, coefficients = predict_ridge(
                train,
                test,
                target,
                selected_value,
            )
            selection_kind = "ridge_alpha"
        else:
            selected_value, selection = choose_logistic_c(
                train,
                target,
                random_state=20_000 + 100 * outer_fold + task_offset,
            )
            model_prediction, feature_names, coefficients = predict_logistic(
                train,
                test,
                target,
                selected_value,
                random_state=30_000 + 100 * outer_fold + task_offset,
            )
            selection_kind = "logistic_c"

        oof.loc[test_idx, f"{task_name}_baseline"] = baseline_prediction
        oof.loc[test_idx, f"{task_name}_model"] = model_prediction

        for row in selection:
            selection_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "task": task_name,
                    "selection_kind": selection_kind,
                    "selected_candidate": selected_value,
                    **row,
                }
            )

        if coefficients is not None:
            for name, coefficient in zip(feature_names, coefficients, strict=True):
                coefficient_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "task": task_name,
                        "feature": name,
                        "coefficient": float(coefficient),
                        "selected_candidate": selected_value,
                    }
                )

if oof.isna().any().any():
    missing_columns = sorted(oof.columns[oof.isna().any()].tolist())
    raise RuntimeError(f"OOF prediction matrix has missing values: {missing_columns}")

metric_rows: list[dict[str, Any]] = []
utility_target = task_specs["utility"]["target"]
utility_y = frame[utility_target].to_numpy(dtype=float)
utility_baseline = oof["utility_baseline"].to_numpy(dtype=float)
utility_model = oof["utility_model"].to_numpy(dtype=float)

utility_baseline_mae = float(mean_absolute_error(utility_y, utility_baseline))
utility_model_mae = float(mean_absolute_error(utility_y, utility_model))
utility_baseline_rmse = float(math.sqrt(mean_squared_error(utility_y, utility_baseline)))
utility_model_rmse = float(math.sqrt(mean_squared_error(utility_y, utility_model)))
utility_mae_improvement = float(
    (utility_baseline_mae - utility_model_mae) / max(utility_baseline_mae, 1e-12)
)

metric_rows.append(
    {
        "task": "utility",
        "kind": "regression",
        "baseline_mae": utility_baseline_mae,
        "model_mae": utility_model_mae,
        "relative_mae_improvement": utility_mae_improvement,
        "baseline_rmse": utility_baseline_rmse,
        "model_rmse": utility_model_rmse,
        "baseline_roc_auc": None,
        "model_roc_auc": None,
        "baseline_brier": None,
        "model_brier": None,
        "brier_skill": None,
    }
)

binary_task_rows: dict[str, dict[str, Any]] = {}
for task_name, spec in task_specs.items():
    if spec["kind"] != "binary":
        continue
    y_true = frame[spec["target"]].to_numpy(dtype=int)
    baseline_prediction = oof[f"{task_name}_baseline"].to_numpy(dtype=float)
    model_prediction = oof[f"{task_name}_model"].to_numpy(dtype=float)
    baseline_brier = float(brier_score_loss(y_true, baseline_prediction))
    model_brier = float(brier_score_loss(y_true, model_prediction))
    brier_skill = float(
        (baseline_brier - model_brier) / max(baseline_brier, 1e-12)
    )
    row = {
        "task": task_name,
        "kind": "binary",
        "baseline_mae": None,
        "model_mae": None,
        "relative_mae_improvement": None,
        "baseline_rmse": None,
        "model_rmse": None,
        "baseline_roc_auc": auc_or_none(y_true, baseline_prediction),
        "model_roc_auc": auc_or_none(y_true, model_prediction),
        "baseline_brier": baseline_brier,
        "model_brier": model_brier,
        "brier_skill": brier_skill,
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }
    metric_rows.append(row)
    binary_task_rows[task_name] = row

evaluation = frame.loc[:, keys + ["action", "requested_rank", utility_target]].copy()
evaluation["utility_baseline"] = utility_baseline
evaluation["utility_model"] = utility_model
observed_best = select_action_per_context(
    evaluation,
    utility_target,
    ascending=False,
).rename(
    columns={
        "action": "observed_best_action",
        utility_target: "observed_best_utility",
    }
)[
    ["seed", "proposal_index", "observed_best_action", "observed_best_utility"]
]

selection_diagnostics: dict[str, Any] = {}
for score_column, label in (
    ("utility_baseline", "action_only_baseline"),
    ("utility_model", "action_conditioned_model"),
):
    chosen = select_action_per_context(evaluation, score_column, ascending=False)
    joined = chosen.merge(
        observed_best,
        on=["seed", "proposal_index"],
        how="inner",
        validate="one_to_one",
    )
    selection_diagnostics[label] = {
        "context_count": int(len(joined)),
        "top1_best_action_hit_rate": float(
            (joined["action"] == joined["observed_best_action"]).mean()
        ),
        "mean_selected_observed_utility": float(joined[utility_target].mean()),
        "mean_utility_regret": float(
            (joined["observed_best_utility"] - joined[utility_target]).mean()
        ),
        "median_utility_regret": float(
            (joined["observed_best_utility"] - joined[utility_target]).median()
        ),
        "chosen_action_distribution": {
            str(key): int(value)
            for key, value in joined["action"].value_counts().sort_index().items()
        },
    }

rank_hit_gain = float(
    selection_diagnostics["action_conditioned_model"]["top1_best_action_hit_rate"]
    - selection_diagnostics["action_only_baseline"]["top1_best_action_hit_rate"]
)
regret_reduction = float(
    selection_diagnostics["action_only_baseline"]["mean_utility_regret"]
    - selection_diagnostics["action_conditioned_model"]["mean_utility_regret"]
)

qualified_risk_heads = []
for task_name, row in binary_task_rows.items():
    auc = row["model_roc_auc"]
    skill = float(row["brier_skill"])
    qualifies = bool(auc is not None and auc >= 0.60 and skill >= 0.02)
    row["predeclared_risk_head_qualified"] = qualifies
    if qualifies:
        qualified_risk_heads.append(task_name)

utility_gate = bool(
    utility_mae_improvement >= 0.05
    and rank_hit_gain >= 0.05
    and regret_reduction > 0.0
)
risk_gate = len(qualified_risk_heads) >= 2
status = (
    "V5F_MULTIHEAD_RUNTIME_SIGNAL_PRESENT"
    if utility_gate and risk_gate
    else "V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT"
)

decision = {
    "status": status,
    "baseline_commit": expected_head,
    "v5e_decision_path": str(v5e_decision_path),
    "v5e_decision_sha256": sha256(v5e_decision_path),
    "action_effects_path": str(action_path),
    "action_effects_sha256": sha256(action_path),
    "proposal_manifest_path": str(proposal_path),
    "proposal_manifest_sha256": sha256(proposal_path),
    "runtime_feature_contract": {
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "action_feature": "action",
        "action_interactions": "numeric_feature x action",
        "forbidden_outcome_or_oracle_inputs": True,
    },
    "protocol": {
        "outer_cv": {
            "splitter": "GroupKFold",
            "group": "seed",
            "n_splits": OUTER_SPLITS,
            "shuffle": True,
            "random_state": 17,
        },
        "inner_cv": {
            "splitter": "GroupKFold",
            "group": "seed",
            "n_splits": INNER_SPLITS,
            "purpose": "select regularization strength inside each outer training partition",
        },
        "model_families": {
            "utility": "action-conditioned Ridge regression",
            "risk_heads": "action-conditioned L2 logistic regression",
            "baseline": "outer-train action-specific mean or event rate",
        },
        "no_final_model_artifact": True,
        "no_threshold_selected": True,
        "no_calibration_executed": True,
        "no_confirmation_executed": True,
    },
    "predeclared_viability_gates": {
        "utility_relative_mae_improvement_minimum": 0.05,
        "utility_top1_best_action_hit_rate_gain_minimum": 0.05,
        "utility_regret_reduction_required": True,
        "risk_minimum_model_roc_auc": 0.60,
        "risk_minimum_brier_skill": 0.02,
        "risk_minimum_qualified_head_count": 2,
    },
    "utility_gate_pass": utility_gate,
    "risk_gate_pass": risk_gate,
    "qualified_risk_heads": qualified_risk_heads,
    "utility_ranking_diagnostics": {
        "rank_hit_gain": rank_hit_gain,
        "regret_reduction": regret_reduction,
        **selection_diagnostics,
    },
    "source_modified": False,
    "config_modified": False,
    "cross_validated_models_fitted": True,
    "final_serving_model_artifact_created": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "If status is V5F_MULTIHEAD_RUNTIME_SIGNAL_PRESENT, freeze the "
        "pre-registered model-family and proceed to a train-only artifact build. "
        "If status is V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT, do not fit a "
        "final action model; return to runtime-state feature instrumentation."
    ),
}

output_root.mkdir(parents=True, exist_ok=True)
oof.to_csv(output_root / "v5f_outer_oof_predictions.csv", index=False)
pd.DataFrame(outer_rows).to_csv(output_root / "v5f_outer_fold_ledger.csv", index=False)
pd.DataFrame(selection_rows).to_csv(output_root / "v5f_inner_regularization_ledger.csv", index=False)
pd.DataFrame(coefficient_rows).to_csv(output_root / "v5f_outer_coefficient_ledger.csv", index=False)
pd.DataFrame(metric_rows).to_csv(output_root / "v5f_head_metrics.csv", index=False)
write_json(output_root / "v5f_multihead_viability_decision.json", decision)

report = [
    "# V5-F Seed-Disjoint Multi-Head Viability Audit",
    "",
    f"- Status: `{status}`",
    f"- Outer protocol: 6-fold GroupKFold grouped by seed.",
    f"- Input rows: {len(frame)} action labels across {frame['seed'].nunique()} seeds.",
    f"- Utility MAE improvement: {utility_mae_improvement:.6f}",
    f"- Utility top-1 action hit gain: {rank_hit_gain:.6f}",
    f"- Utility regret reduction: {regret_reduction:.6f}",
    f"- Qualified risk heads: {qualified_risk_heads}",
    "",
    "This audit emits only out-of-fold diagnostics. It does not persist a final serving model, select a threshold, calibrate probabilities, or use confirmation seeds.",
]
(output_root / "v5f_multihead_viability_report.md").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("===== V5-F SEED-DISJOINT MULTI-HEAD MODEL VIABILITY AUDIT =====")
print(
    json.dumps(
        {
            "decision": decision,
            "head_metrics": metric_rows,
            "outer_fold_ledger": outer_rows,
        },
        indent=2,
        sort_keys=True,
        default=native,
    )
)
