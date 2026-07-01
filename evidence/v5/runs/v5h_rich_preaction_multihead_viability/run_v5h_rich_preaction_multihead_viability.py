from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


v5d_decision_path = Path(sys.argv[1]).resolve()
v5e_decision_path = Path(sys.argv[2]).resolve()
v5f_decision_path = Path(sys.argv[3]).resolve()
v5g_decision_path = Path(sys.argv[4]).resolve()
recovery_decision_path = Path(sys.argv[5]).resolve()
output_root = Path(sys.argv[6]).resolve()
expected_head = sys.argv[7]

RANDOM_STATE = 17
OUTER_SPLITS = 6
INNER_SPLITS = 3
BOOTSTRAP_REPS = 20_000

ACTION_ORDER = (
    "PLACE_AT_1",
    "PLACE_AT_2",
    "PLACE_AT_3",
    "PLACE_AT_5",
    "PLACE_AT_10",
)

ACTION_TO_TARGET_RANK = {
    "PLACE_AT_1": 1,
    "PLACE_AT_2": 2,
    "PLACE_AT_3": 3,
    "PLACE_AT_5": 5,
    "PLACE_AT_10": 10,
}

CONTEXT_NUMERIC_FEATURES = (
    "base_score",
    "base_rank",
    "bm25_score",
    "dense_score",
    "retrieval_score",
    "semantic_rank_score",
    "semantic_component",
    "qrsbt_relevance_probability",
    "qrsbt_confidence",
    "qrsbt_dispersion",
    "qrsbt_compatibility",
    "qrsbt_irrelevant_probability",
    "price",
    "price_log",
    "quality",
    "query_candidate_count",
    "query_base_score_max",
    "query_base_score_min",
    "query_base_score_mean",
    "query_base_score_std",
    "query_base_score_range",
    "query_top1_base_score",
    "query_top2_base_score",
    "query_top1_to_top2_gap",
    "candidate_score_minus_query_mean",
    "candidate_score_z",
    "candidate_rank_fraction",
    "candidate_gap_to_above",
    "candidate_gap_to_below",
    "query_qrsbt_confidence_max",
    "query_qrsbt_confidence_mean",
    "query_qrsbt_confidence_std",
    "query_qrsbt_relevance_probability_max",
    "query_qrsbt_relevance_probability_mean",
    "query_qrsbt_relevance_probability_std",
    "query_qrsbt_irrelevant_probability_max",
    "query_qrsbt_irrelevant_probability_mean",
    "query_qrsbt_irrelevant_probability_std",
)

ACTION_NUMERIC_FEATURES = (
    "target_rank",
    "action_move_distance",
    "action_required_lift",
    "action_score_gap",
    "action_boundary_score",
)

CATEGORICAL_FEATURES = (
    "action",
    "candidate_source",
    "category",
    "query_category",
    "query_intent",
    "relation",
)

FEATURE_COLUMNS = (
    *CONTEXT_NUMERIC_FEATURES,
    *ACTION_NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
)

KEY_COLUMNS = (
    "seed",
    "proposal_index",
    "query_id",
    "product_id",
    "action",
)

UTILITY_TARGET = (
    "mean_scenario_replication_utility_delta"
)

RISK_TARGETS = {
    "irrelevant_exposure_increase": (
        "irrelevant_exposure_delta",
        lambda series: series.gt(0.0),
    ),
    "false_warmup_increase": (
        "false_warmup_delta",
        lambda series: series.gt(0.0),
    ),
    "negative_worst_scenario_utility": (
        "worst_scenario_utility_delta",
        lambda series: series.lt(0.0),
    ),
    "negative_p10_scenario_utility": (
        "p10_scenario_replication_utility_delta",
        lambda series: series.lt(0.0),
    ),
}

FORBIDDEN_SELECTED_FEATURE_TOKENS = (
    "teacher",
    "oracle",
    "future",
    "outcome",
    "label",
    "utility_delta",
    "discovery_delta",
    "exposure_delta",
    "warmup_delta",
    "clicked",
    "purchased",
    "judged",
    "logging_propensity",
    "position",
    "time_block",
    "user_id",
    "prior_",
    "smoothed_",
    "behavior_velocity",
    "first_observed_age",
    "final_score",
    "qrsbt_boost",
    "gate_action",
    "gate_reason",
)

UTILITY_CANDIDATES = (
    ("ridge_alpha_0p1", "ridge", {"alpha": 0.1}),
    ("ridge_alpha_1", "ridge", {"alpha": 1.0}),
    ("ridge_alpha_10", "ridge", {"alpha": 10.0}),
    ("ridge_alpha_100", "ridge", {"alpha": 100.0}),
    (
        "extra_trees_leaf_2",
        "extra_trees",
        {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": 0.70,
        },
    ),
    (
        "extra_trees_leaf_5",
        "extra_trees",
        {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "max_features": 0.70,
        },
    ),
    (
        "extra_trees_leaf_10",
        "extra_trees",
        {
            "n_estimators": 300,
            "min_samples_leaf": 10,
            "max_features": 0.70,
        },
    ),
)

RISK_CANDIDATES = (
    ("logistic_c_0p1", "logistic", {"C": 0.1}),
    ("logistic_c_1", "logistic", {"C": 1.0}),
    ("logistic_c_10", "logistic", {"C": 10.0}),
    (
        "extra_trees_leaf_5",
        "extra_trees",
        {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "max_features": 0.70,
        },
    ),
    (
        "extra_trees_leaf_10",
        "extra_trees",
        {
            "n_estimators": 300,
            "min_samples_leaf": 10,
            "max_features": 0.70,
        },
    ),
    (
        "extra_trees_leaf_20",
        "extra_trees",
        {
            "n_estimators": 300,
            "min_samples_leaf": 20,
            "max_features": 0.70,
        },
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def validate_decision(
    decision: dict[str, Any],
    *,
    label: str,
    expected_status: str,
) -> None:
    if decision.get("status") != expected_status:
        raise RuntimeError(
            f"{label}: unexpected status {decision.get('status')}"
        )

    if decision.get("baseline_commit") != expected_head:
        raise RuntimeError(
            f"{label}: baseline commit mismatch."
        )

    calibration = decision.get(
        "calibration_seeds_executed"
    )

    confirmation = decision.get(
        "confirmation_seeds_executed"
    )

    if calibration not in (None, [], False):
        raise RuntimeError(
            f"{label}: calibration execution detected."
        )

    if confirmation not in (None, [], False):
        raise RuntimeError(
            f"{label}: confirmation execution detected."
        )


def assert_feature_contract() -> None:
    duplicates = [
        value
        for value in FEATURE_COLUMNS
        if FEATURE_COLUMNS.count(value) > 1
    ]

    if duplicates:
        raise RuntimeError(
            "Feature contract contains duplicates: "
            f"{sorted(set(duplicates))}"
        )

    forbidden = [
        feature
        for feature in FEATURE_COLUMNS
        if any(
            token in feature.lower()
            for token in FORBIDDEN_SELECTED_FEATURE_TOKENS
        )
    ]

    if forbidden:
        raise RuntimeError(
            "Feature contract contains forbidden token(s): "
            f"{forbidden}"
        )


def make_preprocessor() -> ColumnTransformer:
    numeric_features = [
        *CONTEXT_NUMERIC_FEATURES,
        *ACTION_NUMERIC_FEATURES,
    ]

    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipe,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipe,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_utility_model(
    family: str,
    params: dict[str, Any],
) -> Pipeline:
    preprocessor = make_preprocessor()

    if family == "ridge":
        estimator = Ridge(
            alpha=float(params["alpha"]),
        )
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(
                params["min_samples_leaf"]
            ),
            max_features=float(params["max_features"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise RuntimeError(
            f"Unknown utility family: {family}"
        )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("estimator", estimator),
        ]
    )


def make_risk_model(
    family: str,
    params: dict[str, Any],
) -> Pipeline:
    preprocessor = make_preprocessor()

    if family == "logistic":
        estimator = LogisticRegression(
            C=float(params["C"]),
            max_iter=5_000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif family == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(
                params["min_samples_leaf"]
            ),
            max_features=float(params["max_features"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise RuntimeError(
            f"Unknown risk family: {family}"
        )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("estimator", estimator),
        ]
    )


def action_means(
    frame: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    grouped = (
        frame.groupby("action")[target]
        .mean()
        .to_dict()
    )

    missing = [
        action
        for action in ACTION_ORDER
        if action not in grouped
    ]

    if missing:
        raise RuntimeError(
            f"Action baseline missing action means: {missing}"
        )

    return {
        str(action): float(value)
        for action, value in grouped.items()
    }


def action_rates(
    frame: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    grouped = (
        frame.groupby("action")[target]
        .mean()
        .to_dict()
    )

    missing = [
        action
        for action in ACTION_ORDER
        if action not in grouped
    ]

    if missing:
        raise RuntimeError(
            f"Action baseline missing action rates: {missing}"
        )

    return {
        str(action): float(value)
        for action, value in grouped.items()
    }


def select_policy_rows(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
) -> pd.DataFrame:
    needed = {
        "seed",
        "proposal_index",
        "action",
        "target_rank",
        UTILITY_TARGET,
        prediction_column,
    }

    missing = sorted(needed - set(frame.columns))

    if missing:
        raise RuntimeError(
            f"Policy selection missing columns: {missing}"
        )

    counts = (
        frame.groupby(
            ["seed", "proposal_index"],
            as_index=False,
        )
        .agg(
            action_count=("action", "size"),
            unique_action_count=("action", "nunique"),
        )
    )

    if not counts["action_count"].eq(5).all():
        raise RuntimeError(
            "At least one context does not contain five actions."
        )

    if not counts["unique_action_count"].eq(5).all():
        raise RuntimeError(
            "At least one context has duplicate action labels."
        )

    selected = (
        frame.sort_values(
            [
                "seed",
                "proposal_index",
                prediction_column,
                "target_rank",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(
            ["seed", "proposal_index"],
            as_index=False,
            sort=False,
        )
        .first()
    )

    return selected


def policy_metrics(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = select_policy_rows(
        frame,
        prediction_column=prediction_column,
    )

    oracle = (
        frame.sort_values(
            [
                "seed",
                "proposal_index",
                UTILITY_TARGET,
                "target_rank",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(
            ["seed", "proposal_index"],
            as_index=False,
            sort=False,
        )
        .first()
        .loc[
            :,
            [
                "seed",
                "proposal_index",
                "action",
                "target_rank",
                UTILITY_TARGET,
            ],
        ]
        .rename(
            columns={
                "action": "oracle_action",
                "target_rank": "oracle_target_rank",
                UTILITY_TARGET: "oracle_utility",
            }
        )
    )

    selected = selected.merge(
        oracle,
        on=["seed", "proposal_index"],
        how="inner",
        validate="one_to_one",
    )

    if len(selected) != 144:
        raise RuntimeError(
            f"{label}: expected 144 selected contexts, found "
            f"{len(selected)}."
        )

    selected["utility_regret"] = (
        selected["oracle_utility"]
        - selected[UTILITY_TARGET]
    )

    selected["oracle_action_hit"] = (
        selected["action"]
        .eq(selected["oracle_action"])
        .astype(int)
    )

    selected["strict_safe_improving"] = (
        selected["benefit_positive"]
        & selected["no_irrelevant_increase"]
        & selected["no_false_warmup_increase"]
        & selected["nonnegative_worst_utility"]
        & selected["nonnegative_p10_utility"]
    ).astype(int)

    selected["core_safe_improving"] = (
        selected["benefit_positive"]
        & selected["no_irrelevant_increase"]
        & selected["no_false_warmup_increase"]
        & selected["nonnegative_worst_utility"]
    ).astype(int)

    by_seed = (
        selected.groupby("seed", as_index=False)
        .agg(
            context_count=("proposal_index", "size"),
            selected_mean_utility=(
                UTILITY_TARGET,
                "mean",
            ),
            selected_relevant_discovery=(
                "relevant_discovery_delta",
                "mean",
            ),
            selected_irrelevant_exposure=(
                "irrelevant_exposure_delta",
                "mean",
            ),
            selected_false_warmup=(
                "false_warmup_delta",
                "mean",
            ),
            selected_worst_utility=(
                "worst_scenario_utility_delta",
                "mean",
            ),
            selected_p10_utility=(
                "p10_scenario_replication_utility_delta",
                "mean",
            ),
            mean_utility_regret=(
                "utility_regret",
                "mean",
            ),
            oracle_action_hit_rate=(
                "oracle_action_hit",
                "mean",
            ),
            strict_safe_improving_rate=(
                "strict_safe_improving",
                "mean",
            ),
            core_safe_improving_rate=(
                "core_safe_improving",
                "mean",
            ),
        )
    )

    return selected, by_seed


def bootstrap_seed_mean_delta(
    seed_frame: pd.DataFrame,
    *,
    delta_column: str,
) -> dict[str, float]:
    values = seed_frame[delta_column].to_numpy(
        dtype=float
    )

    if len(values) != 18:
        raise RuntimeError(
            "Seed bootstrap requires exactly 18 seed values."
        )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Seed bootstrap received non-finite values."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_indices = rng.integers(
        low=0,
        high=len(values),
        size=(BOOTSTRAP_REPS, len(values)),
    )

    samples = values[sampled_indices].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "ci_low_95": float(
            np.quantile(samples, 0.025)
        ),
        "ci_high_95": float(
            np.quantile(samples, 0.975)
        ),
    }


def mean_context_utility(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> tuple[float, float]:
    scored = frame.loc[
        :,
        [
            "seed",
            "proposal_index",
            "action",
            "target_rank",
            UTILITY_TARGET,
        ],
    ].copy()

    scored["_prediction"] = prediction

    selected = select_policy_rows(
        scored,
        prediction_column="_prediction",
    )

    oracle = (
        scored.sort_values(
            [
                "seed",
                "proposal_index",
                UTILITY_TARGET,
                "target_rank",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(
            ["seed", "proposal_index"],
            as_index=False,
            sort=False,
        )
        .first()
        .loc[
            :,
            [
                "seed",
                "proposal_index",
                UTILITY_TARGET,
            ],
        ]
        .rename(
            columns={
                UTILITY_TARGET: "oracle_utility",
            }
        )
    )

    selected = selected.merge(
        oracle,
        on=["seed", "proposal_index"],
        how="inner",
        validate="one_to_one",
    )

    return (
        float(selected[UTILITY_TARGET].mean()),
        float(
            (
                selected["oracle_utility"]
                - selected[UTILITY_TARGET]
            ).mean()
        ),
    )


def make_outer_group_cv(
    n_splits: int,
) -> GroupKFold:
    return GroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def choose_utility_candidate(
    train_frame: pd.DataFrame,
) -> tuple[tuple[str, str, dict[str, Any]], pd.DataFrame]:
    groups = train_frame["seed"].to_numpy()
    inner_cv = make_outer_group_cv(INNER_SPLITS)
    rows: list[dict[str, Any]] = []

    for name, family, params in UTILITY_CANDIDATES:
        fold_utilities: list[float] = []
        fold_regrets: list[float] = []
        fold_maes: list[float] = []

        for inner_train_idx, inner_test_idx in inner_cv.split(
            train_frame,
            groups=groups,
        ):
            inner_train = train_frame.iloc[
                inner_train_idx
            ]
            inner_test = train_frame.iloc[
                inner_test_idx
            ]

            model = make_utility_model(family, params)
            model.fit(
                inner_train.loc[:, list(FEATURE_COLUMNS)],
                inner_train[UTILITY_TARGET],
            )

            prediction = model.predict(
                inner_test.loc[:, list(FEATURE_COLUMNS)]
            )

            selected_utility, selected_regret = (
                mean_context_utility(
                    inner_test,
                    prediction,
                )
            )

            fold_utilities.append(selected_utility)
            fold_regrets.append(selected_regret)
            fold_maes.append(
                float(
                    mean_absolute_error(
                        inner_test[UTILITY_TARGET],
                        prediction,
                    )
                )
            )

        rows.append(
            {
                "candidate": name,
                "family": family,
                "params_json": json.dumps(
                    params,
                    sort_keys=True,
                ),
                "inner_selected_mean_utility": float(
                    np.mean(fold_utilities)
                ),
                "inner_selected_mean_regret": float(
                    np.mean(fold_regrets)
                ),
                "inner_row_mae": float(
                    np.mean(fold_maes)
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        [
            "inner_selected_mean_utility",
            "inner_selected_mean_regret",
            "inner_row_mae",
            "candidate",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    chosen_name = str(table.iloc[0]["candidate"])

    for candidate in UTILITY_CANDIDATES:
        if candidate[0] == chosen_name:
            return candidate, table

    raise RuntimeError(
        "Selected utility candidate was not found."
    )


def constant_probability(
    train_frame: pd.DataFrame,
    risk_target: str,
    test_length: int,
) -> np.ndarray:
    value = float(train_frame[risk_target].mean())

    return np.repeat(
        np.clip(value, 1e-6, 1.0 - 1e-6),
        test_length,
    )


def predict_risk(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    family: str,
    params: dict[str, Any],
    risk_target: str,
) -> np.ndarray:
    labels = train_frame[risk_target].to_numpy(
        dtype=int
    )

    if len(np.unique(labels)) < 2:
        return constant_probability(
            train_frame,
            risk_target,
            len(test_frame),
        )

    model = make_risk_model(family, params)
    model.fit(
        train_frame.loc[:, list(FEATURE_COLUMNS)],
        labels,
    )

    probability = model.predict_proba(
        test_frame.loc[:, list(FEATURE_COLUMNS)]
    )

    class_indices = list(
        model.named_steps["estimator"].classes_
    )

    positive_index = class_indices.index(1)

    return probability[:, positive_index]


def choose_risk_candidate(
    train_frame: pd.DataFrame,
    risk_target: str,
) -> tuple[tuple[str, str, dict[str, Any]], pd.DataFrame]:
    groups = train_frame["seed"].to_numpy()
    inner_cv = make_outer_group_cv(INNER_SPLITS)
    rows: list[dict[str, Any]] = []

    for name, family, params in RISK_CANDIDATES:
        fold_briers: list[float] = []

        for inner_train_idx, inner_test_idx in inner_cv.split(
            train_frame,
            groups=groups,
        ):
            inner_train = train_frame.iloc[
                inner_train_idx
            ]
            inner_test = train_frame.iloc[
                inner_test_idx
            ]

            probability = predict_risk(
                inner_train,
                inner_test,
                family=family,
                params=params,
                risk_target=risk_target,
            )

            fold_briers.append(
                float(
                    brier_score_loss(
                        inner_test[risk_target],
                        probability,
                    )
                )
            )

        rows.append(
            {
                "candidate": name,
                "family": family,
                "params_json": json.dumps(
                    params,
                    sort_keys=True,
                ),
                "inner_brier": float(
                    np.mean(fold_briers)
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        [
            "inner_brier",
            "candidate",
        ],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    chosen_name = str(table.iloc[0]["candidate"])

    for candidate in RISK_CANDIDATES:
        if candidate[0] == chosen_name:
            return candidate, table

    raise RuntimeError(
        f"Selected risk candidate not found for {risk_target}."
    )


def action_rate_prediction(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    risk_target: str,
) -> np.ndarray:
    rates = action_rates(train_frame, risk_target)

    return (
        test_frame["action"]
        .map(rates)
        .to_numpy(dtype=float)
    )


def action_mean_prediction(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> np.ndarray:
    means = action_means(train_frame, UTILITY_TARGET)

    return (
        test_frame["action"]
        .map(means)
        .to_numpy(dtype=float)
    )


def safe_auc(
    labels: np.ndarray,
    probability: np.ndarray,
) -> float | None:
    if len(np.unique(labels)) < 2:
        return None

    return float(
        roc_auc_score(labels, probability)
    )


assert_feature_contract()

v5d = read_json(v5d_decision_path)
v5e = read_json(v5e_decision_path)
v5f = read_json(v5f_decision_path)
v5g = read_json(v5g_decision_path)
recovery = read_json(recovery_decision_path)

validate_decision(
    v5d,
    label="V5-D",
    expected_status=(
        "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
    ),
)

validate_decision(
    v5e,
    label="V5-E",
    expected_status=(
        "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_AUDIT_COMPLETE"
    ),
)

validate_decision(
    v5f,
    label="V5-F",
    expected_status=(
        "V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT"
    ),
)

validate_decision(
    v5g,
    label="V5-G",
    expected_status=(
        "V5G_RUNTIME_STATE_REINSTRUMENTATION_REQUIRED"
    ),
)

validate_decision(
    recovery,
    label="V5-G1",
    expected_status=(
        "V5G1_PREACTION_RUNTIME_STATE_RECOVERED_AND_CONTRACT_READY"
    ),
)

if int(v5d.get("action_count", -1)) != 720:
    raise RuntimeError("V5-D action count mismatch.")

if int(v5d.get("proposal_count", -1)) != 144:
    raise RuntimeError("V5-D proposal count mismatch.")

if int(recovery.get("context_count", -1)) != 144:
    raise RuntimeError("Recovery context count mismatch.")

if int(recovery.get("state_action_row_count", -1)) != 720:
    raise RuntimeError(
        "Recovery state-action count mismatch."
    )

if int(recovery.get("exact_context_match_count", -1)) != 144:
    raise RuntimeError(
        "Recovery exact-context provenance mismatch."
    )

if recovery.get("forbidden_outcome_or_oracle_inputs") is not True:
    raise RuntimeError(
        "Recovery leakage-exclusion contract failed."
    )

state_action_path = Path(
    recovery["state_action_path"]
).resolve()

action_effects_path = Path(
    v5d["action_effects_path"]
).resolve()

for path in (
    state_action_path,
    action_effects_path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Required input artifact missing: {path}"
        )

state_action = pd.read_csv(state_action_path)
outcomes = pd.read_csv(action_effects_path)

for frame, label in (
    (state_action, "recovered state-action"),
    (outcomes, "V5-D action effects"),
):
    missing_keys = sorted(
        set(KEY_COLUMNS) - set(frame.columns)
    )

    if missing_keys:
        raise RuntimeError(
            f"{label} missing key columns: {missing_keys}"
        )

if state_action.duplicated(
    ["seed", "proposal_index", "action"]
).any():
    raise RuntimeError(
        "Recovered state-action matrix has duplicate keys."
    )

if outcomes.duplicated(
    ["seed", "proposal_index", "action"]
).any():
    raise RuntimeError(
        "V5-D action labels have duplicate keys."
    )

missing_features = sorted(
    set(FEATURE_COLUMNS) - set(state_action.columns)
)

if missing_features:
    raise RuntimeError(
        "Recovered state-action matrix lacks frozen "
        f"feature-contract columns: {missing_features}"
    )

required_outcomes = {
    UTILITY_TARGET,
    "relevant_discovery_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
    "requested_rank",
    "achieved_rank",
}

missing_outcomes = sorted(
    required_outcomes - set(outcomes.columns)
)

if missing_outcomes:
    raise RuntimeError(
        f"V5-D action labels missing outcomes: {missing_outcomes}"
    )

frame = state_action.merge(
    outcomes.loc[
        :,
        [
            *KEY_COLUMNS,
            *sorted(required_outcomes),
        ],
    ],
    on=list(KEY_COLUMNS),
    how="inner",
    validate="one_to_one",
)

if len(frame) != 720:
    raise RuntimeError(
        f"Expected 720 joined state-action rows, got {len(frame)}."
    )

if not frame["action"].isin(ACTION_ORDER).all():
    raise RuntimeError(
        "Recovered matrix action space mismatch."
    )

expected_target = frame["action"].map(
    ACTION_TO_TARGET_RANK
)

if not frame["target_rank"].eq(expected_target).all():
    raise RuntimeError(
        "Recovered action target_rank does not match action."
    )

if not frame["requested_rank"].eq(
    frame["target_rank"]
).all():
    raise RuntimeError(
        "Requested action rank does not match recovered state."
    )

if not frame["achieved_rank"].eq(
    frame["requested_rank"]
).all():
    raise RuntimeError(
        "At least one action effect violated exact placement."
    )

if frame.duplicated(
    ["seed", "proposal_index", "action"]
).any():
    raise RuntimeError(
        "Join created duplicate state-action keys."
    )

context_counts = (
    frame.groupby(
        ["seed", "proposal_index"],
        as_index=False,
    )
    .agg(
        action_count=("action", "size"),
        unique_action_count=("action", "nunique"),
    )
)

if len(context_counts) != 144:
    raise RuntimeError(
        "Expected 144 state-action contexts."
    )

if not context_counts["action_count"].eq(5).all():
    raise RuntimeError(
        "At least one context lacks action rows."
    )

if not context_counts["unique_action_count"].eq(5).all():
    raise RuntimeError(
        "At least one context has duplicate actions."
    )

if frame["seed"].nunique() != 18:
    raise RuntimeError(
        "Expected 18 seed groups."
    )

for column in (
    *CONTEXT_NUMERIC_FEATURES,
    *ACTION_NUMERIC_FEATURES,
):
    frame[column] = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

for column in (
    *CATEGORICAL_FEATURES,
):
    frame[column] = (
        frame[column]
        .astype("string")
        .fillna("__MISSING__")
    )

for column in (
    UTILITY_TARGET,
    "relevant_discovery_delta",
    "irrelevant_exposure_delta",
    "false_warmup_delta",
    "worst_scenario_utility_delta",
    "p10_scenario_replication_utility_delta",
):
    frame[column] = pd.to_numeric(
        frame[column],
        errors="raise",
    )

if not np.isfinite(
    frame.loc[
        :,
        [
            UTILITY_TARGET,
            "relevant_discovery_delta",
            "irrelevant_exposure_delta",
            "false_warmup_delta",
            "worst_scenario_utility_delta",
            "p10_scenario_replication_utility_delta",
        ],
    ].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Non-finite V5-D action outcomes."
    )

frame["benefit_positive"] = (
    frame["relevant_discovery_delta"] > 0.0
)

frame["no_irrelevant_increase"] = (
    frame["irrelevant_exposure_delta"] <= 0.0
)

frame["no_false_warmup_increase"] = (
    frame["false_warmup_delta"] <= 0.0
)

frame["nonnegative_worst_utility"] = (
    frame["worst_scenario_utility_delta"] >= 0.0
)

frame["nonnegative_p10_utility"] = (
    frame["p10_scenario_replication_utility_delta"] >= 0.0
)

for risk_name, (outcome_column, transform) in RISK_TARGETS.items():
    frame[risk_name] = transform(
        frame[outcome_column]
    ).astype(int)

# The original risk labels are retained in the output only.
# They are not candidate features.
feature_missing_rate = (
    frame.loc[:, list(FEATURE_COLUMNS)]
    .isna()
    .mean()
    .rename("missing_rate")
    .reset_index()
    .rename(columns={"index": "feature"})
)

feature_unique_count = (
    frame.loc[:, list(FEATURE_COLUMNS)]
    .nunique(dropna=True)
    .rename("unique_count")
    .reset_index()
    .rename(columns={"index": "feature"})
)

feature_contract_report = feature_missing_rate.merge(
    feature_unique_count,
    on="feature",
    how="inner",
    validate="one_to_one",
)

feature_contract_report["feature_role"] = np.where(
    feature_contract_report["feature"].isin(
        CONTEXT_NUMERIC_FEATURES
    ),
    "context_numeric",
    np.where(
        feature_contract_report["feature"].isin(
            ACTION_NUMERIC_FEATURES
        ),
        "action_numeric",
        "categorical",
    ),
)

feature_contract_report["constant"] = (
    feature_contract_report["unique_count"] <= 1
)

constant_features = feature_contract_report.loc[
    feature_contract_report["constant"],
    "feature",
].tolist()

if constant_features:
    raise RuntimeError(
        "Frozen V5-H feature contract has constant feature(s): "
        f"{constant_features}"
    )

outer_cv = make_outer_group_cv(OUTER_SPLITS)
groups = frame["seed"].to_numpy()
oof = frame.loc[
    :,
    [
        *KEY_COLUMNS,
        "target_rank",
        UTILITY_TARGET,
        "relevant_discovery_delta",
        "irrelevant_exposure_delta",
        "false_warmup_delta",
        "worst_scenario_utility_delta",
        "p10_scenario_replication_utility_delta",
        "benefit_positive",
        "no_irrelevant_increase",
        "no_false_warmup_increase",
        "nonnegative_worst_utility",
        "nonnegative_p10_utility",
        *RISK_TARGETS.keys(),
    ],
].copy()

oof["utility_prediction_baseline"] = np.nan
oof["utility_prediction_model"] = np.nan

for risk_name in RISK_TARGETS:
    oof[f"{risk_name}_probability_baseline"] = np.nan
    oof[f"{risk_name}_probability_model"] = np.nan

fold_rows: list[dict[str, Any]] = []
utility_inner_tables: list[pd.DataFrame] = []
risk_inner_tables: list[pd.DataFrame] = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(frame, groups=groups),
    start=1,
):
    train = frame.iloc[train_idx].copy()
    test = frame.iloc[test_idx].copy()

    train_seeds = sorted(
        int(seed)
        for seed in train["seed"].unique()
    )
    test_seeds = sorted(
        int(seed)
        for seed in test["seed"].unique()
    )

    if set(train_seeds) & set(test_seeds):
        raise RuntimeError(
            f"Outer fold {fold} leaks seeds."
        )

    utility_candidate, utility_table = (
        choose_utility_candidate(train)
    )

    utility_table.insert(0, "outer_fold", fold)
    utility_inner_tables.append(utility_table)

    utility_name, utility_family, utility_params = (
        utility_candidate
    )

    utility_model = make_utility_model(
        utility_family,
        utility_params,
    )

    utility_model.fit(
        train.loc[:, list(FEATURE_COLUMNS)],
        train[UTILITY_TARGET],
    )

    baseline_utility = action_mean_prediction(
        train,
        test,
    )

    model_utility = utility_model.predict(
        test.loc[:, list(FEATURE_COLUMNS)]
    )

    oof.loc[
        oof.index[test_idx],
        "utility_prediction_baseline",
    ] = baseline_utility

    oof.loc[
        oof.index[test_idx],
        "utility_prediction_model",
    ] = model_utility

    chosen_risks: dict[str, str] = {}

    for risk_name in RISK_TARGETS:
        risk_candidate, risk_table = (
            choose_risk_candidate(
                train,
                risk_name,
            )
        )

        risk_table.insert(0, "outer_fold", fold)
        risk_table.insert(1, "risk_head", risk_name)
        risk_inner_tables.append(risk_table)

        risk_name_candidate, risk_family, risk_params = (
            risk_candidate
        )

        probability_model = predict_risk(
            train,
            test,
            family=risk_family,
            params=risk_params,
            risk_target=risk_name,
        )

        probability_baseline = action_rate_prediction(
            train,
            test,
            risk_name,
        )

        oof.loc[
            oof.index[test_idx],
            f"{risk_name}_probability_baseline",
        ] = probability_baseline

        oof.loc[
            oof.index[test_idx],
            f"{risk_name}_probability_model",
        ] = probability_model

        chosen_risks[risk_name] = risk_name_candidate

    fold_rows.append(
        {
            "outer_fold": fold,
            "train_seed_count": len(train_seeds),
            "test_seed_count": len(test_seeds),
            "train_seeds": "|".join(
                str(seed)
                for seed in train_seeds
            ),
            "test_seeds": "|".join(
                str(seed)
                for seed in test_seeds
            ),
            "utility_candidate": utility_name,
            "utility_family": utility_family,
            "utility_params_json": json.dumps(
                utility_params,
                sort_keys=True,
            ),
            **{
                f"{risk_name}_candidate": candidate
                for risk_name, candidate in chosen_risks.items()
            },
        }
    )

if oof[
    [
        "utility_prediction_baseline",
        "utility_prediction_model",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing utility OOF predictions."
    )

for risk_name in RISK_TARGETS:
    if oof[
        [
            f"{risk_name}_probability_baseline",
            f"{risk_name}_probability_model",
        ]
    ].isna().any().any():
        raise RuntimeError(
            f"Missing risk OOF probabilities for {risk_name}."
        )

baseline_selected, baseline_by_seed = policy_metrics(
    oof,
    prediction_column="utility_prediction_baseline",
    label="action_only_baseline",
)

model_selected, model_by_seed = policy_metrics(
    oof,
    prediction_column="utility_prediction_model",
    label="rich_preaction_model",
)

policy_by_seed = baseline_by_seed.merge(
    model_by_seed,
    on="seed",
    how="inner",
    validate="one_to_one",
    suffixes=("_baseline", "_model"),
)

policy_delta_columns = {
    "selected_mean_utility": "selected_mean_utility_delta_model_minus_baseline",
    "selected_relevant_discovery": "selected_relevant_discovery_delta_model_minus_baseline",
    "selected_irrelevant_exposure": "selected_irrelevant_exposure_delta_model_minus_baseline",
    "selected_false_warmup": "selected_false_warmup_delta_model_minus_baseline",
    "selected_worst_utility": "selected_worst_utility_delta_model_minus_baseline",
    "selected_p10_utility": "selected_p10_utility_delta_model_minus_baseline",
    "mean_utility_regret": "mean_utility_regret_delta_model_minus_baseline",
    "oracle_action_hit_rate": "oracle_action_hit_rate_delta_model_minus_baseline",
    "strict_safe_improving_rate": "strict_safe_improving_rate_delta_model_minus_baseline",
    "core_safe_improving_rate": "core_safe_improving_rate_delta_model_minus_baseline",
}

for base_name, delta_name in policy_delta_columns.items():
    policy_by_seed[delta_name] = (
        policy_by_seed[f"{base_name}_model"]
        - policy_by_seed[f"{base_name}_baseline"]
    )

utility_bootstrap = bootstrap_seed_mean_delta(
    policy_by_seed,
    delta_column=(
        "selected_mean_utility_delta_model_minus_baseline"
    ),
)

regret_bootstrap = bootstrap_seed_mean_delta(
    policy_by_seed,
    delta_column=(
        "mean_utility_regret_delta_model_minus_baseline"
    ),
)

policy_summary = {
    "baseline": {
        "selected_mean_utility": float(
            baseline_selected[UTILITY_TARGET].mean()
        ),
        "mean_utility_regret": float(
            baseline_selected["utility_regret"].mean()
        ),
        "oracle_action_hit_rate": float(
            baseline_selected["oracle_action_hit"].mean()
        ),
        "strict_safe_improving_rate": float(
            baseline_selected[
                "strict_safe_improving"
            ].mean()
        ),
        "core_safe_improving_rate": float(
            baseline_selected[
                "core_safe_improving"
            ].mean()
        ),
        "selected_action_counts": (
            baseline_selected["action"]
            .value_counts()
            .reindex(ACTION_ORDER, fill_value=0)
            .astype(int)
            .to_dict()
        ),
    },
    "rich_preaction_model": {
        "selected_mean_utility": float(
            model_selected[UTILITY_TARGET].mean()
        ),
        "mean_utility_regret": float(
            model_selected["utility_regret"].mean()
        ),
        "oracle_action_hit_rate": float(
            model_selected["oracle_action_hit"].mean()
        ),
        "strict_safe_improving_rate": float(
            model_selected[
                "strict_safe_improving"
            ].mean()
        ),
        "core_safe_improving_rate": float(
            model_selected[
                "core_safe_improving"
            ].mean()
        ),
        "selected_action_counts": (
            model_selected["action"]
            .value_counts()
            .reindex(ACTION_ORDER, fill_value=0)
            .astype(int)
            .to_dict()
        ),
    },
    "utility_model_minus_baseline_seed_bootstrap": (
        utility_bootstrap
    ),
    "regret_model_minus_baseline_seed_bootstrap": (
        regret_bootstrap
    ),
}

utility_row_metrics = {
    "baseline_mae": float(
        mean_absolute_error(
            oof[UTILITY_TARGET],
            oof["utility_prediction_baseline"],
        )
    ),
    "model_mae": float(
        mean_absolute_error(
            oof[UTILITY_TARGET],
            oof["utility_prediction_model"],
        )
    ),
}

risk_summary_rows: list[dict[str, Any]] = []

for risk_name in RISK_TARGETS:
    labels = oof[risk_name].to_numpy(dtype=int)
    baseline_probability = oof[
        f"{risk_name}_probability_baseline"
    ].to_numpy(dtype=float)

    model_probability = oof[
        f"{risk_name}_probability_model"
    ].to_numpy(dtype=float)

    baseline_brier = float(
        brier_score_loss(
            labels,
            baseline_probability,
        )
    )

    model_brier = float(
        brier_score_loss(
            labels,
            model_probability,
        )
    )

    brier_skill = (
        1.0 - (model_brier / baseline_brier)
        if baseline_brier > 0.0
        else None
    )

    risk_summary_rows.append(
        {
            "risk_head": risk_name,
            "positive_rate": float(labels.mean()),
            "baseline_brier": baseline_brier,
            "model_brier": model_brier,
            "brier_skill": brier_skill,
            "baseline_auc": safe_auc(
                labels,
                baseline_probability,
            ),
            "model_auc": safe_auc(
                labels,
                model_probability,
            ),
        }
    )

risk_summary = pd.DataFrame(
    risk_summary_rows
)

risk_heads_with_positive_skill = int(
    risk_summary["brier_skill"].fillna(-np.inf).ge(0.02).sum()
)

utility_signal_present = bool(
    utility_bootstrap["ci_low_95"] > 0.0
    and regret_bootstrap["ci_high_95"] < 0.0
)

risk_signal_present = bool(
    risk_heads_with_positive_skill >= 2
)

if utility_signal_present and risk_signal_present:
    status = "V5H_RICH_PREACTION_RUNTIME_SIGNAL_PRESENT"
elif utility_signal_present:
    status = "V5H_RICH_PREACTION_UTILITY_SIGNAL_ONLY"
else:
    status = "V5H_RICH_PREACTION_RUNTIME_SIGNAL_INSUFFICIENT"

output_root.mkdir(parents=True, exist_ok=True)

oof_path = (
    output_root
    / "v5h_outer_fold_oof_predictions.csv"
)

baseline_selected_path = (
    output_root
    / "v5h_action_only_baseline_selected_actions.csv"
)

model_selected_path = (
    output_root
    / "v5h_rich_preaction_model_selected_actions.csv"
)

policy_seed_path = (
    output_root
    / "v5h_policy_metrics_by_seed.csv"
)

fold_ledger_path = (
    output_root
    / "v5h_outer_fold_model_ledger.csv"
)

utility_inner_path = (
    output_root
    / "v5h_utility_inner_selection_ledger.csv"
)

risk_inner_path = (
    output_root
    / "v5h_risk_inner_selection_ledger.csv"
)

risk_summary_path = (
    output_root
    / "v5h_risk_head_oof_summary.csv"
)

feature_contract_path = (
    output_root
    / "v5h_frozen_feature_contract.csv"
)

oof.to_csv(oof_path, index=False)
baseline_selected.to_csv(
    baseline_selected_path,
    index=False,
)
model_selected.to_csv(
    model_selected_path,
    index=False,
)
policy_by_seed.to_csv(policy_seed_path, index=False)
pd.DataFrame(fold_rows).to_csv(
    fold_ledger_path,
    index=False,
)
pd.concat(
    utility_inner_tables,
    ignore_index=True,
).to_csv(
    utility_inner_path,
    index=False,
)
pd.concat(
    risk_inner_tables,
    ignore_index=True,
).to_csv(
    risk_inner_path,
    index=False,
)
risk_summary.to_csv(
    risk_summary_path,
    index=False,
)
feature_contract_report.to_csv(
    feature_contract_path,
    index=False,
)

decision = {
    "status": status,
    "baseline_commit": expected_head,
    "v5d_decision_path": str(v5d_decision_path),
    "v5d_decision_sha256": sha256(v5d_decision_path),
    "v5e_decision_path": str(v5e_decision_path),
    "v5e_decision_sha256": sha256(v5e_decision_path),
    "v5f_decision_path": str(v5f_decision_path),
    "v5f_decision_sha256": sha256(v5f_decision_path),
    "v5g_decision_path": str(v5g_decision_path),
    "v5g_decision_sha256": sha256(v5g_decision_path),
    "recovery_decision_path": str(recovery_decision_path),
    "recovery_decision_sha256": sha256(
        recovery_decision_path
    ),
    "state_action_path": str(state_action_path),
    "state_action_sha256": sha256(state_action_path),
    "action_effects_path": str(action_effects_path),
    "action_effects_sha256": sha256(action_effects_path),
    "scikit_learn_version": sklearn.__version__,
    "outer_cv": {
        "splitter": "GroupKFold",
        "n_splits": OUTER_SPLITS,
        "group": "seed",
        "shuffle": True,
        "random_state": RANDOM_STATE,
    },
    "inner_cv": {
        "splitter": "GroupKFold",
        "n_splits": INNER_SPLITS,
        "group": "seed",
        "shuffle": True,
        "random_state": RANDOM_STATE,
    },
    "feature_contract": {
        "context_numeric_count": len(
            CONTEXT_NUMERIC_FEATURES
        ),
        "action_numeric_count": len(
            ACTION_NUMERIC_FEATURES
        ),
        "categorical_count": len(CATEGORICAL_FEATURES),
        "total_feature_column_count": len(
            FEATURE_COLUMNS
        ),
        "context_numeric": list(
            CONTEXT_NUMERIC_FEATURES
        ),
        "action_numeric": list(
            ACTION_NUMERIC_FEATURES
        ),
        "categorical": list(CATEGORICAL_FEATURES),
        "contract_selection_rule": (
            "Fixed before V5-H outcome modeling. The contract "
            "uses only recovered pre-action ranked-frame fields "
            "and source-native rank geometry. It excludes "
            "teacher/oracle/future/outcome/behavior and "
            "post-gate scoring fields."
        ),
    },
    "utility_model_selection": {
        "candidates": [
            {
                "name": name,
                "family": family,
                "params": params,
            }
            for name, family, params in UTILITY_CANDIDATES
        ],
        "inner_objective": (
            "maximize observed mean utility of the action selected "
            "within held-out inner-fold contexts; tie-break by "
            "lower selected-action regret, then lower row MAE."
        ),
        "row_level_oof_metrics": utility_row_metrics,
        "policy_summary": policy_summary,
    },
    "risk_model_selection": {
        "risk_heads": list(RISK_TARGETS),
        "candidates": [
            {
                "name": name,
                "family": family,
                "params": params,
            }
            for name, family, params in RISK_CANDIDATES
        ],
        "inner_objective": "minimize Brier score",
        "oof_summary": risk_summary.to_dict(
            orient="records"
        ),
        "positive_brier_skill_head_count": (
            risk_heads_with_positive_skill
        ),
    },
    "signal_gate": {
        "utility_signal_present": utility_signal_present,
        "risk_signal_present": risk_signal_present,
        "qualification_rule": (
            "Utility signal requires a positive seed-bootstrap "
            "95% lower bound for selected mean utility delta and "
            "a negative seed-bootstrap 95% upper bound for regret "
            "delta. Risk signal requires at least two of four "
            "risk heads with OOF Brier skill >= 0.02 versus an "
            "action-only prevalence baseline."
        ),
    },
    "artifacts": {
        "oof_predictions_path": str(oof_path),
        "oof_predictions_sha256": sha256(oof_path),
        "baseline_selected_actions_path": (
            str(baseline_selected_path)
        ),
        "model_selected_actions_path": str(model_selected_path),
        "policy_by_seed_path": str(policy_seed_path),
        "outer_fold_ledger_path": str(fold_ledger_path),
        "utility_inner_selection_path": str(
            utility_inner_path
        ),
        "risk_inner_selection_path": str(risk_inner_path),
        "risk_summary_path": str(risk_summary_path),
        "feature_contract_path": str(feature_contract_path),
    },
    "final_serving_model_trained": False,
    "temporary_fold_models_fit_for_cv": True,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "source_modified": False,
    "config_modified": False,
    "dynamic_replay_executed_at_release_scale": False,
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Freeze a train-only candidate model family only if both "
        "utility and risk signal gates pass. Otherwise, inspect "
        "the pre-registered OOF result and do not fit a final "
        "artifact, choose a serving threshold, or execute "
        "calibration/confirmation seeds."
    ),
}

decision_path = (
    output_root
    / "v5h_rich_preaction_multihead_viability_decision.json"
)

write_json(decision_path, decision)

print("===== V5-H RICH PRE-ACTION MULTI-HEAD VIABILITY AUDIT =====")
print(
    json.dumps(
        {
            "decision": decision,
            "outer_fold_model_ledger": fold_rows,
            "risk_summary": risk_summary.to_dict(
                orient="records"
            ),
        },
        indent=2,
        sort_keys=True,
        default=native,
    )
)

print("===== V5-H DECISION =====")
print(
    json.dumps(
        {
            "status": status,
            "utility_signal_present": utility_signal_present,
            "risk_signal_present": risk_signal_present,
            "utility_model_minus_baseline_seed_bootstrap": (
                utility_bootstrap
            ),
            "regret_model_minus_baseline_seed_bootstrap": (
                regret_bootstrap
            ),
            "risk_heads_with_brier_skill_ge_0p02": (
                risk_heads_with_positive_skill
            ),
        },
        indent=2,
        sort_keys=True,
    )
)