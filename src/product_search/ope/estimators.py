from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def generate_known_propensity_lab(
    seed: int = 42, n: int = 5000, n_actions: int = 5
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    action_weights = rng.normal(scale=0.7, size=(n_actions, 3))
    reward_weights = rng.normal(scale=0.8, size=(n_actions, 3))
    logging_logits = x @ action_weights.T
    logging_prob = _softmax(logging_logits / 1.3)
    candidate_logits = logging_logits + 0.55 * (x @ reward_weights.T)
    candidate_prob = _softmax(candidate_logits / 0.9)
    actions = np.array([rng.choice(n_actions, p=p) for p in logging_prob])
    reward_prob_all = 1 / (1 + np.exp(-(x @ reward_weights.T)))
    rewards = np.array(
        [rng.random() < reward_prob_all[i, a] for i, a in enumerate(actions)], dtype=float
    )
    rows = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    rows["action"] = actions
    rows["reward"] = rewards
    rows["logging_propensity"] = logging_prob[np.arange(n), actions]
    rows["candidate_propensity"] = candidate_prob[np.arange(n), actions]
    rows["true_candidate_value"] = np.sum(candidate_prob * reward_prob_all, axis=1)
    rows.attrs["candidate_prob"] = candidate_prob
    rows.attrs["reward_prob_all"] = reward_prob_all
    return rows


def _cross_fitted_outcomes(
    frame: pd.DataFrame,
    candidate_prob: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    reward = frame.reward.to_numpy(dtype=int)
    n_actions = candidate_prob.shape[1]
    mu_logged = np.zeros(len(frame), dtype=float)
    dm_values = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for train_index, validation_index in splitter.split(frame, reward):
        train = frame.iloc[train_index]
        validation = frame.iloc[validation_index]
        model = LogisticRegression(max_iter=750, random_state=seed, solver="lbfgs")
        model.fit(train[["x0", "x1", "x2", "action"]], train.reward)
        mu_logged[validation_index] = model.predict_proba(
            validation[["x0", "x1", "x2", "action"]]
        )[:, 1]
        mu_all = np.zeros((len(validation_index), n_actions), dtype=float)
        for action in range(n_actions):
            action_frame = validation[["x0", "x1", "x2"]].copy()
            action_frame["action"] = action
            mu_all[:, action] = model.predict_proba(action_frame)[:, 1]
        dm_values[validation_index] = np.sum(
            candidate_prob[validation_index] * mu_all, axis=1
        )
    return mu_logged, dm_values


def _mean_ci(values: np.ndarray, *, seed: int, samples: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(np.mean(values[rng.integers(0, n, n)]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def estimate_ope(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    clip: float = 20.0,
    seed: int = 42,
) -> dict[str, float]:
    if "candidate_prob" not in frame.attrs:
        raise ValueError("Known-propensity lab is missing candidate-policy probabilities")
    logging = np.clip(frame.logging_propensity.to_numpy(dtype=float), 1e-8, None)
    candidate = frame.candidate_propensity.to_numpy(dtype=float)
    w = candidate / logging
    reward = frame.reward.to_numpy(dtype=float)
    clipped = np.minimum(w, clip)
    ips_values = w * reward
    clipped_ips_values = clipped * reward
    ips = float(np.mean(ips_values))
    snips = float(np.sum(ips_values) / max(np.sum(w), 1e-12))
    clipped_ips = float(np.mean(clipped_ips_values))

    candidate_prob = np.asarray(frame.attrs["candidate_prob"], dtype=float)
    mu_logged, dm_values = _cross_fitted_outcomes(
        frame, candidate_prob, folds=folds, seed=seed
    )
    dm = float(dm_values.mean())
    dr_values = dm_values + w * (reward - mu_logged)
    dr = float(dr_values.mean())
    true_values = frame.true_candidate_value.to_numpy(dtype=float)
    true_value = float(true_values.mean())
    ess = float((w.sum() ** 2) / max(np.sum(w**2), 1e-12))
    dr_low, dr_high = _mean_ci(dr_values, seed=seed + 11)
    true_low, true_high = _mean_ci(true_values, seed=seed + 12)
    support_overlap = float(np.mean(logging > 1e-4))
    return {
        "true_value": true_value,
        "true_value_ci_low": true_low,
        "true_value_ci_high": true_high,
        "dm": dm,
        "ips": ips,
        "snips": snips,
        "clipped_ips": clipped_ips,
        "dr": dr,
        "dr_ci_low": dr_low,
        "dr_ci_high": dr_high,
        "dr_covers_true_value": float(dr_low <= true_value <= dr_high),
        "dm_abs_error": abs(dm - true_value),
        "ips_abs_error": abs(ips - true_value),
        "snips_abs_error": abs(snips - true_value),
        "dr_abs_error": abs(dr - true_value),
        "effective_sample_size": ess,
        "weight_max": float(w.max()),
        "weight_mean": float(w.mean()),
        "weight_clipped_rate": float(np.mean(w > clip)),
        "support_overlap_rate": support_overlap,
        "cross_fit_folds": float(folds),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)
