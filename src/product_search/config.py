from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_SECTIONS = ("retrieval", "ranking", "qrsbt", "simulation", "release")
_CONFIG_SCHEMA_VERSION = "6.0"
_TOP_LEVEL_KEYS = {
    "config_schema_version",
    "seed",
    "mode",
    "output_dir",
    "data",
    "synthetic",
    *_REQUIRED_SECTIONS,
}
_SECTION_KEYS = {
    "data": {"source", "canonical_dir"},
    "synthetic": {
        "n_products",
        "n_queries",
        "n_users",
        "n_time_blocks",
        "impressions_per_query_time",
    },
    "retrieval": {"bm25_k1", "bm25_b", "dense_dim", "candidate_k", "final_k"},
    "ranking": {
        "xgb_estimators",
        "dcn_epochs",
        "dcn_batch_size",
        "learning_rate",
        "semantic_weight",
        "behavior_weight",
    },
    "qrsbt": {
        "neighbors",
        "min_support",
        "semantic_threshold",
        "confidence_threshold",
        "compatibility_threshold",
        "irrelevant_risk_threshold",
        "max_boost",
        "promotion_window",
        "max_promotions_per_query",
    },
    "simulation": {"days", "traffic_per_day", "replications"},
}


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    source_path: Path

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "smoke"))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", "artifacts/smoke"))

    @property
    def data_source(self) -> str:
        return str(self.raw.get("data", {}).get("source", "synthetic"))

    @property
    def canonical_data_dir(self) -> Path:
        value = self.raw.get("data", {}).get("canonical_dir")
        if value is None:
            raise ValueError("data.canonical_dir is required for canonical data")
        path = Path(str(value))
        return path if path.is_absolute() else (self.source_path.parent / path).resolve()


def _require_number(mapping: dict[str, Any], key: str, *, low: float | None = None) -> float:
    if (
        key not in mapping
        or isinstance(mapping[key], bool)
        or not isinstance(mapping[key], (int, float))
    ):
        raise ValueError(f"Configuration value '{key}' must be numeric")
    value = float(mapping[key])
    if low is not None and value < low:
        raise ValueError(f"Configuration value '{key}' must be >= {low}")
    return value


def _require_integer(mapping: dict[str, Any], key: str, *, low: int | None = None) -> int:
    if key not in mapping or isinstance(mapping[key], bool) or not isinstance(mapping[key], int):
        raise ValueError(f"Configuration value '{key}' must be an integer")
    value = int(mapping[key])
    if low is not None and value < low:
        raise ValueError(f"Configuration value '{key}' must be >= {low}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], *, section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration values in {section}: {unknown}")


def validate_config(raw: dict[str, Any]) -> None:
    _reject_unknown(raw, _TOP_LEVEL_KEYS, section="root")
    if str(raw.get("config_schema_version", "")) != _CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"config_schema_version must be {_CONFIG_SCHEMA_VERSION!r} for this package version"
        )
    missing = [section for section in _REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise ValueError(f"Configuration missing sections: {missing}")
    for section in _REQUIRED_SECTIONS:
        if not isinstance(raw[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
    if isinstance(raw.get("seed", 42), bool) or not isinstance(raw.get("seed", 42), int):
        raise ValueError("Configuration 'seed' must be an integer")
    if int(raw.get("seed", 42)) < 0:
        raise ValueError("Configuration 'seed' must be nonnegative")
    output = str(raw.get("output_dir", "")).strip()
    if not output:
        raise ValueError("Configuration 'output_dir' must not be empty")

    data = raw.get("data", {"source": "synthetic"})
    if not isinstance(data, dict):
        raise ValueError("Configuration section 'data' must be a mapping")
    _reject_unknown(data, _SECTION_KEYS["data"], section="data")
    source = str(data.get("source", "synthetic"))
    if source not in {"synthetic", "canonical"}:
        raise ValueError("data.source must be either 'synthetic' or 'canonical'")
    if source == "synthetic":
        if "synthetic" not in raw or not isinstance(raw["synthetic"], dict):
            raise ValueError("synthetic configuration is required for synthetic data")
        synthetic = raw["synthetic"]
        _reject_unknown(synthetic, _SECTION_KEYS["synthetic"], section="synthetic")
        for key in (
            "n_products",
            "n_queries",
            "n_users",
            "n_time_blocks",
            "impressions_per_query_time",
        ):
            _require_integer(synthetic, key, low=1)
        if int(synthetic["n_time_blocks"]) < 5:
            raise ValueError(
                "At least five time blocks are required for train/validation/test isolation"
            )
    else:
        if not str(data.get("canonical_dir", "")).strip():
            raise ValueError("data.canonical_dir is required for canonical data")

    retrieval = raw["retrieval"]
    _reject_unknown(retrieval, _SECTION_KEYS["retrieval"], section="retrieval")
    _require_number(retrieval, "bm25_k1", low=1e-12)
    bm25_b = _require_number(retrieval, "bm25_b", low=0)
    if bm25_b > 1:
        raise ValueError("retrieval.bm25_b must be <= 1")
    _require_integer(retrieval, "dense_dim", low=2)
    _require_integer(retrieval, "candidate_k", low=2)
    _require_integer(retrieval, "final_k", low=1)
    if int(retrieval["candidate_k"]) < int(retrieval["final_k"]):
        raise ValueError("retrieval.candidate_k must be >= retrieval.final_k")
    if source == "synthetic" and int(retrieval["candidate_k"]) > int(
        raw["synthetic"]["n_products"]
    ):
        raise ValueError("retrieval.candidate_k cannot exceed synthetic.n_products")

    qrsbt = raw["qrsbt"]
    _reject_unknown(
        qrsbt,
        _SECTION_KEYS["qrsbt"] | {"promotion_mode"},
        section="qrsbt",
    )
    for key in (
        "semantic_threshold",
        "confidence_threshold",
        "compatibility_threshold",
        "irrelevant_risk_threshold",
        "max_boost",
    ):
        value = _require_number(qrsbt, key, low=0)
        if value > 1:
            raise ValueError(f"qrsbt.{key} must be <= 1")
    _require_integer(qrsbt, "neighbors", low=1)
    _require_integer(qrsbt, "min_support", low=1)
    _require_integer(qrsbt, "promotion_window", low=0)
    _require_integer(qrsbt, "max_promotions_per_query", low=1)
    if int(qrsbt["min_support"]) > int(qrsbt["neighbors"]):
        raise ValueError("qrsbt.min_support cannot exceed qrsbt.neighbors")

    promotion_mode = str(qrsbt.get("promotion_mode", "in_window"))
    if promotion_mode not in {"in_window", "boundary_entry_only"}:
        raise ValueError("qrsbt.promotion_mode must be one of: in_window, boundary_entry_only")

    ranking = raw["ranking"]
    _reject_unknown(ranking, _SECTION_KEYS["ranking"], section="ranking")
    _require_integer(ranking, "xgb_estimators", low=1)
    _require_integer(ranking, "dcn_epochs", low=1)
    _require_integer(ranking, "dcn_batch_size", low=1)
    _require_number(ranking, "learning_rate", low=1e-12)
    semantic_weight = _require_number(ranking, "semantic_weight", low=0)
    behavior_weight = _require_number(ranking, "behavior_weight", low=0)
    if semantic_weight + behavior_weight <= 0:
        raise ValueError("At least one ranking fusion weight must be positive")
    if abs((semantic_weight + behavior_weight) - 1.0) > 1e-9:
        raise ValueError("ranking semantic_weight and behavior_weight must sum to 1")

    simulation = raw["simulation"]
    _reject_unknown(simulation, _SECTION_KEYS["simulation"], section="simulation")
    _require_integer(simulation, "days", low=1)
    _require_integer(simulation, "traffic_per_day", low=1)
    _require_integer(simulation, "replications", low=1)

    release = raw["release"]
    required_release = (
        "max_overall_ndcg_drop",
        "max_overall_ndcg_drop_ci",
        "min_cold_ndcg_lift",
        "min_cold_ndcg_lift_ci_low",
        "max_warm_ndcg_drop",
        "max_warm_ndcg_drop_ci",
        "max_irrelevant_exposure_increase",
        "max_irrelevant_exposure_ci_high",
        "min_ope_ess",
        "max_ope_dr_abs_error",
        "min_ope_support_overlap",
        "max_dynamic_relevant_discovery_drop_rate",
        "max_dynamic_false_warmup_increase",
        "max_dynamic_irrelevant_exposure_increase",
        "min_worst_scenario_utility_delta",
        "min_p10_scenario_replication_utility_delta",
        "max_serving_p95_ms",
        "min_judgment_coverage_at_10",
        "min_future_behavior_roc_auc",
        "max_future_behavior_brier",
        "max_future_behavior_ece",
        "max_future_logged_ndcg_drop",
        "max_relation_eligibility_brier",
        "max_relation_eligibility_ece",
    )
    missing_release = [key for key in required_release if key not in release]
    if missing_release:
        raise ValueError(f"release section missing values: {missing_release}")
    _reject_unknown(release, set(required_release), section="release")
    for key in required_release:
        _require_number(release, key)
    if not 0 <= float(release["min_ope_support_overlap"]) <= 1:
        raise ValueError("release.min_ope_support_overlap must be in [0, 1]")
    if not 0 <= float(release["min_judgment_coverage_at_10"]) <= 1:
        raise ValueError("release.min_judgment_coverage_at_10 must be in [0, 1]")
    if not 0 <= float(release["min_future_behavior_roc_auc"]) <= 1:
        raise ValueError("release.min_future_behavior_roc_auc must be in [0, 1]")
    for key in (
        "max_future_behavior_brier",
        "max_future_behavior_ece",
        "max_relation_eligibility_brier",
        "max_relation_eligibility_ece",
    ):
        if not 0 <= float(release[key]) <= 1:
            raise ValueError(f"release.{key} must be in [0, 1]")
    for key in (
        "max_overall_ndcg_drop",
        "max_overall_ndcg_drop_ci",
        "max_warm_ndcg_drop",
        "max_warm_ndcg_drop_ci",
        "max_irrelevant_exposure_increase",
        "max_irrelevant_exposure_ci_high",
        "max_ope_dr_abs_error",
        "max_dynamic_relevant_discovery_drop_rate",
        "max_dynamic_false_warmup_increase",
        "max_dynamic_irrelevant_exposure_increase",
        "max_serving_p95_ms",
        "max_future_behavior_brier",
        "max_future_behavior_ece",
        "max_future_logged_ndcg_drop",
        "max_relation_eligibility_brier",
        "max_relation_eligibility_ece",
    ):
        if float(release[key]) < 0:
            raise ValueError(f"release.{key} must be nonnegative")
    if float(release["min_ope_ess"]) < 1:
        raise ValueError("release.min_ope_ess must be >= 1")


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(raw)
    return ProjectConfig(raw=raw, source_path=source)
