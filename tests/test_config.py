from copy import deepcopy

import pytest
import yaml

from product_search.config import load_config, validate_config


def _raw():
    return yaml.safe_load(open("configs/smoke.yaml", encoding="utf-8"))


def test_smoke_config_is_valid():
    config = load_config("configs/smoke.yaml")
    assert config.mode == "smoke"


def test_config_rejects_insufficient_temporal_blocks():
    raw = deepcopy(_raw())
    raw["synthetic"]["n_time_blocks"] = 4
    with pytest.raises(ValueError, match="five time blocks"):
        validate_config(raw)


def test_config_rejects_zero_fusion_weight():
    raw = deepcopy(_raw())
    raw["ranking"]["semantic_weight"] = 0
    raw["ranking"]["behavior_weight"] = 0
    with pytest.raises(ValueError, match="fusion weight"):
        validate_config(raw)


def test_config_requires_promotion_budget():
    raw = deepcopy(_raw())
    del raw["qrsbt"]["max_promotions_per_query"]
    with pytest.raises(ValueError):
        validate_config(raw)


def test_config_accepts_canonical_data_without_synthetic_section(tmp_path):
    raw = deepcopy(_raw())
    raw.pop("synthetic")
    raw["data"] = {"source": "canonical", "canonical_dir": "../canonical"}
    validate_config(raw)


def test_config_rejects_boolean_numeric_values():
    raw = deepcopy(_raw())
    raw["retrieval"]["candidate_k"] = True
    with pytest.raises(ValueError, match="integer"):
        validate_config(raw)


def test_config_rejects_fractional_integer_fields():
    raw = deepcopy(_raw())
    raw["qrsbt"]["neighbors"] = 3.5
    with pytest.raises(ValueError, match="integer"):
        validate_config(raw)


def test_config_rejects_support_larger_than_neighbor_count():
    raw = deepcopy(_raw())
    raw["qrsbt"]["neighbors"] = 2
    raw["qrsbt"]["min_support"] = 3
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_config(raw)


def test_config_rejects_unknown_keys_instead_of_silently_ignoring_typos():
    raw = deepcopy(_raw())
    raw["retrieval"]["candiate_k"] = 10
    with pytest.raises(ValueError, match="Unknown configuration"):
        validate_config(raw)


def test_config_requires_normalized_fusion_weights():
    raw = deepcopy(_raw())
    raw["ranking"]["semantic_weight"] = 0.8
    raw["ranking"]["behavior_weight"] = 0.4
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_config(raw)


def test_config_requires_current_schema_version():
    raw = deepcopy(_raw())
    raw["config_schema_version"] = "3.0"
    with pytest.raises(ValueError, match="config_schema_version"):
        validate_config(raw)


def test_dev_dependencies_use_official_httpx_package():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = payload["project"]["optional-dependencies"]["dev"]
    assert any(value.startswith("httpx>=") for value in dependencies)
    assert any(value.startswith("httpx2>=") for value in dependencies)


# Boundary-entry-only config regression tests


def test_config_accepts_boundary_entry_only_promotion_mode():
    from copy import deepcopy

    raw = deepcopy(_raw())
    raw["qrsbt"]["promotion_mode"] = "boundary_entry_only"

    validate_config(raw)


def test_config_rejects_unknown_promotion_mode():
    from copy import deepcopy

    raw = deepcopy(_raw())
    raw["qrsbt"]["promotion_mode"] = "unsupported_mode"

    with pytest.raises(ValueError, match="promotion_mode"):
        validate_config(raw)
