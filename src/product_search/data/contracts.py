from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


Validator = Callable[[pd.DataFrame], None]


@dataclass(frozen=True)
class FrameContract:
    name: str
    required_columns: tuple[str, ...]
    unique_columns: tuple[str, ...] = ()
    validators: tuple[Validator, ...] = ()

    def validate(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{self.name}: expected pandas DataFrame")
        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.name}: missing columns {missing}")
        if frame.empty:
            raise ValueError(f"{self.name}: frame is empty")
        null_columns = [column for column in self.required_columns if frame[column].isna().any()]
        if null_columns:
            raise ValueError(f"{self.name}: required columns contain nulls {null_columns}")
        for column in self.unique_columns:
            if frame[column].duplicated().any():
                raise ValueError(f"{self.name}: {column} must be unique")
        for validator in self.validators:
            validator(frame)


def _nonnegative(columns: tuple[str, ...]) -> Validator:
    def validate(frame: pd.DataFrame) -> None:
        bad = [column for column in columns if (pd.to_numeric(frame[column], errors="coerce") < 0).any()]
        if bad:
            raise ValueError(f"columns must be nonnegative: {bad}")

    return validate


def _bounded(column: str, low: float, high: float) -> Validator:
    def validate(frame: pd.DataFrame) -> None:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(low, high).all():
            raise ValueError(f"{column} must be in [{low}, {high}]")

    return validate


def _allowed(column: str, values: set[str]) -> Validator:
    def validate(frame: pd.DataFrame) -> None:
        observed = set(frame[column].astype(str).unique())
        if not observed.issubset(values):
            raise ValueError(f"{column} contains unsupported values: {sorted(observed - values)}")

    return validate


PRODUCTS = FrameContract(
    "products",
    (
        "product_id",
        "title",
        "brand",
        "category",
        "attribute",
        "model",
        "color",
        "price",
        "launch_block",
        "quality",
    ),
    ("product_id",),
    (_nonnegative(("product_id", "price", "launch_block")), _bounded("quality", 0.0, 1.0)),
)

def _optional_nonnegative(column: str) -> Validator:
    def validate(frame: pd.DataFrame) -> None:
        if column not in frame.columns:
            return
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"{column} must be nonnegative when provided")

    return validate


def _unique_pair(left: str, right: str) -> Validator:
    def validate(frame: pd.DataFrame) -> None:
        if frame.duplicated([left, right]).any():
            raise ValueError(f"({left}, {right}) pairs must be unique")

    return validate


QUERIES = FrameContract(
    "queries",
    ("query_id", "query", "intent", "category"),
    ("query_id",),
    (_nonnegative(("query_id",)), _optional_nonnegative("target_product_id")),
)

RELEVANCE = FrameContract(
    "relevance",
    ("query_id", "product_id", "relevance", "relation", "attribute_compatible"),
    validators=(
        _bounded("relevance", 0, 3),
        _bounded("attribute_compatible", 0, 1),
        _allowed("relation", {"exact", "substitute", "complement", "irrelevant"}),
        _unique_pair("query_id", "product_id"),
    ),
)

INTERACTIONS = FrameContract(
    "interactions",
    (
        "time_block",
        "user_id",
        "query_id",
        "product_id",
        "position",
        "impressed",
        "clicked",
        "purchased",
        "logging_propensity",
    ),
    validators=(
        _nonnegative(("time_block", "user_id", "query_id", "product_id", "position")),
        _bounded("impressed", 0, 1),
        _bounded("clicked", 0, 1),
        _bounded("purchased", 0, 1),
        _bounded("logging_propensity", np.finfo(float).eps, 1.0),
    ),
)


def validate_bundle_foreign_keys(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    relevance: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    product_ids = set(products.product_id)
    query_ids = set(queries.query_id)
    checks = {
        "relevance.product_id": set(relevance.product_id) - product_ids,
        "relevance.query_id": set(relevance.query_id) - query_ids,
        "interactions.product_id": set(interactions.product_id) - product_ids,
        "interactions.query_id": set(interactions.query_id) - query_ids,
    }
    if "target_product_id" in queries.columns:
        checks["queries.target_product_id"] = set(queries.target_product_id) - product_ids
    failures = {name: sorted(values)[:10] for name, values in checks.items() if values}
    if failures:
        raise ValueError(f"Foreign-key violations: {failures}")
    if (interactions.purchased > interactions.clicked).any():
        raise ValueError("Purchased interactions must also be clicked")
    if (interactions.clicked > interactions.impressed).any():
        raise ValueError("Clicked interactions must also be impressed")
