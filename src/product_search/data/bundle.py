from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from product_search.provenance import atomic_write_csv

from .contracts import (
    INTERACTIONS,
    PRODUCTS,
    QUERIES,
    RELEVANCE,
    validate_bundle_foreign_keys,
)


@dataclass
class DataBundle:
    """Canonical in-memory contract shared by synthetic and public-data pipelines."""

    products: pd.DataFrame
    queries: pd.DataFrame
    relevance: pd.DataFrame
    interactions: pd.DataFrame

    def validate(self) -> None:
        PRODUCTS.validate(self.products)
        QUERIES.validate(self.queries)
        RELEVANCE.validate(self.relevance)
        INTERACTIONS.validate(self.interactions)
        validate_bundle_foreign_keys(
            self.products, self.queries, self.relevance, self.interactions
        )

    def write(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(destination / "products.csv", self.products)
        atomic_write_csv(destination / "queries.csv", self.queries)
        atomic_write_csv(destination / "relevance.csv", self.relevance)
        atomic_write_csv(destination / "interactions.csv", self.interactions)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "DataBundle":
        source = Path(directory)
        if not source.exists():
            raise FileNotFoundError(f"Canonical data directory does not exist: {source}")
        required = {
            "products": source / "products.csv",
            "queries": source / "queries.csv",
            "relevance": source / "relevance.csv",
            "interactions": source / "interactions.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Canonical data files are missing: {missing}")
        bundle = cls(**{name: pd.read_csv(path) for name, path in required.items()})
        bundle.validate()
        return bundle
