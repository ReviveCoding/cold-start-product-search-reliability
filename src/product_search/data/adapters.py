from __future__ import annotations

from pathlib import Path

import pandas as pd


class KuaiSearchAdapter:
    """Normalize user-downloaded KuaiSearch(-Lite) tables into project contracts.

    The public release contains recall, relevance, ranking, user, and item tables. File names
    can vary with snapshots, so this adapter accepts explicit paths and maps common field names.
    """

    @staticmethod
    def read_items(
        path: str | Path, *, launch_block_column: str | None = None
    ) -> pd.DataFrame:
        frame = _read_table(path).copy()
        rename = {
            "item_id": "product_id",
            "brand_name": "brand",
            "category_l3": "category",
        }
        frame = frame.rename(columns=rename)
        required = ["product_id", "title"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"KuaiSearch item table missing {missing}")
        frame["product_id"] = pd.to_numeric(frame["product_id"], errors="raise").astype("int64")
        if frame["product_id"].duplicated().any():
            raise ValueError("KuaiSearch item table contains duplicate product IDs")
        frame["title"] = frame["title"].astype(str)
        if launch_block_column is not None:
            if launch_block_column not in frame.columns:
                raise ValueError(f"KuaiSearch item table missing launch column {launch_block_column!r}")
            frame = frame.rename(columns={launch_block_column: "launch_block"})
        if "launch_block" not in frame.columns:
            raise ValueError(
                "KuaiSearch item metadata has no catalog launch time. Derive first-observed "
                "launch_block from a temporally normalized ranking table before creating a "
                "canonical cold-start bundle."
            )
        frame["launch_block"] = pd.to_numeric(frame["launch_block"], errors="raise").astype("int64")
        if (frame["launch_block"] < 0).any():
            raise ValueError("KuaiSearch launch_block must be nonnegative")
        for column, default in {
            "brand": "unknown",
            "category": "unknown",
            "attribute": "unknown",
            "model": "unknown",
            "color": "unknown",
            "price": 0.0,
            "quality": 0.5,
        }.items():
            if column not in frame.columns:
                frame[column] = default
        return frame

    @staticmethod
    def derive_first_observed_launch_blocks(ranking: pd.DataFrame) -> pd.DataFrame:
        required = {"product_id", "time_block"}
        missing = sorted(required - set(ranking.columns))
        if missing:
            raise ValueError(f"Temporal ranking table missing {missing}")
        observed = (
            ranking[["product_id", "time_block"]]
            .assign(
                product_id=lambda x: pd.to_numeric(x.product_id, errors="raise").astype("int64"),
                time_block=lambda x: pd.to_numeric(x.time_block, errors="raise").astype("int64"),
            )
            .groupby("product_id", as_index=False, sort=True)["time_block"]
            .min()
            .rename(columns={"time_block": "launch_block"})
        )
        return observed

    @staticmethod
    def read_recall(path: str | Path) -> pd.DataFrame:
        frame = _read_table(path)
        rename = {
            "item_id": "product_id",
            "timestamp": "time_block",
        }
        frame = frame.rename(columns=rename)
        required = ["user_id", "session_id", "query"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"KuaiSearch recall table missing {missing}")
        return frame

    @staticmethod
    def read_ranking(
        path: str | Path, *, time_column: str | None = None
    ) -> pd.DataFrame:
        """Normalize a ranking snapshot without inventing temporal or propensity fields.

        KuaiSearch snapshots have changed field names.  Callers must either provide ``time_column``
        or include one of the recognized relative-time columns.  The adapter deliberately fails
        rather than assigning every event to time zero, because that would invalidate temporal
        cold-start and leakage claims.
        """
        frame = _read_table(path).copy()
        if "target_item_id" in frame.columns:
            frame = frame.rename(columns={"target_item_id": "product_id"})
            if "item_id" in frame.columns:
                frame = frame.drop(columns=["item_id"])
        elif "item_id" in frame.columns:
            frame = frame.rename(columns={"item_id": "product_id"})
        if time_column is not None:
            if time_column not in frame.columns:
                raise ValueError(f"KuaiSearch ranking table missing time column {time_column!r}")
            frame = frame.rename(columns={time_column: "time_block"})
        elif "time_block" not in frame.columns:
            recognized = next(
                (
                    column
                    for column in ("relative_time", "time_idx", "timestamp")
                    if column in frame.columns
                ),
                None,
            )
            if recognized is None:
                raise ValueError(
                    "KuaiSearch ranking table requires an explicit temporal column; pass "
                    "time_column=... rather than fabricating launch or event time"
                )
            frame = frame.rename(columns={recognized: "time_block"})
        frame = frame.rename(
            columns={
                "is_clicked": "clicked",
                "is_purchased": "purchased",
            }
        )
        required = ["time_block", "user_id", "query", "product_id", "clicked", "purchased"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"KuaiSearch ranking table missing {missing}")
        for column in ("time_block", "user_id", "product_id", "clicked", "purchased"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (frame["time_block"] < 0).any():
            raise ValueError("KuaiSearch relative time must be nonnegative")
        if not frame["clicked"].isin([0, 1]).all() or not frame["purchased"].isin([0, 1]).all():
            raise ValueError("KuaiSearch click and purchase labels must be binary")
        if (frame["purchased"] > frame["clicked"]).any():
            raise ValueError("Purchased KuaiSearch rows must also be clicked")
        return frame

    @staticmethod
    def read_relevance(path: str | Path) -> pd.DataFrame:
        frame = _read_table(path).rename(
            columns={"item_id": "product_id", "score": "relevance", "brand_name": "brand"}
        )
        required = ["query", "title", "relevance"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"KuaiSearch relevance table missing {missing}")
        return frame


class ESCIAdapter:
    """Load joined ESCI examples/products into the external semantic benchmark schema."""

    LABEL_MAP = {"E": 3, "S": 2, "C": 1, "I": 0}

    @classmethod
    def read_joined(cls, examples_path: str | Path, products_path: str | Path) -> pd.DataFrame:
        examples = _read_table(examples_path)
        products = _read_table(products_path)
        required_examples = {"product_id", "product_locale", "query", "esci_label"}
        required_products = {"product_id", "product_locale", "product_title"}
        if missing := sorted(required_examples - set(examples.columns)):
            raise ValueError(f"ESCI examples table missing {missing}")
        if missing := sorted(required_products - set(products.columns)):
            raise ValueError(f"ESCI products table missing {missing}")
        frame = examples.merge(
            products, on=["product_id", "product_locale"], how="left", validate="many_to_one"
        )
        frame["relevance"] = frame["esci_label"].map(cls.LABEL_MAP)
        if frame["relevance"].isna().any():
            unknown = sorted(frame.loc[frame.relevance.isna(), "esci_label"].astype(str).unique())
            raise ValueError(f"Unsupported ESCI labels: {unknown}")
        if frame["product_title"].isna().any():
            raise ValueError("ESCI examples contain products missing from the product catalog")
        return frame


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")
