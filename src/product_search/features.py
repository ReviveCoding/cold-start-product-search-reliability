from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


BEHAVIOR_COLUMNS = [
    "prior_impressions",
    "prior_clicks",
    "prior_purchases",
    "smoothed_ctr",
    "smoothed_purchase_rate",
    "behavior_velocity",
    "first_observed_age",
    "zero_history",
    "sparse_history",
]


def build_temporal_behavior_features(
    interactions: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """Build product-level features using only blocks strictly before each observation block."""
    product_launch = products.set_index("product_id").launch_block.to_dict()
    stats = defaultdict(lambda: {"imp": 0.0, "click": 0.0, "purchase": 0.0, "velocity": 0.0})
    rows: list[dict] = []
    ordered = interactions.sort_values(["time_block", "query_id", "position", "product_id"]).copy()

    for time_block, block in ordered.groupby("time_block", sort=True):
        # Read all features before applying any event from the current time block. This prevents
        # within-block leakage even when the same product appears for multiple queries.
        for row in block.itertuples(index=False):
            state = stats[int(row.product_id)]
            imp = state["imp"]
            clicks = state["click"]
            purchases = state["purchase"]
            rows.append(
                {
                    "time_block": int(time_block),
                    "user_id": int(row.user_id),
                    "query_id": int(row.query_id),
                    "product_id": int(row.product_id),
                    "position": int(row.position),
                    "clicked": int(row.clicked),
                    "purchased": int(row.purchased),
                    "logging_propensity": float(row.logging_propensity),
                    "prior_impressions": imp,
                    "prior_clicks": clicks,
                    "prior_purchases": purchases,
                    "smoothed_ctr": (clicks + 1.0) / (imp + 5.0),
                    "smoothed_purchase_rate": (purchases + 0.5) / (imp + 8.0),
                    "behavior_velocity": state["velocity"],
                    "first_observed_age": max(
                        0, int(time_block) - int(product_launch[int(row.product_id)])
                    ),
                    "zero_history": int(clicks + purchases == 0),
                    "sparse_history": int(clicks + purchases <= 2),
                }
            )
        for row in block.itertuples(index=False):
            state = stats[int(row.product_id)]
            state["imp"] += float(row.impressed)
            state["click"] += float(row.clicked)
            state["purchase"] += float(row.purchased)
            state["velocity"] = 0.75 * state["velocity"] + float(row.clicked)
    return pd.DataFrame(rows)


def build_product_behavior_snapshot(
    interactions: pd.DataFrame,
    products: pd.DataFrame,
    *,
    cutoff_block: int,
) -> pd.DataFrame:
    """Create one cutoff-safe behavior row per product for full-corpus scoring and serving."""
    past = interactions[interactions.time_block < cutoff_block].copy()
    grouped = (
        past.groupby("product_id", as_index=False)
        .agg(
            prior_impressions=("impressed", "sum"),
            prior_clicks=("clicked", "sum"),
            prior_purchases=("purchased", "sum"),
        )
        if not past.empty
        else pd.DataFrame(columns=["product_id", "prior_impressions", "prior_clicks", "prior_purchases"])
    )
    result = products[["product_id", "launch_block"]].merge(grouped, on="product_id", how="left")
    for column in ("prior_impressions", "prior_clicks", "prior_purchases"):
        result[column] = result[column].fillna(0.0).astype(float)

    recent = past[past.time_block >= max(0, cutoff_block - 2)]
    velocity = recent.groupby("product_id")["clicked"].sum().rename("behavior_velocity")
    result = result.merge(velocity, on="product_id", how="left")
    result["behavior_velocity"] = result["behavior_velocity"].fillna(0.0).astype(float)
    result["smoothed_ctr"] = (result.prior_clicks + 1.0) / (result.prior_impressions + 5.0)
    result["smoothed_purchase_rate"] = (result.prior_purchases + 0.5) / (
        result.prior_impressions + 8.0
    )
    result["first_observed_age"] = np.maximum(0, cutoff_block - result.launch_block.astype(int))
    result["zero_history"] = ((result.prior_clicks + result.prior_purchases) == 0).astype(int)
    result["sparse_history"] = ((result.prior_clicks + result.prior_purchases) <= 2).astype(int)
    return result.drop(columns=["launch_block"])


def add_retrieval_features(
    frame: pd.DataFrame,
    queries: pd.DataFrame,
    products: pd.DataFrame,
    bm25,
    dense,
) -> pd.DataFrame:
    query_map = queries.set_index("query_id")["query"].to_dict()
    bm25_index = {int(pid): idx for idx, pid in enumerate(bm25.product_ids_)}
    dense_index = {int(pid): idx for idx, pid in enumerate(dense.product_ids_)}
    bm25_scores: dict[int, np.ndarray] = {}
    dense_scores: dict[int, np.ndarray] = {}
    for query_id, query in query_map.items():
        bm25_scores[int(query_id)] = bm25.score(str(query))
        dense_scores[int(query_id)] = dense.score(str(query))
    result = frame.copy()
    result["bm25_score"] = [
        bm25_scores[int(qid)][bm25_index[int(pid)]]
        for qid, pid in zip(result.query_id, result.product_id)
    ]
    result["dense_score"] = [
        dense_scores[int(qid)][dense_index[int(pid)]]
        for qid, pid in zip(result.query_id, result.product_id)
    ]
    prod = products.set_index("product_id")
    result["price_log"] = np.log1p([float(prod.loc[int(pid), "price"]) for pid in result.product_id])
    result["quality"] = [float(prod.loc[int(pid), "quality"]) for pid in result.product_id]
    result["launch_block"] = [int(prod.loc[int(pid), "launch_block"]) for pid in result.product_id]
    result["query_text"] = [str(query_map[int(qid)]) for qid in result.query_id]
    return result


def build_temporal_product_behavior_reference(
    interactions: pd.DataFrame,
    products: pd.DataFrame,
    *,
    time_blocks: list[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a global, cutoff-safe product behavior store for selected scoring blocks.

    Each emitted row contains only events from blocks strictly earlier than ``time_block``. Unlike
    impression-row features, the reference covers every product available at the cutoff, allowing
    substitute transfer to remain invariant to the retrieved candidate subset.
    """
    product_view = products[["product_id", "launch_block"]].copy()
    product_ids = product_view.product_id.astype(int).to_numpy()
    id_to_pos = {int(pid): pos for pos, pid in enumerate(product_ids)}
    launch = product_view.launch_block.astype(int).to_numpy()
    n_products = len(product_ids)
    impressions = np.zeros(n_products, dtype=float)
    clicks = np.zeros(n_products, dtype=float)
    purchases = np.zeros(n_products, dtype=float)
    velocity = np.zeros(n_products, dtype=float)

    grouped = {
        int(block): frame.groupby("product_id", as_index=False).agg(
            impressed=("impressed", "sum"),
            clicked=("clicked", "sum"),
            purchased=("purchased", "sum"),
        )
        for block, frame in interactions.groupby("time_block", sort=True)
    }
    if time_blocks is None:
        requested = sorted(int(x) for x in interactions.time_block.unique())
    else:
        requested = sorted({int(x) for x in time_blocks})
    if not requested:
        raise ValueError("time_blocks must contain at least one cutoff")

    rows: list[pd.DataFrame] = []
    max_requested = max(requested)
    requested_set = set(requested)
    for block in range(0, max_requested + 1):
        if block in requested_set:
            available = launch <= block
            snapshot = pd.DataFrame(
                {
                    "time_block": block,
                    "product_id": product_ids[available],
                    "prior_impressions": impressions[available],
                    "prior_clicks": clicks[available],
                    "prior_purchases": purchases[available],
                    "behavior_velocity": velocity[available],
                }
            )
            snapshot["smoothed_ctr"] = (
                snapshot.prior_clicks + 1.0
            ) / (snapshot.prior_impressions + 5.0)
            snapshot["smoothed_purchase_rate"] = (
                snapshot.prior_purchases + 0.5
            ) / (snapshot.prior_impressions + 8.0)
            snapshot["first_observed_age"] = np.maximum(
                0,
                block
                - np.array(
                    [launch[id_to_pos[int(pid)]] for pid in snapshot.product_id],
                    dtype=int,
                ),
            )
            snapshot["zero_history"] = (
                (snapshot.prior_clicks + snapshot.prior_purchases) == 0
            ).astype(int)
            snapshot["sparse_history"] = (
                (snapshot.prior_clicks + snapshot.prior_purchases) <= 2
            ).astype(int)
            rows.append(snapshot)

        current = grouped.get(block)
        velocity *= 0.75
        if current is not None:
            for row in current.itertuples(index=False):
                pos = id_to_pos.get(int(row.product_id))
                if pos is None:
                    continue
                impressions[pos] += float(row.impressed)
                clicks[pos] += float(row.clicked)
                purchases[pos] += float(row.purchased)
                velocity[pos] += float(row.clicked)

    if not rows:
        raise RuntimeError("No temporal behavior reference rows were generated")
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["time_block", "product_id"]).any():
        raise RuntimeError("Temporal behavior reference contains duplicate cutoff-product rows")
    return result
