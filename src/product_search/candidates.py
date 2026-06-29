from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_product_behavior_snapshot
from .retrieval.hybrid import hybrid_retrieve


def build_candidate_snapshot(
    *,
    products: pd.DataFrame,
    queries: pd.DataFrame,
    relevance: pd.DataFrame,
    interactions: pd.DataFrame,
    bm25,
    dense,
    cutoff_block: int,
    candidate_k: int,
    query_anchor_map: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve full-corpus candidates and attach cutoff-safe behavior.

    Relevance judgments are joined after retrieval and may be sparse. Unjudged candidates remain in
    the serving/evaluation artifact with ``judged=0`` instead of causing a lookup failure. Metrics that
    require semantic labels are expected to use judged rows only.
    """
    behavior = build_product_behavior_snapshot(
        interactions, products, cutoff_block=cutoff_block
    )
    product_columns = [
        "product_id",
        "title",
        "brand",
        "category",
        "attribute",
        "model",
        "color",
        "price",
        "quality",
        "launch_block",
    ]
    product_view = products[product_columns].copy()
    rows: list[pd.DataFrame] = []

    for query in queries.itertuples(index=False):
        retrieved = hybrid_retrieve(
            str(query.query),
            products=products,
            bm25=bm25,
            dense=dense,
            candidate_k=candidate_k,
            cutoff_block=cutoff_block,
        )
        if retrieved.empty:
            continue
        anchor = (
            int(query_anchor_map[int(query.query_id)])
            if query_anchor_map is not None and int(query.query_id) in query_anchor_map
            else int(retrieved.iloc[0].product_id)
        )
        retrieved["time_block"] = int(cutoff_block)
        retrieved["user_id"] = -1
        retrieved["query_id"] = int(query.query_id)
        retrieved["query_text"] = str(query.query)
        retrieved["query_intent"] = str(getattr(query, "intent", "generic"))
        retrieved["query_category"] = str(getattr(query, "category", "unknown"))
        retrieved["query_anchor_product_id"] = anchor
        retrieved["position"] = 0
        retrieved["clicked"] = 0
        retrieved["purchased"] = 0
        retrieved["logging_propensity"] = 1.0
        rows.append(retrieved)

    if not rows:
        raise RuntimeError("Full-corpus retrieval produced no candidates")
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates.merge(product_view, on="product_id", how="left", validate="many_to_one")
    candidates["price_log"] = np.log1p(candidates.price.astype(float))

    judgment_columns = [
        "query_id",
        "product_id",
        "relevance",
        "relation",
        "attribute_compatible",
    ]
    judgment = relevance[judgment_columns].copy()
    candidates = candidates.merge(
        judgment,
        on=["query_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    candidates["judged"] = candidates.relevance.notna().astype(int)
    candidates = candidates.merge(behavior, on="product_id", how="left", validate="many_to_one")
    return candidates, behavior
