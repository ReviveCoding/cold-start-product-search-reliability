from __future__ import annotations

import numpy as np
import pandas as pd


def _percentile(values: np.ndarray) -> np.ndarray:
    """Deterministic percentile ranks with average ties."""
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy()


def hybrid_retrieve(
    query: str,
    *,
    products: pd.DataFrame,
    bm25,
    dense,
    candidate_k: int,
    cutoff_block: int | None = None,
    lexical_weight: float = 0.45,
    dense_weight: float = 0.55,
) -> pd.DataFrame:
    """Return a product-ID-safe union of lexical and dense candidates.

    The function is shared by offline candidate generation, query-anchor construction, and serving,
    preventing silent differences in availability filters, tie-breaking, or score normalization.
    """
    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")
    if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
        raise ValueError("hybrid weights must be nonnegative with a positive sum")

    lexical_scores = np.asarray(bm25.score(query), dtype=float)
    dense_scores = np.asarray(dense.score(query), dtype=float)
    bm25_ids = np.asarray(bm25.product_ids_, dtype=int)
    dense_ids = np.asarray(dense.product_ids_, dtype=int)
    if lexical_scores.shape[0] != bm25_ids.shape[0]:
        raise RuntimeError("BM25 score and product-ID lengths differ")
    if dense_scores.shape[0] != dense_ids.shape[0]:
        raise RuntimeError("Dense score and product-ID lengths differ")

    if cutoff_block is None:
        available_ids = set(products.product_id.astype(int))
    else:
        available_ids = set(
            products.loc[
                products.launch_block.astype(int) <= int(cutoff_block), "product_id"
            ].astype(int)
        )
    if not available_ids:
        return pd.DataFrame(
            columns=[
                "product_id",
                "bm25_score",
                "dense_score",
                "candidate_source",
                "retrieval_score",
            ]
        )

    lexical_available = np.fromiter(
        (int(pid) in available_ids for pid in bm25_ids), dtype=bool, count=len(bm25_ids)
    )
    dense_available = np.fromiter(
        (int(pid) in available_ids for pid in dense_ids), dtype=bool, count=len(dense_ids)
    )

    lexical_order = np.argsort(
        np.where(lexical_available, -lexical_scores, np.inf), kind="stable"
    )[:candidate_k]
    dense_order = np.argsort(
        np.where(dense_available, -dense_scores, np.inf), kind="stable"
    )[:candidate_k]
    lexical_top = [int(bm25_ids[pos]) for pos in lexical_order if lexical_available[pos]]
    dense_top = [int(dense_ids[pos]) for pos in dense_order if dense_available[pos]]
    product_ids = list(dict.fromkeys([*lexical_top, *dense_top]))
    if not product_ids:
        return pd.DataFrame(
            columns=[
                "product_id",
                "bm25_score",
                "dense_score",
                "candidate_source",
                "retrieval_score",
            ]
        )

    lexical_map = {int(pid): float(score) for pid, score in zip(bm25_ids, lexical_scores, strict=True)}
    dense_map = {int(pid): float(score) for pid, score in zip(dense_ids, dense_scores, strict=True)}
    lexical_set, dense_set = set(lexical_top), set(dense_top)
    frame = pd.DataFrame(
        {
            "product_id": product_ids,
            "bm25_score": [lexical_map.get(pid, 0.0) for pid in product_ids],
            "dense_score": [dense_map.get(pid, 0.0) for pid in product_ids],
        }
    )
    frame["candidate_source"] = [
        "hybrid"
        if pid in lexical_set and pid in dense_set
        else ("bm25" if pid in lexical_set else "dense")
        for pid in product_ids
    ]
    total_weight = lexical_weight + dense_weight
    frame["retrieval_score"] = (
        lexical_weight * _percentile(frame.bm25_score.to_numpy())
        + dense_weight * _percentile(frame.dense_score.to_numpy())
    ) / total_weight
    return frame.sort_values(
        ["retrieval_score", "product_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def build_query_anchor_map(
    queries: pd.DataFrame,
    *,
    products: pd.DataFrame,
    bm25,
    dense,
    candidate_k: int,
    cutoff_block: int | None = None,
) -> dict[int, int]:
    """Build observable query anchors from the same hybrid retriever used online."""
    anchors: dict[int, int] = {}
    for row in queries.itertuples(index=False):
        candidates = hybrid_retrieve(
            str(row.query),
            products=products,
            bm25=bm25,
            dense=dense,
            candidate_k=candidate_k,
            cutoff_block=cutoff_block,
        )
        if candidates.empty:
            raise RuntimeError(f"No available anchor candidate for query_id={int(row.query_id)}")
        anchors[int(row.query_id)] = int(candidates.iloc[0].product_id)
    return anchors


def build_temporal_query_anchor_frame(
    queries: pd.DataFrame,
    *,
    products: pd.DataFrame,
    bm25,
    dense,
    candidate_k: int,
    time_blocks: list[int] | np.ndarray,
) -> pd.DataFrame:
    """Build availability-safe query anchors for every requested scoring block.

    Historical feature rows must not reuse a later release-time anchor because a product that did
    not yet exist could otherwise change relation features for earlier observations.  The returned
    table is keyed by ``(time_block, query_id)`` and can be merged directly into temporal frames.
    """
    blocks = sorted({int(block) for block in time_blocks})
    if not blocks:
        raise ValueError("time_blocks must contain at least one cutoff")
    rows: list[dict[str, int]] = []
    for block in blocks:
        anchors = build_query_anchor_map(
            queries,
            products=products,
            bm25=bm25,
            dense=dense,
            candidate_k=candidate_k,
            cutoff_block=block,
        )
        rows.extend(
            {
                "time_block": block,
                "query_id": int(query_id),
                "query_anchor_product_id": int(product_id),
            }
            for query_id, product_id in anchors.items()
        )
    result = pd.DataFrame(rows)
    if result.duplicated(["time_block", "query_id"]).any():
        raise RuntimeError("Temporal query-anchor frame contains duplicate keys")
    return result
