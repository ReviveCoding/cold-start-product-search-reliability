import pandas as pd

from product_search.pipeline import _add_strict_temporal_retrieval_context


def _products(include_future: bool) -> pd.DataFrame:
    rows = [
        {
            "product_id": 1,
            "title": "wireless headphones",
            "brand": "a",
            "category": "audio",
            "attribute": "wireless",
            "model": "h1",
            "price": 50.0,
            "quality": 0.8,
            "launch_block": 0,
        },
        {
            "product_id": 2,
            "title": "wired headphones",
            "brand": "b",
            "category": "audio",
            "attribute": "wired",
            "model": "h2",
            "price": 30.0,
            "quality": 0.7,
            "launch_block": 0,
        },
    ]
    if include_future:
        rows.append(
            {
                "product_id": 3,
                "title": "wireless future headphones headphones",
                "brand": "c",
                "category": "audio",
                "attribute": "wireless",
                "model": "h3",
                "price": 70.0,
                "quality": 0.9,
                "launch_block": 3,
            }
        )
    return pd.DataFrame(rows)


def test_historical_retrieval_statistics_ignore_future_catalog_documents():
    queries = pd.DataFrame(
        {"query_id": [10], "query": ["wireless headphones"], "category": ["audio"]}
    )
    frame = pd.DataFrame({"time_block": [1], "query_id": [10], "product_id": [1]})
    cfg = {"bm25_k1": 1.5, "bm25_b": 0.75, "dense_dim": 4, "candidate_k": 2}

    without_future, anchors_without = _add_strict_temporal_retrieval_context(
        {"train": frame},
        queries=queries,
        products=_products(False),
        retrieval_config=cfg,
        seed=42,
    )
    with_future, anchors_with = _add_strict_temporal_retrieval_context(
        {"train": frame},
        queries=queries,
        products=_products(True),
        retrieval_config=cfg,
        seed=42,
    )

    left = without_future["train"].iloc[0]
    right = with_future["train"].iloc[0]
    assert left.bm25_score == right.bm25_score
    assert left.dense_score == right.dense_score
    pd.testing.assert_frame_equal(anchors_without, anchors_with)
