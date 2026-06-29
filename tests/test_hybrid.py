from product_search.data.synthetic import generate_synthetic_bundle
from product_search.retrieval.bm25 import BM25Retriever
from product_search.retrieval.dense import DenseRetriever
from product_search.retrieval.hybrid import (
    build_query_anchor_map,
    build_temporal_query_anchor_frame,
    hybrid_retrieve,
)


def test_query_anchor_is_top_shared_hybrid_result():
    bundle = generate_synthetic_bundle(
        n_products=64,
        n_queries=8,
        n_users=16,
        n_time_blocks=5,
        impressions_per_query_time=8,
    )
    bm25 = BM25Retriever().fit(bundle.products)
    dense = DenseRetriever(dimension=14).fit(bundle.products)
    anchors = build_query_anchor_map(
        bundle.queries,
        products=bundle.products,
        bm25=bm25,
        dense=dense,
        candidate_k=12,
        cutoff_block=4,
    )
    for query in bundle.queries.itertuples(index=False):
        result = hybrid_retrieve(
            str(query.query),
            products=bundle.products,
            bm25=bm25,
            dense=dense,
            candidate_k=12,
            cutoff_block=4,
        )
        assert anchors[int(query.query_id)] == int(result.iloc[0].product_id)


def test_temporal_query_anchors_never_reference_future_products():
    bundle = generate_synthetic_bundle(
        seed=13,
        n_products=96,
        n_queries=12,
        n_users=24,
        n_time_blocks=6,
        impressions_per_query_time=8,
    )
    bm25 = BM25Retriever().fit(bundle.products)
    dense = DenseRetriever(dimension=16).fit(bundle.products)
    anchors = build_temporal_query_anchor_frame(
        bundle.queries,
        products=bundle.products,
        bm25=bm25,
        dense=dense,
        candidate_k=16,
        time_blocks=[0, 2, 4],
    )
    launch = bundle.products.set_index("product_id").launch_block.astype(int)
    assert all(
        int(launch.loc[int(row.query_anchor_product_id)]) <= int(row.time_block)
        for row in anchors.itertuples(index=False)
    )
