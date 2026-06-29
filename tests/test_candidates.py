from product_search.candidates import build_candidate_snapshot
from product_search.data.synthetic import generate_synthetic_bundle
from product_search.retrieval.bm25 import BM25Retriever
from product_search.retrieval.dense import DenseRetriever


def test_candidate_snapshot_is_cutoff_safe_and_unique():
    bundle = generate_synthetic_bundle(
        n_products=64,
        n_queries=8,
        n_users=12,
        n_time_blocks=5,
        impressions_per_query_time=6,
    )
    bm25 = BM25Retriever().fit(bundle.products)
    dense = DenseRetriever(dimension=12).fit(bundle.products)
    candidates, snapshot = build_candidate_snapshot(
        products=bundle.products,
        queries=bundle.queries,
        relevance=bundle.relevance,
        interactions=bundle.interactions,
        bm25=bm25,
        dense=dense,
        cutoff_block=3,
        candidate_k=12,
    )
    assert not candidates.duplicated(["query_id", "product_id"]).any()
    launch = bundle.products.set_index("product_id").launch_block
    assert all(launch.loc[candidates.product_id].to_numpy() <= 3)
    past = bundle.interactions[bundle.interactions.time_block < 3]
    expected = past.groupby("product_id").clicked.sum()
    observed = snapshot.set_index("product_id").prior_clicks
    for product_id, clicks in expected.items():
        assert observed.loc[product_id] == clicks


def test_candidate_snapshot_accepts_sparse_relevance_judgments():
    from product_search.candidates import build_candidate_snapshot
    from product_search.data.synthetic import generate_synthetic_bundle
    from product_search.retrieval.bm25 import BM25Retriever
    from product_search.retrieval.dense import DenseRetriever
    from product_search.retrieval.hybrid import build_query_anchor_map

    bundle = generate_synthetic_bundle(
        n_products=72,
        n_queries=10,
        n_users=20,
        n_time_blocks=5,
        impressions_per_query_time=8,
    )
    sparse = bundle.relevance.groupby("query_id", sort=False).head(5).copy()
    bm25 = BM25Retriever().fit(bundle.products)
    dense = DenseRetriever(dimension=16).fit(bundle.products)
    anchors = build_query_anchor_map(
        bundle.queries,
        products=bundle.products,
        bm25=bm25,
        dense=dense,
        candidate_k=15,
        cutoff_block=4,
    )
    candidates, _ = build_candidate_snapshot(
        products=bundle.products,
        queries=bundle.queries,
        relevance=sparse,
        interactions=bundle.interactions,
        bm25=bm25,
        dense=dense,
        cutoff_block=4,
        candidate_k=15,
        query_anchor_map=anchors,
    )
    assert {0, 1}.issubset(set(candidates.judged.unique()))
    assert candidates.loc[candidates.judged == 0, "relevance"].isna().all()
