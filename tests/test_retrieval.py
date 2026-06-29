from product_search.data.synthetic import generate_synthetic_bundle
from product_search.retrieval.bm25 import BM25Retriever
from product_search.retrieval.dense import DenseRetriever


def test_bm25_returns_unique_products():
    bundle = generate_synthetic_bundle(n_products=80, n_queries=12, n_users=20, n_time_blocks=4, impressions_per_query_time=8)
    model = BM25Retriever().fit(bundle.products)
    result = model.search(bundle.queries.iloc[0].query, k=10)
    assert len(result) == 10
    assert result.product_id.is_unique


def test_dense_scores_are_finite():
    bundle = generate_synthetic_bundle(n_products=80, n_queries=12, n_users=20, n_time_blocks=4, impressions_per_query_time=8)
    model = DenseRetriever(dimension=16).fit(bundle.products)
    scores = model.score(bundle.queries.iloc[0].query)
    assert len(scores) == len(bundle.products)
    assert all(map(lambda value: value == value, scores))
