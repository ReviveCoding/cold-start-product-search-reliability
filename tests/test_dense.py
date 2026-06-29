import pandas as pd

from product_search.retrieval.dense import DenseRetriever


def test_dense_retriever_handles_one_product_and_one_feature_catalog():
    products = pd.DataFrame({"product_id": [1], "title": ["headphones"]})
    retriever = DenseRetriever(dimension=16, seed=42).fit(products)
    assert retriever.embedding_mode_ == "tfidf_fallback"
    assert retriever.score("headphones").shape == (1,)
    assert retriever.search("headphones", k=1).product_id.tolist() == [1]
