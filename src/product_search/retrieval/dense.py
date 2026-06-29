from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class DenseRetriever:
    dimension: int = 64
    seed: int = 42

    def fit(self, products: pd.DataFrame) -> "DenseRetriever":
        self.product_ids_ = products.product_id.to_numpy()
        text = products.title.fillna("").astype(str).tolist()
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = self.vectorizer_.fit_transform(text)
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            # TruncatedSVD is undefined for a one-document or one-feature catalog. Preserve a
            # deterministic dense-cosine contract with normalized TF-IDF until the catalog grows.
            self.svd_ = None
            self.embeddings_ = normalize(matrix).toarray()
            self.embedding_mode_ = "tfidf_fallback"
        else:
            max_dim = max(
                1,
                min(self.dimension, matrix.shape[0] - 1, matrix.shape[1] - 1),
            )
            self.svd_ = TruncatedSVD(n_components=max_dim, random_state=self.seed)
            self.embeddings_ = normalize(self.svd_.fit_transform(matrix))
            self.embedding_mode_ = "truncated_svd"
        return self

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        matrix = self.vectorizer_.transform(queries)
        if self.svd_ is None:
            return normalize(matrix).toarray()
        return normalize(self.svd_.transform(matrix))

    def score(self, query: str) -> np.ndarray:
        vector = self.encode_queries([query])[0]
        return self.embeddings_ @ vector

    def search(self, query: str, k: int = 20) -> pd.DataFrame:
        scores = self.score(query)
        order = np.argsort(-scores)[:k]
        return pd.DataFrame({"product_id": self.product_ids_[order], "dense_score": scores[order]})
