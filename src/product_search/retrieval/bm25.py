from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


_TOKEN = re.compile(r"(?u)\b\w\w+\b")


@dataclass
class BM25Retriever:
    k1: float = 1.5
    b: float = 0.75

    def fit(self, products: pd.DataFrame) -> "BM25Retriever":
        self.product_ids_ = products.product_id.to_numpy()
        text = products["title"].fillna("").astype(str).tolist()
        self.vectorizer_ = CountVectorizer(token_pattern=_TOKEN.pattern, lowercase=True)
        counts = self.vectorizer_.fit_transform(text).astype(np.float64).tocsr()
        self.doc_len_ = np.asarray(counts.sum(axis=1)).ravel()
        self.avgdl_ = float(max(self.doc_len_.mean(), 1e-9))
        n_docs = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf_ = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.counts_ = counts
        return self

    def score(self, query: str) -> np.ndarray:
        q = self.vectorizer_.transform([query])
        terms = q.indices
        if len(terms) == 0:
            return np.zeros(len(self.product_ids_), dtype=float)
        scores = np.zeros(len(self.product_ids_), dtype=float)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len_ / self.avgdl_)
        for term in terms:
            tf = self.counts_[:, term].toarray().ravel()
            scores += self.idf_[term] * ((tf * (self.k1 + 1)) / (tf + norm + 1e-12))
        return scores

    def search(self, query: str, k: int = 20) -> pd.DataFrame:
        scores = self.score(query)
        order = np.argsort(-scores)[:k]
        return pd.DataFrame({"product_id": self.product_ids_[order], "bm25_score": scores[order]})
