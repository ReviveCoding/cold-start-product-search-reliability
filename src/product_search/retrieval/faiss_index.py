"""Optional FAISS index for exact or approximate cosine-similarity retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FaissIndexConfig:
    index_type: str = "flat"  # flat or hnsw
    hnsw_m: int = 32


class FaissProductIndex:
    def __init__(self, config: FaissIndexConfig | None = None):
        self.config = config or FaissIndexConfig()
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependencies with `pip install -e .[faiss]` to use FAISS."
            ) from exc
        self.faiss = faiss

    def fit(self, product_ids: np.ndarray, embeddings: np.ndarray) -> "FaissProductIndex":
        vectors = np.asarray(embeddings, dtype=np.float32).copy()
        self.faiss.normalize_L2(vectors)
        dimension = int(vectors.shape[1])
        if self.config.index_type == "flat":
            index = self.faiss.IndexFlatIP(dimension)
        elif self.config.index_type == "hnsw":
            index = self.faiss.IndexHNSWFlat(dimension, self.config.hnsw_m)
            index.metric_type = self.faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.config.index_type}")
        index.add(vectors)
        self.index_ = index
        self.product_ids_ = np.asarray(product_ids)
        return self

    def search(self, query_embeddings: np.ndarray, k: int = 20) -> list[pd.DataFrame]:
        queries = np.asarray(query_embeddings, dtype=np.float32).copy()
        self.faiss.normalize_L2(queries)
        scores, indices = self.index_.search(queries, k)
        results = []
        for row_scores, row_indices in zip(scores, indices):
            valid = row_indices >= 0
            results.append(
                pd.DataFrame(
                    {
                        "product_id": self.product_ids_[row_indices[valid]],
                        "dense_score": row_scores[valid],
                    }
                )
            )
        return results
