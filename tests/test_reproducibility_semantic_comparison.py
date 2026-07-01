from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from product_search.retrieval.bm25 import BM25Retriever
from product_search.retrieval.dense import DenseRetriever


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "reproducibility_check.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "reproducibility_semantic_test_module",
        MODULE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [101, 102, 103, 104],
            "title": [
                "wireless noise cancelling headphones",
                "wireless sport headphones",
                "business laptop ultrabook",
                "trail running shoes",
            ],
        }
    )


def test_bm25_semantic_fingerprint_ignores_mapping_insertion_order() -> None:
    replay = _module()
    model_a = BM25Retriever().fit(_products())
    model_b = copy.deepcopy(model_a)
    model_b.vectorizer_.vocabulary_ = dict(
        reversed(list(model_b.vectorizer_.vocabulary_.items()))
    )
    queries = ["wireless headphones", "business laptop", "trail shoes"]

    assert replay._bm25_semantic_fingerprint(
        model_a,
        queries,
    ) == replay._bm25_semantic_fingerprint(model_b, queries)

    model_b.idf_[0] += 1e-6

    assert replay._bm25_semantic_fingerprint(
        model_a,
        queries,
    ) != replay._bm25_semantic_fingerprint(model_b, queries)


def test_dense_semantic_fingerprint_canonicalizes_svd_sign() -> None:
    replay = _module()
    model_a = DenseRetriever(dimension=2, seed=42).fit(_products())
    model_b = copy.deepcopy(model_a)
    model_b.vectorizer_.vocabulary_ = dict(
        reversed(list(model_b.vectorizer_.vocabulary_.items()))
    )

    if model_b.svd_ is not None:
        model_b.svd_.components_[0] *= -1.0
        model_b.embeddings_[:, 0] *= -1.0

    queries = ["wireless headphones", "business laptop", "trail shoes"]

    assert replay._dense_semantic_fingerprint(
        model_a,
        queries,
    ) == replay._dense_semantic_fingerprint(model_b, queries)

    model_b.embeddings_[0, 0] += np.float64(1e-6)

    assert replay._dense_semantic_fingerprint(
        model_a,
        queries,
    ) != replay._dense_semantic_fingerprint(model_b, queries)
