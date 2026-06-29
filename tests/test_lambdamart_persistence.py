from pathlib import Path

import numpy as np
import pandas as pd

from product_search.ranking.lambdamart import LambdaMARTRanker


def test_lambdamart_native_model_roundtrip(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "query_id": [0, 0, 0, 1, 1, 1],
            "product_id": [0, 1, 2, 0, 1, 2],
            "feature": [0.9, 0.4, 0.1, 0.2, 0.8, 0.3],
            "relevance": [3, 1, 0, 0, 3, 1],
        }
    )
    ranker = LambdaMARTRanker(n_estimators=4, seed=7).fit(
        frame, ["feature"], label="relevance"
    )
    expected = ranker.predict(frame)
    model_path = tmp_path / "ranker.json"
    metadata_path = tmp_path / "ranker_metadata.json"
    ranker.save(model_path, metadata_path)
    restored = LambdaMARTRanker.load(model_path, metadata_path)
    actual = restored.predict(frame)
    assert restored.features_ == ["feature"]
    assert np.allclose(actual, expected)
