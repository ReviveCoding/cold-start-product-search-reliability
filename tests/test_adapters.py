from pathlib import Path

import pandas as pd

from product_search.data.adapters import ESCIAdapter, KuaiSearchAdapter


def test_kuaisearch_ranking_adapter(tmp_path: Path):
    path = tmp_path / "ranking.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "query": ["headphones"],
            "target_item_id": [11],
            "is_clicked": [1],
            "is_purchased": [0],
            "relative_time": [3],
        }
    ).to_csv(path, index=False)
    frame = KuaiSearchAdapter.read_ranking(path)
    assert {"time_block", "product_id", "clicked", "purchased"}.issubset(frame.columns)


def test_esci_adapter_maps_labels(tmp_path: Path):
    examples = tmp_path / "examples.csv"
    products = tmp_path / "products.csv"
    pd.DataFrame(
        {
            "product_id": [1, 2],
            "product_locale": ["us", "us"],
            "query": ["mouse", "mouse"],
            "esci_label": ["E", "I"],
        }
    ).to_csv(examples, index=False)
    pd.DataFrame(
        {
            "product_id": [1, 2],
            "product_locale": ["us", "us"],
            "product_title": ["mouse", "keyboard"],
        }
    ).to_csv(products, index=False)
    frame = ESCIAdapter.read_joined(examples, products)
    assert frame.relevance.tolist() == [3, 0]


def test_kuaisearch_ranking_adapter_refuses_missing_time(tmp_path: Path):
    path = tmp_path / "ranking_without_time.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "query": ["headphones"],
            "target_item_id": [11],
            "is_clicked": [1],
            "is_purchased": [0],
        }
    ).to_csv(path, index=False)
    import pytest

    with pytest.raises(ValueError, match="temporal column"):
        KuaiSearchAdapter.read_ranking(path)


def test_kuaisearch_items_require_observed_launch_or_explicit_column(tmp_path: Path):
    path = tmp_path / "items.csv"
    pd.DataFrame({"item_id": [11], "title": ["headphones"]}).to_csv(path, index=False)
    import pytest

    with pytest.raises(ValueError, match="first-observed"):
        KuaiSearchAdapter.read_items(path)


def test_kuaisearch_derives_first_observed_launch_blocks():
    ranking = pd.DataFrame(
        {"product_id": [2, 1, 2, 1], "time_block": [4, 3, 2, 5]}
    )
    observed = KuaiSearchAdapter.derive_first_observed_launch_blocks(ranking)
    assert observed.set_index("product_id").launch_block.to_dict() == {1: 3, 2: 2}
