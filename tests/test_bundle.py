from pathlib import Path

from product_search.data.bundle import DataBundle
from product_search.data.synthetic import generate_synthetic_bundle


def test_canonical_bundle_round_trip_without_oracle_query_target(tmp_path: Path):
    bundle = generate_synthetic_bundle(
        seed=7,
        n_products=64,
        n_queries=12,
        n_users=24,
        n_time_blocks=6,
        impressions_per_query_time=6,
    )
    bundle.queries = bundle.queries.drop(columns=["target_product_id"])
    directory = tmp_path / "canonical"
    bundle.write(directory)

    loaded = DataBundle.from_directory(directory)
    assert "target_product_id" not in loaded.queries.columns
    assert len(loaded.products) == 64
    assert len(loaded.interactions) > 0
