from product_search.data.synthetic import generate_synthetic_bundle
from product_search.features import build_temporal_behavior_features


def test_synthetic_contracts():
    bundle = generate_synthetic_bundle(n_products=80, n_queries=16, n_users=20, n_time_blocks=5, impressions_per_query_time=8)
    bundle.validate()


def test_temporal_features_do_not_use_current_click():
    bundle = generate_synthetic_bundle(n_products=80, n_queries=12, n_users=20, n_time_blocks=5, impressions_per_query_time=7)
    features = build_temporal_behavior_features(bundle.interactions, bundle.products)
    first = features.sort_values("time_block").iloc[0]
    assert first.prior_clicks == 0
    assert first.prior_purchases == 0


def test_first_observed_age_nonnegative():
    bundle = generate_synthetic_bundle(n_products=60, n_queries=10, n_users=10, n_time_blocks=4, impressions_per_query_time=6)
    features = build_temporal_behavior_features(bundle.interactions, bundle.products)
    assert (features.first_observed_age >= 0).all()


def test_synthetic_queries_have_coherent_main_product_intents():
    bundle = generate_synthetic_bundle(
        n_products=80,
        n_queries=20,
        n_users=20,
        n_time_blocks=5,
        impressions_per_query_time=6,
    )
    assert "accessory" not in set(bundle.queries.intent)
    assert bundle.interactions.logging_propensity.between(0, 1).all()
