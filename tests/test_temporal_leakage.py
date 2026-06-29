import pandas as pd

from product_search.features import build_temporal_behavior_features


def test_same_block_events_see_identical_prior_state():
    products = pd.DataFrame({"product_id": [1], "launch_block": [0]})
    interactions = pd.DataFrame(
        {
            "time_block": [0, 0, 1],
            "user_id": [1, 2, 3],
            "query_id": [10, 11, 10],
            "product_id": [1, 1, 1],
            "position": [1, 1, 1],
            "impressed": [1, 1, 1],
            "clicked": [1, 0, 0],
            "purchased": [0, 0, 0],
            "logging_propensity": [0.5, 0.5, 0.5],
        }
    )
    features = build_temporal_behavior_features(interactions, products)
    block_zero = features[features.time_block == 0]
    assert block_zero.prior_impressions.eq(0).all()
    assert block_zero.prior_clicks.eq(0).all()
    next_block = features[features.time_block == 1].iloc[0]
    assert next_block.prior_impressions == 2
    assert next_block.prior_clicks == 1
