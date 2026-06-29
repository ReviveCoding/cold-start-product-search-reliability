import numpy as np

from product_search.ope.estimators import estimate_ope, generate_known_propensity_lab


def test_propensities_are_valid():
    frame = generate_known_propensity_lab(n=1000)
    assert frame.logging_propensity.between(0, 1).all()
    assert frame.candidate_propensity.between(0, 1).all()


def test_ope_outputs_finite():
    metrics = estimate_ope(generate_known_propensity_lab(n=2000))
    for key in ["dm", "ips", "snips", "dr", "effective_sample_size"]:
        assert np.isfinite(metrics[key])
    assert metrics["effective_sample_size"] > 0


def test_cross_fitted_dr_recovers_known_value():
    metrics = estimate_ope(generate_known_propensity_lab(n=4000, seed=17), seed=17)
    assert metrics["support_overlap_rate"] == 1.0
    assert metrics["dr_abs_error"] < 0.05
    assert metrics["dr_covers_true_value"] == 1.0


def test_ope_lab_is_seed_deterministic():
    left = estimate_ope(generate_known_propensity_lab(n=1500, seed=23), seed=23)
    right = estimate_ope(generate_known_propensity_lab(n=1500, seed=23), seed=23)
    assert left == right
