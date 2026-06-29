import pandas as pd

from product_search.evaluation.metrics import irrelevant_exposure_at_k, ndcg_at_k


def test_perfect_ranking_has_ndcg_one():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "relevance": [3, 2, 0],
            "score": [3.0, 2.0, 0.0],
        }
    )
    assert ndcg_at_k(frame, "score", 3) == 1.0


def test_irrelevant_exposure():
    frame = pd.DataFrame(
        {"query_id": [1, 1, 1], "relevance": [0, 3, 2], "score": [3.0, 2.0, 1.0]}
    )
    assert irrelevant_exposure_at_k(frame, "score", 2) == 0.5


def test_cohort_ndcg_preserves_actual_top_k_positions():
    from product_search.evaluation.metrics import cohort_ndcg

    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "product_id": [1, 2, 3, 4],
            "relevance": [0, 0, 3, 2],
            "zero_history": [0, 0, 1, 1],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    # The cold items occupy ranks 3 and 4, so a top-2 cohort metric must be zero rather than
    # compressing them into cohort-only ranks 1 and 2.
    assert cohort_ndcg(frame, "score", "zero_history", 1, 2) == 0.0


def test_sparse_judgments_preserve_rank_positions_and_report_coverage():
    from product_search.evaluation.metrics import (
        judgment_coverage_at_k,
        ndcg_at_k,
        unjudged_exposure_at_k,
    )

    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": [1, 2, 3],
            "relevance": [None, 3.0, 0.0],
            "judged": [0, 1, 1],
            "score": [3.0, 2.0, 1.0],
        }
    )
    # The unjudged result occupies rank 1 and is treated as zero gain, not silently removed.
    assert 0.0 < ndcg_at_k(frame, "score", 3) < 1.0
    assert judgment_coverage_at_k(frame, "score", 2) == 0.5
    assert unjudged_exposure_at_k(frame, "score", 2) == 0.5


def test_irrelevant_exposure_uses_only_judged_results():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "relevance": [None, 0.0, 3.0],
            "judged": [0, 1, 1],
            "score": [3.0, 2.0, 1.0],
        }
    )
    assert irrelevant_exposure_at_k(frame, "score", 3) == 0.5
