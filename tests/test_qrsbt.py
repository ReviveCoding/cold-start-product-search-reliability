import numpy as np
import pandas as pd
import pytest

from product_search.data.synthetic import generate_synthetic_bundle
from product_search.features import add_retrieval_features, build_temporal_behavior_features
from product_search.retrieval.bm25 import BM25Retriever
from product_search.retrieval.dense import DenseRetriever
from product_search.substitutes.qrsbt import QRSBT, QRSBTConfig


@pytest.fixture(scope="module")
def transferred():
    bundle = generate_synthetic_bundle(
        n_products=100,
        n_queries=16,
        n_users=30,
        n_time_blocks=5,
        impressions_per_query_time=10,
    )
    behavior = build_temporal_behavior_features(bundle.interactions, bundle.products).merge(
        bundle.relevance, on=["query_id", "product_id"], how="left"
    )
    bm25 = BM25Retriever().fit(bundle.products)
    dense = DenseRetriever(dimension=20).fit(bundle.products)
    behavior = add_retrieval_features(behavior, bundle.queries, bundle.products, bm25, dense)
    model = QRSBT(QRSBTConfig(neighbors=5, min_support=1)).fit(
        bundle.products, bundle.queries, bundle.relevance, dense.embeddings_, dense.product_ids_
    )
    return model, behavior, model.transfer(behavior)


def test_no_self_neighbor(transferred):
    _, _, frame = transferred
    for row in frame.itertuples(index=False):
        neighbors = {int(x) for x in str(row.qrsbt_neighbors).split(",") if x}
        assert row.product_id not in neighbors


def test_transfer_confidence_bounded(transferred):
    _, _, frame = transferred
    assert frame.qrsbt_confidence.between(0, 1).all()


def test_transfer_does_not_read_row_level_relation_labels(transferred):
    model, behavior, expected = transferred
    inference = behavior.drop(columns=["relation", "relevance", "attribute_compatible"])
    actual = model.transfer(inference)
    columns = [
        "qrsbt_ctr",
        "qrsbt_purchase",
        "qrsbt_confidence",
        "qrsbt_irrelevant_probability",
    ]
    assert np.allclose(expected[columns], actual[columns])


def test_relation_probabilities_sum_to_one(transferred):
    model, behavior, _ = transferred
    row = behavior.iloc[0]
    probabilities = model.predict_relation_probabilities(
        str(row.query_text),
        behavior.product_id.head(8).astype(int).tolist(),
        query_category=str(row.query_category) if "query_category" in behavior else "unknown",
    )
    assert np.allclose(
        probabilities[["exact", "substitute", "complement", "irrelevant"]].sum(axis=1), 1.0
    )


def test_relation_model_ignores_synthetic_oracle_target():
    bundle = generate_synthetic_bundle(
        n_products=80,
        n_queries=12,
        n_users=20,
        n_time_blocks=5,
        impressions_per_query_time=8,
    )
    dense = DenseRetriever(dimension=16).fit(bundle.products)
    anchor_ids = {
        int(qid): int(pid)
        for qid, pid in zip(
            bundle.queries.query_id,
            bundle.queries.target_product_id,
            strict=True,
        )
    }
    original = QRSBT(QRSBTConfig(neighbors=4, min_support=1)).fit(
        bundle.products,
        bundle.queries,
        bundle.relevance,
        dense.embeddings_,
        dense.product_ids_,
        query_anchor_ids=anchor_ids,
    )
    changed_queries = bundle.queries.copy()
    changed_queries["target_product_id"] = np.roll(changed_queries.target_product_id.to_numpy(), 1)
    changed = QRSBT(QRSBTConfig(neighbors=4, min_support=1)).fit(
        bundle.products,
        changed_queries,
        bundle.relevance,
        dense.embeddings_,
        dense.product_ids_,
        query_anchor_ids=anchor_ids,
    )
    query = bundle.queries.iloc[0]
    ids = bundle.products.product_id.head(10).astype(int).tolist()
    left = original.predict_relation_probabilities(
        str(query.query),
        ids,
        query_category=str(query.category),
        query_intent=str(query.intent),
        anchor_product_id=anchor_ids[int(query.query_id)],
    )
    right = changed.predict_relation_probabilities(
        str(query.query),
        ids,
        query_category=str(query.category),
        query_intent=str(query.intent),
        anchor_product_id=anchor_ids[int(query.query_id)],
    )
    assert np.allclose(
        left[["exact", "substitute", "complement", "irrelevant"]],
        right[["exact", "substitute", "complement", "irrelevant"]],
    )


def test_global_reference_makes_transfer_candidate_subset_invariant():
    from product_search.features import build_temporal_product_behavior_reference
    from product_search.retrieval.hybrid import build_query_anchor_map

    bundle = generate_synthetic_bundle(
        n_products=96,
        n_queries=12,
        n_users=24,
        n_time_blocks=5,
        impressions_per_query_time=10,
    )
    behavior = build_temporal_behavior_features(bundle.interactions, bundle.products).merge(
        bundle.relevance, on=["query_id", "product_id"], how="left"
    )
    bm25 = BM25Retriever().fit(bundle.products)
    dense = DenseRetriever(dimension=18).fit(bundle.products)
    behavior = add_retrieval_features(behavior, bundle.queries, bundle.products, bm25, dense)
    anchors = build_query_anchor_map(
        bundle.queries,
        products=bundle.products,
        bm25=bm25,
        dense=dense,
        candidate_k=20,
        cutoff_block=4,
    )
    context = bundle.queries[["query_id", "intent", "category"]].rename(
        columns={"intent": "query_intent", "category": "query_category"}
    )
    context["query_anchor_product_id"] = context.query_id.map(anchors)
    behavior = behavior.merge(context, on="query_id", how="left", validate="many_to_one")
    reference = build_temporal_product_behavior_reference(
        bundle.interactions, bundle.products, time_blocks=[4]
    )
    group = behavior[(behavior.time_block == 4) & (behavior.query_id == 0)].copy()
    assert len(group) >= 4
    model = QRSBT(QRSBTConfig(neighbors=5, min_support=1)).fit(
        bundle.products,
        bundle.queries,
        bundle.relevance,
        dense.embeddings_,
        dense.product_ids_,
        query_anchor_ids=anchors,
    )
    full = model.transfer(group, behavior_reference=reference).set_index("product_id")
    subset_input = group.iloc[:-1].copy()
    subset = model.transfer(subset_input, behavior_reference=reference).set_index("product_id")
    shared = subset.index
    columns = [
        "qrsbt_ctr",
        "qrsbt_purchase",
        "qrsbt_confidence",
        "qrsbt_support",
        "qrsbt_neighbors",
    ]
    for column in columns[:-1]:
        assert np.allclose(full.loc[shared, column], subset.loc[shared, column])
    assert (
        full.loc[shared, "qrsbt_neighbors"].tolist()
        == subset.loc[shared, "qrsbt_neighbors"].tolist()
    )


def test_relation_cache_is_thread_safe_and_serialization_drops_runtime_cache(transferred, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    import joblib

    model, behavior, _ = transferred
    row = behavior.iloc[0]
    ids = behavior.product_id.head(12).astype(int).tolist()

    def call_once():
        return model.predict_relation_probabilities(
            str(row.query_text),
            ids,
            query_category=str(getattr(row, "query_category", "unknown")),
        )[["exact", "substitute", "complement", "irrelevant"]].to_numpy()

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(lambda _: call_once(), range(32)))
    for output in outputs[1:]:
        assert np.allclose(outputs[0], output)

    # Exercise the mutable LRU before persistence, then ensure traffic-specific entries are omitted.
    context = {
        "query_text": str(row.query_text),
        "query_category": str(getattr(row, "query_category", "unknown")),
        "query_intent": "generic",
        "anchor_product_id": int(ids[0]),
    }
    model._relation_lookup(context, set(ids))
    assert model._relation_probability_cache_
    path = tmp_path / "qrsbt.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert not restored._relation_probability_cache_
    restored._relation_lookup(context, set(ids))
    assert restored._relation_probability_cache_


def _tied_neighbor_graph(
    product_order: list[int],
) -> dict[int, tuple[int, ...]]:
    model = QRSBT(
        QRSBTConfig(
            neighbors=2,
            neighbor_candidates_multiplier=1,
        )
    )

    model.products_ = pd.DataFrame(
        {"category": ["audio"] * len(product_order)},
        index=pd.Index(product_order, name="product_id"),
    )

    model.product_ids_ = np.asarray(
        product_order,
        dtype=int,
    )

    model.product_id_to_idx_ = {
        int(product_id): index for index, product_id in enumerate(product_order)
    }

    model.embeddings_ = np.tile(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        (len(product_order), 1),
    )

    model._fit_neighbor_graph(None)

    return {
        int(product_id): tuple(int(neighbor_id) for neighbor_id in neighbor_ids)
        for product_id, (neighbor_ids, _) in model.neighbor_graph_.items()
    }


def test_neighbor_ties_are_catalog_order_invariant():
    left = _tied_neighbor_graph([40, 10, 30, 20])
    right = _tied_neighbor_graph([20, 30, 40, 10])

    assert left == right
    assert left[40] == (10, 20)
    assert left[10] == (20, 30)
