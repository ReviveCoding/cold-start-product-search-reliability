from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .candidates import build_candidate_snapshot
from .config import ProjectConfig
from .data.bundle import DataBundle
from .data.synthetic import generate_synthetic_bundle
from .evaluation.metrics import ndcg_at_k, prediction_metrics, ranking_report
from .features import (
    BEHAVIOR_COLUMNS,
    add_retrieval_features,
    build_temporal_behavior_features,
    build_temporal_product_behavior_reference,
)
from .policy.gate import GateConfig, apply_coverage_overreach_gate
from .provenance import atomic_joblib_dump, atomic_write_csv, atomic_write_json
from .ranking.behavior import TemporalCalibratedBehaviorModel
from .ranking.lambdamart import LambdaMARTRanker
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import build_query_anchor_map
from .substitutes.qrsbt import QRSBT, QRSBTConfig


BASE_RANK_FEATURES = [
    "bm25_score",
    "dense_score",
    "price_log",
    "quality",
    "first_observed_age",
    "zero_history",
    "sparse_history",
]

BEHAVIOR_FEATURES = (
    BASE_RANK_FEATURES
    + BEHAVIOR_COLUMNS[:-3]
    + [
        "semantic_rank_score",
        "qrsbt_ctr",
        "qrsbt_purchase",
        "qrsbt_confidence",
        "qrsbt_dispersion",
        "qrsbt_support_score",
        "qrsbt_relevance_probability",
        "qrsbt_irrelevant_probability",
    ]
)


def _load_bundle(config: ProjectConfig) -> DataBundle:
    if config.data_source == "synthetic":
        return generate_synthetic_bundle(seed=config.seed, **config.raw["synthetic"])
    if config.data_source == "canonical":
        return DataBundle.from_directory(config.canonical_data_dir)
    raise ValueError(f"Unsupported data source: {config.data_source}")


def _gate_config(raw: dict) -> GateConfig:
    qcfg = raw["qrsbt"]
    rank_cfg = raw["ranking"]
    return GateConfig(
        semantic_threshold=float(qcfg["semantic_threshold"]),
        confidence_threshold=float(qcfg["confidence_threshold"]),
        compatibility_threshold=float(qcfg["compatibility_threshold"]),
        irrelevant_risk_threshold=float(qcfg["irrelevant_risk_threshold"]),
        min_support=int(qcfg["min_support"]),
        max_boost=float(qcfg["max_boost"]),
        semantic_weight=float(rank_cfg["semantic_weight"]),
        behavior_weight=float(rank_cfg["behavior_weight"]),
        top_k=int(raw["retrieval"]["final_k"]),
        promotion_window=int(qcfg["promotion_window"]),
        max_promotions_per_query=int(qcfg["max_promotions_per_query"]),
        promotion_mode=str(qcfg.get("promotion_mode", "in_window")),
    )


def _add_strict_temporal_retrieval_context(
    frames: dict[str, pd.DataFrame],
    *,
    queries: pd.DataFrame,
    products: pd.DataFrame,
    retrieval_config: dict,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit retrieval statistics only on products available at each scoring block.

    Availability filtering alone is insufficient because BM25 document frequencies and dense
    decomposition components can still be influenced by products that did not yet exist. This
    helper refits both retrievers for every observed scoring block, enriches rows with those
    block-local scores, and builds query anchors from the same block-local indexes.
    """
    tagged: list[pd.DataFrame] = []
    for partition, frame in frames.items():
        current = frame.copy()
        current["__partition"] = partition
        current["__row_order"] = np.arange(len(current), dtype=int)
        tagged.append(current)
    combined = pd.concat(tagged, ignore_index=True)
    enriched_parts: list[pd.DataFrame] = []
    anchor_rows: list[dict[str, int]] = []
    for block, block_rows in combined.groupby("time_block", sort=True):
        block_int = int(block)
        available = (
            products[products.launch_block.astype(int) <= block_int].copy().reset_index(drop=True)
        )
        if available.empty:
            raise RuntimeError(f"No products are available at scoring block {block_int}")
        observed_ids = set(block_rows.product_id.astype(int))
        missing = sorted(observed_ids - set(available.product_id.astype(int)))
        if missing:
            raise RuntimeError(
                f"Temporal rows reference unavailable products at block {block_int}: {missing[:10]}"
            )
        block_bm25 = BM25Retriever(
            k1=float(retrieval_config["bm25_k1"]),
            b=float(retrieval_config["bm25_b"]),
        ).fit(available)
        block_dense = DenseRetriever(dimension=int(retrieval_config["dense_dim"]), seed=seed).fit(
            available
        )
        enriched_parts.append(
            add_retrieval_features(
                block_rows,
                queries,
                available,
                block_bm25,
                block_dense,
            )
        )
        anchors = build_query_anchor_map(
            queries,
            products=available,
            bm25=block_bm25,
            dense=block_dense,
            candidate_k=min(int(retrieval_config["candidate_k"]), len(available)),
            cutoff_block=block_int,
        )
        anchor_rows.extend(
            {
                "time_block": block_int,
                "query_id": int(query_id),
                "query_anchor_product_id": int(product_id),
            }
            for query_id, product_id in anchors.items()
        )
    enriched = pd.concat(enriched_parts, ignore_index=True)
    outputs: dict[str, pd.DataFrame] = {}
    for partition in frames:
        subset = (
            enriched[enriched["__partition"] == partition]
            .sort_values("__row_order", kind="stable")
            .drop(columns=["__partition", "__row_order"])
            .reset_index(drop=True)
        )
        outputs[partition] = subset
    anchors = pd.DataFrame(anchor_rows).drop_duplicates(["time_block", "query_id"])
    if anchors.duplicated(["time_block", "query_id"]).any():
        raise RuntimeError("Temporal query-anchor frame contains duplicate keys")
    return outputs, anchors.reset_index(drop=True)


def run_model_stage(config: ProjectConfig) -> dict:
    """Run deterministic data, retrieval, ranking, Q-RSBT, and static evaluation stages."""
    started = time.perf_counter()
    verbose = os.getenv("PRODUCT_SEARCH_VERBOSE", "0") == "1"

    def stage(name: str) -> None:
        if verbose:
            print(f"[model-stage] {name}: {time.perf_counter() - started:.3f}s", flush=True)

    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = config.raw

    bundle = _load_bundle(config)
    bundle.validate()
    observed_blocks = sorted(bundle.interactions["time_block"].astype(int).unique())
    if len(observed_blocks) < 5:
        raise ValueError("At least five observed time blocks are required for temporal isolation")
    bundle.write(out / "data")
    stage("data_generated_and_validated")

    behavior = build_temporal_behavior_features(bundle.interactions, bundle.products)
    behavior = behavior.merge(
        bundle.relevance, on=["query_id", "product_id"], how="left", validate="many_to_one"
    )
    max_block = int(behavior.time_block.max())
    validation_block = max_block - 2
    test_block = max_block - 1
    future_audit_block = max_block
    release_products = (
        bundle.products[bundle.products.launch_block.astype(int) <= test_block]
        .copy()
        .reset_index(drop=True)
    )
    release_product_ids = set(release_products.product_id.astype(int))
    release_relevance = bundle.relevance[
        bundle.relevance.product_id.astype(int).isin(release_product_ids)
    ].copy()
    if int(raw["retrieval"]["candidate_k"]) > len(release_products):
        raise ValueError(
            "retrieval.candidate_k cannot exceed the release-time product catalog size"
        )
    atomic_write_csv(out / "release_catalog.csv", release_products)
    train = behavior[behavior.time_block < validation_block].copy().reset_index(drop=True)
    validation = behavior[behavior.time_block == validation_block].copy().reset_index(drop=True)
    logged_test = behavior[behavior.time_block == test_block].copy().reset_index(drop=True)
    future_audit = (
        behavior[
            (behavior.time_block == future_audit_block)
            & behavior.product_id.astype(int).isin(release_product_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )
    if min(len(train), len(validation), len(logged_test), len(future_audit)) == 0:
        raise RuntimeError("Temporal train/validation/test split produced an empty partition")
    stage("temporal_features_and_split")

    retrieval_cfg = raw["retrieval"]
    bm25 = BM25Retriever(k1=float(retrieval_cfg["bm25_k1"]), b=float(retrieval_cfg["bm25_b"])).fit(
        release_products
    )
    dense = DenseRetriever(dimension=int(retrieval_cfg["dense_dim"]), seed=config.seed).fit(
        release_products
    )
    temporal_frames, temporal_anchor_frame = _add_strict_temporal_retrieval_context(
        {
            "train": train,
            "validation": validation,
            "test": logged_test,
            "future_audit": future_audit,
        },
        queries=bundle.queries,
        products=release_products,
        retrieval_config=retrieval_cfg,
        seed=config.seed,
    )
    train = temporal_frames["train"]
    validation = temporal_frames["validation"]
    logged_test = temporal_frames["test"]
    future_audit = temporal_frames["future_audit"]
    query_context = bundle.queries[["query_id", "intent", "category"]].rename(
        columns={
            "intent": "query_intent",
            "category": "query_category",
        }
    )
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", logged_test),
        ("future_audit", future_audit),
    ):
        merged = frame.merge(query_context, on="query_id", how="left", validate="many_to_one")
        merged = merged.merge(
            temporal_anchor_frame,
            on=["time_block", "query_id"],
            how="left",
            validate="many_to_one",
        )
        if merged.query_anchor_product_id.isna().any():
            raise RuntimeError(f"{name} rows are missing availability-safe query anchors")
        if name == "train":
            train = merged
        elif name == "validation":
            validation = merged
        elif name == "test":
            logged_test = merged
        else:
            future_audit = merged
    relation_anchor_map = {
        int(row.query_id): int(row.query_anchor_product_id)
        for row in temporal_anchor_frame[
            temporal_anchor_frame.time_block.astype(int) == validation_block
        ].itertuples(index=False)
    }
    test_anchor_map = {
        int(row.query_id): int(row.query_anchor_product_id)
        for row in temporal_anchor_frame[
            temporal_anchor_frame.time_block.astype(int) == test_block
        ].itertuples(index=False)
    }
    behavior_reference = build_temporal_product_behavior_reference(
        bundle.interactions,
        release_products,
        time_blocks=sorted(
            {
                *train.time_block.astype(int).unique(),
                *validation.time_block.astype(int).unique(),
                *logged_test.time_block.astype(int).unique(),
                *future_audit.time_block.astype(int).unique(),
            }
        ),
    )
    stage("retrieval_features_and_observable_query_anchors")

    query_ids = np.sort(bundle.queries.query_id.astype(int).unique())
    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(query_ids)
    holdout_count = max(1, int(round(0.25 * len(shuffled))))
    relation_validation_ids = set(map(int, shuffled[:holdout_count]))
    relation_fit_ids = set(map(int, shuffled[holdout_count:]))
    qcfg = raw["qrsbt"]
    qrsbt = QRSBT(
        QRSBTConfig(
            neighbors=int(qcfg["neighbors"]),
            min_support=int(qcfg["min_support"]),
            seed=config.seed,
        )
    ).fit(
        release_products,
        bundle.queries,
        release_relevance,
        dense.embeddings_,
        dense.product_ids_,
        fit_query_ids=relation_fit_ids,
        validation_query_ids=relation_validation_ids,
        query_anchor_ids=relation_anchor_map,
    )
    train = qrsbt.transfer(train, behavior_reference=behavior_reference)
    validation = qrsbt.transfer(validation, behavior_reference=behavior_reference)
    logged_test = qrsbt.transfer(logged_test, behavior_reference=behavior_reference)
    future_audit = qrsbt.transfer(future_audit, behavior_reference=behavior_reference)
    stage("qrsbt_relation_model_and_transfer")

    rank_cfg = raw["ranking"]
    semantic_train = train[train["relevance"].notna()].copy()
    if semantic_train.empty or semantic_train["query_id"].nunique() < 2:
        raise RuntimeError(
            "Semantic ranker training requires judged rows from at least two queries"
        )
    lambdamart = LambdaMARTRanker(int(rank_cfg["xgb_estimators"]), config.seed).fit(
        semantic_train, BASE_RANK_FEATURES, label="relevance"
    )
    for frame in (train, validation, logged_test, future_audit):
        frame["semantic_rank_score"] = lambdamart.predict(frame)
    stage("semantic_ranker")

    behavior_model = TemporalCalibratedBehaviorModel(seed=config.seed).fit(
        train, validation, BEHAVIOR_FEATURES, label="clicked"
    )
    for frame in (train, validation, logged_test, future_audit):
        frame["behavior_score_uncalibrated"] = behavior_model.predict_uncalibrated(frame)
        frame["behavior_score"] = behavior_model.predict_proba(frame)[:, 1]
    gate_config = _gate_config(raw)
    logged_test = apply_coverage_overreach_gate(logged_test, gate_config)
    future_audit = apply_coverage_overreach_gate(future_audit, gate_config)
    stage("temporally_calibrated_behavior_model")

    candidates, behavior_snapshot = build_candidate_snapshot(
        products=release_products,
        queries=bundle.queries,
        relevance=release_relevance,
        interactions=bundle.interactions,
        bm25=bm25,
        dense=dense,
        cutoff_block=test_block,
        candidate_k=int(retrieval_cfg["candidate_k"]),
        query_anchor_map=test_anchor_map,
    )
    serving_reference = behavior_snapshot.copy()
    serving_reference["time_block"] = test_block
    candidates = qrsbt.transfer(candidates, behavior_reference=serving_reference)
    candidates["semantic_rank_score"] = lambdamart.predict(candidates)
    candidates["behavior_score"] = behavior_model.predict_proba(candidates)[:, 1]
    candidates = apply_coverage_overreach_gate(candidates, gate_config)
    stage("full_corpus_candidate_ranking")

    metrics = ranking_report(candidates, bootstrap_samples=250, seed=config.seed)
    calibrated = prediction_metrics(logged_test, "behavior_score")
    uncalibrated = prediction_metrics(logged_test, "behavior_score_uncalibrated")
    metrics.update({f"behavior_{key}": value for key, value in calibrated.items()})
    metrics.update({f"behavior_uncalibrated_{key}": value for key, value in uncalibrated.items()})
    future_prediction = prediction_metrics(future_audit, "behavior_score")
    metrics.update({f"future_behavior_{key}": value for key, value in future_prediction.items()})
    metrics["future_logged_base_ndcg_at_10"] = ndcg_at_k(future_audit, "base_score", 10)
    metrics["future_logged_final_ndcg_at_10"] = ndcg_at_k(future_audit, "final_score", 10)
    metrics.update({f"relation_{key}": value for key, value in qrsbt.relation_validation_.items()})
    stage("evaluation")

    for frame in (train, validation, logged_test, future_audit):
        frame["row_id"] = np.arange(len(frame), dtype=int)
    atomic_write_csv(out / "behavior_train_features.csv", train)
    atomic_write_csv(out / "behavior_validation_features.csv", validation)
    atomic_write_csv(out / "behavior_test_features.csv", logged_test)
    atomic_write_csv(out / "behavior_future_audit_features.csv", future_audit)
    atomic_write_csv(out / "product_behavior_snapshot.csv", behavior_snapshot)
    atomic_write_csv(out / "ranked_test.csv", candidates)
    atomic_write_json(out / "behavior_features.json", BEHAVIOR_FEATURES)
    atomic_write_json(out / "metrics.json", metrics)
    atomic_joblib_dump(out / "bm25.joblib", bm25)
    atomic_joblib_dump(out / "dense.joblib", dense)
    lambdamart.save(out / "lambdamart_model.json", out / "lambdamart_metadata.json")
    atomic_joblib_dump(out / "behavior_model.joblib", behavior_model)
    atomic_joblib_dump(out / "qrsbt.joblib", qrsbt)

    metadata = {
        "runtime_seconds": time.perf_counter() - started,
        "n_products": int(len(bundle.products)),
        "release_catalog_products": int(len(release_products)),
        "n_interactions": int(len(bundle.interactions)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(logged_test)),
        "future_audit_rows": int(len(future_audit)),
        "candidate_rows": int(len(candidates)),
        "behavior_champion": "temporally_calibrated_logistic",
        "validation_block": validation_block,
        "test_block": test_block,
        "future_audit_block": future_audit_block,
        "relation_fit_queries": len(relation_fit_ids),
        "relation_validation_queries": len(relation_validation_ids),
        "query_anchor_source": "availability_safe_temporal_hybrid_retriever",
        "retriever_catalog_cutoff_block": test_block,
        "historical_retriever_policy": "refit_on_catalog_available_at_each_scoring_block",
        "qrsbt_neighbor_graph_source": qrsbt.neighbor_graph_source_,
        "qrsbt_behavior_reference": "global_cutoff_safe_catalog",
        "data_source": config.data_source,
        "judged_train_rows": int(len(semantic_train)),
        "stage_status": "complete",
    }
    atomic_write_json(out / "model_stage_metadata.json", metadata)
    stage("artifacts_saved")
    return {"metrics": metrics, "metadata": metadata, "output_dir": str(out)}


run_pipeline = run_model_stage
