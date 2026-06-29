from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_TOKEN = re.compile(r"(?u)\b\w+\b")
_RELATIONS = ("exact", "substitute", "complement", "irrelevant")
_INTENTS = ("generic", "exact_model", "attribute", "brand_category", "accessory")
_REFERENCE_COLUMNS = (
    "time_block",
    "product_id",
    "prior_impressions",
    "smoothed_ctr",
    "smoothed_purchase_rate",
)


@dataclass(frozen=True)
class QRSBTConfig:
    neighbors: int = 8
    min_support: int = 2
    relation_training_limit: int = 50_000
    neighbor_candidates_multiplier: int = 4
    exact_neighbor_limit: int = 100_000
    neighbor_chunk_size: int = 512
    seed: int = 42


class QRSBT:
    """Query-conditioned reliable substitute behavioral transfer.

    Relation judgments train a compact query-product classifier, but inference uses only observable
    query text, metadata, and a retriever-derived query anchor. Substitute behavior is drawn from a
    catalog-level neighbor graph and a cutoff-safe behavior reference, so the transfer for one target
    does not change merely because another candidate was omitted from the retrieved list.
    """

    def __init__(self, config: QRSBTConfig):
        self.config = config
        self._cache_lock_ = threading.RLock()
        self._relation_probability_cache_ = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Runtime caches are query-traffic dependent, can be large, and must not be persisted.
        state["_relation_probability_cache_"] = OrderedDict()
        state.pop("_cache_lock_", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._relation_probability_cache_ = OrderedDict()
        self._cache_lock_ = threading.RLock()

    def clear_runtime_cache(self) -> None:
        with self._cache_lock_:
            self._relation_probability_cache_.clear()

    def fit(
        self,
        products: pd.DataFrame,
        queries: pd.DataFrame,
        relevance: pd.DataFrame,
        dense_embeddings: np.ndarray,
        dense_product_ids: np.ndarray,
        *,
        fit_query_ids: set[int] | None = None,
        validation_query_ids: set[int] | None = None,
        query_anchor_ids: dict[int, int] | None = None,
        precomputed_neighbors: pd.DataFrame | None = None,
    ) -> "QRSBT":
        self.products_ = products.set_index("product_id").copy()
        self.queries_ = queries.set_index("query_id").copy()
        dense_ids = np.asarray(dense_product_ids, dtype=int)
        if len(dense_ids) != len(set(map(int, dense_ids))):
            raise ValueError("Dense product IDs must be unique")
        if set(map(int, dense_ids)) != set(self.products_.index.astype(int)):
            raise ValueError("Dense embedding product IDs must match the product catalog")
        embeddings = np.asarray(dense_embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(dense_ids):
            raise ValueError("Dense embeddings must be a two-dimensional catalog-aligned matrix")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings_ = embeddings / np.maximum(norms, 1e-12)
        self.product_ids_ = dense_ids
        self.product_id_to_idx_ = {int(pid): idx for idx, pid in enumerate(dense_ids)}
        self.category_map_ = self.products_["category"].astype(str).to_dict()
        self.attribute_map_ = self.products_["attribute"].astype(str).to_dict()
        self.product_records_ = self.products_.to_dict(orient="index")
        self.query_anchor_ids_ = {
            int(query_id): int(product_id)
            for query_id, product_id in (query_anchor_ids or {}).items()
        }
        unknown_anchors = set(self.query_anchor_ids_.values()) - set(self.products_.index.astype(int))
        if unknown_anchors:
            raise ValueError(f"Query anchors reference unknown products: {sorted(unknown_anchors)[:10]}")

        self._fit_neighbor_graph(precomputed_neighbors)

        training = relevance.copy()
        if fit_query_ids is not None:
            training = training[training.query_id.isin(fit_query_ids)]
        if len(training) > self.config.relation_training_limit:
            training = training.sample(
                self.config.relation_training_limit, random_state=self.config.seed
            )
        x = self._relation_features_for_rows(training)
        y = training.relation.astype(str).to_numpy()
        if np.unique(y).size < 2:
            raise ValueError("Q-RSBT relation training requires at least two relation classes")
        self.relation_model_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=self.config.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        ).fit(x, y)
        self.relation_classes_ = list(self.relation_model_.named_steps["model"].classes_)
        self.relation_validation_ = self._evaluate_relation_model(
            relevance, validation_query_ids
        )
        self.clear_runtime_cache()
        self.relation_feature_contract_ = (
            "query_title_overlap",
            "query_category_token",
            "query_attribute_token",
            "brand_in_query",
            "model_in_query",
            "query_category_match",
            "anchor_category_match",
            "anchor_attribute_match",
            "anchor_brand_match",
            "anchor_model_match",
            "accessory_token",
            "query_length",
            *[f"intent_{intent}" for intent in _INTENTS],
            "quality",
            "price_log_scaled",
        )
        return self

    def _fit_neighbor_graph(self, precomputed: pd.DataFrame | None) -> None:
        if precomputed is not None:
            required = {"product_id", "neighbor_product_id", "similarity"}
            missing = sorted(required - set(precomputed.columns))
            if missing:
                raise ValueError(f"Precomputed neighbor table missing columns: {missing}")
            graph: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for pid, group in precomputed.groupby("product_id", sort=False):
                ordered = group.sort_values(
                    ["similarity", "neighbor_product_id"], ascending=[False, True]
                )
                graph[int(pid)] = (
                    ordered.neighbor_product_id.astype(int).to_numpy(),
                    ordered.similarity.astype(float).to_numpy(),
                )
            self.neighbor_graph_ = graph
            self.neighbor_graph_source_ = "precomputed"
            return

        if len(self.product_ids_) > self.config.exact_neighbor_limit:
            raise ValueError(
                "Exact Q-RSBT neighbor construction is disabled above "
                f"{self.config.exact_neighbor_limit:,} products; provide a precomputed FAISS/HNSW "
                "neighbor table with product_id, neighbor_product_id, and similarity"
            )
        pool_size = max(
            self.config.neighbors,
            self.config.neighbors * self.config.neighbor_candidates_multiplier,
        )
        graph: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        category_series = self.products_["category"].astype(str)
        for _, category_ids_index in category_series.groupby(category_series).groups.items():
            category_ids = np.asarray(list(map(int, category_ids_index)), dtype=int)
            if category_ids.size <= 1:
                for pid in category_ids:
                    graph[int(pid)] = (np.array([], dtype=int), np.array([], dtype=float))
                continue
            category_positions = np.asarray(
                [self.product_id_to_idx_[int(pid)] for pid in category_ids], dtype=int
            )
            category_vectors = self.embeddings_[category_positions]
            for start in range(0, len(category_ids), self.config.neighbor_chunk_size):
                stop = min(start + self.config.neighbor_chunk_size, len(category_ids))
                similarities = category_vectors[start:stop] @ category_vectors.T
                for local_row, global_row in enumerate(range(start, stop)):
                    similarities[local_row, global_row] = -np.inf
                    keep = min(pool_size, len(category_ids) - 1)
                    if keep <= 0:
                        selected = np.array([], dtype=int)
                    else:
                        selected = np.argpartition(-similarities[local_row], keep - 1)[:keep]
                        selected = selected[
                            np.lexsort(
                                (
                                    category_ids[selected],
                                    -similarities[local_row, selected],
                                )
                            )
                        ]
                    graph[int(category_ids[global_row])] = (
                        category_ids[selected].astype(int, copy=True),
                        similarities[local_row, selected].astype(float, copy=True),
                    )
        self.neighbor_graph_ = graph
        self.neighbor_graph_source_ = "exact_chunked"

    def _query_context(self, query_id: int) -> dict[str, object]:
        query = self.queries_.loc[int(query_id)]
        return {
            "query_text": str(query["query"]),
            "query_category": str(query.get("category", "unknown")),
            "query_intent": str(query.get("intent", "generic")),
            "anchor_product_id": self.query_anchor_ids_.get(int(query_id)),
        }

    def _evaluate_relation_model(
        self, relevance: pd.DataFrame, validation_query_ids: set[int] | None
    ) -> dict[str, float]:
        if not validation_query_ids:
            return {}
        validation = relevance[relevance.query_id.isin(validation_query_ids)].copy()
        if validation.empty:
            return {}
        x = self._relation_features_for_rows(validation)
        y = validation.relation.astype(str).to_numpy()
        probability = self.relation_model_.predict_proba(x)
        prediction = np.asarray(self.relation_classes_)[np.argmax(probability, axis=1)]
        eligible = np.isin(y, ["exact", "substitute"]).astype(int)
        eligible_probability = np.zeros(len(validation), dtype=float)
        for label in ("exact", "substitute"):
            if label in self.relation_classes_:
                eligible_probability += probability[:, self.relation_classes_.index(label)]
        eligible_probability = np.clip(eligible_probability, 1e-6, 1 - 1e-6)
        edges = np.linspace(0.0, 1.0, 11)
        eligible_ece = 0.0
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            mask = (eligible_probability >= low) & (
                eligible_probability < high if high < 1.0 else eligible_probability <= high
            )
            if mask.any():
                eligible_ece += float(mask.mean()) * abs(
                    float(eligible[mask].mean()) - float(eligible_probability[mask].mean())
                )
        return {
            "accuracy": float(accuracy_score(y, prediction)),
            "log_loss": float(log_loss(y, probability, labels=self.relation_classes_)),
            "eligibility_brier": float(brier_score_loss(eligible, eligible_probability)),
            "eligibility_ece": float(eligible_ece),
            "rows": float(len(validation)),
        }

    def _relation_features_for_rows(self, rows: pd.DataFrame) -> np.ndarray:
        contexts = [self._query_context(int(qid)) for qid in rows.query_id]
        return self._relation_features(contexts, rows.product_id.astype(int).tolist())

    def _relation_features(
        self, contexts: list[dict[str, object]], product_ids: list[int]
    ) -> np.ndarray:
        matrix: list[list[float]] = []
        for context, pid in zip(contexts, product_ids, strict=True):
            product = self.product_records_[int(pid)]
            query = str(context.get("query_text", "")).lower()
            query_category = str(context.get("query_category", "unknown"))
            intent = str(context.get("query_intent", "generic"))
            anchor_id = context.get("anchor_product_id")
            anchor = (
                self.product_records_[int(anchor_id)]
                if anchor_id is not None and int(anchor_id) in self.product_records_
                else None
            )

            q_tokens = set(_TOKEN.findall(query))
            title_tokens = set(_TOKEN.findall(str(product["title"]).lower()))
            union = q_tokens | title_tokens
            overlap = len(q_tokens & title_tokens) / max(len(union), 1)
            category_tokens = set(
                _TOKEN.findall(str(product["category"]).replace("_", " ").lower())
            )
            attribute_tokens = set(_TOKEN.findall(str(product["attribute"]).lower()))
            brand = str(product["brand"]).lower()
            model = str(product["model"]).lower()
            anchor_category_match = float(
                anchor is not None and product["category"] == anchor["category"]
            )
            anchor_attribute_match = float(
                anchor is not None and product["attribute"] == anchor["attribute"]
            )
            anchor_brand_match = float(anchor is not None and product["brand"] == anchor["brand"])
            anchor_model_match = float(anchor is not None and product["model"] == anchor["model"])
            intent_features = [float(intent == candidate) for candidate in _INTENTS]
            matrix.append(
                [
                    overlap,
                    float(bool(q_tokens & category_tokens)),
                    float(bool(q_tokens & attribute_tokens)),
                    float(bool(brand) and brand in query),
                    float(bool(model) and model in query),
                    float(product["category"] == query_category),
                    anchor_category_match,
                    anchor_attribute_match,
                    anchor_brand_match,
                    anchor_model_match,
                    float(
                        any(
                            token in q_tokens
                            for token in {"case", "cover", "charger", "strap", "replacement"}
                        )
                    ),
                    min(len(q_tokens), 12) / 12.0,
                    *intent_features,
                    float(product["quality"]),
                    float(np.log1p(max(float(product["price"]), 0.0)) / 10.0),
                ]
            )
        return np.asarray(matrix, dtype=np.float64)

    def predict_relation_probabilities(
        self,
        query_text: str,
        product_ids: list[int] | np.ndarray,
        *,
        query_category: str = "unknown",
        query_intent: str = "generic",
        anchor_product_id: int | None = None,
    ) -> pd.DataFrame:
        ids = [int(pid) for pid in product_ids]
        if not ids:
            return pd.DataFrame(columns=("product_id", *_RELATIONS))
        context = {
            "query_text": query_text,
            "query_category": query_category,
            "query_intent": query_intent,
            "anchor_product_id": anchor_product_id,
        }
        x = self._relation_features([context] * len(ids), ids)
        raw = self.relation_model_.predict_proba(x)
        frame = pd.DataFrame(0.0, index=np.arange(len(ids)), columns=_RELATIONS)
        for column_index, relation in enumerate(self.relation_classes_):
            frame[str(relation)] = raw[:, column_index]
        frame.insert(0, "product_id", ids)
        return frame

    def _relation_lookup(
        self,
        context: dict[str, object],
        product_ids: set[int],
    ) -> dict[int, tuple[float, float, float, float]]:
        key = (
            str(context["query_text"]),
            str(context["query_category"]),
            str(context["query_intent"]),
            int(context["anchor_product_id"])
            if context["anchor_product_id"] is not None
            else None,
        )
        requested = sorted(map(int, product_ids))
        # Synchronous FastAPI handlers may run concurrently.  Keep OrderedDict mutations inside
        # the lock but execute the read-only sklearn prediction outside it so unrelated queries do
        # not serialize behind one expensive inference call. Duplicate work during a race is safe.
        with self._cache_lock_:
            cache = self._relation_probability_cache_.setdefault(key, {})
            self._relation_probability_cache_.move_to_end(key)
            missing = [pid for pid in requested if pid not in cache]

        predicted: dict[int, tuple[float, float, float, float]] = {}
        if missing:
            probabilities = self.predict_relation_probabilities(
                str(context["query_text"]),
                missing,
                query_category=str(context["query_category"]),
                query_intent=str(context["query_intent"]),
                anchor_product_id=(
                    int(context["anchor_product_id"])
                    if context["anchor_product_id"] is not None
                    else None
                ),
            )
            predicted = {
                int(row.product_id): (
                    float(row.exact),
                    float(row.substitute),
                    float(row.complement),
                    float(row.irrelevant),
                )
                for row in probabilities.itertuples(index=False)
            }

        with self._cache_lock_:
            cache = self._relation_probability_cache_.setdefault(key, {})
            cache.update(predicted)
            self._relation_probability_cache_.move_to_end(key)
            while len(self._relation_probability_cache_) > 1024:
                self._relation_probability_cache_.popitem(last=False)
            return {pid: cache[pid] for pid in requested}

    def _reference_cache(
        self, reference: pd.DataFrame
    ) -> tuple[
        dict[int, dict[int, tuple[float, float, float]]],
        dict[tuple[int, str], tuple[float, float]],
        dict[int, tuple[float, float]],
    ]:
        values_by_time: dict[int, dict[int, tuple[float, float, float]]] = {}
        category_priors: dict[tuple[int, str], tuple[float, float]] = {}
        global_priors: dict[int, tuple[float, float]] = {}
        working = reference.copy()
        working["_category"] = working.product_id.astype(int).map(self.category_map_)
        for time_block, block in working.groupby("time_block", sort=False):
            time_block = int(time_block)
            values_by_time[time_block] = {
                int(row.product_id): (
                    float(row.prior_impressions),
                    float(row.smoothed_ctr),
                    float(row.smoothed_purchase_rate),
                )
                for row in block.itertuples(index=False)
            }
            historical = block[block.prior_impressions.astype(float) > 0]
            fallback = historical if not historical.empty else block
            weights = np.maximum(fallback.prior_impressions.to_numpy(dtype=float), 1.0)
            global_priors[time_block] = (
                float(np.average(fallback.smoothed_ctr.to_numpy(dtype=float), weights=weights)),
                float(
                    np.average(
                        fallback.smoothed_purchase_rate.to_numpy(dtype=float), weights=weights
                    )
                ),
            )
            for category, category_block in historical.groupby("_category", sort=False):
                category_weights = np.maximum(
                    category_block.prior_impressions.to_numpy(dtype=float), 1.0
                )
                category_priors[(time_block, str(category))] = (
                    float(
                        np.average(
                            category_block.smoothed_ctr.to_numpy(dtype=float),
                            weights=category_weights,
                        )
                    ),
                    float(
                        np.average(
                            category_block.smoothed_purchase_rate.to_numpy(dtype=float),
                            weights=category_weights,
                        )
                    ),
                )
        return values_by_time, category_priors, global_priors

    def prepare_behavior_reference(
        self, behavior_reference: pd.DataFrame
    ) -> tuple[
        dict[int, dict[int, tuple[float, float, float]]],
        dict[tuple[int, str], tuple[float, float]],
        dict[int, tuple[float, float]],
    ]:
        """Validate and pre-index an immutable behavior snapshot for repeated serving calls."""
        placeholder = behavior_reference.head(1).copy()
        reference = self._prepare_reference(placeholder, behavior_reference)
        return self._reference_cache(reference)

    def transfer(
        self,
        frame: pd.DataFrame,
        *,
        behavior_reference: pd.DataFrame | None = None,
        prepared_reference: tuple[
            dict[int, dict[int, tuple[float, float, float]]],
            dict[tuple[int, str], tuple[float, float]],
            dict[int, tuple[float, float]],
        ] | None = None,
    ) -> pd.DataFrame:
        required = {
            "time_block",
            "query_id",
            "product_id",
            "prior_impressions",
            "smoothed_ctr",
            "smoothed_purchase_rate",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Q-RSBT transfer missing columns: {missing}")
        base = frame.reset_index(drop=True).copy()
        if prepared_reference is None:
            reference = self._prepare_reference(base, behavior_reference)
            reference_by_time, category_priors, global_priors = self._reference_cache(reference)
        else:
            reference_by_time, category_priors, global_priors = prepared_reference

        n = len(base)
        outputs: dict[str, list] = {
            "qrsbt_ctr": [0.0] * n,
            "qrsbt_purchase": [0.0] * n,
            "qrsbt_ctr_lower": [0.0] * n,
            "qrsbt_purchase_lower": [0.0] * n,
            "qrsbt_confidence": [0.0] * n,
            "qrsbt_dispersion": [1.0] * n,
            "qrsbt_support": [0] * n,
            "qrsbt_neighbors": [""] * n,
            "qrsbt_support_score": [0.0] * n,
            "qrsbt_relevance_probability": [0.0] * n,
            "qrsbt_irrelevant_probability": [1.0] * n,
            "qrsbt_compatibility": [0.0] * n,
            "qrsbt_transfer_source": ["none"] * n,
        }

        for (time_block, query_id), group in base.groupby(
            ["time_block", "query_id"], sort=False
        ):
            time_block = int(time_block)
            query_id = int(query_id)
            if time_block not in reference_by_time:
                raise ValueError(f"Behavior reference has no rows for time_block={time_block}")
            reference_values = reference_by_time[time_block]
            positions = group.index.to_numpy()
            product_ids = group.product_id.astype(int).to_numpy()
            context = self._resolve_query_context(group, query_id)

            relation_ids: set[int] = set(map(int, product_ids))
            for pid in product_ids:
                ids, _ = self.neighbor_graph_.get(
                    int(pid), (np.array([], dtype=int), np.array([], dtype=float))
                )
                relation_ids.update(map(int, ids))
            relation = self._relation_lookup(context, relation_ids)

            for local_i, global_i in enumerate(positions):
                pid = int(product_ids[local_i])
                exact, substitute, complement, irrelevant = relation[pid]
                target_utility = exact + 0.85 * substitute + 0.08 * complement
                target_compatibility = float(np.clip(exact + substitute, 0.0, 1.0))
                outputs["qrsbt_relevance_probability"][global_i] = target_utility
                outputs["qrsbt_irrelevant_probability"][global_i] = irrelevant
                outputs["qrsbt_compatibility"][global_i] = target_compatibility

                category = self.category_map_[pid]
                category_ctr, category_purchase = category_priors.get(
                    (time_block, category), global_priors[time_block]
                )
                outputs["qrsbt_ctr"][global_i] = category_ctr
                outputs["qrsbt_purchase"][global_i] = category_purchase
                outputs["qrsbt_ctr_lower"][global_i] = category_ctr
                outputs["qrsbt_purchase_lower"][global_i] = category_purchase
                outputs["qrsbt_transfer_source"][global_i] = "category_prior"

                neighbor_ids, neighbor_similarity = self.neighbor_graph_.get(
                    pid, (np.array([], dtype=int), np.array([], dtype=float))
                )
                selected_rows: list[tuple[int, float, float, float]] = []
                target_attribute = self.attribute_map_[pid]
                for neighbor_id, similarity in zip(
                    neighbor_ids, neighbor_similarity, strict=True
                ):
                    neighbor_id = int(neighbor_id)
                    behavior = reference_values.get(neighbor_id)
                    if behavior is None or behavior[0] <= 0:
                        continue
                    n_exact, n_substitute, n_complement, _ = relation[neighbor_id]
                    relation_utility = n_exact + 0.85 * n_substitute + 0.08 * n_complement
                    pair_compatible = float(
                        self.attribute_map_.get(neighbor_id) == target_attribute
                    )
                    weight = (
                        max(float(similarity), 0.0)
                        * relation_utility
                        * (0.20 + 0.80 * pair_compatible)
                    )
                    if weight > 0:
                        selected_rows.append((neighbor_id, weight, relation_utility, pair_compatible))
                selected_rows.sort(key=lambda item: (-item[1], item[0]))
                selected_rows = selected_rows[: self.config.neighbors]
                support = len(selected_rows)
                outputs["qrsbt_support"][global_i] = support
                outputs["qrsbt_neighbors"][global_i] = ",".join(
                    str(item[0]) for item in selected_rows
                )
                support_score = min(1.0, support / max(self.config.neighbors, 1))
                outputs["qrsbt_support_score"][global_i] = support_score
                if support < self.config.min_support:
                    continue

                raw_weights = np.asarray([item[1] for item in selected_rows], dtype=float)
                normalized = raw_weights / max(float(raw_weights.sum()), 1e-12)
                ctr = np.asarray(
                    [reference_values[item[0]][1] for item in selected_rows], dtype=float
                )
                purchase = np.asarray(
                    [reference_values[item[0]][2] for item in selected_rows], dtype=float
                )
                neighbor_ctr = float(np.sum(normalized * ctr))
                neighbor_purchase = float(np.sum(normalized * purchase))
                dispersion = float(
                    np.sqrt(np.sum(normalized * (ctr - neighbor_ctr) ** 2))
                )
                relation_quality = float(
                    np.sum(normalized * np.asarray([item[2] for item in selected_rows]))
                )
                pair_compatibility = float(
                    np.sum(normalized * np.asarray([item[3] for item in selected_rows]))
                )
                confidence = float(
                    np.clip(
                        relation_quality
                        * pair_compatibility
                        * target_compatibility
                        * support_score
                        * max(0.0, 1.0 - min(dispersion, 1.0)),
                        0.0,
                        1.0,
                    )
                )
                transfer_ctr = confidence * neighbor_ctr + (1.0 - confidence) * category_ctr
                transfer_purchase = (
                    confidence * neighbor_purchase
                    + (1.0 - confidence) * category_purchase
                )
                outputs["qrsbt_ctr"][global_i] = transfer_ctr
                outputs["qrsbt_purchase"][global_i] = transfer_purchase
                outputs["qrsbt_ctr_lower"][global_i] = max(0.0, transfer_ctr - dispersion)
                outputs["qrsbt_purchase_lower"][global_i] = max(
                    0.0, transfer_purchase - 0.5 * dispersion
                )
                outputs["qrsbt_confidence"][global_i] = confidence
                outputs["qrsbt_dispersion"][global_i] = dispersion
                outputs["qrsbt_transfer_source"][global_i] = "catalog_substitute_shrunk"

        return pd.concat([base, pd.DataFrame(outputs)], axis=1)

    def _prepare_reference(
        self, base: pd.DataFrame, behavior_reference: pd.DataFrame | None
    ) -> pd.DataFrame:
        reference = base if behavior_reference is None else behavior_reference
        reference = reference.copy()
        if "time_block" not in reference.columns:
            unique_blocks = sorted(base.time_block.astype(int).unique())
            if len(unique_blocks) != 1:
                raise ValueError(
                    "Behavior reference without time_block can only serve one scoring block"
                )
            reference["time_block"] = unique_blocks[0]
        missing = sorted(set(_REFERENCE_COLUMNS) - set(reference.columns))
        if missing:
            raise ValueError(f"Behavior reference missing columns: {missing}")
        reference = reference.drop_duplicates(["time_block", "product_id"], keep="last")
        known = set(self.products_.index.astype(int))
        unknown = set(reference.product_id.astype(int)) - known
        if unknown:
            raise ValueError(f"Behavior reference contains unknown products: {sorted(unknown)[:10]}")
        return reference

    def _resolve_query_context(self, group: pd.DataFrame, query_id: int) -> dict[str, object]:
        if query_id in self.queries_.index:
            context = self._query_context(query_id)
        else:
            context = {
                "query_text": "",
                "query_category": "unknown",
                "query_intent": "generic",
                "anchor_product_id": None,
            }
        if "query_text" in group.columns and group.query_text.notna().any():
            context["query_text"] = str(group.query_text.dropna().iloc[0])
        if "query_category" in group.columns and group.query_category.notna().any():
            context["query_category"] = str(group.query_category.dropna().iloc[0])
        if "query_intent" in group.columns and group.query_intent.notna().any():
            context["query_intent"] = str(group.query_intent.dropna().iloc[0])
        if (
            "query_anchor_product_id" in group.columns
            and group.query_anchor_product_id.notna().any()
        ):
            context["anchor_product_id"] = int(
                group.query_anchor_product_id.dropna().iloc[0]
            )
        if context["anchor_product_id"] is None:
            score = group.get("retrieval_score")
            if score is None:
                score = group.get("bm25_score", pd.Series(0.0, index=group.index)) + group.get(
                    "dense_score", pd.Series(0.0, index=group.index)
                )
            context["anchor_product_id"] = int(group.loc[score.idxmax(), "product_id"])
        if not str(context["query_text"]):
            raise ValueError("Q-RSBT requires query_text for unseen query identifiers")
        return context
