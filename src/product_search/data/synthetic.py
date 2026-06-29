from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle


SyntheticBundle = DataBundle


CATEGORIES = {
    "headphones": ["wireless", "noise cancelling", "over ear", "earbuds"],
    "coffee": ["espresso", "dark roast", "decaf", "whole bean"],
    "running_shoes": ["cushioned", "trail", "lightweight", "stability"],
    "laptop": ["gaming", "ultrabook", "business", "creator"],
    "skincare": ["moisturizer", "serum", "cleanser", "sunscreen"],
    "camera": ["mirrorless", "compact", "action", "instant"],
    "cookware": ["nonstick", "stainless", "cast iron", "ceramic"],
    "backpack": ["travel", "hiking", "laptop", "commuter"],
}
COLORS = ["black", "white", "blue", "red", "green", "silver"]
BRANDS = ["Aster", "Nimbus", "Orion", "Vela", "Juniper", "Atlas", "Kite", "Mosaic"]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_bundle(
    *,
    seed: int = 42,
    n_products: int = 320,
    n_queries: int = 48,
    n_users: int = 160,
    n_time_blocks: int = 8,
    impressions_per_query_time: int = 18,
) -> SyntheticBundle:
    rng = np.random.default_rng(seed)
    categories = list(CATEGORIES)
    product_rows = []
    for product_id in range(n_products):
        category = categories[product_id % len(categories)]
        attribute = rng.choice(CATEGORIES[category])
        brand = rng.choice(BRANDS)
        color = rng.choice(COLORS)
        model = f"{chr(65 + (product_id % 20))}{100 + product_id % 90}"
        launch_block = int(rng.choice(np.arange(n_time_blocks), p=_launch_probabilities(n_time_blocks)))
        quality = float(np.clip(rng.normal(0.62, 0.16), 0.1, 0.98))
        price = float(np.round(np.exp(rng.normal(3.5, 0.55)), 2))
        title = f"{brand} {model} {color} {attribute} {category.replace('_', ' ')}"
        product_rows.append(
            {
                "product_id": product_id,
                "title": title,
                "brand": brand,
                "category": category,
                "attribute": attribute,
                "color": color,
                "model": model,
                "price": price,
                "launch_block": launch_block,
                "quality": quality,
            }
        )
    products = pd.DataFrame(product_rows)

    query_rows = []
    for query_id in range(n_queries):
        category = categories[query_id % len(categories)]
        pool = products[products.category == category]
        target = pool.sample(1, random_state=seed + query_id).iloc[0]
        intent = rng.choice(
            ["generic", "exact_model", "attribute", "brand_category"],
            p=[0.38, 0.22, 0.25, 0.15],
        )
        if intent == "exact_model":
            query = f"{target.brand} {target.model}"
        elif intent == "attribute":
            query = f"{target.attribute} {category.replace('_', ' ')}"
        elif intent == "brand_category":
            query = f"{target.brand} {category.replace('_', ' ')}"
        else:
            query = category.replace("_", " ")
        query_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "intent": intent,
                "category": category,
                "target_product_id": int(target.product_id),
            }
        )
    queries = pd.DataFrame(query_rows)

    relevance_rows = []
    for query in queries.itertuples(index=False):
        target = products.loc[products.product_id == query.target_product_id].iloc[0]
        for product in products.itertuples(index=False):
            same_category = product.category == query.category
            exact = product.product_id == query.target_product_id
            same_brand = product.brand == target.brand
            same_attr = product.attribute == target.attribute
            same_model = product.model == target.model
            compatible = same_category and (same_attr or query.intent in {"generic", "brand_category"})
            if exact or (query.intent == "exact_model" and same_model and same_brand):
                relation, rel = "exact", 3
            elif same_category and compatible:
                relation, rel = "substitute", 2
            elif same_category and not compatible:
                relation, rel = "complement", 1
            else:
                relation, rel = "irrelevant", 0
            relevance_rows.append(
                {
                    "query_id": query.query_id,
                    "product_id": product.product_id,
                    "relevance": rel,
                    "relation": relation,
                    "attribute_compatible": int(compatible or exact),
                }
            )
    relevance = pd.DataFrame(relevance_rows)

    interaction_rows = []
    item_popularity = np.zeros(n_products, dtype=float)
    for time_block in range(n_time_blocks):
        for query in queries.itertuples(index=False):
            available = products[products.launch_block <= time_block].copy()
            rel = relevance[relevance.query_id == query.query_id][["product_id", "relevance"]]
            available = available.merge(rel, on="product_id", how="left")
            # Production-like logging score with a popularity feedback loop and small exploration.
            score = (
                1.1 * available.relevance.to_numpy()
                + 0.8 * np.log1p(item_popularity[available.product_id.to_numpy()])
                + 0.6 * available.quality.to_numpy()
                - 0.05 * np.log1p(available.price.to_numpy())
                + rng.normal(0, 0.35, len(available))
            )
            order = np.argsort(-score)
            head = available.iloc[order[: max(impressions_per_query_time * 3, 30)]].copy()
            logits = score[order[: len(head)]] / 1.3
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            replace = len(head) < impressions_per_query_time
            selected_idx = rng.choice(
                np.arange(len(head)),
                size=impressions_per_query_time,
                replace=replace,
                p=probs,
            )
            sampled = head.iloc[selected_idx].copy()
            sampled["draw_propensity"] = probs[selected_idx]
            selected = (
                sampled.drop_duplicates("product_id")
                .head(impressions_per_query_time)
                .sort_values(["relevance", "quality"], ascending=False)
                .reset_index(drop=True)
            )
            for position, product in enumerate(selected.itertuples(index=False), start=1):
                user_id = int(rng.integers(0, n_users))
                exam = 1.0 / np.log2(position + 1.5)
                cold_penalty = 0.40 if item_popularity[product.product_id] == 0 else 0.0
                click_logit = -2.25 + 0.95 * product.relevance + 1.0 * product.quality + np.log(exam) - cold_penalty
                click_prob = float(_sigmoid(click_logit))
                clicked = int(rng.random() < click_prob)
                purchase_prob = float(_sigmoid(-3.1 + 0.85 * product.relevance + 1.2 * product.quality - 0.12 * np.log1p(product.price)))
                purchased = int(clicked and (rng.random() < purchase_prob))
                item_popularity[product.product_id] += clicked + 2.5 * purchased
                interaction_rows.append(
                    {
                        "time_block": time_block,
                        "user_id": user_id,
                        "query_id": query.query_id,
                        "product_id": product.product_id,
                        "position": position,
                        "impressed": 1,
                        "clicked": clicked,
                        "purchased": purchased,
                        "logging_propensity": float(max(1e-6, product.draw_propensity)),
                    }
                )
    interactions = pd.DataFrame(interaction_rows)
    bundle = SyntheticBundle(products, queries, relevance, interactions)
    bundle.validate()
    return bundle


def _launch_probabilities(n_time_blocks: int) -> np.ndarray:
    weights = np.linspace(2.2, 0.5, n_time_blocks)
    return weights / weights.sum()
