# Data Contracts v0.5.0

## Canonical bundle

A runnable bundle contains:

- `products.csv`: stable product ID, text/category fields, and explicit `launch_block` or documented first-observed block;
- `queries.csv`: stable query ID and query text; no oracle target is required;
- `relevance.csv`: sparse or complete query-product graded judgments;
- `interactions.csv`: user, query, product, time block, impression, click, and purchase outcomes.

Validation requires unique keys, valid foreign keys, binary click/purchase fields, `purchase <= click`, nonnegative integer time blocks, and no interaction with a product before launch/first-observed time.

## Temporal contract

- train blocks fit model parameters;
- the next block calibrates behavioral probability and informs frozen policy design;
- the test block performs final static evaluation and defines the release-catalog cutoff;
- the untouched final block audits the frozen model on products already in the release catalog;
- every historical BM25 and dense retriever is fit only on products available at that block;
- release serving and every fallback use only the frozen catalog.

## Judgment contract

Sparse judgments are allowed. Unjudged products are not silently treated as judged negatives for coverage reporting. The release records judgment coverage and unjudged exposure, and the smoke contract requires complete top-10 judgment coverage.

## Public adapters

KuaiSearch recall, relevance, and ranking stages remain separate. Missing time, launch, propensity, or stable product-ID joins are never fabricated. ESCI Exact/Substitute/Complement/Irrelevant judgments remain an independent semantic benchmark unless a valid operational join exists.
