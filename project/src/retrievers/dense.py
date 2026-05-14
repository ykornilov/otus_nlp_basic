"""Dense retriever: cosine top-k over pre-computed embeddings.

Supports two modes per call:
  - dataset_filter: list of recipe_ids restricted to the current dataset
  - exclude_self: optional recipe_id to drop from results (for sanity tests)

Embeddings are expected to be L2-normalized — cosine ≡ dot product.
"""
from __future__ import annotations

import numpy as np


def cosine_top_k(
    query_emb: np.ndarray,           # shape (D,) — L2-normalized
    corpus_emb: np.ndarray,          # shape (N, D) — L2-normalized
    corpus_ids: list[str],           # length N
    k: int,
    allowed_ids: set[str] | None = None,
    exclude_id: str | None = None,
) -> list[tuple[str, float]]:
    """Return list of (recipe_id, score) sorted by score desc, length ≤ k."""
    scores = corpus_emb @ query_emb  # shape (N,)
    # Build a mask of eligible indices
    indices = np.arange(len(corpus_ids))
    if allowed_ids is not None or exclude_id is not None:
        mask = np.ones(len(corpus_ids), dtype=bool)
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            mask &= np.array([rid in allowed for rid in corpus_ids])
        if exclude_id is not None:
            mask &= np.array([rid != exclude_id for rid in corpus_ids])
        indices = indices[mask]
        scores = scores[mask]
    # Top-k
    if len(indices) == 0:
        return []
    k_eff = min(k, len(indices))
    top_idx_local = np.argpartition(-scores, k_eff - 1)[:k_eff]
    top_idx_local = top_idx_local[np.argsort(-scores[top_idx_local])]
    return [(corpus_ids[indices[i]], float(scores[i])) for i in top_idx_local]


def retrieve_in_dataset(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    corpus_datasets: list[str],   # parallel array, dataset id per recipe
    current_dataset_id: str,
    k: int,
) -> list[str]:
    """Retrieve top-k from current dataset only, return ids."""
    allowed = {rid for rid, ds in zip(corpus_ids, corpus_datasets) if ds == current_dataset_id}
    res = cosine_top_k(query_emb, corpus_emb, corpus_ids, k, allowed_ids=allowed)
    return [rid for rid, _ in res]


def retrieve_full(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    k: int,
) -> list[str]:
    """Retrieve top-k over the whole corpus."""
    res = cosine_top_k(query_emb, corpus_emb, corpus_ids, k)
    return [rid for rid, _ in res]
