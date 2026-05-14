"""Recall@k metrics with per-dataset breakdown, macro/micro aggregation,
and split between in-dataset / cross-dataset evaluation modes.

The retrieval system produces, for each query, two top-k lists:
  - top_k_in_ds:    retrieval restricted to current_dataset_id
  - top_k_full:     retrieval over the whole corpus

Metrics:
  recall@k = |retrieved ∩ relevant| / |relevant|  (skip if |relevant| == 0)

In-dataset metric: relevant = relevant_template ∩ {recipes in current dataset}
                   retrieved = top_k_in_ds
Cross-dataset metric (only on T6 / fallback_label == "cross_dataset"):
                   relevant = relevant_template ∩ {recipes in OTHER datasets}
                   retrieved = top_k_full

`relevant_strict` produces a parallel set of metrics (narrow, mostly Hit@k).
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


def _recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Recall@k = |relevant ∩ top-k(retrieved)| / |relevant|.
    Returns None if |relevant| == 0 (caller should skip)."""
    if not relevant:
        return None
    rel_set = set(relevant)
    hits = sum(1 for r in retrieved[:k] if r in rel_set)
    return hits / len(rel_set)


def compute_metrics(
    queries: list[dict],
    retrieved_in_ds: dict[str, list[str]],
    retrieved_full: dict[str, list[str]],
    recipes_by_id: dict[str, dict],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Compute the full metrics bundle.

    Args:
      queries: list of eval records (from queries.jsonl). Each must have
               query_id, current_dataset_id, query_type, fallback_label,
               relevant_strict_recipe_ids, relevant_template_recipe_ids.
      retrieved_in_ds: query_id -> top-K retrieved recipe_ids
                      (retrieval restricted to current_dataset_id).
      retrieved_full:  query_id -> top-K retrieved recipe_ids
                      (retrieval over full corpus).
      recipes_by_id:   recipe_id -> recipe object (for datasetId lookup).
      k_values:        which @k values to compute.

    Returns dict with:
      recall@{K}_template_macro            (main headline at K=5)
      recall@{K}_template_micro
      recall@{K}_template_per_dataset      (dict)
      recall@{K}_template_cross_dataset
      recall@{K}_strict_macro / _micro / _per_dataset
      n_queries_with_template_in_dataset
      n_queries_t6_with_template_cross
    """
    out: dict[str, object] = {}

    def ds_of(recipe_id: str) -> str:
        return recipes_by_id[recipe_id]["recipe"]["datasetId"]

    for k in k_values:
        for label_kind in ("template", "strict"):
            label_key = f"relevant_{label_kind}_recipe_ids"

            # In-dataset: per-query recall, then per-dataset & macro
            per_q_recall: dict[str, list[float]] = defaultdict(list)
            for q in queries:
                cur_ds = q["current_dataset_id"]
                rel_in_cur = [r for r in q[label_key] if ds_of(r) == cur_ds]
                if not rel_in_cur:
                    continue
                top_k = retrieved_in_ds.get(q["query_id"], [])
                r = _recall_at_k(top_k, rel_in_cur, k)
                if r is not None:
                    per_q_recall[cur_ds].append(r)

            per_dataset = {ds: mean(vals) for ds, vals in per_q_recall.items() if vals}
            macro = mean(per_dataset.values()) if per_dataset else 0.0
            all_recalls = [r for vals in per_q_recall.values() for r in vals]
            micro = mean(all_recalls) if all_recalls else 0.0

            out[f"recall@{k}_{label_kind}_macro"] = macro
            out[f"recall@{k}_{label_kind}_micro"] = micro
            out[f"recall@{k}_{label_kind}_per_dataset"] = per_dataset
            if k == 5 and label_kind == "template":
                out["n_queries_with_template_in_dataset"] = len(all_recalls)

            # Cross-dataset: only on T6 / fallback "cross_dataset"
            cross_recalls: list[float] = []
            for q in queries:
                if q.get("fallback_label") != "cross_dataset":
                    continue
                cur_ds = q["current_dataset_id"]
                rel_other = [r for r in q[label_key] if ds_of(r) != cur_ds]
                if not rel_other:
                    continue
                top_k = retrieved_full.get(q["query_id"], [])
                r = _recall_at_k(top_k, rel_other, k)
                if r is not None:
                    cross_recalls.append(r)
            out[f"recall@{k}_{label_kind}_cross_dataset"] = (
                mean(cross_recalls) if cross_recalls else 0.0
            )
            if k == 5 and label_kind == "template":
                out["n_queries_t6_with_template_cross"] = len(cross_recalls)

    return out
