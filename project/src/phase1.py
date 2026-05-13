"""Phase 1 grid runner: 10 strategies × 3 models = 30 experiments.

Each run produces a JSON file in experiments/results/{exp_id}.json.
Idempotent: skips experiments whose result file already exists.

Steps for one (model, strategy):
  1. For each recipe r in the corpus, build the strategy text (or pair of
     texts for S3) and embed it as a "passage" — uses cached embed_texts.
  2. For S3 strategies, combine α·v(desc) + (1-α)·v(schema) and re-normalize.
  3. Embed all queries as "query" — uses cached embed_texts.
  4. For each query, run two retrievals: in-dataset (filtered) and full.
  5. Compute metrics with src.eval.metrics.compute_metrics.
  6. Write the JSON result.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

from src.config import (
    MODELS, STRATEGIES, RECIPES_PATH, QUERIES_PATH, RESULTS_DIR,
    EXTRA_K_VALUES, short_model_name, parse_strategy,
)
from src.indexers import serialize as S
from src.indexers.embed import embed_texts
from src.eval.metrics import compute_metrics
from src.retrievers.dense import retrieve_in_dataset, retrieve_full


def _l2_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-9)
    return x / n


def _build_corpus_embeddings(
    model: str,
    strategy: str,
    recipes: list[dict],
) -> np.ndarray:
    """Return (N, D) corpus embedding matrix for a given strategy.
    Uses embed_texts cache; for S3 combines two embeddings linearly."""
    cfg = parse_strategy(strategy)
    kind = cfg["kind"]
    fmt = cfg["format"]

    if kind == "S3":
        # Embed description and schema separately, then combine with α.
        descs = [r["description"] for r in recipes]
        if fmt == "F1":
            schemas = [S.json_dump(r["recipe"]) for r in recipes]
        else:
            schemas = [S.flat_kv(r["recipe"]) for r in recipes]
        v_desc = embed_texts(model, descs, kind="passage")
        v_schema = embed_texts(model, schemas, kind="passage")
        alpha = cfg["alpha"]
        combined = alpha * v_desc + (1.0 - alpha) * v_schema
        return _l2_norm(combined)

    # S1, S2, S4 — single text per recipe
    texts: list[str] = []
    for r in recipes:
        t = S.text_for_strategy(kind, fmt, r)
        assert isinstance(t, str), f"Strategy {strategy} should produce a single text"
        texts.append(t)
    return embed_texts(model, texts, kind="passage")


def _build_query_embeddings(
    model: str,
    queries: list[dict],
) -> np.ndarray:
    """Embed all queries as 'query' kind."""
    return embed_texts(model, [q["query"] for q in queries], kind="query")


def run_one(
    model: str,
    strategy: str,
    recipes: list[dict],
    queries: list[dict],
    k_top: int = 10,
) -> dict:
    """Run one (model, strategy) experiment, return the result dict.

    `k_top` is the max k we retrieve (must be ≥ max(EXTRA_K_VALUES)).
    """
    by_id = {r["recipe_id"]: r for r in recipes}
    corpus_ids = [r["recipe_id"] for r in recipes]
    corpus_datasets = [r["recipe"]["datasetId"] for r in recipes]

    t0 = time.time()
    corpus_emb = _build_corpus_embeddings(model, strategy, recipes)
    t_corpus = time.time() - t0

    t0 = time.time()
    query_emb = _build_query_embeddings(model, queries)
    t_query = time.time() - t0

    # Retrieve for each query
    retrieved_in_ds: dict[str, list[str]] = {}
    retrieved_full: dict[str, list[str]] = {}
    latencies_ms: list[float] = []
    for i, q in enumerate(queries):
        qe = query_emb[i]
        t0 = time.time()
        retrieved_in_ds[q["query_id"]] = retrieve_in_dataset(
            qe, corpus_emb, corpus_ids, corpus_datasets,
            current_dataset_id=q["current_dataset_id"], k=k_top,
        )
        retrieved_full[q["query_id"]] = retrieve_full(
            qe, corpus_emb, corpus_ids, k=k_top,
        )
        latencies_ms.append((time.time() - t0) * 1000.0)

    metrics = compute_metrics(
        queries=queries,
        retrieved_in_ds=retrieved_in_ds,
        retrieved_full=retrieved_full,
        recipes_by_id=by_id,
        k_values=EXTRA_K_VALUES,
    )

    metrics["latency_ms_p50"] = float(median(latencies_ms)) if latencies_ms else 0.0
    metrics["embed_corpus_seconds"] = round(t_corpus, 2)
    metrics["embed_queries_seconds"] = round(t_query, 2)

    cfg = parse_strategy(strategy)
    return {
        "exp_id": f"{strategy}_{short_model_name(model)}",
        "phase": 1,
        "config": {
            "strategy": strategy,
            "model": model,
            "alpha": cfg["alpha"],
            "format": cfg["format"],
            "k_top": k_top,
        },
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(recipes),
        "eval_size": len(queries),
    }


def run_phase1(
    models: Iterable[str] = MODELS,
    strategies: Iterable[str] = STRATEGIES,
    skip_existing: bool = True,
) -> list[Path]:
    """Iterate the 30-point grid. Returns list of paths written."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recipes = json.loads(RECIPES_PATH.read_text())
    queries = [json.loads(l) for l in QUERIES_PATH.read_text().splitlines()]

    written: list[Path] = []
    for model in models:
        for strategy in strategies:
            exp_id = f"{strategy}_{short_model_name(model)}"
            out = RESULTS_DIR / f"{exp_id}.json"
            if skip_existing and out.exists():
                # Read the cached result to show its key metric, not just "skip".
                try:
                    cached = json.loads(out.read_text())
                    macro = cached["metrics"]["recall@5_template_macro"]
                    print(f"[skip] {exp_id:36s}  recall@5_template_macro={macro:.3f}")
                except Exception:
                    print(f"[skip] {exp_id}")
                continue
            print(f"[run]  {exp_id} ...", flush=True)
            t0 = time.time()
            try:
                result = run_one(model, strategy, recipes, queries)
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                written.append(out)
                m = result["metrics"]
                print(
                    f"       done in {time.time()-t0:.1f}s · "
                    f"recall@5_template_macro={m['recall@5_template_macro']:.3f} · "
                    f"corpus_emb={m['embed_corpus_seconds']}s queries_emb={m['embed_queries_seconds']}s"
                )
            except Exception as e:
                print(f"       FAILED: {type(e).__name__}: {e}")
                raise
    return written
