"""Phase 2 routing experiment: (a) cascade with threshold τ vs (c) fixed mix.

Both run on the Phase 1 winner (M*, S*) = (Qodo-Embed-1-1.5B, S4).

Routing decisions:

  (a) Cascade with τ
      1. Retrieve top-K with scores from current dataset.
      2. If max_score < τ OR len < k_min → fallback to full corpus.
      3. Return top-K from chosen mode.

  (c) Fixed mix (k1, k2)
      Always return top-k1 from current dataset + top-k2 from OTHER datasets.

Metrics (computed on the routed top-5):
  - recall@5_template_unified       — single per-query recall against
                                      relevant_template (dataset-agnostic),
                                      then macro-avg by current_dataset_id.
  - recall@5_strict_unified         — same on strict.
  - routing_accuracy (cascade only) — fraction of queries where cascade's
                                      decision matches fallback_label:
                                      "in_dataset"     ↔ stay,
                                      "cross_dataset"  ↔ fallback,
                                      "no_match"       ↔ fallback (both modes
                                                                   give nothing,
                                                                   but full
                                                                   corpus is
                                                                   safer).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

from src.config import (
    RECIPES_PATH, QUERIES_PATH, RESULTS_DIR,
    short_model_name, parse_strategy,
)
from src.indexers import serialize as S
from src.indexers.embed import embed_texts
from src.retrievers.dense import cosine_top_k

# Phase 1 winner — set here so Phase 2 always uses the same baseline.
WINNER_MODEL = "Qodo/Qodo-Embed-1-1.5B"
WINNER_STRATEGY = "S4"


# ----------------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------------

def cascade_route(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    corpus_datasets: list[str],
    current_dataset_id: str,
    tau: float,
    k_min: int,
    k_final: int,
) -> tuple[list[str], str]:
    """Cascade routing: try in-dataset first, fallback if confidence low.

    Returns (top_k_recipe_ids, decision) where decision ∈ {"in_dataset", "fallback"}.
    """
    in_ds_allowed = {rid for rid, ds in zip(corpus_ids, corpus_datasets) if ds == current_dataset_id}
    in_ds_results = cosine_top_k(query_emb, corpus_emb, corpus_ids, k=max(k_final, k_min),
                                 allowed_ids=in_ds_allowed)
    if not in_ds_results:
        # No in-dataset recipes at all → fallback
        full_results = cosine_top_k(query_emb, corpus_emb, corpus_ids, k=k_final)
        return [rid for rid, _ in full_results], "fallback"

    max_score = in_ds_results[0][1]
    if max_score < tau or len(in_ds_results) < k_min:
        full_results = cosine_top_k(query_emb, corpus_emb, corpus_ids, k=k_final)
        return [rid for rid, _ in full_results], "fallback"
    return [rid for rid, _ in in_ds_results[:k_final]], "in_dataset"


def mix_route(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    corpus_datasets: list[str],
    current_dataset_id: str,
    k1: int,
    k2: int,
) -> tuple[list[str], str]:
    """Fixed mix routing: k1 from current dataset + k2 from others."""
    in_ds_allowed = {rid for rid, ds in zip(corpus_ids, corpus_datasets) if ds == current_dataset_id}
    other_allowed = {rid for rid, ds in zip(corpus_ids, corpus_datasets) if ds != current_dataset_id}
    top_in = cosine_top_k(query_emb, corpus_emb, corpus_ids, k=k1, allowed_ids=in_ds_allowed)
    top_other = cosine_top_k(query_emb, corpus_emb, corpus_ids, k=k2, allowed_ids=other_allowed)
    ids = [rid for rid, _ in top_in] + [rid for rid, _ in top_other]
    return ids, "mix"


# ----------------------------------------------------------------------------
# Phase 1 baseline (in-dataset only retrieval, no fallback) — for reference
# ----------------------------------------------------------------------------

def in_dataset_only_route(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    corpus_datasets: list[str],
    current_dataset_id: str,
    k_final: int,
) -> tuple[list[str], str]:
    """Phase 1 baseline: just in-dataset top-k, no fallback."""
    allowed = {rid for rid, ds in zip(corpus_ids, corpus_datasets) if ds == current_dataset_id}
    results = cosine_top_k(query_emb, corpus_emb, corpus_ids, k=k_final, allowed_ids=allowed)
    return [rid for rid, _ in results], "in_dataset_only"


# ----------------------------------------------------------------------------
# Metrics for Phase 2
# ----------------------------------------------------------------------------

def _per_query_recall(retrieved: list[str], relevant: list[str]) -> float | None:
    if not relevant:
        return None
    rel_set = set(relevant)
    hits = sum(1 for r in retrieved if r in rel_set)
    return hits / len(rel_set)


def compute_phase2_metrics(
    queries: list[dict],
    routed: dict[str, tuple[list[str], str]],
    label_kind: str = "template",
) -> dict:
    """Returns a metrics bundle for one routing config.

    `routed[query_id]` = (top_k_ids, decision_str).
    """
    label_key = f"relevant_{label_kind}_recipe_ids"
    per_q_recall: dict[str, list[float]] = {}
    for q in queries:
        rel = q[label_key]
        if not rel: continue
        top = routed[q["query_id"]][0]
        r = _per_query_recall(top, rel)
        if r is None: continue
        per_q_recall.setdefault(q["current_dataset_id"], []).append(r)

    per_ds = {ds: mean(vs) for ds, vs in per_q_recall.items() if vs}
    macro = mean(per_ds.values()) if per_ds else 0.0
    all_recalls = [r for vs in per_q_recall.values() for r in vs]
    micro = mean(all_recalls) if all_recalls else 0.0

    return {
        f"recall@5_{label_kind}_macro": macro,
        f"recall@5_{label_kind}_micro": micro,
        f"recall@5_{label_kind}_per_dataset": per_ds,
        f"n_queries_{label_kind}": len(all_recalls),
    }


def compute_routing_accuracy(queries: list[dict], routed: dict[str, tuple[list[str], str]]) -> float:
    """Routing accuracy for cascade.

    Correct = (predicted "fallback" when label is cross_dataset / no_match)
              OR (predicted "in_dataset" when label is in_dataset).
    """
    correct = total = 0
    for q in queries:
        if q["query_id"] not in routed:
            continue
        decision = routed[q["query_id"]][1]
        truth = q.get("fallback_label")
        if truth == "in_dataset":
            ok = decision == "in_dataset"
        else:  # cross_dataset or no_match
            ok = decision == "fallback"
        correct += int(ok)
        total += 1
    return correct / total if total else 0.0


# ----------------------------------------------------------------------------
# Building winner embeddings
# ----------------------------------------------------------------------------

def _winner_corpus_embeddings(
    model: str,
    strategy: str,
    recipes: list[dict],
) -> np.ndarray:
    """Same as src.phase1._build_corpus_embeddings — duplicated here to avoid
    circular import. Reuses cache, so essentially free."""
    cfg = parse_strategy(strategy)
    kind, fmt = cfg["kind"], cfg["format"]
    if kind == "S3":
        descs = [r["description"] for r in recipes]
        schemas = [(S.json_dump(r["recipe"]) if fmt == "F1" else S.flat_kv(r["recipe"])) for r in recipes]
        v_d = embed_texts(model, descs, kind="passage")
        v_s = embed_texts(model, schemas, kind="passage")
        alpha = cfg["alpha"]
        out = alpha * v_d + (1.0 - alpha) * v_s
        n = np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-9)
        return out / n
    texts: list[str] = []
    for r in recipes:
        t = S.text_for_strategy(kind, fmt, r)
        assert isinstance(t, str)
        texts.append(t)
    return embed_texts(model, texts, kind="passage")


# ----------------------------------------------------------------------------
# Top-level runner: τ-sweep cascade + fixed mix + baselines
# ----------------------------------------------------------------------------

def run_phase2(
    tau_grid: list[float] | None = None,
    k_min: int = 3,
    k_final: int = 5,
    mix_configs: list[tuple[int, int]] | None = None,
) -> list[Path]:
    """Run Phase 2 routing experiments and write per-config JSONs."""
    if tau_grid is None:
        # Reasonable defaults based on cosine score scale on L2-normalized embeddings.
        # Will be refined after seeing the score distribution from Phase 1.
        tau_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    if mix_configs is None:
        mix_configs = [(3, 2), (2, 3), (4, 1)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recipes = json.loads(RECIPES_PATH.read_text())
    queries = [json.loads(l) for l in QUERIES_PATH.read_text().splitlines()]

    corpus_emb = _winner_corpus_embeddings(WINNER_MODEL, WINNER_STRATEGY, recipes)
    corpus_ids = [r["recipe_id"] for r in recipes]
    corpus_ds = [r["recipe"]["datasetId"] for r in recipes]
    query_emb = embed_texts(WINNER_MODEL, [q["query"] for q in queries], kind="query")

    out_paths: list[Path] = []

    # Phase 1 in_dataset_only baseline (reference)
    routed = {}
    for i, q in enumerate(queries):
        ids, dec = in_dataset_only_route(
            query_emb[i], corpus_emb, corpus_ids, corpus_ds,
            q["current_dataset_id"], k_final=k_final,
        )
        routed[q["query_id"]] = (ids, dec)
    metrics = {
        **compute_phase2_metrics(queries, routed, "template"),
        **compute_phase2_metrics(queries, routed, "strict"),
        "routing_accuracy": None,
    }
    record = {
        "exp_id": "phase2_baseline_in_dataset_only",
        "phase": 2,
        "config": {
            "router": "in_dataset_only",
            "model": WINNER_MODEL, "strategy": WINNER_STRATEGY,
            "k_final": k_final,
        },
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(recipes),
        "eval_size": len(queries),
    }
    out = RESULTS_DIR / f"{record['exp_id']}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    out_paths.append(out)
    print(f"[done] {record['exp_id']:50s}  recall@5_template_macro={metrics['recall@5_template_macro']:.3f}")

    # (a) Cascade τ-sweep
    for tau in tau_grid:
        routed = {}
        for i, q in enumerate(queries):
            ids, dec = cascade_route(
                query_emb[i], corpus_emb, corpus_ids, corpus_ds,
                q["current_dataset_id"], tau=tau, k_min=k_min, k_final=k_final,
            )
            routed[q["query_id"]] = (ids, dec)
        metrics = {
            **compute_phase2_metrics(queries, routed, "template"),
            **compute_phase2_metrics(queries, routed, "strict"),
            "routing_accuracy": compute_routing_accuracy(queries, routed),
        }
        # Also count fallback rate
        fallback_count = sum(1 for v in routed.values() if v[1] == "fallback")
        metrics["fallback_rate"] = fallback_count / len(routed)
        record = {
            "exp_id": f"phase2_cascade_tau{tau:.2f}",
            "phase": 2,
            "config": {
                "router": "cascade", "tau": tau,
                "k_min": k_min, "k_final": k_final,
                "model": WINNER_MODEL, "strategy": WINNER_STRATEGY,
            },
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus_size": len(recipes),
            "eval_size": len(queries),
        }
        out = RESULTS_DIR / f"{record['exp_id']}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        out_paths.append(out)
        print(
            f"[done] {record['exp_id']:50s}  "
            f"recall@5_template_macro={metrics['recall@5_template_macro']:.3f}  "
            f"routing_acc={metrics['routing_accuracy']:.3f}  "
            f"fb_rate={metrics['fallback_rate']:.2f}"
        )

    # (c) Fixed mix
    for k1, k2 in mix_configs:
        routed = {}
        for i, q in enumerate(queries):
            ids, dec = mix_route(
                query_emb[i], corpus_emb, corpus_ids, corpus_ds,
                q["current_dataset_id"], k1=k1, k2=k2,
            )
            routed[q["query_id"]] = (ids, dec)
        metrics = {
            **compute_phase2_metrics(queries, routed, "template"),
            **compute_phase2_metrics(queries, routed, "strict"),
            "routing_accuracy": None,  # mix doesn't make a fallback decision
        }
        record = {
            "exp_id": f"phase2_mix_k1_{k1}_k2_{k2}",
            "phase": 2,
            "config": {
                "router": "mix", "k1": k1, "k2": k2, "k_final": k1 + k2,
                "model": WINNER_MODEL, "strategy": WINNER_STRATEGY,
            },
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus_size": len(recipes),
            "eval_size": len(queries),
        }
        out = RESULTS_DIR / f"{record['exp_id']}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        out_paths.append(out)
        print(
            f"[done] {record['exp_id']:50s}  "
            f"recall@5_template_macro={metrics['recall@5_template_macro']:.3f}"
        )

    return out_paths


def show_phase2_results():
    """Build summary table for Phase 2 results with per-dataset breakdown."""
    rows = []
    for p in sorted(RESULTS_DIR.glob("phase2_*.json")):
        d = json.loads(p.read_text())
        cfg = d["config"]
        m = d["metrics"]
        per_ds = m.get("recall@5_template_per_dataset", {})
        rows.append({
            "exp_id": d["exp_id"],
            "router": cfg["router"],
            "tau": cfg.get("tau"),
            "k1/k2": f"{cfg.get('k1', '-')}/{cfg.get('k2', '-')}" if cfg["router"] == "mix" else "-",
            "macro":  m["recall@5_template_macro"],
            "films":  per_ds.get("omhpbh1k83ao8"),
            "taxi":   per_ds.get("33h8c3n5nbien"),
            "retail": per_ds.get("b60rhj4luj0y3"),
            "observ": per_ds.get("pzf0mu9kgqz4k"),
            "strict_macro": m["recall@5_strict_macro"],
            "routing_acc":  m.get("routing_accuracy"),
            "fb_rate":      m.get("fallback_rate"),
        })

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("macro", ascending=False).reset_index(drop=True)
    print("== Phase 2 — все routing-конфиги (sorted by macro desc) ==\n")
    print("ВАЖНО: метрика отличается от Phase 1 — здесь recall@5 считается на ВСЁМ relevant_template")
    print("(включая шаблоны из других датасетов), а не на in_dataset-срезе.\n")
    cols = ["exp_id", "router", "tau", "k1/k2", "macro", "films", "taxi", "retail", "observ",
            "strict_macro", "routing_acc", "fb_rate"]
    print(df[cols].to_string(index=False, float_format='%.3f', na_rep='—'))

    best = df.iloc[0]
    print(f"\n→ Лучший роутер: {best['exp_id']}")
    print(f"   macro={best['macro']:.3f}, strict_macro={best['strict_macro']:.3f}")
    return df
