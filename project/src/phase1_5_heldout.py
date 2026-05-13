"""Phase 1.5 — held-out leave-one-out: production-realistic strategy comparison.

Production scenario: user asks for a chart whose exact recipe does NOT exist
in the corpus; the agent retrieves the best STRUCTURALLY similar recipes
as seeds for the downstream agent that generates the new recipe.

Phase 1 measured the easy case (source-recipe IS in corpus). Phase 1.5 simulates
production by removing each recipe in turn from the corpus and checking whether
retrieval finds the OTHER structurally-relevant recipes.

Method (per (model, strategy) pair):

  - T1-T4 (268 queries, main metric):
      For each query q with source_recipe_id=r_X:
        - exclude r_X from the corpus (leave-one-out)
        - retrieve top-5 from current_dataset_id only (in_dataset_only — same
          retrieval mode as Phase 1, so the comparison is apples-to-apples)
        - relevant_held = (relevant_template - {r_X}) ∩ {recipes in current_ds}
        - skip the query if relevant_held is empty (source was the only
          in-dataset structural match — nothing to recover)
        - recall@5 = |top5 ∩ relevant_held| / |relevant_held|
      Aggregate macro by dataset (4 datasets).

  - T5/T6 (38 queries, sidecar — full corpus, no leave-one-out):
      T5 (hard negative twin) and T6 (cross-dataset trigger) are already
      "no exact match" by design, so leave-one-out doesn't apply.
      Retrieval is full corpus here (T6's relevant_template lives in OTHER
      datasets — in_dataset_only would mechanically give 0).

  - T7 ignored (precision metric, separate concern).

Embeddings are reused via the on-disk cache from Phase 1 — running all
30 pairs costs only the matrix ops, not re-embedding.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from src.config import (
    MODELS, STRATEGIES, RECIPES_PATH, QUERIES_PATH, RESULTS_DIR,
    short_model_name, parse_strategy,
)
from src.indexers import serialize as S
from src.indexers.embed import embed_texts
from src.retrievers.dense import cosine_top_k

HELDOUT_TYPES = {"T1", "T2", "T3", "T4"}
SIDECAR_TYPES = {"T5", "T6"}
K_FINAL = 5


def _l2_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-9)
    return x / n


def _build_corpus_embeddings(model: str, strategy: str, recipes: list[dict]) -> np.ndarray:
    cfg = parse_strategy(strategy)
    kind, fmt = cfg["kind"], cfg["format"]
    if kind == "S3":
        descs = [r["description"] for r in recipes]
        schemas = [
            (S.json_dump(r["recipe"]) if fmt == "F1" else S.flat_kv(r["recipe"]))
            for r in recipes
        ]
        v_d = embed_texts(model, descs, kind="passage")
        v_s = embed_texts(model, schemas, kind="passage")
        alpha = cfg["alpha"]
        return _l2_norm(alpha * v_d + (1.0 - alpha) * v_s)
    texts: list[str] = []
    for r in recipes:
        t = S.text_for_strategy(kind, fmt, r)
        assert isinstance(t, str)
        texts.append(t)
    return embed_texts(model, texts, kind="passage")


def _recall(retrieved: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def run_one_heldout(
    model: str,
    strategy: str,
    recipes: list[dict],
    queries: list[dict],
) -> dict:
    """Run one (model, strategy) Phase 1.5 experiment."""
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for r in recipes:
        by_dataset[r["recipe"]["datasetId"]].add(r["recipe_id"])

    corpus_ids = [r["recipe_id"] for r in recipes]
    corpus_emb = _build_corpus_embeddings(model, strategy, recipes)
    query_emb = embed_texts(model, [q["query"] for q in queries], kind="query")

    # T1-T4 held-out: leave-one-out, in_dataset_only retrieval
    heldout_by_ds: dict[str, list[float]] = defaultdict(list)
    heldout_by_type: dict[str, list[float]] = defaultdict(list)
    n_evaluated = 0
    n_skipped_no_relevant = 0

    for i, q in enumerate(queries):
        qt = q["query_type"]
        if qt not in HELDOUT_TYPES:
            continue
        cur_ds = q["current_dataset_id"]
        src = q["source_recipe_id"]
        rel = set(q["relevant_template_recipe_ids"]) - {src}
        rel_in_ds = rel & by_dataset[cur_ds]
        if not rel_in_ds:
            n_skipped_no_relevant += 1
            continue
        results = cosine_top_k(
            query_emb[i], corpus_emb, corpus_ids,
            k=K_FINAL, allowed_ids=by_dataset[cur_ds], exclude_id=src,
        )
        top_ids = [rid for rid, _ in results]
        r = _recall(top_ids, rel_in_ds)
        if r is None:
            continue
        heldout_by_ds[cur_ds].append(r)
        heldout_by_type[qt].append(r)
        n_evaluated += 1

    per_ds = {ds: mean(vs) for ds, vs in heldout_by_ds.items() if vs}
    per_qt = {qt: mean(vs) for qt, vs in heldout_by_type.items() if vs}
    macro = mean(per_ds.values()) if per_ds else 0.0
    all_recalls = [v for vs in heldout_by_ds.values() for v in vs]
    micro = mean(all_recalls) if all_recalls else 0.0

    heldout_metrics = {
        "recall@5_template_macro": macro,
        "recall@5_template_micro": micro,
        "recall@5_template_per_dataset": per_ds,
        "recall@5_template_per_query_type": per_qt,
        "n_queries_evaluated": n_evaluated,
        "n_queries_skipped_no_relevant": n_skipped_no_relevant,
    }

    # Sidecar T5/T6 — full corpus retrieval, no leave-one-out
    t5_template: list[float] = []
    t6_template: list[float] = []
    t6_strict: list[float] = []

    for i, q in enumerate(queries):
        qt = q["query_type"]
        if qt not in SIDECAR_TYPES:
            continue
        results = cosine_top_k(query_emb[i], corpus_emb, corpus_ids, k=K_FINAL)
        top_ids = [rid for rid, _ in results]
        rel_t = set(q["relevant_template_recipe_ids"])
        r_t = _recall(top_ids, rel_t)
        if r_t is not None:
            if qt == "T5":
                t5_template.append(r_t)
            else:
                t6_template.append(r_t)
        if qt == "T6":
            r_s = _recall(top_ids, set(q["relevant_strict_recipe_ids"]))
            if r_s is not None:
                t6_strict.append(r_s)

    sidecar_metrics = {
        "T5_recall@5_template": mean(t5_template) if t5_template else 0.0,
        "T5_n": len(t5_template),
        "T6_recall@5_template_full": mean(t6_template) if t6_template else 0.0,
        "T6_recall@5_strict_full": mean(t6_strict) if t6_strict else 0.0,
        "T6_n": len(t6_template),
    }

    cfg = parse_strategy(strategy)
    return {
        "exp_id": f"phase1_5_{strategy}_{short_model_name(model)}",
        "phase": 1.5,
        "config": {
            "strategy": strategy,
            "model": model,
            "alpha": cfg["alpha"],
            "format": cfg["format"],
            "k_top": K_FINAL,
            "retrieval_mode": "in_dataset_only",
            "holdout": "leave_one_out",
        },
        "metrics": {
            "heldout": heldout_metrics,
            "sidecar": sidecar_metrics,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(recipes),
        "eval_size": len(queries),
    }


def run_phase1_5(
    models: Iterable[str] = MODELS,
    strategies: Iterable[str] = STRATEGIES,
    skip_existing: bool = True,
) -> list[Path]:
    """Iterate the 30-point grid in held-out mode. Returns paths written."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recipes = json.loads(RECIPES_PATH.read_text())
    queries = [json.loads(l) for l in QUERIES_PATH.read_text().splitlines()]

    written: list[Path] = []
    for model in models:
        for strategy in strategies:
            exp_id = f"phase1_5_{strategy}_{short_model_name(model)}"
            out = RESULTS_DIR / f"{exp_id}.json"
            if skip_existing and out.exists():
                try:
                    cached = json.loads(out.read_text())
                    macro = cached["metrics"]["heldout"]["recall@5_template_macro"]
                    print(f"[skip] {exp_id:50s}  heldout_macro={macro:.3f}")
                except Exception:
                    print(f"[skip] {exp_id}")
                continue
            print(f"[run]  {exp_id} ...", flush=True)
            t0 = time.time()
            try:
                result = run_one_heldout(model, strategy, recipes, queries)
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                written.append(out)
                m = result["metrics"]["heldout"]
                print(
                    f"       done in {time.time()-t0:.1f}s · "
                    f"heldout_macro={m['recall@5_template_macro']:.3f} · "
                    f"n_eval={m['n_queries_evaluated']} · "
                    f"n_skipped={m['n_queries_skipped_no_relevant']}"
                )
            except Exception as e:
                print(f"       FAILED: {type(e).__name__}: {e}")
                raise
    return written


def show_phase1_5_results():
    """Summary tables across all 30 (M, S) pairs, plus Phase 1 comparison."""
    rows = []
    for p in sorted(RESULTS_DIR.glob("phase1_5_*.json")):
        d = json.loads(p.read_text())
        h = d["metrics"]["heldout"]
        s = d["metrics"]["sidecar"]
        per_ds = h.get("recall@5_template_per_dataset", {})
        per_qt = h.get("recall@5_template_per_query_type", {})
        rows.append({
            "exp_id":   d["exp_id"].replace("phase1_5_", ""),
            "strategy": d["config"]["strategy"],
            "model":    short_model_name(d["config"]["model"]),
            "ho_macro": h["recall@5_template_macro"],
            "ho_micro": h["recall@5_template_micro"],
            "films":    per_ds.get("omhpbh1k83ao8"),
            "taxi":     per_ds.get("33h8c3n5nbien"),
            "retail":   per_ds.get("b60rhj4luj0y3"),
            "observ":   per_ds.get("pzf0mu9kgqz4k"),
            "min_ds":   min(per_ds.values()) if per_ds else None,
            "T1": per_qt.get("T1"), "T2": per_qt.get("T2"),
            "T3": per_qt.get("T3"), "T4": per_qt.get("T4"),
            "T5":        s.get("T5_recall@5_template"),
            "T6_tmpl":   s.get("T6_recall@5_template_full"),
            "T6_strict": s.get("T6_recall@5_strict_full"),
        })
    if not rows:
        print("Нет результатов Phase 1.5 — запусти run_phase1_5() сначала.")
        return None

    import pandas as pd
    df = pd.DataFrame(rows)

    # 1) Pivot strategy × model на главной метрике
    print("== Pivot Phase 1.5: held-out recall@5_template_macro (rows=strategy, cols=model) ==\n")
    pivot = df.pivot(index="strategy", columns="model", values="ho_macro").round(3)
    pivot = pivot.reindex(STRATEGIES)
    print(pivot.to_string())
    print()

    # 2) Per-model breakdown — все 10 стратегий с per-dataset + per-query-type
    cols_show = ["strategy", "films", "taxi", "retail", "observ",
                 "ho_macro", "min_ds", "T1", "T2", "T3", "T4",
                 "T5", "T6_tmpl", "T6_strict"]
    strategy_order = {s: i for i, s in enumerate(STRATEGIES)}
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        sub["_order"] = sub["strategy"].map(strategy_order)
        sub = sub.sort_values("_order")
        print(f"== {model} — все 10 стратегий (Phase 1.5 held-out) ==\n")
        print(sub[cols_show].to_string(index=False, float_format='%.3f', na_rep='—'))
        print()

    # 3) TOP-10 over all 30
    print("== TOP-10 by held-out recall@5_template_macro (tie-break min_ds desc) ==\n")
    top = df.sort_values(["ho_macro", "min_ds"], ascending=[False, False]).head(10)
    cols = ["exp_id", "ho_macro", "min_ds", "films", "taxi", "retail", "observ",
            "T5", "T6_tmpl", "T6_strict"]
    print(top[cols].to_string(index=False, float_format='%.3f', na_rep='—'))
    print()

    # 3) Сравнение с Phase 1 — насколько Phase 1 «надувал» цифры
    p1_rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        name = p.name
        if name.startswith("phase1_5_") or name.startswith("phase2_"):
            continue
        try:
            d = json.loads(p.read_text())
            if d.get("phase") != 1:
                continue
            p1_rows.append({
                "exp_id": d["exp_id"],
                "p1_macro": d["metrics"]["recall@5_template_macro"],
            })
        except Exception:
            continue
    if p1_rows:
        p1_df = pd.DataFrame(p1_rows)
        merged = df.merge(p1_df, on="exp_id", how="left")
        merged["delta"] = merged["ho_macro"] - merged["p1_macro"]
        merged = merged.sort_values("ho_macro", ascending=False)
        print("== Phase 1 → Phase 1.5: насколько 'надулась' оценка ==\n")
        cmp_cols = ["exp_id", "p1_macro", "ho_macro", "delta"]
        print(merged[cmp_cols].to_string(index=False, float_format='%.3f', na_rep='—'))
        print()

    # 4) Per-query-type breakdown победителя
    winner = df.sort_values(["ho_macro", "min_ds"], ascending=[False, False]).iloc[0]
    print(f"→ Phase 1.5 победитель: {winner['exp_id']}")
    print(f"   held-out macro={winner['ho_macro']:.3f}, min_ds={winner['min_ds']:.3f}")
    print(f"   T1={winner['T1']:.3f}  T2={winner['T2']:.3f}  "
          f"T3={winner['T3']:.3f}  T4={winner['T4']:.3f}")
    print(f"   sidecar: T5={winner['T5']:.3f}  "
          f"T6_template={winner['T6_tmpl']:.3f}  T6_strict={winner['T6_strict']:.3f}")
    return df
