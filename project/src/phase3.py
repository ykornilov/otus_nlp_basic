"""Phase 3 — лексические/гибридные baselines поверх Phase 1.5 winner.

Цель: атаковать конкретные дыры Phase 1.5 на (M*, S*) = (Qodo-Embed-1-1.5B, S4):
  - taxi in-dataset (held-out macro = 0.239) — dense ставит тематически-близкие
    выше структурно-близких.
  - T6 cross-dataset (recall@5_template = 0.150) — dense из коробки не работает
    на cross-domain структурном поиске.

Eval-сетап — тот же, что и Phase 1.5 (production-realistic):
  - T1–T4: in_dataset_only retrieval с leave-one-out source-рецепта.
  - T5/T6: sidecar на full corpus retrieval, без leave-one-out.

Эксперименты:
  1) BM25 alone (description + linearize_to_text, regex-tokenized, lowercase).
     Лексический поиск без эмбеддингов.
  2) Hybrid (BM25 + dense via RRF): top-20 от BM25 + top-20 от Qodo+S4 →
     Reciprocal Rank Fusion (k=60) → top-5.

Эмбеддинги dense переиспользуются из кеша Phase 1; BM25-индекс строится
один раз (для маленького корпуса 67 — пренебрежимо быстро).
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
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

# Winner pair from Phase 1.5 (same as Phase 2 winner).
WINNER_MODEL = "Qodo/Qodo-Embed-1-1.5B"
WINNER_STRATEGY = "S4"

HELDOUT_TYPES = {"T1", "T2", "T3", "T4"}
SIDECAR_TYPES = {"T5", "T6"}
K_FINAL = 5
TOP_K_BM25 = 20      # candidates from BM25 for fusion
TOP_K_DENSE = 20     # candidates from dense for fusion
RRF_K_CONSTANT = 60  # standard RRF


# ----------------------------------------------------------------------------
# BM25 helpers
# ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Simple regex tokenizer: lowercased Unicode word characters.
    Robust to mixed RU/EN corpora; no morphological normalization (corpus is
    small enough that lemmatization didn't help in pilot).
    """
    return _TOKEN_RE.findall(text.lower())


def _build_bm25(recipes: list[dict]):
    """Build BM25 index over S4-serialized recipe text."""
    from rank_bm25 import BM25Okapi
    texts = [S.text_for_strategy("S4", None, r) for r in recipes]
    tokens = [_tokenize(t) for t in texts]
    return BM25Okapi(tokens)


def _bm25_topk(
    bm25,
    query_tokens: list[str],
    corpus_ids: list[str],
    k: int,
    allowed_ids: set[str] | None = None,
    exclude_id: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k by BM25, with allowed/exclude filters (matches dense API shape)."""
    scores = bm25.get_scores(query_tokens)
    candidates: list[tuple[str, float]] = []
    for i, rid in enumerate(corpus_ids):
        if exclude_id is not None and rid == exclude_id:
            continue
        if allowed_ids is not None and rid not in allowed_ids:
            continue
        candidates.append((rid, float(scores[i])))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:k]


# ----------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ----------------------------------------------------------------------------

def _rrf_fuse(rank_lists: list[list[str]], k_constant: int = RRF_K_CONSTANT,
              k_final: int = K_FINAL) -> list[str]:
    """Reciprocal Rank Fusion. Each rank list is recipe_ids sorted best-first.
    Score = sum over lists of 1 / (k_constant + rank_in_list)."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for r, rid in enumerate(ranks):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k_constant + r + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [rid for rid, _ in fused[:k_final]]


# ----------------------------------------------------------------------------
# Dense winner embeddings (duplicated from phase1/2 to avoid circular imports)
# ----------------------------------------------------------------------------

def _winner_corpus_embeddings(recipes: list[dict]) -> np.ndarray:
    cfg = parse_strategy(WINNER_STRATEGY)
    kind, fmt = cfg["kind"], cfg["format"]
    if kind == "S3":
        # Not used for our winner (S4), kept for completeness.
        descs = [r["description"] for r in recipes]
        schemas = [(S.json_dump(r["recipe"]) if fmt == "F1" else S.flat_kv(r["recipe"]))
                   for r in recipes]
        v_d = embed_texts(WINNER_MODEL, descs, kind="passage")
        v_s = embed_texts(WINNER_MODEL, schemas, kind="passage")
        alpha = cfg["alpha"]
        out = alpha * v_d + (1.0 - alpha) * v_s
        n = np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-9)
        return out / n
    texts = [S.text_for_strategy(kind, fmt, r) for r in recipes]
    return embed_texts(WINNER_MODEL, texts, kind="passage")


# ----------------------------------------------------------------------------
# Generic Phase 1.5 held-out evaluation harness
# ----------------------------------------------------------------------------

def _eval_heldout(
    retrieve_in_ds_fn,    # (query_idx, current_dataset_id, exclude_id, k) -> list[recipe_id]
    retrieve_full_fn,     # (query_idx, k) -> list[recipe_id]
    recipes: list[dict],
    queries: list[dict],
) -> dict:
    """Evaluates a retrieval system under Phase 1.5 setup:
    - T1-T4: in_dataset_only with leave-one-out source.
    - T5/T6: full corpus retrieval, no held-out.
    Returns the same metrics dict shape as Phase 1.5."""
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for r in recipes:
        by_dataset[r["recipe"]["datasetId"]].add(r["recipe_id"])

    # T1-T4 held-out
    heldout_by_ds: dict[str, list[float]] = defaultdict(list)
    heldout_by_type: dict[str, list[float]] = defaultdict(list)
    n_evaluated = 0
    n_skipped = 0

    for i, q in enumerate(queries):
        qt = q["query_type"]
        if qt not in HELDOUT_TYPES:
            continue
        cur_ds = q["current_dataset_id"]
        src = q["source_recipe_id"]
        rel = set(q["relevant_template_recipe_ids"]) - {src}
        rel_in_ds = rel & by_dataset[cur_ds]
        if not rel_in_ds:
            n_skipped += 1
            continue
        top_ids = retrieve_in_ds_fn(i, cur_ds, src, K_FINAL)
        recall = sum(1 for r in top_ids if r in rel_in_ds) / len(rel_in_ds)
        heldout_by_ds[cur_ds].append(recall)
        heldout_by_type[qt].append(recall)
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
        "n_queries_skipped_no_relevant": n_skipped,
    }

    # T5/T6 sidecar — full corpus, no held-out
    t5_template: list[float] = []
    t6_template: list[float] = []
    t6_strict: list[float] = []

    for i, q in enumerate(queries):
        qt = q["query_type"]
        if qt not in SIDECAR_TYPES:
            continue
        top_ids = retrieve_full_fn(i, K_FINAL)
        rel_t = set(q["relevant_template_recipe_ids"])
        if rel_t:
            r_t = sum(1 for r in top_ids if r in rel_t) / len(rel_t)
            (t5_template if qt == "T5" else t6_template).append(r_t)
        if qt == "T6":
            rel_s = set(q["relevant_strict_recipe_ids"])
            if rel_s:
                r_s = sum(1 for r in top_ids if r in rel_s) / len(rel_s)
                t6_strict.append(r_s)

    sidecar_metrics = {
        "T5_recall@5_template": mean(t5_template) if t5_template else 0.0,
        "T5_n": len(t5_template),
        "T6_recall@5_template_full": mean(t6_template) if t6_template else 0.0,
        "T6_recall@5_strict_full": mean(t6_strict) if t6_strict else 0.0,
        "T6_n": len(t6_template),
    }

    return {"heldout": heldout_metrics, "sidecar": sidecar_metrics}


# ----------------------------------------------------------------------------
# Experiment 1: BM25 alone
# ----------------------------------------------------------------------------

def run_bm25(recipes: list[dict], queries: list[dict]) -> dict:
    bm25 = _build_bm25(recipes)
    corpus_ids = [r["recipe_id"] for r in recipes]
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for r in recipes:
        by_dataset[r["recipe"]["datasetId"]].add(r["recipe_id"])
    query_tokens = [_tokenize(q["query"]) for q in queries]

    def retrieve_in_ds(i, cur_ds, exclude_id, k):
        res = _bm25_topk(bm25, query_tokens[i], corpus_ids, k,
                         allowed_ids=by_dataset[cur_ds], exclude_id=exclude_id)
        return [rid for rid, _ in res]

    def retrieve_full(i, k):
        res = _bm25_topk(bm25, query_tokens[i], corpus_ids, k)
        return [rid for rid, _ in res]

    metrics = _eval_heldout(retrieve_in_ds, retrieve_full, recipes, queries)
    return {
        "exp_id": "phase3_bm25_S4",
        "phase": 3,
        "config": {
            "approach": "bm25",
            "text_strategy": "S4",
            "tokenizer": "regex_word_lowercase",
            "k_final": K_FINAL,
            "retrieval_mode": "in_dataset_only",
            "holdout": "leave_one_out",
        },
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(recipes),
        "eval_size": len(queries),
    }


# ----------------------------------------------------------------------------
# Experiment 2: Hybrid (BM25 + dense, RRF)
# ----------------------------------------------------------------------------

def run_hybrid_rrf(recipes: list[dict], queries: list[dict]) -> dict:
    bm25 = _build_bm25(recipes)
    corpus_ids = [r["recipe_id"] for r in recipes]
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for r in recipes:
        by_dataset[r["recipe"]["datasetId"]].add(r["recipe_id"])
    query_tokens = [_tokenize(q["query"]) for q in queries]

    corpus_emb = _winner_corpus_embeddings(recipes)
    query_emb = embed_texts(WINNER_MODEL, [q["query"] for q in queries], kind="query")

    def retrieve_in_ds(i, cur_ds, exclude_id, k):
        bm25_res = _bm25_topk(bm25, query_tokens[i], corpus_ids, TOP_K_BM25,
                              allowed_ids=by_dataset[cur_ds], exclude_id=exclude_id)
        bm25_rank = [rid for rid, _ in bm25_res]
        dense_res = cosine_top_k(query_emb[i], corpus_emb, corpus_ids,
                                 k=TOP_K_DENSE,
                                 allowed_ids=by_dataset[cur_ds],
                                 exclude_id=exclude_id)
        dense_rank = [rid for rid, _ in dense_res]
        return _rrf_fuse([bm25_rank, dense_rank], k_final=k)

    def retrieve_full(i, k):
        bm25_res = _bm25_topk(bm25, query_tokens[i], corpus_ids, TOP_K_BM25)
        bm25_rank = [rid for rid, _ in bm25_res]
        dense_res = cosine_top_k(query_emb[i], corpus_emb, corpus_ids, k=TOP_K_DENSE)
        dense_rank = [rid for rid, _ in dense_res]
        return _rrf_fuse([bm25_rank, dense_rank], k_final=k)

    metrics = _eval_heldout(retrieve_in_ds, retrieve_full, recipes, queries)
    return {
        "exp_id": "phase3_hybrid_rrf",
        "phase": 3,
        "config": {
            "approach": "hybrid_rrf",
            "text_strategy": "S4",
            "tokenizer": "regex_word_lowercase",
            "dense_model": WINNER_MODEL,
            "dense_strategy": WINNER_STRATEGY,
            "bm25_top_k": TOP_K_BM25,
            "dense_top_k": TOP_K_DENSE,
            "rrf_k_constant": RRF_K_CONSTANT,
            "k_final": K_FINAL,
            "retrieval_mode": "in_dataset_only",
            "holdout": "leave_one_out",
        },
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(recipes),
        "eval_size": len(queries),
    }


# ----------------------------------------------------------------------------
# Top-level runner
# ----------------------------------------------------------------------------

EXPERIMENTS = [
    ("phase3_bm25_S4", run_bm25),
    ("phase3_hybrid_rrf", run_hybrid_rrf),
]


def run_phase3(skip_existing: bool = True) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recipes = json.loads(RECIPES_PATH.read_text())
    queries = [json.loads(l) for l in QUERIES_PATH.read_text().splitlines()]

    written: list[Path] = []
    for exp_id, runner in EXPERIMENTS:
        out = RESULTS_DIR / f"{exp_id}.json"
        if skip_existing and out.exists():
            try:
                cached = json.loads(out.read_text())
                macro = cached["metrics"]["heldout"]["recall@5_template_macro"]
                print(f"[skip] {exp_id:40s}  heldout_macro={macro:.3f}")
            except Exception:
                print(f"[skip] {exp_id}")
            continue
        print(f"[run]  {exp_id} ...", flush=True)
        t0 = time.time()
        try:
            result = runner(recipes, queries)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            written.append(out)
            m = result["metrics"]["heldout"]
            s = result["metrics"]["sidecar"]
            print(
                f"       done in {time.time()-t0:.1f}s · "
                f"heldout_macro={m['recall@5_template_macro']:.3f} · "
                f"T6_tmpl={s['T6_recall@5_template_full']:.3f} · "
                f"n_eval={m['n_queries_evaluated']}"
            )
        except Exception as e:
            print(f"       FAILED: {type(e).__name__}: {e}")
            raise
    return written


# ----------------------------------------------------------------------------
# Results summary
# ----------------------------------------------------------------------------

def _row_from_result(d: dict, label: str) -> dict:
    h = d["metrics"]["heldout"]
    s = d["metrics"]["sidecar"]
    per_ds = h.get("recall@5_template_per_dataset", {})
    per_qt = h.get("recall@5_template_per_query_type", {})
    return {
        "approach":  label,
        "macro":     h["recall@5_template_macro"],
        "min_ds":    min(per_ds.values()) if per_ds else None,
        "films":     per_ds.get("omhpbh1k83ao8"),
        "taxi":      per_ds.get("33h8c3n5nbien"),
        "retail":    per_ds.get("b60rhj4luj0y3"),
        "observ":    per_ds.get("pzf0mu9kgqz4k"),
        "T1": per_qt.get("T1"), "T2": per_qt.get("T2"),
        "T3": per_qt.get("T3"), "T4": per_qt.get("T4"),
        "T5":        s.get("T5_recall@5_template"),
        "T6_tmpl":   s.get("T6_recall@5_template_full"),
        "T6_strict": s.get("T6_recall@5_strict_full"),
    }


def show_phase3_results():
    """Phase 3 vs Phase 1.5 winner — side-by-side."""
    import pandas as pd

    rows: list[dict] = []

    # Phase 1.5 baseline (winner)
    p15 = RESULTS_DIR / "phase1_5_S4_Qodo-Embed-1-1.5B.json"
    if p15.exists():
        rows.append(_row_from_result(json.loads(p15.read_text()), "dense S4+Qodo (P1.5)"))

    # Phase 3 experiments
    for exp_id, _ in EXPERIMENTS:
        p = RESULTS_DIR / f"{exp_id}.json"
        if not p.exists():
            continue
        label = exp_id.replace("phase3_", "")
        rows.append(_row_from_result(json.loads(p.read_text()), label))

    if not rows:
        print("Нет результатов Phase 3 — запусти run_phase3() сначала.")
        return None

    df = pd.DataFrame(rows)

    # Per-dataset taxonomy
    print("== Phase 3 vs Phase 1.5 baseline (held-out, T1–T4 main; T5/T6 sidecar) ==\n")
    cols_main = ["approach", "macro", "min_ds", "films", "taxi", "retail", "observ"]
    print(df[cols_main].to_string(index=False, float_format='%.3f', na_rep='—'))
    print()

    print("== Per query-type (held-out T1–T4) + sidecar T5/T6 ==\n")
    cols_qt = ["approach", "T1", "T2", "T3", "T4", "T5", "T6_tmpl", "T6_strict"]
    print(df[cols_qt].to_string(index=False, float_format='%.3f', na_rep='—'))
    print()

    # Delta vs baseline
    if rows[0]["approach"].startswith("dense"):
        base = df.iloc[0]
        print(f"== Δ vs dense S4+Qodo baseline (positive = approach лучше) ==\n")
        delta_rows = []
        for _, r in df.iloc[1:].iterrows():
            delta_rows.append({
                "approach": r["approach"],
                "Δ_macro":  r["macro"] - base["macro"],
                "Δ_taxi":   r["taxi"] - base["taxi"],
                "Δ_retail": r["retail"] - base["retail"],
                "Δ_observ": r["observ"] - base["observ"],
                "Δ_T6_tmpl":   r["T6_tmpl"] - base["T6_tmpl"],
                "Δ_T6_strict": r["T6_strict"] - base["T6_strict"],
            })
        ddf = pd.DataFrame(delta_rows)
        print(ddf.to_string(index=False, float_format='%+.3f'))
    return df
