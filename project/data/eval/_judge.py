"""Judge-pass: produces data/eval/queries.jsonl with relevance labels.

Approach:
- Pre-compute per-recipe template-compatible candidates with FUZZY matching
  (see _compute_signatures.py → data/recipe_template_candidates.json).
- For each query, the template_candidates = compat-list of source recipe.
  Average ~5–6 candidates per query, matching user's expectation that the
  retriever should return ~5 relevant schemas per query.

Labels per query:
  - relevant_template_recipe_ids: structurally compatible recipes (chart type
    group + role pattern with ≤1 fuzz for tables + color-agnostic for
    bars/lines). Includes source for T1–T6.
  - relevant_strict_recipe_ids: subset of template that also semantically
    matches the query. Mostly {source} for T1–T4 + curated extensions for
    semantically duplicate recipes (Wizard L/M/S, error-action overlap, etc.)
    For T5 = {} (decoy by design). For T6 = {source}. For T7 = {}.

fallback_label:
  - "in_dataset" if any strict in current_dataset_id
  - "cross_dataset" if no strict in cur, but template exists in other datasets
  - "no_match" otherwise
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = ROOT / "data" / "recipes.json"
COMPAT_PATH = ROOT / "data" / "recipe_template_candidates.json"
QUERIES_RAW_PATH = ROOT / "data" / "eval" / "queries_raw.jsonl"
OUT_PATH = ROOT / "data" / "eval" / "queries.jsonl"


# Wizard L/M/S — same chart at three sizes; queries for any match all three.
WIZARD_TRIO = {"r_0052", "r_0053", "r_0054"}

# r_0055 (errors by action, excludes data fetches) ↔ r_0059 (data errors by
# action). Generic queries match both; T3 of r_0055 explicitly excludes data.
ERRORS_ACTION_GENERIC_TYPES = {"T1", "T2", "T4"}

# r_0065 (5xx by service) ⊃ r_0066 (500 in BI by service). Generic queries
# match both; T3 with explicit responseStatus narrows to source.
ERRORS_5XX_GENERIC_TYPES = {"T1", "T2", "T4"}

# r_0001 (avg rating series) ⊂ r_0005 (rating by Тип multi-line, covers all
# media types including series). Add r_0005 to r_0001 queries.
RATING_SERIES_TYPES = {"T1", "T2", "T3", "T4"}


def build_extensions(by_src: dict[str, list[dict]]) -> dict[str, set[str]]:
    """Return dict[query_id] -> set of additional recipe_ids that should be
    in relevant_strict beyond the source recipe."""
    ext: dict[str, set[str]] = {}

    # Wizard trio: any T1–T4 query for one matches all three
    for rid in WIZARD_TRIO:
        for q in by_src.get(rid, []):
            if q["query_type"] in {"T1", "T2", "T3", "T4"}:
                ext[q["query_id"]] = WIZARD_TRIO - {rid}

    # r_0055 → +r_0059 for generic queries
    for q in by_src.get("r_0055", []):
        if q["query_type"] in ERRORS_ACTION_GENERIC_TYPES:
            ext.setdefault(q["query_id"], set()).add("r_0059")

    # r_0065 ↔ r_0066 (500 ⊂ 5xx)
    for q in by_src.get("r_0065", []):
        if q["query_type"] in ERRORS_5XX_GENERIC_TYPES:
            ext.setdefault(q["query_id"], set()).add("r_0066")
    for q in by_src.get("r_0066", []):
        if q["query_type"] in ERRORS_5XX_GENERIC_TYPES:
            ext.setdefault(q["query_id"], set()).add("r_0065")

    # r_0001 → +r_0005
    for q in by_src.get("r_0001", []):
        if q["query_type"] in RATING_SERIES_TYPES:
            ext.setdefault(q["query_id"], set()).add("r_0005")

    return ext


def main() -> None:
    recipes = json.loads(RECIPES_PATH.read_text())
    by_id = {r["recipe_id"]: r for r in recipes}
    compat: dict[str, list[str]] = json.loads(COMPAT_PATH.read_text())
    queries_raw = [json.loads(l) for l in QUERIES_RAW_PATH.read_text().splitlines()]

    by_src: dict[str, list[dict]] = {}
    for q in queries_raw:
        src = q.get("source_recipe_id")
        if src:
            by_src.setdefault(src, []).append(q)

    extensions = build_extensions(by_src)

    out_records = []
    for q in queries_raw:
        qtype = q["query_type"]
        src = q.get("source_recipe_id")
        cur_ds = q["current_dataset_id"]

        # template
        if src is None:
            template = []
        else:
            template = list(compat[src])

        # strict
        if qtype == "T7":
            strict = []
        elif qtype == "T5":
            strict = sorted(extensions.get(q["query_id"], set()))
        elif qtype in {"T1", "T2", "T3", "T4", "T6"}:
            base = {src} if src else set()
            base |= extensions.get(q["query_id"], set())
            strict = sorted(base)
        else:
            strict = []

        # fallback_label
        strict_in_cur = any(by_id[r]["recipe"]["datasetId"] == cur_ds for r in strict)
        template_in_other = any(
            by_id[r]["recipe"]["datasetId"] != cur_ds for r in template
        )
        if strict_in_cur:
            fallback = "in_dataset"
        elif template_in_other:
            fallback = "cross_dataset"
        else:
            fallback = "no_match"

        rec = {
            "query_id": q["query_id"],
            "query": q["query"],
            "current_dataset_id": cur_ds,
            "query_type": qtype,
            "source_recipe_id": src,
            "fallback_label": fallback,
            "relevant_strict_recipe_ids": sorted(strict),
            "relevant_template_recipe_ids": sorted(template),
        }
        out_records.append(rec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    print(f"Wrote {len(out_records)} records to {OUT_PATH}")
    print()
    fb = Counter(r["fallback_label"] for r in out_records)
    print(f"fallback_label distribution: {dict(fb)}")
    avg_strict = sum(len(r["relevant_strict_recipe_ids"]) for r in out_records) / len(out_records)
    avg_template = sum(len(r["relevant_template_recipe_ids"]) for r in out_records) / len(out_records)
    print(f"Avg |strict|   = {avg_strict:.2f}")
    print(f"Avg |template| = {avg_template:.2f}")
    print()
    print("By query_type:")
    for qt in ["T1","T2","T3","T4","T5","T6","T7"]:
        recs = [r for r in out_records if r["query_type"] == qt]
        if not recs: continue
        avg_s = sum(len(r["relevant_strict_recipe_ids"]) for r in recs) / len(recs)
        avg_t = sum(len(r["relevant_template_recipe_ids"]) for r in recs) / len(recs)
        print(f"  {qt}: n={len(recs):3d}  avg_strict={avg_s:.2f}  avg_template={avg_t:.2f}")


if __name__ == "__main__":
    main()
