"""Compute per-recipe template-compatible candidates with FUZZY matching.

Replaces the previous equivalence-class signature approach. Outputs:
  - data/recipe_template_candidates.json — for each recipe_id, the list of
    UP TO K=7 most-similar template-compatible recipe_ids (sorted by
    similarity to source). The compatibility relation is symmetric, but
    after the K-cap the resulting per-recipe lists may not be — A may keep
    B in its top-7 while B prefers other neighbours.

Why K=7: bigger structural clusters reach 14+ recipes; recall@5 caps at
5/14 = 36% there, hurting model discrimination. K=7 keeps avg template
size near 5 while letting the metric reach 1.0 on most queries.

Compatibility rules (matching project.ipynb spec, with explicit relaxations
to keep the average template-candidate count near ~5 per query):

  - Multi-layer: recipes must have the same number of layers; layers are
    matched after sorting by chart-type group; all pairs must be compatible.

  Per layer, by chart-type group:
  - bars / lines (column/bar/area/line/*100p): SET of x roles and SET of y
    roles must match. Color presence/role and FIELD COUNT ignored — stacked
    vs plain, single-y vs multi-y with same role(s) are compatible templates.
  - circles (donut, pie): SET of measures roles must match (count ignored).
  - metric: SET of measures roles must match.
  - flatTable: column-role MULTISETS may differ by ≤2 (one step beyond
    spec's ≤1 — the corpus has 4-col tables that differ pairwise by
    exactly 2 in role mix; with ≤1 those stayed singletons).
  - pivotTable: combined multiset of (rows ∪ columns ∪ measures) may differ
    by ≤2 (same relaxation as flatTable for symmetry; spec said "по всем
    pivot-axes" without specific tolerance).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = ROOT / "data" / "recipes.json"
ROLES_PATH = ROOT / "data" / "field_roles.json"
OUT_PATH = ROOT / "data" / "recipe_template_candidates.json"


TYPE_GROUP: dict[str, str] = {
    "column": "bars", "bar": "bars",
    "column100p": "bars", "bar100p": "bars",
    "line": "lines", "area": "lines", "area100p": "lines",
    "donut": "circles", "pie": "circles",
    "metric": "metric",
    "flatTable": "flatTable",
    "pivotTable": "pivotTable",
}


def _roles_at(layer, position, roles, ds):
    items = layer.get(position) or []
    return [
        roles.get(ds, {}).get(it.get("title", ""), "categorical")
        for it in items if it.get("title")
    ]


def _layer_repr(layer, roles, ds):
    """Layer-level compatibility data — kind-tagged tuple for matching."""
    g = TYPE_GROUP.get(layer.get("type"), layer.get("type", "?"))
    if g in ("bars", "lines"):
        return ("xy", g,
                frozenset(_roles_at(layer, "x", roles, ds)),
                frozenset(_roles_at(layer, "y", roles, ds)))
    if g in ("circles", "metric"):
        return ("measures", g,
                frozenset(_roles_at(layer, "measures", roles, ds)))
    if g == "flatTable":
        return ("multiset_columns", g,
                Counter(_roles_at(layer, "columns", roles, ds)))
    if g == "pivotTable":
        combined = (Counter(_roles_at(layer, "rows", roles, ds))
                    + Counter(_roles_at(layer, "columns", roles, ds))
                    + Counter(_roles_at(layer, "measures", roles, ds)))
        return ("multiset_pivot", g, combined)
    return ("?", g)


def _layers_compat(la, lb) -> bool:
    if la[0] != lb[0] or la[1] != lb[1]:
        return False
    kind = la[0]
    if kind == "xy":
        return la[2] == lb[2] and la[3] == lb[3]
    if kind == "measures":
        return la[2] == lb[2]
    if kind in ("multiset_columns", "multiset_pivot"):
        ca, cb = la[2], lb[2]
        keys = set(ca) | set(cb)
        diff = sum(abs(ca[k] - cb[k]) for k in keys)
        return diff <= 2
    return False


def _recipe_layers(recipe, roles):
    ds = recipe["recipe"]["datasetId"]
    layers = recipe["recipe"].get("layers", [])
    return [_layer_repr(L, roles, ds) for L in layers]


def recipes_compat(a_layers, b_layers) -> bool:
    if len(a_layers) != len(b_layers):
        return False
    # Sort both by chart group to align them
    a_sorted = sorted(a_layers, key=lambda x: x[1])
    b_sorted = sorted(b_layers, key=lambda x: x[1])
    return all(_layers_compat(la, lb) for la, lb in zip(a_sorted, b_sorted))


K_CAP = 7  # max template candidates per recipe (incl. self)


def _similarity(source_recipe: dict, cand_recipe: dict) -> float:
    """Score how good `cand_recipe` is as a template for queries about
    `source_recipe`. Higher = more similar. Source vs source = max score."""
    s, c = source_recipe["recipe"], cand_recipe["recipe"]
    score = 0.0

    # Same dataset bonus
    if s["datasetId"] == c["datasetId"]:
        score += 0.5

    # Layer count matches (always true if compat, but count for safety)
    s_layers, c_layers = s.get("layers", []), c.get("layers", [])
    if len(s_layers) != len(c_layers):
        return score

    # For each layer pair (sorted by chart group for stability)
    s_sorted = sorted(s_layers, key=lambda L: TYPE_GROUP.get(L.get("type"), "?"))
    c_sorted = sorted(c_layers, key=lambda L: TYPE_GROUP.get(L.get("type"), "?"))
    for sl, cl in zip(s_sorted, c_sorted):
        # +2 for exact chart type (column == column, not column ≈ bar)
        if sl.get("type") == cl.get("type"):
            score += 2.0
        # +1 for matching field count in main positions
        for pos in ("x", "y", "colors", "measures", "columns", "rows"):
            sn = len(sl.get(pos) or [])
            cn = len(cl.get(pos) or [])
            if sn == cn:
                score += 0.5
        # +1 if filter count diff ≤ 2
        sf = len(sl.get("filters") or [])
        cf = len(cl.get("filters") or [])
        if abs(sf - cf) <= 2:
            score += 1.0
    return score


def main() -> None:
    recipes = json.loads(RECIPES_PATH.read_text())
    roles = json.loads(ROLES_PATH.read_text())
    by_id = {r["recipe_id"]: r for r in recipes}

    layers_of = {rid: _recipe_layers(r, roles) for rid, r in by_id.items()}

    compat: dict[str, list[str]] = {}
    for rid_a in by_id:
        all_mates = [
            rid_b for rid_b in by_id
            if recipes_compat(layers_of[rid_a], layers_of[rid_b])
        ]
        # Rank by similarity to source, cap at K
        all_mates.sort(
            key=lambda rid_b: -_similarity(by_id[rid_a], by_id[rid_b])
        )
        compat[rid_a] = all_mates[:K_CAP]

    OUT_PATH.write_text(json.dumps(compat, ensure_ascii=False, indent=2))

    # Stats
    sizes = Counter(len(v) for v in compat.values())
    print(f"Recipes: {len(by_id)}")
    print()
    print("Template-compatible-set size distribution (incl. self):")
    for size, count in sorted(sizes.items()):
        print(f"  size {size}: {count} recipes")
    print()
    avg = sum(len(v) for v in compat.values()) / len(compat)
    print(f"Average compat set size: {avg:.2f}")
    print()

    # Show top 3 largest sets
    largest = sorted(compat.items(), key=lambda x: -len(x[1]))[:3]
    print("3 largest compat sets:")
    for rid, mates in largest:
        desc = by_id[rid]["description"].replace("\n", " ")[:50]
        print(f"  {rid} ({desc!r}): size {len(mates)}")

    # How many template candidates per query (using queries_raw.jsonl)
    qr_path = ROOT / "data" / "eval" / "queries_raw.jsonl"
    queries = [json.loads(l) for l in qr_path.read_text().splitlines()]
    cand_counts = []
    for q in queries:
        src = q.get("source_recipe_id")
        if src is None:
            cand_counts.append(0)
        else:
            cand_counts.append(len(compat[src]))
    n_with_src = sum(1 for c in cand_counts if c > 0)
    avg_q = sum(cand_counts) / max(1, n_with_src)
    print(f"\nQueries with source: {n_with_src}, avg template candidates: {avg_q:.2f}")
    pair_count = sum(cand_counts)
    print(f"Total (query, template-candidate) pairs: {pair_count}")


if __name__ == "__main__":
    main()
