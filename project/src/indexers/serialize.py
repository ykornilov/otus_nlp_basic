"""Recipe-to-text serialization functions for the 3 formats:

- json_dump(recipe)         — F1: raw JSON dump (compact, ensure_ascii=False)
- flat_kv(recipe)           — F2: flat "key: value; ..." string
- linearize_to_text(recipe) — S4: deterministic NL template

All accept a recipe dict (the full recipe object, including `datasetId` and
`layers`) and return a single string. Multi-layer recipes have all layers
concatenated with a separator that depends on the format.
"""
from __future__ import annotations

import json
from typing import Any

# ----------------------------------------------------------------------------
# F1 — JSON dump
# ----------------------------------------------------------------------------

def json_dump(recipe: dict) -> str:
    """Compact JSON dump of the recipe object."""
    return json.dumps(recipe, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------------------
# F2 — flat key=value
# ----------------------------------------------------------------------------

def _titles(items: list[dict] | None) -> list[str]:
    return [it.get("title", "") for it in (items or []) if it.get("title")]


def _filters_str(filters: list[dict] | None) -> str:
    parts = []
    for f in filters or []:
        title = (f.get("field") or {}).get("title", "?")
        op = f.get("operation", "?")
        vals = f.get("values") or []
        # Truncate long value lists for readability
        if len(vals) > 5:
            vals_str = ", ".join(str(v) for v in vals[:5]) + f", ... ({len(vals)} total)"
        else:
            vals_str = ", ".join(str(v) for v in vals)
        parts.append(f"{title} {op} [{vals_str}]")
    return " | ".join(parts)


def _layer_flat_kv(layer: dict) -> str:
    parts = [f"type: {layer.get('type', '?')}"]
    for pos in ("x", "y", "y2", "colors", "measures", "rows", "columns"):
        titles = _titles(layer.get(pos))
        if titles:
            parts.append(f"{pos}: {', '.join(titles)}")
    fs = _filters_str(layer.get("filters"))
    if fs:
        parts.append(f"filters: {fs}")
    sortings = layer.get("sorting") or []
    if sortings:
        parts.append(
            "sorting: "
            + ", ".join(f"{s.get('title', '?')} {s.get('direction', '?')}" for s in sortings)
        )
    return "; ".join(parts)


def flat_kv(recipe: dict) -> str:
    """Flat key=value serialization (F2). Multi-layer joined with ' | '."""
    layers = recipe.get("layers", [])
    return " | ".join(_layer_flat_kv(L) for L in layers)


# ----------------------------------------------------------------------------
# S4 — deterministic NL template
# ----------------------------------------------------------------------------

_TYPE_RU = {
    "line": "line", "area": "area", "area100p": "area-100%",
    "column": "column", "bar": "bar",
    "column100p": "column-100%", "bar100p": "bar-100%",
    "donut": "donut", "pie": "pie",
    "flatTable": "flatTable", "pivotTable": "pivotTable",
    "metric": "metric (single value)",
}

_GROUP_BARS = {"column", "bar", "column100p", "bar100p"}
_GROUP_LINES = {"line", "area", "area100p"}
_GROUP_CIRCLES = {"donut", "pie"}


def _filters_nl(filters: list[dict] | None) -> str:
    if not filters:
        return ""
    parts = []
    for f in filters:
        title = (f.get("field") or {}).get("title", "?")
        op = f.get("operation", "?")
        vals = f.get("values") or []
        if len(vals) > 5:
            vals_str = ", ".join(str(v) for v in vals[:5]) + f" и ещё ({len(vals)} всего)"
        else:
            vals_str = ", ".join(str(v) for v in vals)
        parts.append(f"{title} {op} [{vals_str}]")
    return " и ".join(parts)


def _linearize_layer(layer: dict) -> str:
    t = layer.get("type", "?")
    t_human = _TYPE_RU.get(t, t)
    parts = [f"Чарт типа «{t_human}»."]

    if t in _GROUP_BARS or t in _GROUP_LINES:
        x = _titles(layer.get("x"))
        y = _titles(layer.get("y"))
        y2 = _titles(layer.get("y2"))
        cols = _titles(layer.get("colors"))
        if x:
            parts.append(f"По оси X — {', '.join(f'поле «{i}»' for i in x)}.")
        if y:
            parts.append(f"По оси Y — {', '.join(f'поле «{i}»' for i in y)}.")
        if y2:
            parts.append(f"По второй оси Y — {', '.join(f'поле «{i}»' for i in y2)}.")
        if cols:
            parts.append(f"Цветовая разбивка по {', '.join(f'полю «{i}»' for i in cols)}.")

    elif t in _GROUP_CIRCLES:
        m = _titles(layer.get("measures"))
        if m:
            parts.append(f"Круговая диаграмма по измерению {', '.join(f'«{i}»' for i in m)}.")

    elif t == "metric":
        m = _titles(layer.get("measures"))
        if m:
            parts.append(f"Одно значение — {', '.join(f'«{i}»' for i in m)}.")

    elif t == "flatTable":
        cols = _titles(layer.get("columns"))
        if cols:
            parts.append(f"Таблица с колонками: {', '.join(f'«{i}»' for i in cols)}.")
        sortings = layer.get("sorting") or []
        if sortings:
            parts.append(
                "Сортировка: "
                + ", ".join(f"«{s.get('title', '?')}» {s.get('direction', '?')}" for s in sortings)
                + "."
            )

    elif t == "pivotTable":
        rows = _titles(layer.get("rows"))
        cols = _titles(layer.get("columns"))
        m = _titles(layer.get("measures"))
        if rows:
            parts.append(f"Строки: {', '.join(f'«{i}»' for i in rows)}.")
        if cols:
            parts.append(f"Колонки: {', '.join(f'«{i}»' for i in cols)}.")
        if m:
            parts.append(f"Измерения: {', '.join(f'«{i}»' for i in m)}.")

    fs = _filters_nl(layer.get("filters"))
    if fs:
        parts.append(f"Фильтры: {fs}.")

    return " ".join(parts)


def linearize_to_text(recipe: dict) -> str:
    """Deterministic NL linearization (S4). Multi-layer: subsequent layers prefixed
    with 'Дополнительный слой:'."""
    layers = recipe.get("layers", [])
    if not layers:
        return ""
    pieces = [_linearize_layer(layers[0])]
    for L in layers[1:]:
        pieces.append("Дополнительный слой: " + _linearize_layer(L))
    return " ".join(pieces)


# ----------------------------------------------------------------------------
# Compose the full text to embed for a given strategy
# ----------------------------------------------------------------------------

def text_for_strategy(strategy_kind: str, fmt: str | None, recipe_obj: dict) -> str | tuple[str, str]:
    """Build the text(s) to embed for a given strategy.

    `recipe_obj` is the FULL recipe object (with description and recipe).

    For S1, S2, S4: returns a single string.
    For S3 (separate-then-combine): returns (description_text, schema_text)
      so the caller can embed them separately and combine.
    """
    desc = recipe_obj["description"]
    recipe = recipe_obj["recipe"]
    if strategy_kind == "S1":
        return desc
    if strategy_kind == "S2":
        if fmt == "F1":
            return desc + " " + json_dump(recipe)
        if fmt == "F2":
            return desc + " " + flat_kv(recipe)
    if strategy_kind == "S3":
        # Separate texts; caller embeds each and combines with α.
        if fmt == "F1":
            return (desc, json_dump(recipe))
        if fmt == "F2":
            return (desc, flat_kv(recipe))
    if strategy_kind == "S4":
        return desc + " " + linearize_to_text(recipe)
    raise ValueError(f"Unknown (kind, format): {strategy_kind}, {fmt}")
