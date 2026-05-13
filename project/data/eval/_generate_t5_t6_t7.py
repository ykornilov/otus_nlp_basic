"""Appends T5 (hard negatives), T6 (cross-dataset triggers) and T7 (no-match)
queries to queries_raw.jsonl.

Each block is curated by the agent, drawing on the recipe summary in the
notebook ("Замечания по корпусу"). Run AFTER _generate_t1_t4.py.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = ROOT / "data" / "recipes.json"
OUT_PATH = ROOT / "data" / "eval" / "queries_raw.jsonl"


# ---------------------------------------------------------------------------
# T5 — hard negatives ("twins"). The query *looks like* it would match
# `decoy_recipe_id` (e.g. shares wording or filters) but flips one critical
# property (different category, city, metric, chart-type-by-implication).
# `source_recipe_id` is set to the decoy — the recipe the query is "tempting"
# the retriever toward but should NOT be relevant. Judge will determine the
# actual relevant set; some T5 may end up `no_match`.
# ---------------------------------------------------------------------------

T5_QUERIES: list[tuple[str, str, str]] = [
    # (decoy_recipe_id, current_dataset_id, query)
    ("r_0001", "omhpbh1k83ao8", "Как менялся средний рейтинг фильмов с годами"),
    ("r_0002", "omhpbh1k83ao8", "В каком году вышло больше всего комедийных фильмов"),
    ("r_0000", "omhpbh1k83ao8", "Топ драматических сериалов 90-х"),
    ("r_0006", "omhpbh1k83ao8", "Сколько сериалов вышло в 2025 по жанрам"),
    ("r_0014", "33h8c3n5nbien", "Сравни в динамике долю детского в Эконом, Комфорт и Комфорт+ в Питере"),
    ("r_0016", "33h8c3n5nbien", "Доля оклеенных машин в тарифах Эконом, Комфорт, Комфорт+ в Санкт-Петербурге"),
    ("r_0021", "33h8c3n5nbien", "Зависимость среднего чека от комиссии водителя в Москве в Эконом за март"),
    ("r_0011", "33h8c3n5nbien", "WoW Active Riders за последние 3 месяца в динамике в Москве"),
    ("r_0019", "33h8c3n5nbien", "Доля GMV в повышенных классах в Москве за последний месяц"),
    ("r_0018", "33h8c3n5nbien", "Личные поездки в России с начала года"),
    ("r_0029", "33h8c3n5nbien", "Сколько клиентов платило картой в странах СНГ"),
    ("r_0009", "33h8c3n5nbien", "Динамика активных водителей в России по дням за последний месяц"),
    ("r_0027", "33h8c3n5nbien", "Топ типов оплаты по водительским отменам в Азербайджане"),
    ("r_0036", "b60rhj4luj0y3", "Продажи по месяцам подачи заявки"),
    ("r_0048", "b60rhj4luj0y3", "Продажи в иерархии магазинов: округ, тип, адрес"),
    ("r_0050", "b60rhj4luj0y3", "Сводная: продажи по типу доставки и году × магазинам и брендам"),
    ("r_0058", "pzf0mu9kgqz4k", "Ошибки по экшенам во времени"),
    ("r_0062", "pzf0mu9kgqz4k", "Медианная латентность по сервисам во времени"),
    ("r_0065", "pzf0mu9kgqz4k", "5xx-ошибки по экшенам"),
]


# ---------------------------------------------------------------------------
# T6 — cross-dataset trigger. Query asked from `current_dataset_id` whose
# strict relevant recipe DOES NOT exist there, but a structural template
# (chart type + role pattern) exists in another dataset. `source_recipe_id`
# is set to one such template recipe in a different dataset (judge will mark
# all matching templates regardless of dataset).
#
# Each tuple: (template_recipe_id, current_dataset_id, query)
# Template recipe lives in a DIFFERENT dataset than current_dataset_id.
# ---------------------------------------------------------------------------

T6_QUERIES: list[tuple[str, str, str]] = [
    # Films dataset (omhpbh1k83ao8) has NO: metric, pivotTable, bar, *100p,
    # area, scatter. Triggers asking for those should fallback.
    ("r_0046", "omhpbh1k83ao8", "Покажи общее число фильмов одной цифрой"),
    ("r_0018", "omhpbh1k83ao8", "Средний рейтинг фильмов одним числом"),
    ("r_0050", "omhpbh1k83ao8", "Сводная таблица фильмов по жанрам и годам"),
    ("r_0024", "omhpbh1k83ao8", "Доли стран в выпуске фильмов горизонтальными процентами"),
    ("r_0008", "omhpbh1k83ao8", "Структура числа фильмов по типу областями во времени"),

    # Observability (pzf0mu9kgqz4k) has NO: donut/pie, pivotTable, bar.
    ("r_0006", "pzf0mu9kgqz4k", "Покажи кругом долю запросов по сервисам"),
    ("r_0020", "pzf0mu9kgqz4k", "Доля ошибок 5xx круговой диаграммой"),
    ("r_0041", "pzf0mu9kgqz4k", "Кольцевая диаграмма по числу запросов"),
    ("r_0050", "pzf0mu9kgqz4k", "Сводная таблица: сервисы × экшены, число ошибок"),
    ("r_0032", "pzf0mu9kgqz4k", "Сводная по неделям × сервисам с числом запросов"),
    ("r_0027", "pzf0mu9kgqz4k", "Топ хостов по числу ошибок горизонтальными столбиками"),

    # Taxi (33h8c3n5nbien) has donut? No — has pie (r_0020). No donut
    # specifically. But group {donut, pie} — so pie suffices for donut. Need
    # things taxi truly lacks.
    # Taxi has all common types. Skipping pure taxi T6 to avoid false triggers.

    # Retail (b60rhj4luj0y3) has NO: bin-bucketing histograms, surge-style.
    ("r_0022", "b60rhj4luj0y3", "Распределение продаж по бинам суммы чека"),
    ("r_0023", "b60rhj4luj0y3", "Распределение заявок по бинам выручки в Москве"),
    ("r_0007", "b60rhj4luj0y3", "Динамика доли типов доставки 100% столбиками"),

    # Films again — column100p / area100p / bar100p absent.
    ("r_0015", "omhpbh1k83ao8", "Доли типов кино во времени 100% областями"),
    ("r_0024", "omhpbh1k83ao8", "Доля жанров фильмов 100% горизонтальными столбиками"),

    # Observability — column100p / area100p / bar100p absent.
    ("r_0015", "pzf0mu9kgqz4k", "Доли сервисов в общем числе запросов 100% областями"),

    # Films — flatTable with many columns / pivotTable absent.
    ("r_0030", "omhpbh1k83ao8", "Таблица: жанр, число фильмов, число фильмов в прошлом году, изменение MoM"),

    # Observability — multi-column flatTable with WoW/MoM dynamics — observability has flatTable but limited.
    ("r_0030", "pzf0mu9kgqz4k", "Таблица: сервис, число ошибок, число ошибок в прошлом месяце, изменение MoM"),
]


# ---------------------------------------------------------------------------
# T7 — no-match. Queries unrelated to anything in the corpus. They might
# be chart-shaped (asking for a chart) but about a domain not represented
# (weather, currency, recipes, sports). source_recipe_id = None.
# current_dataset_id is sampled to spread across all datasets.
# ---------------------------------------------------------------------------

T7_QUERIES: list[tuple[str | None, str, str]] = [
    (None, "omhpbh1k83ao8", "Прогноз погоды в Москве на завтра"),
    (None, "33h8c3n5nbien", "Курс доллара к рублю за последние полгода"),
    (None, "b60rhj4luj0y3", "Рецепт борща с пампушками"),
    (None, "pzf0mu9kgqz4k", "Расписание электричек Ярославского направления"),
    (None, "omhpbh1k83ao8", "Как настроить Wi-Fi роутер TP-Link"),
    (None, "33h8c3n5nbien", "Биржевые котировки Apple за год"),
    (None, "b60rhj4luj0y3", "Численность населения Москвы по округам"),
    (None, "pzf0mu9kgqz4k", "График солнечной активности"),
    (None, "omhpbh1k83ao8", "Топ книг 2025 года по продажам"),
    (None, "33h8c3n5nbien", "Состав сборной России по футболу"),
    (None, "b60rhj4luj0y3", "Самый популярный язык программирования"),
    (None, "pzf0mu9kgqz4k", "Список пермских перевалов туристических"),
    (None, "omhpbh1k83ao8", "Размер штрафа за превышение скорости"),
    (None, "33h8c3n5nbien", "Сравнение iPhone 15 и iPhone 16"),
    (None, "b60rhj4luj0y3", "Что приготовить из курицы"),
    (None, "pzf0mu9kgqz4k", "Расстояние от Земли до Марса"),
    (None, "omhpbh1k83ao8", "Средняя зарплата дата-сайентиста в России"),
    (None, "33h8c3n5nbien", "Какой подарок выбрать на день рождения"),
    (None, "b60rhj4luj0y3", "ВВП Германии в 2024 году"),
    (None, "pzf0mu9kgqz4k", "Длительность беременности у слона"),
    (None, "omhpbh1k83ao8", "Что такое квантовая запутанность"),
    (None, "33h8c3n5nbien", "Сравнение iOS и Android"),
    (None, "b60rhj4luj0y3", "Почему небо голубое"),
    (None, "pzf0mu9kgqz4k", "Биография Льва Толстого"),
    (None, "omhpbh1k83ao8", "Цена квартиры в Сочи у моря"),
    (None, "33h8c3n5nbien", "Сколько калорий в банане"),
    (None, "b60rhj4luj0y3", "Расписание матчей чемпионата мира"),
    (None, "pzf0mu9kgqz4k", "Стоимость биткоина сегодня"),
]


def main() -> None:
    recipes = json.loads(RECIPES_PATH.read_text())
    by_id = {r["recipe_id"]: r for r in recipes}

    # Validate T5 / T6 references
    for rid, _, _ in T5_QUERIES:
        if rid not in by_id:
            raise RuntimeError(f"T5 references unknown recipe {rid}")
    for rid, ds, _ in T6_QUERIES:
        if rid not in by_id:
            raise RuntimeError(f"T6 references unknown recipe {rid}")
        if by_id[rid]["recipe"]["datasetId"] == ds:
            raise RuntimeError(
                f"T6 source {rid} must be in different dataset than current ({ds})"
            )

    # Read existing queries to find next qid
    existing_lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
    next_qid = len(existing_lines)

    appended = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for src, ds, q in T5_QUERIES:
            f.write(json.dumps({
                "query_id": f"q_{next_qid:04d}",
                "query": q,
                "current_dataset_id": ds,
                "query_type": "T5",
                "source_recipe_id": src,
            }, ensure_ascii=False) + "\n")
            next_qid += 1
            appended += 1
        for src, ds, q in T6_QUERIES:
            f.write(json.dumps({
                "query_id": f"q_{next_qid:04d}",
                "query": q,
                "current_dataset_id": ds,
                "query_type": "T6",
                "source_recipe_id": src,
            }, ensure_ascii=False) + "\n")
            next_qid += 1
            appended += 1
        for src, ds, q in T7_QUERIES:
            rec = {
                "query_id": f"q_{next_qid:04d}",
                "query": q,
                "current_dataset_id": ds,
                "query_type": "T7",
            }
            if src is not None:
                rec["source_recipe_id"] = src
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            next_qid += 1
            appended += 1

    print(f"Appended T5={len(T5_QUERIES)}, T6={len(T6_QUERIES)}, T7={len(T7_QUERIES)} (total {appended}). New file size: {next_qid} queries.")


if __name__ == "__main__":
    main()
