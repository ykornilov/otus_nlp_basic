"""Phase 1 grid configuration: 10 strategies × 3 models = 30 runs."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RECIPES_PATH = DATA_DIR / "recipes.json"
QUERIES_PATH = DATA_DIR / "eval" / "queries.jsonl"
ROLES_PATH = DATA_DIR / "field_roles.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CACHE_DIR = EXPERIMENTS_DIR / "cache"

# Models for Phase 1 grid
MODELS: list[str] = [
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "Qodo/Qodo-Embed-1-1.5B",  # qodo-embed-1.5 generative-style embedder
]

# Strategies: 10 points
STRATEGIES: list[str] = [
    "S1",
    "S2-F1", "S2-F2",
    "S3-F1@0.25", "S3-F1@0.5", "S3-F1@0.75",
    "S3-F2@0.25", "S3-F2@0.5", "S3-F2@0.75",
    "S4",
]

# Datasets in our corpus (also keys for per-dataset metric breakdown)
DATASETS: list[str] = [
    "33h8c3n5nbien",   # taxi
    "b60rhj4luj0y3",   # retail
    "pzf0mu9kgqz4k",   # observability
    "omhpbh1k83ao8",   # films
]

# Chart-type group equivalence (matches data/eval/_compute_signatures.py)
TYPE_GROUP: dict[str, str] = {
    "column": "bars", "bar": "bars",
    "column100p": "bars", "bar100p": "bars",
    "line": "lines", "area": "lines", "area100p": "lines",
    "donut": "circles", "pie": "circles",
    "metric": "metric",
    "flatTable": "flatTable",
    "pivotTable": "pivotTable",
}

# Default top-k for retrieval / metrics
DEFAULT_K = 5
EXTRA_K_VALUES = (1, 5, 10)


def short_model_name(model: str) -> str:
    """Short id used in exp_id and cache directory: 'intfloat/foo' -> 'foo'."""
    return model.split("/")[-1]


def parse_strategy(strategy: str) -> dict:
    """Parse strategy id into structured config.

    Returns:
      {
        "kind": "S1" | "S2" | "S3" | "S4",
        "format": None | "F1" | "F2",
        "alpha": None | float,
      }
    """
    if strategy == "S1":
        return {"kind": "S1", "format": None, "alpha": None}
    if strategy in ("S2-F1", "S2-F2"):
        return {"kind": "S2", "format": strategy.split("-")[1], "alpha": None}
    if strategy.startswith("S3-"):
        # S3-F1@0.25 etc.
        rest = strategy[3:]            # "F1@0.25"
        fmt, alpha_str = rest.split("@")
        return {"kind": "S3", "format": fmt, "alpha": float(alpha_str)}
    if strategy == "S4":
        return {"kind": "S4", "format": None, "alpha": None}
    raise ValueError(f"Unknown strategy: {strategy}")
