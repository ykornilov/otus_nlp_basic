"""Embedding wrappers around HuggingFace transformers + on-disk cache.

Three model families are supported with their own pooling/prefix conventions:

  - intfloat/multilingual-e5-large
        Mean-pooling, L2-normalize. Inputs are prefixed with "query: " or
        "passage: " before tokenization.
  - BAAI/bge-m3
        CLS pooling, L2-normalize. No prefix needed (the model is robust to
        both query and passage without them; we keep inputs uniform).
  - Qodo/Qodo-Embed-1-1.5B
        Generative-style embedder (decoder-only). Use last-token pooling
        (the embedding is taken from the last non-padding token), normalize.
        This convention follows Qodo's published example usage.

Cache: SHA256(f"{model_short}|{kind}|{text}") → .npy in
       experiments/cache/{model_short}/{hash}.npy
where `kind` ∈ {"query", "passage"} disambiguates E5's prefixing (same
text embedded as query vs passage gives different vectors).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from src.config import CACHE_DIR, short_model_name


# Lazy global cache of loaded (tokenizer, model) per model name.
_LOADED: dict[str, tuple] = {}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load(model_name: str):
    if model_name in _LOADED:
        return _LOADED[model_name]
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    mdl.to(_device())
    _LOADED[model_name] = (tok, mdl)
    return tok, mdl


def _cache_key(model_name: str, kind: str, text: str) -> Path:
    short = short_model_name(model_name)
    h = hashlib.sha256(f"{short}|{kind}|{text}".encode("utf-8")).hexdigest()
    d = CACHE_DIR / short
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.npy"


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


# ----------------------------------------------------------------------------
# Per-family encoders. Each takes a list of pre-prefixed strings and returns
# a (N, D) numpy float32 array of L2-normalized embeddings.
# ----------------------------------------------------------------------------

def _encode_e5(model_name: str, texts: list[str]) -> np.ndarray:
    tok, mdl = _load(model_name)
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state          # (B, L, D)
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
    summed = (out * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts                         # mean-pool
    pooled = _normalize(pooled)
    return pooled.cpu().numpy().astype(np.float32)


def _encode_bge(model_name: str, texts: list[str]) -> np.ndarray:
    tok, mdl = _load(model_name)
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state
    pooled = out[:, 0]                               # CLS
    pooled = _normalize(pooled)
    return pooled.cpu().numpy().astype(np.float32)


def _encode_qodo(model_name: str, texts: list[str]) -> np.ndarray:
    tok, mdl = _load(model_name)
    # Qodo is a decoder-only model; use last-token pooling (last non-pad).
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state           # (B, L, D)
    # Index of last non-pad token per sequence
    seq_lens = enc["attention_mask"].sum(dim=1) - 1   # 0-based last position
    seq_lens = seq_lens.clamp(min=0)
    idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, out.size(-1))
    pooled = out.gather(dim=1, index=idx).squeeze(1)
    pooled = _normalize(pooled)
    return pooled.cpu().numpy().astype(np.float32)


def _encode_dispatch(model_name: str, texts: list[str]) -> np.ndarray:
    if "e5" in model_name.lower():
        return _encode_e5(model_name, texts)
    if "bge" in model_name.lower():
        return _encode_bge(model_name, texts)
    if "qodo" in model_name.lower():
        return _encode_qodo(model_name, texts)
    raise ValueError(f"Unknown model family: {model_name}")


def _prefix(model_name: str, kind: str, text: str) -> str:
    """Apply E5-specific prefixes; pass-through for others."""
    if "e5" in model_name.lower():
        if kind == "query":
            return f"query: {text}"
        return f"passage: {text}"
    return text


# ----------------------------------------------------------------------------
# Public API: embed_texts with caching
# ----------------------------------------------------------------------------

def embed_texts(
    model_name: str,
    texts: list[str],
    kind: str,                 # "query" or "passage"
    batch_size: int = 8,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed a list of texts under (model, kind). Cached on disk.

    `kind` matters for E5 (query vs passage prefix); for other models both
    keys point to the same vector but we still hash separately to avoid
    accidental cross-model leakage.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    # Try cache; collect texts that need to be encoded
    cached: list[tuple[int, np.ndarray]] = []
    todo_idx: list[int] = []
    todo_texts: list[str] = []
    for i, t in enumerate(texts):
        if use_cache:
            p = _cache_key(model_name, kind, t)
            if p.exists():
                cached.append((i, np.load(p)))
                continue
        todo_idx.append(i)
        todo_texts.append(t)

    # Encode missing in batches (with prefix applied)
    new_embs: list[np.ndarray] = []
    if todo_texts:
        prefixed = [_prefix(model_name, kind, t) for t in todo_texts]
        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start:start + batch_size]
            arr = _encode_dispatch(model_name, batch)
            new_embs.append(arr)
        new_arr = np.vstack(new_embs)
        # Save to cache
        if use_cache:
            for i_local, t in enumerate(todo_texts):
                p = _cache_key(model_name, kind, t)
                np.save(p, new_arr[i_local])

    # Stitch together in original order
    D = (cached[0][1].shape[0] if cached else new_arr.shape[1])
    out = np.empty((len(texts), D), dtype=np.float32)
    for i, e in cached:
        out[i] = e
    if todo_texts:
        for j, i in enumerate(todo_idx):
            out[i] = new_arr[j]
    return out


def time_encode_one(model_name: str, text: str = "тест") -> float:
    """Return seconds to encode one short string (model warmup + 1 forward pass).
    Useful for sanity / latency measurement."""
    _load(model_name)  # warmup
    t0 = time.time()
    embed_texts(model_name, [text], kind="query", use_cache=False)
    return time.time() - t0
