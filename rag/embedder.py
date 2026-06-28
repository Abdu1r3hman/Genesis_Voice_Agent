"""On-device text embedding via fastembed (BAAI/bge-small-en-v1.5, 384-dim,
L2-normalised so cosine == dot product). Model is a lazy singleton; query
embeddings are LRU-cached since a voice loop re-asks similar phrasings."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # lazy import keeps startup cheap

        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed a batch of documents into (n, EMBED_DIM) L2-normalised float32."""
    model = _get_model()
    vectors = list(model.embed(texts))
    return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=512)
def embed_query(text: str) -> np.ndarray:
    """Embed a single query into an (EMBED_DIM,) float32 vector."""
    model = _get_model()
    vector = next(iter(model.embed([text])))
    return np.asarray(vector, dtype=np.float32)


def warmup() -> None:
    """Force model load + one inference so the first real query is fast."""
    embed_query("warmup")
