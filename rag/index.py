"""
Build & persist the in-process vector index from the scraped inventory.

Reads:
  data/inventory.json             (structured records -> metadata + filtering)
  data/inventory_documents.jsonl  (natural-language text -> embedded)

Writes (to data/index/):
  embeddings.npy   float32 (n, 384), L2-normalised, row-aligned with records
  records.json     the structured records in the same row order

This runs offline/once. Production loads the persisted index at boot, so no
re-embedding happens on startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embedder import embed_documents

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"


def load_source() -> tuple[list[dict], list[str]]:
    records = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))

    # map id -> retrieval text from the jsonl documents
    text_by_id: dict[str, str] = {}
    with (DATA_DIR / "inventory_documents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                text_by_id[doc["id"]] = doc["text"]

    texts = [text_by_id.get(r["id"], _fallback_text(r)) for r in records]
    return records, texts


def _fallback_text(rec: dict) -> str:
    """Defensive: build text from a record if the jsonl is missing an entry."""
    feats = ", ".join(rec.get("features", []))
    return (
        f"{rec.get('full_name')} ({rec.get('year')}). Genesis {rec.get('model')}. "
        f"{rec.get('price')} {rec.get('currency')}. {rec.get('exterior_color')}. "
        f"Features: {feats}."
    )


def build() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records, texts = load_source()

    print(f"Embedding {len(texts)} documents with BGE-small (ONNX) ...")
    embeddings = embed_documents(texts)
    assert embeddings.shape[0] == len(records), "row/record mismatch"

    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved index: {embeddings.shape} -> {INDEX_DIR}")
    print(f"  embeddings.npy ({embeddings.nbytes/1024:.1f} KB)")
    print(f"  records.json   ({len(records)} records)")


if __name__ == "__main__":
    build()
