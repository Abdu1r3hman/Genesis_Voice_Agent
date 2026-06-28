"""Hybrid retriever: deterministic structured filters narrow the candidate set,
then in-process semantic search (NumPy, in-RAM) ranks them, with optional
superlative sort. Wrapped in a class so the vector backend can later be swapped."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import embed_query, warmup
from .query import Filters, parse

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"

_SORT_KEY = {
    "price_asc": (lambda r: r.get("price") or 1e12, False),
    "price_desc": (lambda r: r.get("price") or -1, True),
    "year_desc": (lambda r: r.get("year") or 0, True),
    "mileage_asc": (lambda r: r.get("mileage_km") if r.get("mileage_km") is not None else 1e12, False),
}


def format_vehicle(rec: dict, idx: int, max_features: int = 12) -> str:
    """One compact, LLM-ready line describing a vehicle."""
    price = f"{rec['price']:,} {rec['currency']}" if rec.get("price") else "price on request"
    mileage = "0 km (new)" if rec.get("mileage_km") == 0 else (
        f"{rec['mileage_km']:,} km" if rec.get("mileage_km") else "n/a")
    feats = ", ".join(rec.get("features", [])[:max_features])
    return (
        f"[{idx}] {rec.get('full_name')} ({rec.get('year')}) | {rec.get('model')} "
        f"{rec.get('body_type')} | {price} | {rec.get('availability')} | "
        f"{mileage} | ext: {rec.get('exterior_color')} | int: {rec.get('interior_color')} | "
        f"{rec.get('fuel_type')}/{rec.get('transmission')}/{rec.get('drivetrain')} | "
        f"features: {feats} | url: {rec.get('url')}"
    )


def format_vehicles(records: list[dict], max_features: int = 12) -> str:
    return "\n".join(format_vehicle(r, i, max_features) for i, r in enumerate(records, 1))


@dataclass
class Hit:
    record: dict
    score: float          # semantic similarity (0..1)
    rank_reason: str      # "semantic" | "price_asc" | ...

    @property
    def id(self) -> str:
        return self.record["id"]


@dataclass
class RetrievalResult:
    query: str
    filters: Filters
    hits: list[Hit]
    used_fallback: bool   # True if no record matched the hard filters

    def grounding_context(self) -> str:
        """Compact, LLM-ready block of ONLY the retrieved stock."""
        return format_vehicles([h.record for h in self.hits])


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.embeddings: np.ndarray = np.load(index_dir / "embeddings.npy")
        self.records: list[dict] = json.loads(
            (index_dir / "records.json").read_text(encoding="utf-8")
        )
        assert self.embeddings.shape[0] == len(self.records)
        self._facts: str | None = None

    # Whole-stock aggregates, so the LLM never generalises from the few cars it sees.
    def summary_facts(self) -> str:
        """Cached factual totals for the entire inventory."""
        if self._facts is not None:
            return self._facts
        recs = self.records
        miles = [r.get("mileage_km") for r in recs if r.get("mileage_km") is not None]
        new_n = sum(1 for m in miles if m == 0)
        used = sorted(m for m in miles if m and m > 0)
        prices = [r.get("price") for r in recs if r.get("price")]
        models = sorted({r.get("model") for r in recs if r.get("model")})
        fuels = sorted({r.get("fuel_type") for r in recs if r.get("fuel_type")})
        parts = [f"Total cars in stock: {len(recs)}."]
        if new_n:
            parts.append(f"{new_n} are brand-new at 0 km.")
        if used:
            parts.append(
                f"{len(used)} are pre-owned, mileage from {used[0]:,} to {used[-1]:,} km "
                "(so NOT every car is 0 km).")
        if prices:
            parts.append(f"Prices range about {min(prices):,} to {max(prices):,} SAR.")
        if models:
            parts.append("Models: " + ", ".join(models) + ".")
        if fuels:
            parts.append("Fuel types: " + ", ".join(fuels) + ".")
        self._facts = " ".join(parts)
        return self._facts

    @staticmethod
    def _passes(rec: dict, f: Filters) -> bool:
        if f.models and rec.get("model") not in f.models:
            return False
        # Years are deliberately NOT gated: a mis-read number ("2000 km" heard as
        # year 2000) must never delete the right cars. Year only nudges ranking in search().
        if f.fuels and rec.get("fuel_type") not in f.fuels:
            return False
        if f.drivetrains and rec.get("drivetrain") not in f.drivetrains:
            return False
        if f.body_types and rec.get("body_type") not in f.body_types:
            return False
        if f.coupe and "coupe" not in (rec.get("variant") or "").lower():
            return False
        if f.trims:
            variant = (rec.get("variant") or "").lower()
            if not any(t in variant for t in f.trims):
                return False
        if f.colors:
            # Match exterior and interior, normalising gray/grey on both sides
            # (data mixes "Slate Gray" & "Makalu Grey").
            colour_text = (
                f"{rec.get('exterior_color') or ''} {rec.get('interior_color') or ''}"
            ).lower().replace("gray", "grey").replace("vrown", "brown")
            if not any(c in colour_text for c in f.colors):
                return False
        if f.features:
            owned = " | ".join(rec.get("features", [])).lower()
            if not all(feat.lower() in owned for feat in f.features):
                return False
        price = rec.get("price")
        if f.min_price is not None and (price is None or price < f.min_price):
            return False
        if f.max_price is not None and (price is None or price > f.max_price):
            return False
        if f.brand_new and rec.get("mileage_km") != 0:
            return False
        return True

    def search(self, query: str, k: int = 4, filters: Filters | None = None) -> RetrievalResult:
        # `filters` lets a caller pass an already-understood query (e.g. from the LLM
        # extractor) instead of re-parsing the text with regex.
        f = filters if filters is not None else parse(query)

        cand_idx = [i for i, r in enumerate(self.records) if self._passes(r, f)]
        used_fallback = False
        if not cand_idx:
            cand_idx = list(range(len(self.records)))
            used_fallback = f.has_hard_filter()  # filters were set but matched nothing

        qvec = embed_query(query)
        sims = self.embeddings[cand_idx] @ qvec  # normalised -> cosine

        order = sorted(range(len(cand_idx)), key=lambda j: sims[j], reverse=True)
        reason = "semantic"

        # Soft preference: float year-matches up. Stable sort, so semantic order holds within.
        if f.years:
            order = sorted(order, key=lambda j: self.records[cand_idx[j]].get("year") in f.years,
                           reverse=True)

        # Explicit/superlative sort overrides semantic ordering.
        if f.sort in _SORT_KEY:
            keyfn, rev = _SORT_KEY[f.sort]
            order = sorted(order, key=lambda j: keyfn(self.records[cand_idx[j]]), reverse=rev)
            reason = f.sort
        elif f.target_mileage is not None:
            order = sorted(order, key=lambda j: abs((self.records[cand_idx[j]].get("mileage_km") or 0) - f.target_mileage))
            reason = "mileage_near"
        elif f.target_price is not None:
            order = sorted(order, key=lambda j: abs((self.records[cand_idx[j]].get("price") or 0) - f.target_price))
            reason = "price_near"

        hits = [
            Hit(record=self.records[cand_idx[j]], score=float(sims[j]), rank_reason=reason)
            for j in order[:k]
        ]
        return RetrievalResult(query=query, filters=f, hits=hits, used_fallback=used_fallback)


    def sort_records(self, records: list[dict], sort: str | None) -> list[dict]:
        """Re-rank a given list of records by a sort key (e.g. a 'sort them by price' follow-up)."""
        if sort in _SORT_KEY:
            keyfn, rev = _SORT_KEY[sort]
            return sorted(records, key=keyfn, reverse=rev)
        return records


_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide singleton; warms the embedder on first construction."""
    global _singleton
    if _singleton is None:
        warmup()
        _singleton = Retriever()
    return _singleton
