"""LLM-based query understanding: a free-form message becomes a Pydantic-validated
`CarQuery` mapped onto the retriever's `Filters`. Handles phrasing variety and context
drift the regex parser can't; returns None on any failure so the caller falls back to it."""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, ValidationError

from .query import Filters, MODELS

_SORT_MAP = {
    "cheapest": "price_asc", "most_expensive": "price_desc",
    "newest": "year_desc", "lowest_mileage": "mileage_asc",
}
_BODY_MAP = {"suv": "SUV", "sedan": "Sedan"}
_FUEL_MAP = {"electric": "Electric", "ev": "Electric", "gasoline": "Gasoline", "petrol": "Gasoline"}


class CarQuery(BaseModel):
    intent: str = "search"                       # search | recommend | followup | compare | other
    keep_context: bool = False                   # refers to cars already under discussion
    models: list[str] = []                       # GV80 / G80 / G90
    trims: list[str] = []                        # Platinum / Royal / Prestige / Premium / Advance
    colors: list[str] = []
    fuel: Optional[str] = None
    body: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    target_price: Optional[int] = None
    target_mileage: Optional[int] = None
    sort: Optional[str] = None

    def has_filter(self) -> bool:
        return bool(self.models or self.trims or self.colors or self.fuel or self.body
                    or self.min_price or self.max_price or self.target_price
                    or self.target_mileage or self.sort)

    def to_filters(self) -> Filters:
        """Map the validated query onto the retriever's Filters."""
        f = Filters(raw="")
        f.models = {m.upper() for m in self.models if isinstance(m, str) and m.upper() in MODELS}
        f.trims = {t.lower() for t in self.trims if isinstance(t, str)}
        f.colors = {c.lower() for c in self.colors if isinstance(c, str)}
        if self.fuel and self.fuel.lower() in _FUEL_MAP:
            f.fuels = {_FUEL_MAP[self.fuel.lower()]}
        if self.body and self.body.lower() in _BODY_MAP:
            f.body_types = {_BODY_MAP[self.body.lower()]}
        f.min_price = self.min_price if isinstance(self.min_price, int) else None
        f.max_price = self.max_price if isinstance(self.max_price, int) else None
        f.target_price = self.target_price if isinstance(self.target_price, int) else None
        f.target_mileage = self.target_mileage if isinstance(self.target_mileage, int) else None
        f.sort = _SORT_MAP.get((self.sort or "").lower())
        return f


_SYS = """You convert a car-shopping message into a JSON query for a Genesis dealership.

Inventory: models GV80 (an SUV), G80 (a sedan), G90 (a sedan). Trims: Platinum, Royal, \
Prestige, Premium, Advance. Fuels: Electric, Gasoline. Prices are in SAR (about 99000 to \
335000). Mileage in km.

You get the user's MESSAGE and sometimes CONTEXT (the cars already under discussion). \
If the message refers to those cars ("tell me more", "the cheaper one", "in blue", \
"what about one with 2000 km", "which is better"), set keep_context=true AND carry the \
CONTEXT's model/trim into models/trims. If it is a brand-new or open request ("recommend \
me anything", "show me an SUV", "do you have a G90", "something else"), set keep_context=false.

Reply with ONLY a JSON object with these exact keys:
  intent: one of "search","recommend","followup","compare","other"
  keep_context: boolean
  models: array, each one of "GV80","G80","G90"
  trims: array of trim names
  colors: array of colour words the user mentioned
  fuel: "Electric" or "Gasoline" or null
  body: "SUV" or "Sedan" or null
  min_price: integer SAR or null
  max_price: integer SAR or null
  target_price: integer SAR or null   (use for "around X")
  target_mileage: integer km or null  (use for "with X km")
  sort: one of "cheapest","most_expensive","newest","lowest_mileage" or null

CRITICAL: put a model/trim in the arrays ONLY if the user named it OR it's the specific car \
in CONTEXT. NEVER fill the arrays with every option — for an open or no-preference request, \
leave models and trims as EMPTY arrays []. Only include constraints the user actually expressed.

Examples:
MESSAGE: recommend me anything
{"intent":"recommend","keep_context":false,"models":[],"trims":[],"colors":[],"fuel":null,"body":null,"min_price":null,"max_price":null,"target_price":null,"target_mileage":null,"sort":null}
MESSAGE: the cheapest one
CONTEXT (cars under discussion): platinum
{"intent":"search","keep_context":true,"models":[],"trims":["Platinum"],"colors":[],"fuel":null,"body":null,"min_price":null,"max_price":null,"target_price":null,"target_mileage":null,"sort":"cheapest"}
MESSAGE: what about one with 2000 km
CONTEXT (cars under discussion): gv80 royal
{"intent":"search","keep_context":true,"models":["GV80"],"trims":["Royal"],"colors":[],"fuel":null,"body":null,"min_price":null,"max_price":null,"target_price":null,"target_mileage":2000,"sort":null}
MESSAGE: do you have any electric cars
{"intent":"search","keep_context":false,"models":[],"trims":[],"colors":[],"fuel":"Electric","body":null,"min_price":null,"max_price":null,"target_price":null,"target_mileage":null,"sort":null}
MESSAGE: forget that, show me sedans under 200k
{"intent":"search","keep_context":false,"models":[],"trims":[],"colors":[],"fuel":null,"body":"Sedan","min_price":null,"max_price":200000,"target_price":null,"target_mileage":null,"sort":null}"""


def understand(client, model: str, text: str, subject: str = "") -> Optional[CarQuery]:
    """Groq -> validated CarQuery. Returns None on any failure (caller falls back to regex)."""
    if not client:
        return None
    try:
        user = text if not subject else f"MESSAGE: {text}\nCONTEXT (cars under discussion): {subject}"
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=240,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return CarQuery.model_validate(data)
    except (ValidationError, json.JSONDecodeError, Exception):  # noqa: BLE001
        return None
