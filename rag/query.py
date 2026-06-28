"""Deterministic, LLM-free slot extractor: rules + a vocabulary derived from the
inventory turn a query into `Filters`. Keeps numeric/exact queries (price, model,
colour, features, superlatives) reliable where pure vector search is weak."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

MODELS = ("GV80", "G80", "G90")  # GV80 first so it matches before G80

_INVENTORY = Path(__file__).resolve().parent.parent / "data" / "inventory.json"

COLOR_SYNONYMS = {"gray": "grey", "vrown": "brown"}  # also fix scraped typos

# Common colours, so the agent can recognise (and deny) a colour even if not in stock.
_BASE_COLORS = {
    "white", "black", "blue", "green", "grey", "gray", "silver", "red", "brown",
    "beige", "burgundy", "maroon", "tan", "gold", "bronze", "navy", "ivory",
    "cream", "charcoal", "orange", "purple", "yellow", "pink", "copper", "olive",
    "teal", "turquoise", "anthracite",
}
# words that appear inside colour fields but are materials/textures, not colours
_COLOR_STOP = {
    "aluminum", "birch", "brich", "film", "genuine", "gum", "hairline", "insert",
    "known", "leather", "matte", "metallic", "mettalic", "monotone", "nappa",
    "news", "newspaper", "paper", "pattern", "real", "stitching", "strip", "tone",
    "two", "ultra", "with", "woo", "wood", "velvet", "not", "type", "available",
}


# variant words that are NOT trims (body style, plate, drivetrain, materials...)
_TRIM_STOP = {
    "coupe", "long", "short", "plate", "type", "real", "wood", "dark", "pack",
    "matt", "matte", "awd", "rwd", "sport", "line", "with",
}


@lru_cache(maxsize=1)
def trim_vocab() -> frozenset[str]:
    """Trim names read from the inventory's variant field (data-derived, nothing hardcoded)."""
    vocab = set()
    try:
        for rec in json.loads(_INVENTORY.read_text(encoding="utf-8")):
            for tok in re.findall(r"[a-z]+", (rec.get("variant") or "").lower()):
                # >3 chars, not a stop word, not a model token (g80/gv80)
                if len(tok) > 3 and tok not in _TRIM_STOP and not re.match(r"gv?\d", tok):
                    vocab.add(tok)
    except OSError:
        pass
    return frozenset(vocab)


@lru_cache(maxsize=1)
def color_vocab() -> frozenset[str]:
    """Colour words = common colours + every colour token seen in the inventory."""
    vocab = set(_BASE_COLORS)
    try:
        for rec in json.loads(_INVENTORY.read_text(encoding="utf-8")):
            for fld in (rec.get("exterior_color"), rec.get("interior_color")):
                for tok in re.findall(r"[a-z]+", (fld or "").lower()):
                    if len(tok) > 2 and tok not in _COLOR_STOP:
                        vocab.add(tok)
    except OSError:
        pass
    return frozenset(vocab)

# spoken phrasing -> canonical feature substring (matched against record features)
FEATURE_KEYWORDS = {
    "sunroof": "Sunroof",
    "moonroof": "Sunroof",
    "panoramic": "Sunroof",
    "carplay": "Apple CarPlay",
    "apple car play": "Apple CarPlay",
    "android auto": "Android Auto",
    "navigation": "Navigation",
    "nav": "Navigation",
    "gps": "Navigation",
    "heads up": "Heads-Up Display",
    "heads-up": "Heads-Up Display",
    "hud": "Heads-Up Display",
    "360": "360",
    "surround view": "360",
    "surround camera": "360",
    "blind spot": "Blind-Spot",
    "ventilated": "Ventilated",
    "cooled seat": "Ventilated",
    "heated": "Heated",
    "memory seat": "Memory Seats",
    "wireless charging": "Wireless Charging",
    "ambient": "Ambient Lighting",
    "adaptive cruise": "Adaptive Cruise",
    "cruise control": "Adaptive Cruise",
    "parking sensor": "Parking Sensors",
    "premium audio": "Premium Audio",
    "sound system": "Premium Audio",
    "remote start": "Remote Start",
    "leather": "Leather Seats",
    "lane keep": "Lane Keep",
    "emergency brak": "Automatic Emergency Braking",
    "highway driving": "Highway Driving Assist",
}


# spoken numbers -> digits, so streaming STT ("two hundred and fifty thousand")
# feeds the same price logic as typed digits ("250000").
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "k": 1000, "grand": 1000,
           "million": 1_000_000, "m": 1_000_000}
_STARTERS = set(_ONES) | set(_TENS)


def _eval_run(words: list[str]) -> int:
    total, current = 0, 0
    for w in words:
        if w in _ONES:
            current += _ONES[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        else:  # thousand / million / k / m / grand
            total += (current or 1) * _SCALES[w]
            current = 0
    return total + current


def normalize_numbers(text: str) -> str:
    """Replace runs of English number-words with their digits."""
    out: list[str] = []
    run: list[str] = []

    def flush():
        if run:
            out.append(str(_eval_run([w for w in run if w != "and"])))
            run.clear()

    _BIG = {"thousand", "million", "k", "m", "grand"}
    for tok in text.split():
        low = tok.lower().strip(".,?!")
        if low in _STARTERS:
            run.append(low)
        elif run and low in _SCALES:
            run.append(low)
        # Absorb "and" only inside a number ("two hundred and fifty"), not between
        # two complete amounts ("...thousand AND three hundred...").
        elif run and low == "and" and not any(w in _BIG for w in run):
            run.append(low)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def normalize_models(text: str) -> str:
    """Collapse spoken/spelled model names to canonical tokens ("g v 80" -> "gv80").
    Run AFTER normalize_numbers (so "eighty" is already "80")."""
    text = re.sub(r"\bg\s*v\s*0*(\d{2})\b", r"gv\1", text)   # GV-series first
    text = re.sub(r"\bg\s+0*(\d{2})\b", r"g\1", text)
    return text


@dataclass
class Filters:
    models: set[str] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    fuels: set[str] = field(default_factory=set)          # "Electric" / "Gasoline"
    drivetrains: set[str] = field(default_factory=set)    # "AWD" / "RWD"
    body_types: set[str] = field(default_factory=set)     # "SUV" / "Sedan"
    coupe: bool = False                                   # variant must contain "Coupe"
    trims: set[str] = field(default_factory=set)          # "platinum" / "royal" / "prestige"...
    colors: set[str] = field(default_factory=set)         # base colour words
    features: set[str] = field(default_factory=set)       # canonical feature substrings
    min_price: int | None = None
    max_price: int | None = None
    target_price: int | None = None                      # "around X" -> rank by closeness
    target_mileage: int | None = None                    # "with 300 km" -> rank by closeness
    brand_new: bool = False                              # mileage == 0
    sort: str | None = None  # price_asc | price_desc | year_desc | mileage_asc
    raw: str = ""

    def has_hard_filter(self) -> bool:
        # Years excluded on purpose: a soft preference, not a gate, so it can't empty
        # results and trigger a false "NO VEHICLE MATCHES".
        return any([
            self.models, self.fuels, self.drivetrains,
            self.body_types, self.coupe, self.trims, self.colors, self.features,
            self.min_price is not None, self.max_price is not None,
            self.brand_new,
        ])


def _to_amount(num: str, suffix: str) -> int:
    value = float(num.replace(",", ""))
    if suffix and suffix.lower().startswith("k"):
        value *= 1_000
    elif suffix and suffix.lower().startswith("m"):
        value *= 1_000_000
    elif value < 1000:  # bare "250" almost certainly means 250k in this catalog
        value *= 1_000
    return int(value)


_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?"


def _parse_price(q: str, f: Filters) -> None:
    m = re.search(rf"between\s+{_NUM}\s+(?:and|to|-)\s+{_NUM}", q)
    if not m:
        m = re.search(rf"{_NUM}\s*(?:to|-)\s*{_NUM}", q)
    if m:
        a = _to_amount(m.group(1), m.group(2))
        b = _to_amount(m.group(3), m.group(4))
        f.min_price, f.max_price = min(a, b), max(a, b)
        return

    for pat, kind in [
        (rf"(?:under|below|less than|cheaper than|up to|max(?:imum)?|within|at most|no more than)\s+{_NUM}", "max"),
        (rf"(?:over|above|more than|at least|min(?:imum)?|starting (?:at|from))\s+{_NUM}", "min"),
        (rf"(?:around|about|approximately|roughly|near)\s+{_NUM}", "target"),
    ]:
        m = re.search(pat, q)
        if m:
            amt = _to_amount(m.group(1), m.group(2))
            if kind == "max":
                f.max_price = amt
            elif kind == "min":
                f.min_price = amt
            else:
                f.target_price = amt

    # Bare price with no comparator ("...Royal for 335000") -> target, rank by closeness.
    # Lookbehind skips digits glued to a word ("80" in "gv80"); >= 50000 excludes years/seats/trims.
    if f.min_price is None and f.max_price is None and f.target_price is None:
        for m in re.finditer(r"(?<![a-z0-9.])" + _NUM, q):
            digits, suffix = m.group(1).replace(",", ""), m.group(2)
            if len(digits) < 3 and not suffix:   # skip bare "80", "5", "7"
                continue
            amt = _to_amount(m.group(1), m.group(2))
            if amt >= 50_000:
                f.target_price = amt
                break


def parse(query: str) -> Filters:
    q = " " + normalize_models(normalize_numbers(query.lower().strip())) + " "
    f = Filters(raw=query)

    # Pull mileage out FIRST so a mileage figure is never mistaken for a year or price.
    mil = re.search(r"(\d[\d,]*)\s*(?:k\.?m\.?|kilomet(?:er|re)s?|miles?|mileage)\b", q)
    if mil:
        f.target_mileage = int(mil.group(1).replace(",", ""))
        q = q[:mil.start()] + " " + q[mil.end():]      # remove it before year/price parsing

    for m in MODELS:
        if re.search(rf"\b{m.lower()}\b", q):
            f.models.add(m)

    for y in re.findall(r"\b(20\d{2})\b", q):
        f.years.add(int(y))

    if re.search(r"\b(electric|ev|battery)\b", q):
        f.fuels.add("Electric")
    if re.search(r"\b(petrol|gas|gasoline|combustion)\b", q):
        f.fuels.add("Gasoline")

    if re.search(r"\b(awd|all[- ]wheel|4wd|four[- ]wheel)\b", q):
        f.drivetrains.add("AWD")
    if re.search(r"\b(rwd|rear[- ]wheel)\b", q):
        f.drivetrains.add("RWD")

    if re.search(r"\b(suv|crossover)\b", q):
        f.body_types.add("SUV")
    if re.search(r"\b(sedan|saloon)\b", q):
        f.body_types.add("Sedan")
    if re.search(r"\bcoupe\b", q):
        f.coupe = True

    # Trim and colour vocabularies are data-driven from the inventory.
    for w in trim_vocab():
        if re.search(rf"\b{re.escape(w)}\b", q):
            f.trims.add(w)

    for w in color_vocab():
        if re.search(rf"\b{re.escape(w)}\b", q):
            f.colors.add(COLOR_SYNONYMS.get(w, w))

    for kw, canonical in FEATURE_KEYWORDS.items():
        if kw in q:
            f.features.add(canonical)

    _parse_price(q, f)

    if re.search(r"\b(brand[- ]new|0\s*km|zero km|unused|never driven)\b", q):
        f.brand_new = True
        f.target_mileage = None                         # "0 km" is the brand_new filter, not a target

    # Sort intent, including plain "sort them by price" / "high to low".
    _generic_price = re.search(r"\b(sort|order|arrange|rank|list)\b.{0,25}\bprice\b", q) or \
        re.search(r"\bby price\b|\bprice[- ]wise\b|\bprice order\b", q)
    if re.search(r"\b(most expensive|priciest|highest price|dearest|top of the (?:range|line)|"
                 r"most premium|flagship|high(?:est)? to low|expensive first|descending)\b", q):
        f.sort = "price_desc"
    elif re.search(r"\b(cheapest|least expensive|lowest price|most affordable|best price|budget|"
                   r"low(?:est)? to high|cheapest first|ascending)\b", q) or _generic_price:
        f.sort = "price_asc"
    elif re.search(r"\b(newest|latest|most recent|brand[- ]new(?:est)?)\b", q):
        f.sort = "year_desc"
    elif re.search(r"\b(lowest mileage|least mileage|least driven|fewest (?:km|kilometers|miles))\b", q):
        f.sort = "mileage_asc"

    return f
