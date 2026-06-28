"""
Genesis Certified Pre-Owned (Saudi Arabia) inventory scraper.

Source: https://genesis-cpo.netlify.app/  (Next.js site)

Strategy
--------
The inventory listing page is client-rendered and filtered by model, so we use
Playwright to enumerate every vehicle detail URL across the model filters.
Each *detail* page is server-rendered and contains:
  - a schema.org JSON-LD "Car"/"Product" block (price, year, colors, fuel, etc.)
  - a visible spec grid  (.gvd-spec-item)
  - a rich feature list   (.gvd-features__text)
We merge all three into one clean record per vehicle.

Outputs
-------
  data/inventory.json            -> list of structured vehicle records
  data/inventory_documents.jsonl -> one retrieval-ready text document per vehicle
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = "https://genesis-cpo.netlify.app"
INVENTORY_PATH = "/en/inventory"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def _collect_links_on_page(page) -> set[str]:
    hrefs = page.eval_on_selector_all(
        "a[href*='/inventory/']",
        "els => els.map(e => e.getAttribute('href'))",
    )
    return {
        h.split("#")[0].rstrip("/")
        for h in hrefs
        if h and "/inventory/" in h and "?" not in h
    }


def enumerate_slugs(page) -> list[str]:
    """
    Collect every unique /inventory/<slug> URL.

    The listing shows 12 cars per page and paginates with a client-side
    "Next page" button (no URL param), so we click through every page until
    the Next arrow is disabled.
    """
    slugs: set[str] = set()
    page.goto(BASE + INVENTORY_PATH, wait_until="networkidle", timeout=60_000)
    page.wait_for_selector("a[href*='/inventory/']", timeout=30_000)

    label = page.query_selector(".gi-pager__label")
    print("Pager:", label.inner_text() if label else "(no pager)")

    while True:
        slugs |= _collect_links_on_page(page)

        next_btn = page.query_selector("button[aria-label='Next page']")
        if not next_btn or next_btn.is_disabled():
            break

        prev_label = page.query_selector(".gi-pager__label")
        prev_text = prev_label.inner_text() if prev_label else ""
        next_btn.click()
        # wait until the pager label updates (page content swapped in)
        try:
            page.wait_for_function(
                "(t) => { const e = document.querySelector('.gi-pager__label');"
                " return e && e.innerText !== t; }",
                arg=prev_text,
                timeout=15_000,
            )
        except Exception:
            page.wait_for_timeout(1500)

    return sorted(slugs)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def extract_car_jsonld(soup: BeautifulSoup) -> dict:
    """Return the schema.org Car/Product object from the page's JSON-LD."""
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            t = item.get("@type")
            types = t if isinstance(t, list) else [t]
            if "Car" in types or "Vehicle" in types or "Product" in types:
                return item
    return {}


def extract_spec_grid(soup: BeautifulSoup) -> dict[str, str]:
    """Parse the visible .gvd-spec-item label/value pairs."""
    specs: dict[str, str] = {}
    for item in soup.select(".gvd-spec-item"):
        label = item.select_one(".gvd-spec-item__label")
        value = item.select_one(".gvd-spec-item__value")
        if label and value:
            specs[label.get_text(strip=True)] = value.get_text(strip=True)
    return specs


def extract_features(soup: BeautifulSoup) -> list[str]:
    feats = [e.get_text(strip=True) for e in soup.select(".gvd-features__text")]
    # de-dup, keep order
    seen, out = set(), []
    for f in feats:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def last_segment(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def parse_drivetrain(text: str) -> str | None:
    m = re.search(r"\b(AWD|RWD|FWD|4WD)\b", text, re.I)
    return m.group(1).upper() if m else None


def parse_seats(text: str) -> int | None:
    m = re.search(r"\b(\d)\s*P\b", text, re.I)  # e.g. "7P", "5P"
    return int(m.group(1)) if m else None


def parse_int(value) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def schema_tail(uri: str | None) -> str | None:
    """https://schema.org/InStock -> 'InStock'."""
    if not uri:
        return None
    return str(uri).rstrip("/").split("/")[-1]


# --------------------------------------------------------------------------- #
# Per-vehicle record
# --------------------------------------------------------------------------- #
def build_record(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    ld = extract_car_jsonld(soup)
    specs = extract_spec_grid(soup)
    features = extract_features(soup)

    offers = ld.get("offers", {}) or {}
    odometer = ld.get("mileageFromOdometer", {}) or {}

    # additionalProperty -> dict
    extra = {}
    for prop in ld.get("additionalProperty", []) or []:
        if isinstance(prop, dict) and prop.get("name"):
            extra[prop["name"]] = prop.get("value")

    description = ld.get("description") or ""
    engine = specs.get("Engine") or ""
    drivetrain_src = f"{description} {engine} {last_segment(url)}"

    mileage_km = parse_int(odometer.get("value"))
    if mileage_km is None and specs.get("Mileage", "").lower() not in ("", "not available"):
        mileage_km = parse_int(specs.get("Mileage"))

    certified_flag = extra.get("Genesis Certified")
    certified = (certified_flag or "").strip().lower() == "yes" or "Certified" in (
        offers.get("seller", {}) or {}
    ).get("name", "")

    record = {
        "id": last_segment(url),
        "url": url,
        "model": ld.get("model"),
        "variant": ld.get("name"),                       # e.g. "GV80 3.5T Royal"
        "full_name": description or ld.get("name"),      # e.g. "Genesis GV80 3.5T Royal 7P AWD Long Plate"
        "year": parse_int(ld.get("vehicleModelDate")),
        "body_type": ld.get("bodyType"),
        "price": parse_int(offers.get("price")),
        "currency": offers.get("priceCurrency"),
        "availability": schema_tail(offers.get("availability")),
        "condition": schema_tail(ld.get("itemCondition") or offers.get("itemCondition")),
        "mileage_km": mileage_km,
        "exterior_color": ld.get("color") or specs.get("Exterior"),
        "interior_color": extra.get("Interior color") or specs.get("Interior"),
        "fuel_type": ld.get("fuelType") or specs.get("Fuel Type"),
        "transmission": ld.get("vehicleTransmission") or specs.get("Transmission"),
        "drivetrain": parse_drivetrain(drivetrain_src),
        "seats": parse_seats(drivetrain_src),
        "engine": engine or None,
        "certified": certified,
        "features": features,
        "image": ld.get("image"),
        "brand": (ld.get("brand", {}) or {}).get("name", "Genesis"),
    }
    return record


def record_to_document(rec: dict) -> dict:
    """Build a natural-language document for the retrieval layer."""
    price = (
        f"{rec['price']:,} {rec['currency']}"
        if rec.get("price") and rec.get("currency")
        else "price on request"
    )
    avail = {
        "InStock": "in stock and available",
        "OutOfStock": "currently out of stock",
        "SoldOut": "sold out",
    }.get(rec.get("availability"), rec.get("availability") or "availability unconfirmed")

    mileage = (
        "brand new (0 km)"
        if rec.get("mileage_km") == 0
        else f"{rec['mileage_km']:,} km"
        if rec.get("mileage_km")
        else "mileage not specified"
    )
    seats = f"{rec['seats']}-seater" if rec.get("seats") else None
    certified = "Genesis Certified Pre-Owned" if rec.get("certified") else None

    parts = [
        f"{rec.get('full_name') or rec.get('variant')} ({rec.get('year')}).",
        f"Model: Genesis {rec.get('model')}.",
        f"Body type: {rec.get('body_type')}." if rec.get("body_type") else "",
        f"Total purchase price: {price}.",
        f"Availability: {avail}.",
        f"Mileage: {mileage}.",
        f"Exterior colour: {rec.get('exterior_color')}." if rec.get("exterior_color") else "",
        f"Interior: {rec.get('interior_color')}." if rec.get("interior_color") else "",
        f"Fuel type: {rec.get('fuel_type')}." if rec.get("fuel_type") else "",
        f"Transmission: {rec.get('transmission')}." if rec.get("transmission") else "",
        f"Drivetrain: {rec.get('drivetrain')}." if rec.get("drivetrain") else "",
        f"Engine/spec: {rec.get('engine')}." if rec.get("engine") else "",
        f"Seating: {seats}." if seats else "",
        f"{certified}." if certified else "",
    ]
    if rec.get("features"):
        parts.append("Key features: " + ", ".join(rec["features"]) + ".")

    text = " ".join(p for p in parts if p)
    return {
        "id": rec["id"],
        "model": rec.get("model"),
        "year": rec.get("year"),
        "variant": rec.get("variant"),
        "price": rec.get("price"),
        "url": rec.get("url"),
        "text": text,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    records: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        print("Enumerating inventory ...")
        slugs = enumerate_slugs(page)
        print(f"Found {len(slugs)} vehicle pages.\n")

        for i, slug in enumerate(slugs, 1):
            url = BASE + slug
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector(".gvd-spec-item", state="attached", timeout=30_000)
            rec = build_record(url, page.content())
            records.append(rec)
            print(f"[{i:2d}/{len(slugs)}] {rec['model']} {rec['variant']} "
                  f"{rec['year']} - {rec['price']} {rec['currency']} "
                  f"({len(rec['features'])} features)")

        browser.close()

    # sort for stable output: model, year desc, price
    records.sort(key=lambda r: (r.get("model") or "", -(r.get("year") or 0), r.get("price") or 0))

    inv_path = DATA_DIR / "inventory.json"
    inv_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    docs = [record_to_document(r) for r in records]
    doc_path = DATA_DIR / "inventory_documents.jsonl"
    with doc_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # quick summary
    models = {}
    for r in records:
        models[r.get("model")] = models.get(r.get("model"), 0) + 1

    print(f"\nWrote {len(records)} records -> {inv_path}")
    print(f"Wrote {len(docs)} documents -> {doc_path}")
    print("By model:", ", ".join(f"{k}={v}" for k, v in sorted(models.items())))


if __name__ == "__main__":
    main()
