"""
Retrieval evaluation harness.

Runs a battery of realistic customer questions and asserts that the top hit(s)
satisfy the expected constraint. This is the proof that "the customer's question
reliably pulls the right vehicle(s)" before we wire it into voice.

Run:  python -m rag.eval   (from the project root, venv active)
"""

from __future__ import annotations

import time

from .retriever import get_retriever

# Each case: (question, predicate over the top hit's record, human label)
CASES = [
    ("Do you have any electric Genesis cars?",
     lambda r: r["fuel_type"] == "Electric", "EV only"),
    ("What's the cheapest car you have?",
     lambda r: r["price"] == 99000, "global cheapest = 99k EV"),
    ("Show me the cheapest GV80",
     lambda r: r["model"] == "GV80" and r["price"] == 205000, "cheapest GV80"),
    ("What's your most expensive model?",
     lambda r: r["price"] == 335000, "priciest = 335k GV80"),
    ("I want a G90",
     lambda r: r["model"] == "G90", "model G90"),
    ("Looking for an SUV",
     lambda r: r["body_type"] == "SUV", "SUV body"),
    ("Do you have a sedan under 200k?",
     lambda r: r["body_type"] == "Sedan" and r["price"] < 200000, "sedan < 200k"),
    ("Any GV80 in white?",
     lambda r: r["model"] == "GV80" and "white" in r["exterior_color"].lower(), "white GV80"),
    ("I'd like a G80 with a panoramic sunroof",
     lambda r: r["model"] == "G80" and any("Sunroof" in f for f in r["features"]), "G80 + sunroof"),
    ("Show me cars between 250k and 300k",
     lambda r: 250000 <= r["price"] <= 300000, "price band 250-300k"),
    ("electric car under 150000",
     lambda r: r["fuel_type"] == "Electric" and r["price"] < 150000, "EV < 150k"),
    ("newest GV80 available",
     lambda r: r["model"] == "GV80" and r["year"] == 2026, "newest GV80"),
    ("a 2023 model",
     lambda r: r["year"] == 2023, "year 2023"),
    ("rear wheel drive G80",
     lambda r: r["model"] == "G80" and r["drivetrain"] == "RWD", "RWD G80"),
    ("all wheel drive SUV with ventilated seats",
     lambda r: r["drivetrain"] == "AWD" and any("Ventilated" in f for f in r["features"]), "AWD + ventilated"),
    ("a brand new GV80 with zero km",
     lambda r: r["model"] == "GV80" and r["mileage_km"] == 0, "new GV80 0km"),
    ("luxury car with heads up display and 360 camera",
     lambda r: any("Heads-Up" in f for f in r["features"]) and any("360" in f for f in r["features"]),
     "HUD + 360"),
    ("a comfortable executive sedan for business",
     lambda r: r["body_type"] == "Sedan", "semantic -> sedan"),
    ("around 240k",
     lambda r: abs(r["price"] - 240000) <= 20000, "~240k"),
    ("the coupe",
     lambda r: "coupe" in r["variant"].lower(), "coupe variant"),
]


def main() -> None:
    ret = get_retriever()

    # latency probe (after warmup)
    t = time.perf_counter()
    ret.search("electric SUV under 300k")
    warm_ms = (time.perf_counter() - t) * 1000

    passed = 0
    print(f"{'RESULT':6} | {'QUESTION':48} | TOP HIT")
    print("-" * 110)
    for question, predicate, label in CASES:
        res = ret.search(question, k=3)
        top = res.hits[0].record if res.hits else None
        ok = bool(top and predicate(top))
        passed += ok
        tag = "PASS" if ok else "FAIL"
        summary = (
            f"{top['model']} {top['variant']} {top['year']} - {top['price']:,} {top['currency']}"
            if top else "<none>"
        )
        flag = " [fallback]" if res.used_fallback else ""
        print(f"{tag:6} | {question[:48]:48} | {summary}{flag}")

    print("-" * 110)
    print(f"PASSED {passed}/{len(CASES)}   |   warm query latency ~{warm_ms:.1f} ms")


if __name__ == "__main__":
    main()
