"""Groq LLM client that grounds each turn with RAG and streams the spoken reply
for sentence-by-sentence TTS pipelining."""

from __future__ import annotations

import datetime
import re
from collections import deque
from typing import Iterator

from groq import Groq

from rag.retriever import get_retriever, format_vehicles, RetrievalResult
from rag.query import parse
from rag.understand import understand
from crm.store import get_store, window_phrase

from . import config


def _load_system_prompt() -> str:
    return (config.PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8")


class GroqAgent:
    def __init__(self):
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it before running the agent."
            )
        # timeout + few retries so a rate-limited call fails fast instead of hanging
        self.client = Groq(api_key=config.GROQ_API_KEY, timeout=18.0, max_retries=1)
        self.system_prompt = _load_system_prompt()
        self.retriever = get_retriever()
        self.store = get_store()
        self.customer: dict | None = None
        self._turn = 0
        # booking sub-flow: None | "need_slot" | "need_name"
        self._booking_stage: str | None = None
        self._booking_day: str | None = None
        # the slot Aria last OFFERED, so "yes, that works" (no number spoken) still books it
        self._offered_time: str | None = None
        self._offered_day: str | None = None
        # a free slot the customer picked, held until we have their name to finalise
        self._pending_day: str | None = None
        self._pending_time: str | None = None
        # how the UI should treat car cards this turn: "set" | "keep" | "clear"
        self.card_action: str = "clear"
        # the exact cars grounded this turn, so cards always match what the agent is
        # actually talking about (incl. follow-ups & re-ranks)
        self.display_records: list[dict] = []
        self.history: deque[tuple[str, str]] = deque(maxlen=config.HISTORY_TURNS)
        self.last_result: RetrievalResult | None = None
        # cars discussed last turn, carried forward so follow-ups ("does it have a
        # sunroof?", "the second one") stay grounded
        self.recent_records: list[dict] = []
        # model/trim of the current subject, so a refining follow-up naming no car
        # ("what about one with 2000 km") re-searches in context
        self._subject_terms: str = ""

    def reset(self) -> None:
        """Wipe all conversation memory at the start of a new call/chat so the previous
        customer's cars, booking, name, and history never bleed into it."""
        self.customer = None
        self._booking_stage = None
        self._booking_day = None
        self._offered_time = self._offered_day = None
        self._pending_day = self._pending_time = None
        self.history.clear()
        self.recent_records = []
        self._subject_terms = ""
        self.card_action = "clear"
        self.display_records = []
        self.last_result = None
        self._turn = 0

    # Don't match a bare "come in" — it fires on "what colours does it come in?".
    # Require a viewing-ish continuation ("come in to/and see", "come see/by/over/visit").
    _BOOK_INTENT = re.compile(
        r"\b(book|appointment|viewing|schedule an?|reserve a|test drive|"
        r"come (?:see|by|over|visit|in to|and see|to see)|come to the showroom|"
        r"drop by|stop by|pop in)\b", re.I)
    _BOOK_OFFER = re.compile(   # did Aria's last reply offer to book?
        r"\b(book|appointment|viewing|schedule|come (?:in|by|see)|drop by|test drive)\b", re.I)
    _DAY = re.compile(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|"
                      r"saturday|sunday|weekend)\b", re.I)
    _AFFIRM = re.compile(r"\b(yes|yeah|yep|sure|please|ok|okay|sounds good|let'?s|go ahead|"
                         r"definitely|absolutely|i would|that works)\b", re.I)
    _DECLINE = re.compile(r"\b(no|nope|not now|not yet|maybe later|just looking|i'?m good|no thanks)\b", re.I)
    _AFFIRM_TIME = re.compile(r"\b(\d{1,2}|am|pm|o'?clock|noon|midday|morning|afternoon|evening|"
                              r"works|perfect|sounds good|that one|let'?s)\b", re.I)
    _SPELLED = re.compile(r"\b(?:[a-z][\s.\-]+){2,}[a-z]\b", re.I)   # "h-u-z-a-i-f-a"
    # asking ABOUT availability (a question), as opposed to committing to book
    _AVAIL_INTENT = re.compile(
        r"\b(available|availabilit|free|open(?:ing)?s?|any (?:slot|time|spot)|"
        r"what(?:'?s| is| are) (?:free|open|available)|(?:is|are) .* (?:free|open|taken|available)|"
        r"do you have .* (?:slot|time|spot)|slots?)\b", re.I)
    # customer wants to browse on their own — don't pitch or push
    _JUST_LOOKING = re.compile(
        r"\b(just (?:looking|browsing)|look(?:ing)? (?:around|myself|on my own)|by myself|"
        r"on my own|don'?t need|do not need|not interested|no help|"
        r"i'?ll (?:look|check|see|find|do) (?:it|that)? ?(?:myself|on my own)?)\b", re.I)
    # is this turn about cars at all? (only used to decide whether to refresh cards)
    _BROWSE = re.compile(
        r"\b(show|have|got|looking|recommend|suggest|see one|option|inventory|stock|"
        r"car|cars|vehicle|suv|sedan|coupe|model|something|price|cost|cheap|expensive|"
        r"new|used|mileage|colou?r|fuel|electric|petrol)\b", re.I)
    # customer explicitly starts a new search, so we must drop the pinned subject
    _RESTART = re.compile(
        r"\b(show me|what else|anything else|something else|other (?:car|model|option)s?|"
        r"another (?:car|model|one)|different (?:car|model|option)|move on|forget (?:that|those)|"
        r"new search|start over|instead)\b", re.I)
    # during these steps, stay locked on the car(s) under discussion — never feed a
    # fresh retrieval dump
    _BOOKING_ACTIONS = frozenset(
        {"ask_slot", "pick_slot", "name_book", "check_availability"})

    # customer moved on to a new car/inventory request — lets us break OUT of a
    # half-finished booking instead of trapping them in "what day/time?"
    _NEW_TOPIC = re.compile(
        r"\b(show me|do you (?:have|sell|stock|carry)|cheapest|cheaper|most expensive|priciest|"
        r"price of|how much|tell me about|recommend|looking for|interested in|what (?:do you|cars|"
        r"models)|electric|petrol|hybrid|suv|sedan|coupe|gv80|g80|g90|platinum|royal|prestige|"
        r"premium|advance)\b", re.I)

    def _is_new_request(self, text: str) -> bool:
        """Did they switch from the booking to a fresh car question?"""
        f = parse(text)
        if (f.models or f.trims or f.colors or f.body_types or f.fuels or f.sort
                or f.min_price is not None or f.max_price is not None
                or f.target_price is not None or f.target_mileage is not None):
            return True
        return bool(self._NEW_TOPIC.search(text))

    def _reset_booking(self) -> None:
        self._booking_stage = None
        self._offered_time = self._offered_day = None
        self._pending_day = self._pending_time = None

    def _plan_tool(self, text: str) -> str | None:
        """Decide which deterministic booking action to run (None = normal reply).

        Booking is intentionally short: state the window & ask the day/time -> check it's
        free -> grab the name -> book. No 'do you want to book?', no car disambiguation.
        """
        # mid-booking sub-flow, with an escape hatch so a new question isn't trapped
        if self._booking_stage == "need_slot":
            booking_reply = bool(self._DAY.search(text) or self._has_time(text)
                                 or (self._offered_time and self._AFFIRM.search(text)))
            if booking_reply or not self._is_new_request(text):
                return "pick_slot"
            self._reset_booking()                           # changed topic -> drop the booking
        elif self._booking_stage == "need_name":
            if self._SPELLED.search(text) or not self._is_new_request(text):
                return "name_book"
            self._reset_booking()                           # changed topic -> drop the booking

        # availability question (not a commitment to book) -> answer with real data
        if (self._AVAIL_INTENT.search(text) and not self._BOOK_INTENT.search(text)
                and (self._DAY.search(text) or self._has_time(text))):
            return "check_availability"

        # start booking: go straight to the slot step (state window, ask day+time)
        prev_reply = self.history[-1][1] if self.history else ""
        accepted = bool(self._BOOK_OFFER.search(prev_reply)
                        and self._AFFIRM.search(text) and not self._DECLINE.search(text))
        if (self._BOOK_INTENT.search(text) or accepted) and not self._JUST_LOOKING.search(text):
            self._booking_stage = "need_slot"
            return "pick_slot" if (self._DAY.search(text) or self._has_time(text)) else "ask_slot"
        return None

    _NAME_FILLER = {"no", "nope", "yes", "yeah", "its", "it", "is", "the", "name", "spelled",
                    "spell", "hi", "hello", "um", "uh", "so", "please", "thanks", "you", "sure"}

    def _extract_spelled_name(self, text: str) -> str | None:
        """Build the name from the spelled letters — deterministic fallback, no network."""
        toks = re.findall(r"[A-Za-z]+", text)
        letters = [t for t in toks if len(t) == 1]
        if len(letters) >= 2:
            return "".join(letters).capitalize()
        words = [w for w in toks if w.lower() not in self._NAME_FILLER]
        return words[0].capitalize() if words else None    # STT merged it into one word

    def _capture_name(self, text: str) -> str | None:
        """Read the customer's name. If they actually spell it (>=2 single letters), join
        the letters exactly — never ask Groq, which 'corrects' spellings it half-recognises
        ('h u z a i f a' -> 'Huzafia'). Only let Groq interpret when it's not clearly spelled
        (STT merged or garbled it)."""
        toks = re.findall(r"[A-Za-z]+", text)
        letters = [t for t in toks if len(t) == 1]
        if len(letters) >= 2:
            return "".join(letters).capitalize()                 # spelled out -> exact

        if self.client:                                          # merged/garbled -> let Groq read it
            try:
                resp = self.client.chat.completions.create(
                    model=config.GROQ_MODEL, temperature=0, max_tokens=12,
                    messages=[
                        {"role": "system", "content":
                         "Extract the person's first name from the message and reply with ONLY that "
                         "name, properly capitalised — nothing else. Do NOT invent or 'correct' it; "
                         "use the letters given. If there is no name, reply exactly NONE."},
                        {"role": "user", "content": text},
                    ],
                )
                out = (resp.choices[0].message.content or "").strip().strip(".,!?").split()
                cand = out[0] if out else ""
                if cand and cand.upper() != "NONE" and cand.isalpha() and 2 <= len(cand) <= 20:
                    return cand.capitalize()
            except Exception:  # noqa: BLE001
                pass

        words = [w for w in toks if w.lower() not in self._NAME_FILLER]
        return words[0].capitalize() if words else None

    def _extract_time(self, text: str) -> str:
        m = re.search(r"\b(\d{1,2})(?:[:\s](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", text, re.I)
        if m:
            h, mins = int(m.group(1)), int(m.group(2) or 0)
            ap = (m.group(3) or "").replace(".", "").upper()
            if not ap:                                       # no am/pm spoken -> infer
                if h == 0:
                    h, ap = 12, "AM"
                elif h == 12:
                    ap = "PM"
                elif h > 12:                                 # 24-hour clock, e.g. 16:00 -> 4 PM
                    h, ap = h - 12, "PM"
                elif h in (10, 11):
                    ap = "AM"                                # 10-11 -> morning
                else:
                    ap = "PM"                                # 1-9 in a 10-5 day -> afternoon
            return f"{h}:{mins:02d} {ap}"                    # canonical -> matches a 30-min slot label
        if "noon" in text.lower() or "midday" in text.lower():
            return "12:00 PM"
        return text

    def _has_time(self, text: str) -> bool:
        # "2pm" glues the digit to "pm" (no word boundary), so \b\d{1,2}\b alone misses it;
        # also match an am/pm-suffixed number, bare numbers, and time words.
        return bool(re.search(
            r"\d{1,2}\s*(?::\d{2})?\s*[ap]\.?m\.?|\b\d{1,2}(?::\d{2})?\b|"
            r"\bnoon\b|\bmidday\b|\bhalf past\b|o'?clock", text, re.I))

    def _extract_day_text(self, text: str) -> str:
        m = self._DAY.search(text)
        return m.group(0) if m else self._booking_day or "tomorrow"

    @staticmethod
    def _slot_minutes(label: str) -> int:
        m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)", label, re.I)
        if not m:
            return 0
        h = int(m.group(1)) % 12 + (12 if m.group(3).upper() == "PM" else 0)
        return h * 60 + int(m.group(2))

    def _nearest_free(self, free: list[str], requested: str, n: int = 3) -> list[str]:
        if not free:
            return []
        target = self._slot_minutes(requested)
        return sorted(free, key=lambda f: abs(self._slot_minutes(f) - target))[:n]

    @staticmethod
    def _day_label(iso: str) -> str:
        """ISO date -> spoken weekday, e.g. '2026-06-30' -> 'Tuesday'."""
        try:
            return datetime.date.fromisoformat(iso).strftime("%A")
        except ValueError:
            return iso

    # Scheduling speaks deterministic lines (prefix "SAY|") rather than letting the 8B
    # improvise — it was over-claiming ("you're all set" before a slot, listing random
    # cars). The LLM still handles all the car conversation.
    def _do_action(self, action: str, text: str) -> str | None:
        """Run the SQLite op directly. Booking steps return a 'SAY|<verbatim line>'."""
        win = window_phrase()

        if action == "ask_slot":
            return f"SAY|We're open {win}. What day and time would you like to come in?"

        if action == "check_availability":
            info = self.store.available_slots(self._extract_day_text(text))
            if info.get("closed"):
                self._offered_time = self._offered_day = None
                return f"SAY|I'm afraid we're closed on {self._day_label(info['day'])}. We're open {win} — which day works for you?"
            self._booking_day = self._offered_day = info["day"]
            lbl, free = self._day_label(info["day"]), info["free"]
            if self._has_time(text):
                requested = self._extract_time(text)
                if requested in free:
                    self._offered_time = requested
                    return f"SAY|Yes, {requested} on {lbl} is open. Would you like me to book it for you?"
                near = self._nearest_free(free, requested)
                self._offered_time = near[0] if near else None
                return f"SAY|I'm afraid {requested} on {lbl} isn't free. I do have {self._join(near)} — would any of those work?"
            if not free:
                self._offered_time = None
                return f"SAY|{lbl} is fully booked, I'm afraid. We're open {win} — would another day work?"
            self._offered_time = free[0]
            return f"SAY|On {lbl} I have {self._join(free[:3])} open. What time would you like?"

        if action == "pick_slot":
            affirm_offer = bool(self._offered_time and self._AFFIRM.search(text)
                                and not self._DECLINE.search(text) and not self._has_time(text))
            # no day AND no time (e.g. they blurted their name) -> assume nothing, re-ask
            if not self._DAY.search(text) and not self._has_time(text) and not affirm_offer:
                lead = ("I'll take your name once we've set a time. " if self._SPELLED.search(text) else "")
                return f"SAY|{lead}We're open {win}. What day and time would you like to come in?"
            day_txt = (self._offered_day if (affirm_offer and not self._DAY.search(text) and self._offered_day)
                       else self._extract_day_text(text))
            info = self.store.available_slots(day_txt)
            lbl = self._day_label(info["day"])
            if info.get("closed"):
                self._offered_time = None
                return f"SAY|I'm sorry, we're closed on {lbl}. We're open {win} — which day works for you?"
            self._booking_day, free = info["day"], info["free"]

            if self._has_time(text):
                requested = self._extract_time(text)
            elif affirm_offer:
                requested = self._offered_time
            else:                                            # day but no time -> ask the time
                self._offered_day = info["day"]
                return f"SAY|Sure — what time on {lbl} works for you? We're open {win}."

            if requested not in free:                        # taken / off-grid -> offer nearest
                near = self._nearest_free(free, requested)
                self._offered_time = near[0] if near else None
                self._offered_day = info["day"]
                if not near:
                    return f"SAY|I'm afraid {lbl} is fully booked. We're open {win} — would another day work?"
                return f"SAY|I'm afraid {requested} on {lbl} is taken. I do have {self._join(near)} — would any of those suit you?"

            # slot is free -> hold it. Known customer books now; otherwise take the name next.
            self._pending_day, self._pending_time = info["day"], requested
            if self.customer and self.customer.get("id"):
                return self._finalize_booking()
            self._booking_stage = "need_name"
            return f"SAY|{requested} on {lbl} is open. Could I take your name, spelled out — like A-R-S-A-L-A-N?"

        if action == "name_book":
            name = self._capture_name(text)
            if not name:
                return "SAY|Sorry, I didn't catch that — could you spell your name out for me, letter by letter?"
            self.customer = self.store.identify(name)
            return self._finalize_booking()
        return None

    @staticmethod
    def _join(items: list[str]) -> str:
        """'a, b or c' for a natural spoken list."""
        items = [i for i in items if i]
        if not items:
            return "no open times"
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " or " + items[-1]

    def _finalize_booking(self) -> str:
        """Book the held slot for the known customer and confirm verbatim."""
        model = self.recent_records[0].get("model") if self.recent_records else None
        res = self.store.book(self.customer["id"], self._pending_day, self._pending_time, model)
        self._booking_stage = None
        if res.get("ok"):
            self._offered_time = self._offered_day = None
            self._pending_day = self._pending_time = None
            self.customer["appointments"] = self.store.appointments(self.customer["id"])
            return (f"SAY|You're all set, {self.customer['name']} — "
                    f"{self._day_label(res['day'])} at {res['time']}. We'll see you then!")
        # rare race: the slot was taken between checking and booking
        near = self._nearest_free(res.get("free", []), self._pending_time or "")
        self._offered_time = near[0] if near else None
        self._offered_day = self._pending_day
        lbl = self._day_label(self._pending_day or "")
        self._pending_day = self._pending_time = None
        self._booking_stage = "need_slot"
        return f"SAY|Ah, that slot was just taken. I have {self._join(near)} — would any of those work?"

    def _customer_context(self) -> str | None:
        c = self.customer
        if not c or not c.get("id"):
            return None
        parts = [f"CURRENT CUSTOMER: {c['name']}.",
                 "Returning customer." if c.get("known") else "New customer (first visit)."]
        if c.get("interests"):
            parts.append("Previously looked at: " + ", ".join(c["interests"]) + ".")
        if c.get("appointments"):
            ap = "; ".join(f"{a['day']} at {a['time']}" + (f" for the {a['model']}" if a.get("model") else "")
                           for a in c["appointments"])
            parts.append("Existing appointments: " + ap + ".")
        parts.append("Use their name naturally; don't ask for it again.")
        return " ".join(parts)

    def _build_messages(self, user_text: str, grounding: str, no_match: bool,
                        action_note: str | None = None, booking: bool = False,
                        match_hint: str | None = None) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "system", "content": (
            "INVENTORY FACTS (whole-stock totals — use THESE for any question about the overall "
            "range, mileage, or what's on offer; never generalise from the few cars listed below): "
            + self.retriever.summary_facts())})
        ctx = self._customer_context()
        if ctx:
            messages.append({"role": "system", "content": ctx})
        for u, a in self.history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

        if no_match:
            header = (
                "NO VEHICLE MATCHES THE CUSTOMER'S REQUEST. The cars below are the "
                "closest available but DO NOT meet what was asked. Tell the customer "
                "clearly that you don't have exactly what they asked for, then offer "
                "these as alternatives only if relevant:"
            )
        else:
            header = "RETRIEVED INVENTORY (the only cars you may reference this turn):"

        user_block = (
            f"{header}\n{grounding if grounding else '(none)'}\n\n"
            f"Customer said: {user_text}"
        )
        messages.append({"role": "user", "content": user_block})
        if match_hint:    # the retrieved cars satisfy a mileage/price target — present, don't deny
            messages.append({"role": "system", "content": match_hint})
        if action_note:   # result of a just-run CRM action + how to phrase it
            if booking:
                action_note += (" IMPORTANT: Do exactly this and nothing else — do NOT list, "
                                "name, or describe any cars, do NOT make your own booking offer, "
                                "and do NOT claim anything is booked/confirmed unless this note "
                                "literally says BOOKED. The cars above are only context.")
            messages.append({"role": "system", "content": action_note})
        return messages

    @staticmethod
    def _dedupe(records: list[dict]) -> list[dict]:
        """Collapse truly-identical listings so the agent never reads the same car twice.
        Mileage is part of the key: two same-name/price cars with different mileage are
        different physical cars and must both survive (else a 'with 300 km' match could be
        deduped away)."""
        seen, out = set(), []
        for r in records:
            key = (r.get("full_name"), r.get("price"), r.get("mileage_km"))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    # a refinement = a new attribute value (mileage/price/colour/sort) to apply to
    # whatever we're already discussing
    @staticmethod
    def _is_refine(f) -> bool:
        return bool(f.target_mileage is not None or f.target_price is not None or f.colors
                    or f.sort or f.min_price is not None or f.max_price is not None or f.brand_new)

    @staticmethod
    def _has_anchor(f) -> bool:
        return bool(f.models or f.trims or f.body_types or f.fuels or f.drivetrains or f.coupe)

    def respond_stream(self, user_text: str) -> Iterator[str]:
        """Understand the request (LLM-structured, regex fallback), retrieve, stream reply."""
        # booking router first: deterministic, speaks verbatim SAY| lines, no LLM
        self._turn += 1
        action = self._plan_tool(user_text)
        if action:
            note = self._do_action(action, user_text)
            if note and note.startswith("SAY|"):
                reply = note[4:].strip()
                self.display_records = self.recent_records
                self.card_action = "keep" if self.recent_records else "clear"
                self.history.append((user_text, reply))
                yield reply
                return
        # every booking action returns SAY|, so past here it's a normal car/inventory turn

        # Fast path: if the regex sees a concrete attribute (model, trim, colour, price,
        # mileage, sort...), trust it — no extra LLM call. Slow path: only when the turn
        # has no recognisable attribute ("recommend me anything", "which is better") do we
        # call the LLM to resolve intent/context via the Pydantic schema.
        if self._RESTART.search(user_text):
            self._subject_terms = ""
        pf = parse(user_text)
        has_signal = self._has_anchor(pf) or self._is_refine(pf)
        cq = None
        if config.USE_LLM_UNDERSTANDING and not has_signal:
            cq = understand(self.client, config.GROQ_UNDERSTAND_MODEL, user_text, self._subject_terms)

        if cq is not None:                               # LLM structured path (vague/contextual)
            f = cq.to_filters()
            result = self.retriever.search(user_text, k=config.TOP_K, filters=f)
            current = [h.record for h in result.hits]
            followup = bool(self.recent_records) and cq.keep_context and not cq.has_filter()
            records = self.recent_records if followup else current
            subj = sorted(f.models) + sorted(f.trims)
            if subj:
                self._subject_terms = " ".join(x.lower() for x in subj)
            elif not cq.keep_context:
                self._subject_terms = ""                 # open/fresh request, no specific car
        else:                                            # regex fast path (clear query)
            inherit = (not self._has_anchor(pf) and self._is_refine(pf) and bool(self._subject_terms)
                       and not self._BOOK_INTENT.search(user_text))
            search_text = f"{self._subject_terms} {user_text}" if inherit else user_text
            result = self.retriever.search(search_text, k=config.TOP_K)
            f = result.filters
            current = [h.record for h in result.hits]
            if f.models or f.trims:
                self._subject_terms = " ".join(sorted(x.lower() for x in f.models) + sorted(f.trims))
            new_constraint = bool(f.models or f.fuels or f.drivetrains or f.body_types or f.coupe
                                  or f.trims or f.colors or f.min_price is not None
                                  or f.max_price is not None or f.brand_new)
            followup = (bool(self.recent_records) and not new_constraint and not inherit
                        and not self._is_refine(f) and not self._RESTART.search(user_text))
            records = self.recent_records if followup else current

        self.last_result = result

        # auto-log what an identified customer is looking at
        if self.customer and self.customer.get("id"):
            for m in dict.fromkeys(r.get("model") for r in records[:2] if r.get("model")):
                self.store.log_interest(self.customer["id"], m)

        # cards: show whenever we have cars and they're not waving us off
        self.card_action = "clear" if (self._JUST_LOOKING.search(user_text) or not records) else "set"

        # dedupe + honour a re-rank (sort / nearest mileage / nearest price)
        records = self._dedupe(records)
        if f.sort:
            records = self.retriever.sort_records(records, f.sort)
        elif f.target_mileage is not None:
            records = sorted(records, key=lambda r: abs((r.get("mileage_km") or 0) - f.target_mileage))
        elif f.target_price is not None:
            records = sorted(records, key=lambda r: abs((r.get("price") or 0) - f.target_price))
        self.display_records = records

        # if they asked by mileage/price, name the closest match so it presents (not denies) it
        match_hint = None
        if records:
            if f.target_mileage is not None and records[0].get("mileage_km") is not None:
                match_hint = (f"MATCH NOTE: They asked about ~{f.target_mileage:,} km. The cars below are "
                              f"ordered by closeness to that — the first is {records[0]['mileage_km']:,} km. "
                              "These ARE matches; present the closest affirmatively. Never say we don't have it.")
            elif f.target_price is not None and records[0].get("price"):
                match_hint = (f"MATCH NOTE: They asked about ~{f.target_price:,} SAR. The cars below are ordered "
                              f"by closeness to that price — the first is {records[0]['price']:,}. Present the "
                              "closest affirmatively. Never say we don't have it.")

        grounding = format_vehicles(records, max_features=config.GROUNDING_FEATURES)
        messages = self._build_messages(user_text, grounding, result.used_fallback, match_hint=match_hint)
        full = yield from self._stream_plain(messages)

        self.history.append((user_text, full))
        if not followup:
            self.recent_records = current[:3]

    def _stream_plain(self, messages: list[dict]) -> Iterator[str]:
        """Stream one reply. If a model is rate-limited (429), fall through to the next
        model automatically — each Groq model has its own limit, so this keeps the call
        alive instead of dying silently and dumping the user back to 'listening'."""
        models = [config.GROQ_MODEL, *config.GROQ_FALLBACK_MODELS]
        last_err: Exception | None = None
        for i, model in enumerate(models):
            full: list[str] = []
            yielded = False
            try:
                stream = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS, stream=True,
                )
                for chunk in stream:
                    d = chunk.choices[0].delta.content or ""
                    if d:
                        full.append(d)
                        yielded = True
                        yield d
                return "".join(full)
            except Exception as e:  # noqa: BLE001
                last_err = e
                # only safe to retry on another model if we haven't streamed anything yet
                if yielded or i == len(models) - 1:
                    raise
                print(f"[llm] {model} failed ({str(e)[:80]}); trying {models[i + 1]}", flush=True)
        if last_err:
            raise last_err

    def respond(self, user_text: str) -> str:
        """Non-streaming convenience wrapper (text mode / testing)."""
        return "".join(self.respond_stream(user_text))
