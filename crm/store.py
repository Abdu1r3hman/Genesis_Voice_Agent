"""Local SQLite store for customer identity, memory, and appointment booking."""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from pathlib import Path

import jellyfish
from rapidfuzz import process, fuzz

from voice import config

_THRESHOLD = config.NAME_MATCH_THRESHOLD
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Slots are generated from working hours, not listed by hand, so the dashboard
# can edit hours without touching slot definitions.
_SCHEDULE_FILE = config.ROOT / "data" / "schedule.json"
_DEFAULT_SCHEDULE = {"days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                     "start": "10:00", "end": "17:00", "slot_minutes": 30}


def load_schedule() -> dict:
    try:
        data = json.loads(_SCHEDULE_FILE.read_text(encoding="utf-8"))
        return {**_DEFAULT_SCHEDULE, **data}
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_SCHEDULE)


def save_schedule(sch: dict) -> None:
    _SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SCHEDULE_FILE.write_text(json.dumps(sch, indent=2), encoding="utf-8")


def working_days() -> list[str]:
    return [d.lower() for d in load_schedule().get("days", [])]


def window_phrase() -> str:
    """Human availability window, e.g. 'Monday to Friday, 10 AM to 5 PM'."""
    sch = load_schedule()
    idxs = sorted(_WEEKDAYS.index(d) for d in working_days() if d in _WEEKDAYS)
    if not idxs:
        return "by appointment"
    if idxs == list(range(idxs[0], idxs[-1] + 1)):
        days_str = f"{_WEEKDAYS[idxs[0]].title()} to {_WEEKDAYS[idxs[-1]].title()}"
    else:
        days_str = ", ".join(_WEEKDAYS[i].title() for i in idxs)
    try:
        def hm(s):
            h, m = map(int, str(s).split(":"))
            return _fmt12(h * 60 + m)
        return f"{days_str}, {hm(sch['start'])} to {hm(sch['end'])}"
    except (ValueError, KeyError):
        return days_str


def _fmt12(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def slot_times() -> list[str]:
    """Generate the day's slot labels from the schedule's hours + slot length."""
    sch = load_schedule()
    try:
        sh, sm = map(int, str(sch["start"]).split(":"))
        eh, em = map(int, str(sch["end"]).split(":"))
        step = max(5, int(sch.get("slot_minutes", 30)))
        start, end = sh * 60 + sm, eh * 60 + em
    except (ValueError, KeyError):
        return config.APPOINTMENT_SLOTS
    times, t = [], start
    while t + step <= end:                 # slot must finish by closing time, not just start before it
        times.append(_fmt12(t))
        t += step
    return times


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def normalize_day(text: str) -> str:
    """Resolve 'today' / 'tomorrow' / a weekday to an ISO date; else a cleaned label."""
    s = (text or "").strip().lower()
    today = datetime.date.today()
    if s in ("today", "tonight"):
        return today.isoformat()
    if s in ("tomorrow", "tmrw", "tmr"):
        return (today + datetime.timedelta(days=1)).isoformat()
    for i, wd in enumerate(_WEEKDAYS):
        if wd in s:
            delta = (i - today.weekday()) % 7
            delta = delta or 7  # a named weekday means the next one, never today
            return (today + datetime.timedelta(days=delta)).isoformat()
    return s or today.isoformat()


def _canon_time(text: str) -> str | None:
    """Snap a spoken time ('4 pm', '16:00') to one of the configured slot labels."""
    if not text:
        return None
    best = process.extractOne(text.strip(), slot_times(), scorer=fuzz.WRatio)
    return best[0] if best and best[1] >= 60 else None


class CustomerStore:
    def __init__(self, db_path: Path = config.CRM_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers(
                    id INTEGER PRIMARY KEY,
                    name TEXT, key TEXT, metaphone TEXT,
                    first_seen TEXT, last_seen TEXT);
                CREATE TABLE IF NOT EXISTS interests(
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER, model TEXT, ts TEXT);
                CREATE TABLE IF NOT EXISTS appointments(
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER, day TEXT, time TEXT, model TEXT,
                    status TEXT DEFAULT 'booked', created TEXT,
                    UNIQUE(day, time));
                """
            )
            self.conn.commit()

    def identify(self, name: str) -> dict:
        """Find-or-create a customer, matching by fuzzy name. Returns their profile."""
        name = (name or "").strip()
        if not name:
            return {"id": None, "name": None, "known": False}
        key = name.lower()
        with self.lock:
            rows = self.conn.execute("SELECT * FROM customers").fetchall()
            best, best_score = None, 0.0
            for r in rows:
                # Jaro-Winkler so speech-to-text spelling drift still resolves to the same person
                score = jellyfish.jaro_winkler_similarity(key, r["key"] or "")
                if score > best_score:
                    best, best_score = r, score

            if best and best_score >= _THRESHOLD:
                cid = best["id"]
                self.conn.execute("UPDATE customers SET last_seen=? WHERE id=?", (_now(), cid))
                self.conn.commit()
                known, disp = True, best["name"]
            else:
                cur = self.conn.execute(
                    "INSERT INTO customers(name,key,metaphone,first_seen,last_seen) VALUES(?,?,?,?,?)",
                    (name, key, jellyfish.metaphone(name), _now(), _now()),
                )
                self.conn.commit()
                cid, known, disp = cur.lastrowid, False, name

        return {
            "id": cid, "name": disp, "known": known,
            "interests": self.interests(cid),
            "appointments": self.appointments(cid),
        }

    def interests(self, customer_id: int) -> list[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT DISTINCT model FROM interests WHERE customer_id=? ORDER BY ts DESC LIMIT 6",
                (customer_id,),
            ).fetchall()
        return [r["model"] for r in rows if r["model"]]

    def log_interest(self, customer_id: int, model: str):
        if not customer_id or not model:
            return
        with self.lock:
            self.conn.execute(
                "INSERT INTO interests(customer_id,model,ts) VALUES(?,?,?)",
                (customer_id, model, _now()),
            )
            self.conn.commit()

    def available_slots(self, day: str) -> dict:
        d = normalize_day(day)
        try:
            wd = datetime.date.fromisoformat(d).strftime("%A").lower()
            if wd not in working_days():
                return {"day": d, "free": [], "taken": [], "closed": True,
                        "open_days": working_days()}
        except ValueError:
            pass  # non-date label -> skip the weekday gate
        with self.lock:
            taken = {r["time"] for r in self.conn.execute(
                "SELECT time FROM appointments WHERE day=? AND status='booked'", (d,)).fetchall()}
        free = [t for t in slot_times() if t not in taken]
        return {"day": d, "free": free, "taken": sorted(taken), "closed": False}

    def book(self, customer_id: int, day: str, time: str, model: str | None = None) -> dict:
        d = normalize_day(day)
        t = _canon_time(time)
        if not t:
            return {"ok": False, "reason": "unrecognised_time", "free": self.available_slots(d)["free"]}
        with self.lock:
            taken = self.conn.execute(
                "SELECT 1 FROM appointments WHERE day=? AND time=? AND status='booked'", (d, t)).fetchone()
            if taken:
                free = [x for x in slot_times() if x != t and not self.conn.execute(
                    "SELECT 1 FROM appointments WHERE day=? AND time=? AND status='booked'", (d, x)).fetchone()]
                return {"ok": False, "reason": "slot_taken", "day": d, "time": t, "free": free}
            self.conn.execute(
                "INSERT INTO appointments(customer_id,day,time,model,created) VALUES(?,?,?,?,?)",
                (customer_id, d, t, model, _now()))
            self.conn.commit()
        return {"ok": True, "day": d, "time": t, "model": model}

    def book_manual(self, name: str, day: str, time: str, model: str | None = None) -> dict:
        """Dashboard booking: find-or-create the customer, then book."""
        cust = self.identify((name or "").strip() or "Walk-in")
        return self.book(cust["id"], day, time, model)

    def clear_appointments(self) -> int:
        """Delete all bookings; returns the number removed."""
        with self.lock:
            cur = self.conn.execute("DELETE FROM appointments")
            self.conn.commit()
        return cur.rowcount

    def cancel(self, day: str, time: str) -> bool:
        """Cancel a booking, freeing the slot."""
        d = normalize_day(day)
        with self.lock:
            cur = self.conn.execute(
                "DELETE FROM appointments WHERE day=? AND time=?", (d, time))
            self.conn.commit()
        return cur.rowcount > 0

    def appointments(self, customer_id: int) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT day,time,model FROM appointments WHERE customer_id=? AND status='booked' ORDER BY day,time",
                (customer_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_appointments(self) -> list[dict]:
        """Every booking joined with its customer name, for the admin view."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT a.day, a.time, a.model, c.name, a.created "
                "FROM appointments a JOIN customers c ON c.id=a.customer_id "
                "WHERE a.status='booked' ORDER BY a.day, a.time").fetchall()
        return [dict(r) for r in rows]


_store: CustomerStore | None = None


def get_store() -> CustomerStore:
    """Process-wide singleton (one SQLite connection shared across agents)."""
    global _store
    if _store is None:
        _store = CustomerStore()
    return _store
