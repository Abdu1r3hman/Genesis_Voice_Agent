"""Starlette WebSocket server for the Genesis CPO voice agent: STT/TTS/RAG
pipeline plus the browser UI. Models warm in the background and the server
announces readiness over the socket once loaded."""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import re
import sys
import threading
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = Path(__file__).resolve().parent / "static"

from voice.llm import GroqAgent          # noqa: E402
from voice.stt import make_transcriber   # noqa: E402
from voice.tts import Speaker            # noqa: E402
from voice.agent import split_sentences  # noqa: E402
from voice import config                 # noqa: E402
from crm.store import get_store, slot_times, load_schedule, save_schedule, working_days  # noqa: E402

GREETING = (
    "Hello, and welcome to Genesis Certified Pre-Owned. I'm Aria, your personal concierge. "
    "Are you looking for a particular model today, or shall I show you around the collection?"
)

GOODBYE = re.compile(
    r"\b(bye|goodbye|good bye|see you|farewell|that'?s all|that'?s it|nothing else|"
    r"i'?m done|we'?re done|end (the )?call|hang up|talk later|take care)\b", re.I)

_STT = None
_TTS = None
_READY = threading.Event()


BUILD = "2026-06-28f  (hybrid: instant common Qs, LLM only for vague · 70B reply · buffer @1.3s)"


def _warm():
    global _STT, _TTS
    print(f"=== Genesis Concierge build {BUILD} ===", flush=True)
    print("Loading models (STT + TTS + retriever)...", flush=True)
    _STT = make_transcriber()
    _TTS = Speaker()
    _TTS.synth("Ready.")
    _STT.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
    GroqAgent()                          # warm the retriever singleton
    _READY.set()
    print("Ready -> http://localhost:8000", flush=True)


# colour name -> accent hex, so cards/orb are tinted by the car's real colour
_COLOR_HEX = {
    "capri blue": "#2f5a86", "ceres blue": "#3a5a80", "tasman blue": "#34699a",
    "matira blue": "#2e5a8f", "berling blue": "#2a4d75", "ultramarine": "#27408b",
    "marine": "#2e5a8f", "white": "#e9ecef", "glacier": "#e9eef0",
    "storr green": "#2f4538", "hallasen": "#33503f", "brunswick": "#2c4636",
    "cardiff": "#35513f", "forest": "#2f4538", "olive": "#4a4b2f", "green": "#34503f",
    "makalu": "#6e7479", "melbourne": "#6e7479", "geneva": "#b8bcc0",
    "shooting star": "#aeb4ba", "ash": "#7e8488", "anthracite": "#3a3d40",
    "silver": "#b8bcc0", "grey": "#6e7479", "gray": "#6e7479",
    "black sapphire": "#15181b", "maui black": "#16181a", "obsidian": "#141618",
    "black": "#15181b", "barossa": "#5e2230", "burgundy": "#5e2230",
    "maroon": "#5e2230", "bordeaux": "#5a2330", "red": "#7a2230",
    "havana": "#4a3528", "urban brown": "#4a3528", "camel": "#8a6f4a",
    "dune": "#9a8463", "ecru": "#cabfa6", "beige": "#c9bfa6", "vanilla": "#e6dcc2",
    "brown": "#4a3528", "vrown": "#4a3528", "copper": "#9a5b3b", "orange": "#b5642f",
    "blue": "#2f5a86",
}


def _accent(color: str | None) -> str:
    c = (color or "").lower()
    for key, hexv in _COLOR_HEX.items():
        if key in c:
            return hexv
    return "#5a6470"


def _car_payload(records) -> list[dict]:
    cars = []
    for r in (records or [])[:4]:
        cars.append({
            "name": r.get("variant") or r.get("full_name"),
            "fullName": r.get("full_name"),
            "model": r.get("model"),
            "year": r.get("year"),
            "price": r.get("price"),
            "currency": r.get("currency"),
            "color": r.get("exterior_color"),
            "interior": r.get("interior_color"),
            "mileage": r.get("mileage_km"),
            "fuel": r.get("fuel_type"),
            "drivetrain": r.get("drivetrain"),
            "body": r.get("body_type"),
            "features": (r.get("features") or [])[:5],
            "url": r.get("url"),
            "image": r.get("image"),
            "accent": _accent(r.get("exterior_color")),
        })
    return cars


def _wav_b64(samples: np.ndarray, sr: int) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def _decode_pcm(b64: str, rate: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if rate != config.SAMPLE_RATE and len(samples):
        n = int(len(samples) * config.SAMPLE_RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples), n, endpoint=False),
            np.arange(len(samples)), samples,
        ).astype(np.float32)
    return samples


async def _say_text(ws: WebSocket, text: str):
    """Synthesize sentence-by-sentence; each frame carries its own text for synced captions."""
    sents, rem = split_sentences(text)
    for s in sents + ([rem] if rem.strip() else []):
        s = s.strip()
        if not s:
            continue
        samples, sr = await asyncio.to_thread(_TTS.synth, s)
        await ws.send_json({"type": "audio", "wav": _wav_b64(samples, sr), "text": s})


# Lead-ins spoken only while the answer is being prepared, to avoid dead air. The real
# reply is queued after this finishes (the client's audio queue is gapless+sequential),
# so it flows like one sentence and never cuts Aria off mid-buffer.
_FILLERS = ["Sure, let me check.", "One moment.", "Let me see.", "Of course, just a second.",
            "Right, let me look that up.", "Good question — let me check."]


async def _process(ws: WebSocket, agent: GroqAgent, user_text: str, voice: bool = True):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    spoke = asyncio.Event()      # set the instant real reply audio is about to go out

    async def filler():
        """Speak a lead-in only if the reply is slow, so the caller gets quick
        conversational feedback instead of dead air while the turn is prepared."""
        try:
            await asyncio.wait_for(spoke.wait(), timeout=1.3)
            return                                   # reply was fast — no lead-in needed
        except asyncio.TimeoutError:
            pass
        if spoke.is_set():
            return
        line = _FILLERS[agent._turn % len(_FILLERS)]
        samples, sr = await asyncio.to_thread(_TTS.synth, line)
        if spoke.is_set():                           # reply landed while synthesising
            return
        # "filler": True marks this as a thinking-indicator, not the reply, so the client
        # keeps the mic closed (no barge-in) and doesn't reopen listening afterwards.
        await ws.send_json({"type": "audio", "filler": True,
                            "wav": _wav_b64(samples, sr), "text": line})

    def _emit_cars():
        # only refresh the cards when this turn is genuinely about cars (see GroqAgent.card_action)
        action = getattr(agent, "card_action", "set")
        if action == "set":
            loop.call_soon_threadsafe(q.put_nowait, ("cars", _car_payload(agent.display_records)))
        elif action == "clear":
            loop.call_soon_threadsafe(q.put_nowait, ("cars", []))
        # "keep" -> send nothing, leave whatever is on screen untouched

    def worker():
        full: list[str] = []
        try:
            gen = agent.respond_stream(user_text)
            started = False
            for delta in gen:
                if not started:
                    _emit_cars()
                    loop.call_soon_threadsafe(q.put_nowait, ("reply_start", None))
                    started = True
                full.append(delta)
                loop.call_soon_threadsafe(q.put_nowait, ("delta", delta))
            if not started:
                _emit_cars()
                loop.call_soon_threadsafe(q.put_nowait, ("reply_start", None))
            loop.call_soon_threadsafe(q.put_nowait, ("reply_end", None))
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, ("done", "".join(full)))

    threading.Thread(target=worker, daemon=True).start()
    filler_task = asyncio.create_task(filler()) if voice else None

    pending = ""                                     # text generated but not yet spoken

    async def _speak(sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return
        spoke.set()                                  # real audio going out -> suppress the filler
        samples, sr = await asyncio.to_thread(_TTS.synth, sentence)
        await ws.send_json({"type": "audio", "wav": _wav_b64(samples, sr), "text": sentence})

    while True:
        kind, data = await q.get()
        if kind == "delta":
            await ws.send_json({"type": "reply_delta", "text": data})
            pending += data
            # speak each complete sentence as soon as it's ready, so audio starts before
            # the whole reply has streamed in
            sents, pending = split_sentences(pending)
            for s in sents:
                await _speak(s)
        elif kind == "done":
            if pending.strip():                      # flush the last partial sentence
                await _speak(pending)
                pending = ""
            spoke.set()
            if filler_task:
                try:
                    await filler_task                # let an in-flight filler settle cleanly
                except Exception:                    # noqa: BLE001
                    pass
            await ws.send_json({"type": "speech_end"})
            break
        elif kind == "cars":
            await ws.send_json({"type": "cars", "cars": data})
        elif kind == "reply_start":
            await ws.send_json({"type": "reply_start"})
        elif kind == "reply_end":
            await ws.send_json({"type": "reply_end"})
        elif kind == "error":
            await ws.send_json({"type": "error", "text": data})


async def _greet(ws: WebSocket):
    await ws.send_json({"type": "reply_start"})
    await ws.send_json({"type": "reply_delta", "text": GREETING})
    await ws.send_json({"type": "reply_end"})
    await _say_text(ws, GREETING)
    await ws.send_json({"type": "speech_end"})


async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not _READY.is_set():
        await ws.send_json({"type": "loading"})
        while not _READY.is_set():
            await asyncio.sleep(0.15)
    await ws.send_json({"type": "ready", "build": BUILD})

    agent = GroqAgent()
    try:
        while True:
            msg = await ws.receive_json()
            try:
                mtype = msg.get("type")

                if mtype == "reset":
                    agent.reset()                # new session -> clear conversation memory
                    continue
                if mtype == "start_call":
                    agent.reset()                # each call starts fresh, no bleed from the last
                    await _greet(ws)
                    continue
                if mtype == "audio":
                    samples = _decode_pcm(msg.get("pcm", ""), int(msg.get("rate", config.SAMPLE_RATE)))
                    text = (await asyncio.to_thread(_STT.transcribe, samples)).strip()
                    await ws.send_json({"type": "transcript", "text": text})
                    if not text:
                        # speech_end after the re-prompt so the call resumes listening
                        reprompt = "Sorry, I didn't quite catch that — could you say it again?"
                        await ws.send_json({"type": "reply_start"})
                        await ws.send_json({"type": "reply_delta", "text": reprompt})
                        await ws.send_json({"type": "reply_end"})
                        await _say_text(ws, reprompt)
                        await ws.send_json({"type": "speech_end"})
                        continue
                elif mtype == "text":
                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue
                else:
                    continue

                # deterministic farewell on goodbye, no LLM round-trip
                if GOODBYE.search(text):
                    farewell = "It was a pleasure helping you today. Take care, and goodbye!"
                    await ws.send_json({"type": "reply_start"})
                    await ws.send_json({"type": "reply_delta", "text": farewell})
                    await ws.send_json({"type": "reply_end"})
                    await _say_text(ws, farewell)
                    await ws.send_json({"type": "speech_end"})
                    await ws.send_json({"type": "end_call"})
                    continue

                await _process(ws, agent, text, voice=msg.get("mode") != "chat")
            except WebSocketDisconnect:
                raise
            except Exception as e:  # noqa: BLE001 — never let one bad turn freeze the call
                print(f"[turn error] {e}", flush=True)
                try:
                    await ws.send_json({"type": "error", "text": "Sorry, that glitched — let's try again."})
                    await ws.send_json({"type": "speech_end"})
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass


async def index(_request):
    return FileResponse(STATIC / "index.html")


_DOW = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_DASH_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,system-ui,sans-serif;background:#000;color:#f2f2f6;min-height:100vh}
.bg{position:fixed;inset:0;z-index:-1;background:radial-gradient(120% 80% at 50% 120%,#0c0610,#000 60%)}
.blob{position:fixed;border-radius:50%;filter:blur(90px);opacity:.45}
.b1{width:480px;height:480px;top:-160px;right:-120px;background:radial-gradient(circle,rgba(150,60,230,.5),transparent 70%)}
.b2{width:420px;height:420px;bottom:-160px;left:-120px;background:radial-gradient(circle,rgba(255,122,69,.35),transparent 70%)}
.top{display:flex;align-items:center;justify-content:space-between;padding:22px 34px}
.brand{display:flex;align-items:center;gap:13px}
.logo{width:34px;height:34px;border-radius:11px;background:linear-gradient(150deg,#9b3be6,#d6248f 55%,#ff7a45);box-shadow:0 0 22px rgba(150,60,230,.5)}
.brand b{display:block;font-size:13px;letter-spacing:.18em}.brand span{font-size:11px;color:#9a9aa4}
.back{color:#cdb6ec;text-decoration:none;font-size:13px}.back:hover{color:#fff}
main{max-width:1000px;margin:0 auto;padding:10px 34px 50px;display:flex;flex-direction:column;gap:22px}
.card{background:rgba(255,255,255,.05);backdrop-filter:saturate(160%) blur(24px);border:1px solid rgba(255,255,255,.09);border-radius:22px;padding:26px;box-shadow:0 30px 80px rgba(0,0,0,.5)}
.card h2{font-size:20px;font-weight:800;letter-spacing:-.01em}
.card h2 em{color:#9a9aa4;font-style:normal;font-weight:400;font-size:14px}
.hint{color:#9a9aa4;font-size:13px;margin:6px 0 18px}
.days{display:flex;flex-wrap:wrap;gap:9px;margin-bottom:18px}
.day{cursor:pointer;user-select:none;font-size:13px;padding:9px 15px;border-radius:100px;border:1px solid rgba(255,255,255,.12);color:#9a9aa4;transition:.2s}
.day.on{color:#fff;background:linear-gradient(180deg,rgba(150,60,230,.5),rgba(150,60,230,.15));border-color:rgba(177,75,232,.55);box-shadow:0 0 20px rgba(150,60,230,.35)}
.day input{display:none}
.times{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end}
.times label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#9a9aa4}
.times input{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;color:#fff;padding:10px 12px;font-family:inherit;font-size:14px;color-scheme:dark}
.times button{margin-left:auto;background:linear-gradient(180deg,rgba(150,60,230,.85),rgba(150,60,230,.55));color:#fff;border:none;border-radius:100px;padding:12px 24px;font-weight:600;font-size:14px;cursor:pointer;box-shadow:0 0 26px rgba(150,60,230,.45)}
.times button:hover{filter:brightness(1.08)}
.preview{margin-top:20px;padding-top:18px;border-top:1px solid rgba(255,255,255,.08)}
.preview .lbl{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#5d5d66}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
.chip{font-size:12px;padding:5px 11px;border-radius:8px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#cdc9d4}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.tab{cursor:pointer;font-family:inherit;font-size:13px;display:flex;align-items:center;gap:8px;padding:9px 16px;border-radius:100px;border:1px solid rgba(255,255,255,.12);color:#9a9aa4;background:transparent;transition:.2s}
.tab em{font-style:normal;font-size:11px;min-width:18px;text-align:center;padding:1px 7px;border-radius:100px;background:rgba(255,255,255,.1);color:#cdc9d4}
.tab:hover{color:#fff;border-color:rgba(177,75,232,.4)}
.tab.on{color:#fff;background:linear-gradient(180deg,rgba(150,60,230,.5),rgba(150,60,230,.15));border-color:rgba(177,75,232,.55);box-shadow:0 0 20px rgba(150,60,230,.35)}
.tab.on em{background:rgba(177,75,232,.5);color:#fff}
.panel{display:none}.panel.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:9px}
.slot{position:relative;border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:10px 12px;display:flex;flex-direction:column;gap:3px}
.slot b{font-size:12.5px}.slot span{font-size:11.5px}
.slot.free{color:#5d5d66;cursor:pointer;transition:.15s}
.slot.free span{color:#3f7d57}
.slot.free:hover{border-color:rgba(177,75,232,.5);color:#cdb6ec;background:rgba(150,60,230,.08)}
.slot.free:hover span{color:#cdb6ec}
.slot.taken{background:linear-gradient(150deg,rgba(150,60,230,.2),transparent);border-color:rgba(177,75,232,.45)}
.slot.taken span{color:#cdb6ec}.empty{color:#9a9aa4}
.rm{position:absolute;top:6px;right:6px;margin:0}
.rmbtn{display:flex;align-items:center;justify-content:center;width:20px;height:20px;padding:0;font-size:13px;line-height:1;cursor:pointer;color:#ff9b9b;background:rgba(255,80,80,.14);border:1px solid rgba(255,90,90,.4);border-radius:7px;transition:.15s}
.rmbtn:hover{background:rgba(255,80,80,.32);color:#fff}
.addform{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end}
.addform label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#9a9aa4}
.addform input,.addform select{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;color:#fff;padding:10px 12px;font-family:inherit;font-size:14px;color-scheme:dark;min-width:160px}
.addform button{margin-left:auto;background:linear-gradient(180deg,rgba(150,60,230,.85),rgba(150,60,230,.55));color:#fff;border:none;border-radius:100px;padding:12px 24px;font-weight:600;font-size:14px;cursor:pointer;box-shadow:0 0 26px rgba(150,60,230,.45)}
.addform button:hover{filter:brightness(1.08)}
.clearbtn{background:rgba(255,80,80,.14);border:1px solid rgba(255,90,90,.4);color:#ff9b9b;border-radius:100px;padding:8px 16px;font-family:inherit;font-size:13px;cursor:pointer;transition:.15s}
.clearbtn:hover:not(:disabled){background:rgba(255,80,80,.3);color:#fff}
.clearbtn:disabled{opacity:.4;cursor:default}
.build{position:fixed;bottom:8px;right:12px;font-size:10px;color:#5d5d66;font-family:monospace}
"""


async def dashboard(request):
    """Admin dashboard: manage the weekly schedule + view/add/remove appointments."""
    if request.method == "POST":
        form = await request.form()
        action = form.get("action", "schedule")
        if action == "add":
            get_store().book_manual(form.get("name"), form.get("day"), form.get("time"))
            return RedirectResponse("/dashboard", status_code=303)
        if action == "remove":
            get_store().cancel(form.get("day"), form.get("time"))
            return RedirectResponse("/dashboard", status_code=303)
        if action == "clear_all":
            get_store().clear_appointments()
            return RedirectResponse("/dashboard", status_code=303)
        days = [d for d in form.getlist("days") if d in _DOW]
        try:
            mins = max(5, min(120, int(form.get("slot_minutes", 30))))
        except ValueError:
            mins = 30
        save_schedule({
            "days": days or ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "start": form.get("start") or "10:00",
            "end": form.get("end") or "17:00",
            "slot_minutes": mins,
        })
        return RedirectResponse("/dashboard", status_code=303)

    sch = load_schedule()
    slots = slot_times()
    appts = get_store().all_appointments()

    day_boxes = "".join(
        f'<label class="day {"on" if d in sch["days"] else ""}">'
        f'<input type="checkbox" name="days" value="{d}" {"checked" if d in sch["days"] else ""}>'
        f'{d[:3].title()}</label>' for d in _DOW)
    preview = "".join(f'<span class="chip">{t}</span>' for t in slots) or "<i>no slots — check hours</i>"

    by_day: dict[str, list[dict]] = {}
    for a in appts:
        by_day.setdefault(a["day"], []).append(a)

    # show the whole upcoming working week (even empty days) plus any booked day outside
    # that range, so the admin sees every slot they might cancel
    today = datetime.date.today()
    wdays = set(working_days())
    upcoming: list[str] = []
    for i in range(21):
        d = today + datetime.timedelta(days=i)
        if d.strftime("%A").lower() in wdays:
            upcoming.append(d.isoformat())
        if len(upcoming) >= 7:
            break
    all_days = sorted(set(upcoming) | set(by_day))

    def _label(iso: str) -> str:
        try:
            return datetime.date.fromisoformat(iso).strftime("%a, %b %d")
        except ValueError:
            return iso

    tabs, panels = [], []
    for idx, day in enumerate(all_days):
        booked = {a["time"]: a for a in by_day.get(day, [])}
        cells = []
        for t in slots:
            a = booked.get(t)
            if a:
                sub = f'{a["name"]}{(" · " + a["model"]) if a.get("model") else ""}'
                rm = (f'<form method="post" class="rm"><input type="hidden" name="action" value="remove">'
                      f'<input type="hidden" name="day" value="{day}">'
                      f'<input type="hidden" name="time" value="{t}">'
                      f'<button class="rmbtn" title="Remove appointment">&times;</button></form>')
                cells.append(f'<div class="slot taken"><b>{t}</b><span>{sub}</span>{rm}</div>')
            else:
                cells.append(f'<div class="slot free" data-day="{day}" data-time="{t}" '
                             f'title="Click to add an appointment"><b>{t}</b><span>+ add</span></div>')
        on = " on" if idx == 0 else ""
        tabs.append(f'<button class="tab{on}" data-tab="{idx}">{_label(day)}<em>{len(booked)}</em></button>')
        panels.append(f'<div class="panel{on}" data-panel="{idx}"><div class="grid">{"".join(cells)}</div></div>')
    appt_html = (f'<div class="tabs">{"".join(tabs)}</div>{"".join(panels)}'
                 if all_days else '<p class="empty">No working days configured — set them above.</p>')

    day_opts = "".join(f'<option value="{d}">{_label(d)}</option>' for d in (upcoming or all_days))
    time_opts = "".join(f'<option value="{t}">{t}</option>' for t in slots)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dashboard · Genesis Concierge</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>{_DASH_CSS}</style></head><body>
<div class="bg"><div class="blob b1"></div><div class="blob b2"></div></div>
<header class="top"><div class="brand"><span class="logo"></span><div><b>GENESIS CONCIERGE</b><span>Admin Dashboard</span></div></div>
<a class="back" href="/">← back to Aria</a></header>
<main>
  <section class="card">
    <h2>Weekly availability</h2><p class="hint">Pick working days and hours. {sch["slot_minutes"]}-minute viewing slots are generated automatically.</p>
    <form method="post" class="sched">
      <div class="days">{day_boxes}</div>
      <div class="times">
        <label>Open<input type="time" name="start" value="{sch["start"]}"></label>
        <label>Close<input type="time" name="end" value="{sch["end"]}"></label>
        <label>Slot (min)<input type="number" name="slot_minutes" min="10" max="120" step="5" value="{sch["slot_minutes"]}"></label>
        <button type="submit">Save schedule</button>
      </div>
    </form>
    <div class="preview"><span class="lbl">Generated slots</span><div class="chips">{preview}</div></div>
  </section>
  <section class="card">
    <h2>Add appointment</h2><p class="hint">Book a slot manually, or click any free slot below to fill this in.</p>
    <form method="post" class="addform">
      <input type="hidden" name="action" value="add">
      <label>Name<input name="name" required placeholder="Customer name"></label>
      <label>Day<select name="day" id="addDay">{day_opts}</select></label>
      <label>Time<select name="time" id="addTime">{time_opts}</select></label>
      <button type="submit">Add appointment</button>
    </form>
  </section>
  <section class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
      <h2>Appointments <em>· {len(appts)} total</em></h2>
      <form method="post" onsubmit="return confirm('Delete ALL {len(appts)} appointments? This frees every slot.')" style="margin:0">
        <input type="hidden" name="action" value="clear_all">
        <button class="clearbtn" {'disabled' if not appts else ''}>Clear all</button>
      </form>
    </div>
    <p class="hint">The full working week. Click a free slot to add, or &times; to remove. Auto-refreshes.</p>
    {appt_html}
  </section>
</main>
<div class="build">build {BUILD}</div>
<script>
// day tabs: click a day -> show only that day's slots
function showTab(i){{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.tab===String(i)));
  document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('on',x.dataset.panel===String(i)));
}}
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{{
  showTab(t.dataset.tab); sessionStorage.setItem('apptTab',t.dataset.tab);
}}));
// restore the tab you were on (so the 15s auto-refresh doesn't snap back to Monday)
const _savedTab=sessionStorage.getItem('apptTab');
if(_savedTab!==null && document.querySelector('.tab[data-tab="'+_savedTab+'"]')) showTab(_savedTab);
// make the day chips toggle their 'on' look as you click (purely visual; checkbox drives the save)
document.querySelectorAll('.day input').forEach(cb=>cb.addEventListener('change',e=>e.target.closest('.day').classList.toggle('on',e.target.checked)));
// click a free slot -> prefill the Add-appointment form and jump to it
document.querySelectorAll('.slot.free').forEach(s=>s.addEventListener('click',()=>{{
  document.getElementById('addDay').value=s.dataset.day;
  document.getElementById('addTime').value=s.dataset.time;
  const n=document.querySelector('.addform input[name=name]');
  document.querySelector('.addform').scrollIntoView({{behavior:'smooth',block:'center'}}); n.focus();
}}));
// gentle auto-refresh, paused while you're typing/selecting
setInterval(()=>{{const t=document.activeElement.tagName;if(t!=='INPUT'&&t!=='SELECT')location.reload()}},15000);
</script>
</body></html>"""
    return HTMLResponse(html)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm, daemon=True).start()   # warm models off the event loop
    yield


app = Starlette(
    routes=[
        Route("/", index),
        Route("/dashboard", dashboard, methods=["GET", "POST"]),
        Route("/appointments", lambda r: RedirectResponse("/dashboard")),
        Route("/appointment", lambda r: RedirectResponse("/dashboard")),
        WebSocketRoute("/ws", ws_endpoint),
        Mount("/static", app=StaticFiles(directory=STATIC), name="static"),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import os
    import uvicorn
    # HOST defaults to localhost for local dev; Docker sets HOST=0.0.0.0 so the port is reachable.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")), log_level="warning")
