# Writeup — Genesis CPO Voice Agent

## Approach
A real-time voice concierge built from individual parts (no end-to-end voice platform): each turn
flows **VAD → STT → understand → RAG → LLM → TTS**, with **everything except the LLM running
locally** (ONNX via `onnxruntime`, no PyTorch). Two priorities drove every choice — **low latency**
and **never hallucinating stock**.

The design was **hardened iteratively**. It began with a deterministic, regex-based retrieval core
— fast and exact for clear queries — which I then layered up for real conversation: an
**LLM → Pydantic structured-understanding step** for vague or context-dependent speech ("the cheaper
one", "recommend me anything"), kept *behind* the instant regex path so common questions never pay
for an extra call; a **fully deterministic booking flow** so transactional steps stay reliable
rather than LLM-phrased; and a tuned speech pipeline (**sentence-streamed TTS, a hybrid fast-path,
per-session memory**) so it feels responsive and holds context across turns. The result is layered:
the **LLM understands and phrases, but deterministic code owns every decision that must be exact** —
retrieval, availability, and booking.

## Tech stack — and why
- **STT: Moonshine (local, ONNX) over Whisper.** Whisper pads *every* clip to 30 s, so a short
  spoken turn still costs ~0.6–0.8 s; Moonshine scales with the actual audio length, transcribing
  a typical 2–3 s turn in **~0.2 s — roughly 3–4× faster on short utterances**, and it's free and
  on-device (no API latency or cost). Whisper is kept as a selectable fallback.
- **LLM: Groq.** Groq's LPU gives the fastest streaming inference available (~sub-0.3 s to first
  token). I use **`llama-3.3-70b-versatile`** for the spoken reply (richer phrasing) and a fast
  **`8b-instant`** model for the understanding call, so structured parsing adds ~0.3 s instead of
  a full 70B round-trip.
- **Understanding: Groq → Pydantic.** Free speech is converted into a validated `CarQuery`
  schema. Pydantic **guarantees the structure** (no malformed/hallucinated fields), and the
  deterministic DB query picks the cars — so the model can't invent a car, price, or trim.
- **Embeddings: fastembed / BGE-small (local, ONNX)** — ~12 ms, no API. **TTS: Piper (local)**,
  streamed **sentence-by-sentence** so the first word is spoken ~1 s in, not after the whole reply.
- **Web: Starlette** (pure-Python) — avoids Rust/PyTorch wheels that don't build on Python 3.14.

## Trade-offs & fallbacks (where the engineering is)
- **Hybrid understanding:** clear questions ("GV80 under 250k") take an **instant regex path —
  no second LLM call**; only vague/contextual ones ("recommend me anything") pay for the LLM.
  Common queries stay at a single network hop.
- **Resilient retrieval for noisy speech:** because ASR can mis-hear a number or attribute,
  structured signals *score and rank* candidates rather than hard-filtering them — a slightly
  off value gently re-orders results instead of dropping a valid match, so the right car always
  stays in front of the model.
- **Layered fallbacks for reliability:** a regex understanding path if the structured LLM call is
  unavailable; an automatic **Groq model fallback** on a rate-limit (429); exact letter-by-letter
  name capture (LLM only when letters arrive merged); and a **fully deterministic booking flow**
  (zero LLM, fixed verified lines) so every transactional step stays reliable and predictable.

**Latency result:** first spoken word **~1 s** after you stop talking; one network hop;
**booking turns make zero LLM calls.**

## What I'd improve with more time
Real calendar sync (Cal.com / Google Calendar) behind the existing store interface; **streaming
STT** for partial transcripts; pinned dependency versions + a CI build; a small labelled eval set
for the understanding layer; multi-user sessions/auth; and Arabic speech models for the SA market.

## Assumptions
English audio; a single-user local demo; the catalogue is the 56 cars scraped from the site; the
runner supplies their own free Groq key; and bookings persist in local SQLite (cleared from the
dashboard).
