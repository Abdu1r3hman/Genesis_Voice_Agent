# Genesis CPO — Voice AI Concierge

A real-time **voice** (and text) sales concierge, **Aria**, for the
[Genesis Certified Pre-Owned](https://genesis-cpo.netlify.app/) inventory. You speak;
it retrieves the right cars from the scraped stock, answers — grounded, never
hallucinated — and books a viewing. Assembled by hand: scraper + retrieval + STT/LLM/TTS
+ CRM. **No end-to-end voice-agent platform.**

```
🎤 ─▶ VAD ─▶ Moonshine STT ─▶ understand (regex / LLM+Pydantic) ─▶ RAG ─▶ Groq LLM ─▶ Piper TTS ─▶ 🔊
                                                                          │
                                              everything local except the LLM (one network hop)

## Run it

### Option A — Docker (no Python setup)
```bash
git clone https://github.com/Abdu1r3hman/Genesis_Voice_Agent.git
cd Genesis_Voice_Agent
docker build -t genesis-voice .
docker run --rm -p 8000:8000 -e GROQ_API_KEY=gsk_your_key genesis-voice
```
Open **http://localhost:8000** (the mic works on `localhost` without HTTPS) · dashboard at **/dashboard**.
Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

### Option B — Text mode (lightest, ~2 min, no speech models)
Text mode loads only the RAG brain — **no 260 MB model download.**
```bash
pip install -r requirements.txt
# set GROQ_API_KEY in your environment (or copy .env.example -> .env), then:
python -m voice.run --text
```

### Option C — Full local (voice + web UI + dashboard)
Requires **Python 3.14** (ONNX-only stack, no PyTorch).
```bash
python -m venv venv && venv\Scripts\activate     # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m voice.setup_models                      # one-time: download speech models (~260 MB)
copy .env.example .env                            # then paste your Groq key
python web/server.py                              # http://localhost:8000
python -m voice.run                               # or the terminal voice agent
```

---

## What it does

- **Grounded RAG** — answers only from the 56 real scraped cars; never invents stock.
- **Understands natural speech** — "the cheaper one", "recommend me anything", "GV80 with low miles" — via a regex fast-path plus an LLM→Pydantic structured layer for the vague cases.
- **Books viewings** — a deterministic flow against real free/occupied slots (no double-booking), persisted to SQLite, with an admin dashboard.
- **Low latency** — local STT/TTS/VAD/embeddings; Groq is the only network hop; first spoken word ~1 s after you stop talking.

See **[WRITEUP.md](WRITEUP.md)** for the approach, tech-stack rationale, trade-offs, and latency results.

---

## Architecture

| Stage | Tech | Where |
|---|---|---|
| Scrape | Playwright + BeautifulSoup → JSON | offline |
| Retrieve | rule filters + **fastembed / BGE-small** (ONNX) over an in-process NumPy index | on-device (~12 ms) |
| Understand | regex parser + **Groq → Pydantic** structured query | hybrid |
| Speech in | **Silero VAD** + **Moonshine** STT (sherpa-onnx) | on-device |
| Brain | **Groq** `llama-3.3-70b-versatile` (streaming) | API |
| Speech out | **Piper** `en_US-amy-medium` (sherpa-onnx) | on-device |
| CRM | SQLite + jellyfish (phonetic) + rapidfuzz | local |
| Web | Starlette + uvicorn + WebSockets | pure-Python (3.14-safe) |

```
scraper/  rag/  voice/  crm/  web/  prompts/  data/
```
