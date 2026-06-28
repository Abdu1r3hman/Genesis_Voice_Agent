"""Central configuration for the voice agent (models, audio, paths)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
PROMPTS_DIR = ROOT / "prompts"

load_dotenv(ROOT / ".env")

# --- audio -------------------------------------------------------------- #
SAMPLE_RATE = 16_000          # Whisper + Silero VAD both require 16 kHz mono
CHANNELS = 1
BLOCK_MS = 100                # mic read block size in milliseconds

# --- VAD (Silero) ------------------------------------------------------- #
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_SEC = 0.9     # trailing silence that ends a turn
VAD_MIN_SPEECH_SEC = 0.25     # ignore blips shorter than this

# --- STT: Moonshine base (fast + accurate, on-device) ------------------- #
STT_MODEL = "moonshine"       # "moonshine" or "whisper"
MOONSHINE_DIR = MODELS_DIR / "sherpa-onnx-moonshine-base-en-int8"
MOONSHINE_PREPROCESSOR = MOONSHINE_DIR / "preprocess.onnx"
MOONSHINE_ENCODER = MOONSHINE_DIR / "encode.int8.onnx"
MOONSHINE_UNCACHED_DECODER = MOONSHINE_DIR / "uncached_decode.int8.onnx"
MOONSHINE_CACHED_DECODER = MOONSHINE_DIR / "cached_decode.int8.onnx"
MOONSHINE_TOKENS = MOONSHINE_DIR / "tokens.txt"

# --- STT (Whisper base.en) - alternative transcriber -------------------- #
WHISPER_DIR = MODELS_DIR / "sherpa-onnx-whisper-base.en"
WHISPER_ENCODER = WHISPER_DIR / "base.en-encoder.int8.onnx"
WHISPER_DECODER = WHISPER_DIR / "base.en-decoder.int8.onnx"
WHISPER_TOKENS = WHISPER_DIR / "base.en-tokens.txt"

# --- TTS (Piper VITS via sherpa-onnx) ----------------------------------- #
TTS_DIR = MODELS_DIR / "vits-piper-en_US-amy-medium"
TTS_MODEL = TTS_DIR / "en_US-amy-medium.onnx"
TTS_TOKENS = TTS_DIR / "tokens.txt"
TTS_DATA_DIR = TTS_DIR / "espeak-ng-data"
TTS_SPEED = 1.30              # >1 = faster speech pace

# --- Barge-in (interrupt the agent by speaking) ------------------------- #
# Tuned for HEADPHONES. On open speakers without echo cancellation the agent's
# own voice can leak into the mic; if it interrupts itself, raise
# BARGE_RMS_FLOOR (e.g. 0.05-0.10) or set ENABLE_BARGE_IN = False.
ENABLE_BARGE_IN = True
BARGE_RMS_FLOOR = 0.015       # ignore mic quieter than this (room noise floor)
BARGE_MIN_BLOCKS = 2          # consecutive ~100ms speech blocks before stopping
BARGE_GRACE_SEC = 0.25        # skip playback onset to avoid self-triggering
INTERRUPT_SILENCE_SEC = 2.5   # silence after a barge-in before we re-prompt

# --- LLM (Groq) --------------------------------------------------------- #
# Reply model favours quality (70B); understand model favours speed (8B, tiny JSON).
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_UNDERSTAND_MODEL = os.getenv("GROQ_UNDERSTAND_MODEL", "llama-3.1-8b-instant")
# Each Groq model has its own rate limit, so switching model dodges a per-model 429.
# Order = preference.
GROQ_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GROQ_FALLBACK_MODELS", "llama-3.1-8b-instant,gemma2-9b-it,llama3-8b-8192"
    ).split(",") if m.strip()
]
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_TEMPERATURE = 0.55
LLM_MAX_TOKENS = 100          # hard cap so replies stay short and conversational

# --- Retrieval ---------------------------------------------------------- #
TOP_K = 8                     # wide window so a soft ranking miss can't drop the right car
# LLM-based understanding (Groq -> Pydantic) is robust to phrasing/context drift but
# costs one extra fast call/turn. Set False to fall back to regex.
USE_LLM_UNDERSTANDING = os.getenv("USE_LLM_UNDERSTANDING", "1") not in ("0", "false", "False")
HISTORY_TURNS = 4             # conversation turns kept in context (fewer = less quota)
GROUNDING_FEATURES = 8        # features listed per car in the grounding block

# --- CRM: customers + appointments (local SQLite) ----------------------- #
CRM_DB = ROOT / "data" / "crm.db"
NAME_MATCH_THRESHOLD = 0.85   # Jaro-Winkler: match the same customer across spelling drift
# Fallback slots if data/schedule.json is unreadable; normally generated from the
# dashboard-managed availability — see crm/store.py.
APPOINTMENT_SLOTS = ["10:00 AM", "11:00 AM", "12:00 PM", "2:00 PM",
                     "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM"]
