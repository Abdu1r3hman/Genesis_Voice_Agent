"""Conversation orchestrator: ties RAG + Groq LLM + local STT/TTS into a loop.

Replies are flushed to TTS a sentence at a time as the LLM streams, and each
sentence plays interruptibly so the user can barge in mid-response.
"""

from __future__ import annotations

import re

from . import config
from .llm import GroqAgent

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_CHARS = 12  # don't speak tiny fragments (e.g. "Mr.") alone

GOODBYE = re.compile(r"\b(bye|goodbye|that'?s all|nothing else|end the (call|conversation))\b", re.I)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remaining_partial)."""
    parts = _SENT_BOUNDARY.split(buffer)
    if len(parts) == 1:
        return [], buffer
    complete, remainder = parts[:-1], parts[-1]

    flushed, carry = [], ""
    for s in complete:
        s = (carry + " " + s).strip() if carry else s
        if len(s) < _MIN_SENTENCE_CHARS:
            carry = s
        else:
            flushed.append(s)
            carry = ""
    remainder = (carry + " " + remainder).strip() if carry else remainder
    return flushed, remainder


class VoiceAgent:
    """Speech-driven loop (mic -> STT -> RAG+LLM -> TTS) with barge-in."""

    def __init__(self):
        from .audio_io import AudioIO
        from .stt import make_transcriber
        from .tts import Speaker

        self.audio = AudioIO()
        self.stt = make_transcriber()
        self.tts = Speaker()
        self.agent = GroqAgent()
        self._warmup()

    def _warmup(self) -> None:
        import numpy as np

        self.tts.synth("Ready.")
        self.stt.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))

    def _listen(self, initial_timeout: float | None = None) -> str | None:
        """Capture one utterance and transcribe it."""
        samples = self.audio.capture_utterance(initial_timeout=initial_timeout)
        if samples is None:
            return None
        text = self.stt.transcribe(samples).strip()
        if text:
            print(f"You: {text}")
        return text

    # -- speaking --------------------------------------------------------- #
    def _say(self, text: str) -> bool:
        """Speak one chunk interruptibly. Returns False if the user barged in."""
        text = text.strip()
        if not text:
            return True
        samples, sr = self.tts.synth(text)
        return self.audio.play_interruptible(samples, sr)

    def _respond(self, user_text: str) -> bool:
        """Stream the reply, speaking sentence-by-sentence. Returns True if barged."""
        print("Aria: ", end="", flush=True)
        buffer = ""
        for delta in self.agent.respond_stream(user_text):
            print(delta, end="", flush=True)
            buffer += delta
            sentences, buffer = split_sentences(buffer)
            for s in sentences:
                if not self._say(s):
                    print("  [interrupted]")
                    return True
        if buffer.strip() and not self._say(buffer):
            print("  [interrupted]")
            return True
        print()
        return False

    # -- main loop -------------------------------------------------------- #
    def run(self) -> None:
        greeting = (
            "Hello, and welcome to Genesis Certified Pre-Owned. I'm Aria, your concierge. "
            "Are you looking for a particular model today, or shall I show you around the collection?"
        )
        print(f"\nAria: {greeting}\n")
        self._say(greeting)

        pending = None  # text captured via barge-in, processed on the next turn
        try:
            while True:
                if pending is None:
                    print("[listening...]")
                    user_text = self._listen()
                else:
                    user_text, pending = pending, None

                if not user_text or not user_text.strip():
                    continue
                user_text = user_text.strip()

                if GOODBYE.search(user_text):
                    self._say("It was my pleasure. Visit us anytime — goodbye!")
                    break

                try:
                    barged = self._respond(user_text)
                except Exception as e:  # one bad turn must not end the call
                    print(f"\n[error: {e}]")
                    self._say("Sorry, I lost that for a moment. Could you say it again?")
                    continue

                if barged:
                    pending = self._handle_interrupt()
        except KeyboardInterrupt:
            print("\n[ended]")
        finally:
            self.audio.close()

    def _handle_interrupt(self):
        """After a barge-in, capture what the user says; re-prompt if they go quiet."""
        text = self._listen(initial_timeout=config.INTERRUPT_SILENCE_SEC)
        if text:
            return text
        # barged in but stayed silent: check in and give one more chance
        self._say("Sorry, did you want to add something, or shall I continue?")
        return self._listen(initial_timeout=config.INTERRUPT_SILENCE_SEC + 1.5)


def run_text() -> None:
    """Text-only loop: same RAG+LLM brain with typed I/O."""
    agent = GroqAgent()
    print("Genesis CPO assistant (text mode). Type 'quit' to exit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text or user_text.lower() in {"quit", "exit"}:
            break
        print("Aria: ", end="", flush=True)
        for delta in agent.respond_stream(user_text):
            print(delta, end="", flush=True)
        print("\n")
