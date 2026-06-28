"""Local text-to-speech via sherpa-onnx Piper (VITS, ONNX).

synth(text) returns (samples, sample_rate); speak(text) plays through the
speakers. Synthesis is per-sentence so the agent can start talking while the
LLM is still streaming later sentences.
"""

from __future__ import annotations

import threading

import numpy as np
import sherpa_onnx
import sounddevice as sd

from . import config


class Speaker:
    def __init__(self):
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(config.TTS_MODEL),
                    tokens=str(config.TTS_TOKENS),
                    data_dir=str(config.TTS_DATA_DIR),
                ),
                num_threads=2,
            ),
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self.tts.sample_rate
        self._stop = threading.Event()

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        audio = self.tts.generate(text, sid=0, speed=config.TTS_SPEED)
        return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate

    def speak(self, text: str) -> None:
        """Synthesize and play one chunk of text (blocking until done)."""
        text = text.strip()
        if not text:
            return
        samples, sr = self.synth(text)
        self._stop.clear()
        sd.play(samples, sr)
        sd.wait()

    def stop(self) -> None:
        """Interrupt current playback (for barge-in)."""
        self._stop.set()
        sd.stop()
