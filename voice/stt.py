"""Local speech-to-text via sherpa-onnx (no PyTorch).

Moonshine is the default (one-shot, faster than Whisper at comparable accuracy);
Whisper base.en is kept as an alternative. Pick via config.STT_MODEL.
"""

from __future__ import annotations

import numpy as np
import sherpa_onnx

from . import config


class MoonshineTranscriber:
    """Offline Moonshine (one-shot, fast + accurate)."""

    def __init__(self):
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(config.MOONSHINE_PREPROCESSOR),
            encoder=str(config.MOONSHINE_ENCODER),
            uncached_decoder=str(config.MOONSHINE_UNCACHED_DECODER),
            cached_decoder=str(config.MOONSHINE_CACHED_DECODER),
            tokens=str(config.MOONSHINE_TOKENS),
            num_threads=2,
            decoding_method="greedy_search",
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> str:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


class Transcriber:
    """Offline Whisper base.en (alternative transcriber)."""

    def __init__(self):
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(config.WHISPER_ENCODER),
            decoder=str(config.WHISPER_DECODER),
            tokens=str(config.WHISPER_TOKENS),
            num_threads=2,
            decoding_method="greedy_search",
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> str:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


def make_transcriber():
    """The transcriber selected by config.STT_MODEL ('moonshine' | 'whisper')."""
    return Transcriber() if config.STT_MODEL == "whisper" else MoonshineTranscriber()
