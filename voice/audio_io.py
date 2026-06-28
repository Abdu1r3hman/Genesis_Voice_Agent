"""Full-duplex audio I/O: mic capture + interruptible playback.

A single persistent input stream feeds Silero VAD. capture_utterance() blocks
until the user finishes speaking; play_interruptible() plays audio but stops
the instant the user barges in. Barge-in works best with headphones — see
config.BARGE_RMS_FLOOR for tuning around speaker echo leakage.
"""

from __future__ import annotations

import time

import numpy as np
import sherpa_onnx
import sounddevice as sd

from . import config


class AudioIO:
    def __init__(self):
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(config.VAD_MODEL),
                threshold=config.VAD_THRESHOLD,
                min_silence_duration=config.VAD_MIN_SILENCE_SEC,
                min_speech_duration=config.VAD_MIN_SPEECH_SEC,
                window_size=512,
            ),
            sample_rate=config.SAMPLE_RATE,
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        self.window = 512
        self.block = int(config.BLOCK_MS / 1000 * config.SAMPLE_RATE)

        self._stream = sd.InputStream(
            channels=config.CHANNELS, samplerate=config.SAMPLE_RATE,
            dtype="float32", blocksize=self.block,
        )
        self._stream.start()

    # -- low level -------------------------------------------------------- #
    def _read(self) -> np.ndarray:
        data, _ = self._stream.read(self.block)
        return data.reshape(-1)

    def _drain(self) -> None:
        """Drop buffered mic audio + reset VAD (clears playback echo tail)."""
        avail = self._stream.read_available
        if avail > 0:
            self._stream.read(avail)
        self.vad.reset()

    # -- capture ---------------------------------------------------------- #
    def capture_utterance(self, initial_timeout: float | None = None) -> np.ndarray | None:
        """
        Block until one complete utterance is captured and return its samples.
        If initial_timeout is set and no speech begins within it, return None.
        """
        self._drain()
        buffer = np.empty(0, dtype=np.float32)
        started = False
        t0 = time.time()

        while True:
            buffer = np.concatenate([buffer, self._read()])
            while len(buffer) >= self.window:
                self.vad.accept_waveform(buffer[: self.window])
                buffer = buffer[self.window :]

            if self.vad.is_speech_detected():
                started = True
            if not self.vad.empty():
                samples = np.asarray(self.vad.front.samples, dtype=np.float32)
                self.vad.pop()
                return samples
            if not started and initial_timeout and (time.time() - t0) > initial_timeout:
                return None

    # -- playback --------------------------------------------------------- #
    def play_interruptible(self, samples: np.ndarray, sample_rate: int) -> bool:
        """
        Play audio. Returns True if it finished, False if the user barged in.
        Detects the user via the VAD (speech) plus a loudness floor, so normal
        speaking reliably interrupts the agent.
        """
        self._drain()
        sd.play(samples, sample_rate)

        if not config.ENABLE_BARGE_IN:
            sd.wait()
            return True

        duration = len(samples) / sample_rate
        t0 = time.time()
        buffer = np.empty(0, dtype=np.float32)
        speech_blocks = 0

        while time.time() - t0 < duration:
            block = self._read()
            if time.time() - t0 < config.BARGE_GRACE_SEC:
                continue  # ignore onset so playback doesn't self-trigger barge-in

            rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
            buffer = np.concatenate([buffer, block])
            while len(buffer) >= self.window:
                self.vad.accept_waveform(buffer[: self.window])
                buffer = buffer[self.window :]

            if self.vad.is_speech_detected() and rms > config.BARGE_RMS_FLOOR:
                speech_blocks += 1
                if speech_blocks >= config.BARGE_MIN_BLOCKS:
                    sd.stop()
                    self.vad.reset()
                    return False
            else:
                speech_blocks = 0

        sd.wait()
        return True

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
