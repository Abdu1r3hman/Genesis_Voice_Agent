"""
One-time download of the local ONNX speech models (VAD + STT + TTS).

All models come from the sherpa-onnx model repo and run via onnxruntime
(no PyTorch). Total ~260 MB. Re-running skips anything already present.

Run:  python -m voice.setup_models
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# single-file downloads
FILES = {
    f"{BASE}/asr-models/silero_vad.onnx": "silero_vad.onnx",
}

# tar.bz2 archives -> extracted into MODELS_DIR/<dirname>/
ARCHIVES = {
    # STT (Moonshine) - fast + accurate on-device transcription
    f"{BASE}/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2":
        "sherpa-onnx-moonshine-base-en-int8",
    # STT (Whisper) - alternative transcriber
    f"{BASE}/asr-models/sherpa-onnx-whisper-base.en.tar.bz2": "sherpa-onnx-whisper-base.en",
    f"{BASE}/tts-models/vits-piper-en_US-amy-medium.tar.bz2": "vits-piper-en_US-amy-medium",
}


def _progress(name: str):
    def hook(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  {name}: {pct:3d}%", end="", flush=True)
    return hook


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, _progress(dest.name))
    print()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for url, name in FILES.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"[skip] {name} (exists)")
            continue
        print(f"[get ] {name}")
        _download(url, dest)

    for url, dirname in ARCHIVES.items():
        out_dir = MODELS_DIR / dirname
        if out_dir.exists():
            print(f"[skip] {dirname}/ (exists)")
            continue
        archive = MODELS_DIR / f"{dirname}.tar.bz2"
        print(f"[get ] {dirname}.tar.bz2")
        _download(url, archive)
        print(f"  extracting ...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(MODELS_DIR)
        archive.unlink()  # remove the tarball after extraction

    print("\nAll speech models ready in", MODELS_DIR)


if __name__ == "__main__":
    main()
