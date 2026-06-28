"""
Entry point for the Genesis CPO voice agent.

  python -m voice.run          # voice mode (mic in, speaker out, auto turn-taking)
  python -m voice.run --text   # text mode (typed I/O, same RAG+LLM brain)

Requires GROQ_API_KEY in the environment, and the local speech models
(run `python -m voice.setup_models` once).
"""

from __future__ import annotations

import argparse
import sys

from . import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis CPO voice agent")
    parser.add_argument("--text", action="store_true", help="text mode (no audio)")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        sys.exit("ERROR: set GROQ_API_KEY in your environment first.")

    if args.text:
        from .agent import run_text
        run_text()
        return

    # voice mode
    for path in (config.VAD_MODEL, config.MOONSHINE_ENCODER, config.TTS_MODEL):
        if not path.exists():
            sys.exit(f"ERROR: missing model {path}. Run: python -m voice.setup_models")

    print("Loading speech models...", flush=True)
    from .agent import VoiceAgent
    VoiceAgent().run()


if __name__ == "__main__":
    main()
