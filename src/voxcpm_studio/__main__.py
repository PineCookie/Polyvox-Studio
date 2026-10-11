"""Entry point: ``voxcpm-studio`` / ``python -m voxcpm_studio``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Must be set before tokenizers is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# A Hugging Face mirror also serves models loaded by id (downloaded to the
# Hugging Face cache on first use); huggingface_hub reads it when imported.
if mirror := os.environ.get("VOXCPM_STUDIO_HF_ENDPOINT", "").strip():
    os.environ.setdefault("HF_ENDPOINT", mirror.rstrip("/"))


SHUTDOWN_GRACE_S = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM Studio server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8808)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.platform == "win32" and not sys.flags.utf8_mode:
        # Torch Inductor writes generated sources with the locale encoding otherwise.
        logging.warning("Python UTF-8 mode is off; set PYTHONUTF8=1 (run.bat does) for torch.compile.")

    import uvicorn

    from .server import create_app

    # Ctrl+C waits at most this long for open responses: a paused <audio> player
    # (e.g. a run's samples) holds its download open, and uvicorn would wait for it
    # forever. A running training job still saves its checkpoint afterwards.
    uvicorn.run(create_app(), host=args.host, port=args.port, timeout_graceful_shutdown=SHUTDOWN_GRACE_S)


if __name__ == "__main__":
    main()
