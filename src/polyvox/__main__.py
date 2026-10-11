"""Entry point: ``polyvox`` / ``python -m polyvox``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Must be set before tokenizers is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# A Hugging Face mirror also serves models loaded by id (downloaded to the
# Hugging Face cache on first use); huggingface_hub reads it when imported.
if mirror := os.environ.get("POLYVOX_HF_ENDPOINT", "").strip():
    os.environ.setdefault("HF_ENDPOINT", mirror.rstrip("/"))


SHUTDOWN_GRACE_S = 3


class QuietReads(logging.Filter):
    """Leaves successful GET requests out of uvicorn's access log: the UI polls
    progress, jobs and status every second. Changes (POST) and errors stay."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 5:  # client, method, path, http version, status
            method, status = args[1], args[4]
            return not (method == "GET" and isinstance(status, int) and status < 400)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Polyvox Studio server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--log-requests", action="store_true", help="log every request, also successful GETs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.log_requests:
        logging.getLogger("uvicorn.access").addFilter(QuietReads())
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
