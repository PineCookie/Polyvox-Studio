"""How long each engine's one-off phases (loading the model, compiling) took the last time on
this computer, so synthesis progress can say what to expect: they depend on the GPU, the disk,
whether the files are still in memory and whether torch's compile cache is warm, so no fixed
number fits every machine. Kept in ``<models folder>/.phase-times.json``.

``watching(engine)`` follows a request's phases by polling ``engine.progress()``; it works for any
engine without changes to it. Only requests that finish count, and a load that waited for a
download (models.active_download) doesn't count as a load time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .. import models

logger = logging.getLogger(__name__)

ONE_OFF = ("loading", "compiling")  # phases the first request after a load waits through
POLL_S = 0.1
MIN_S = 0.5  # shorter ones aren't worth a hint


class PhaseTimes:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._times: dict[str, dict[str, float]] = self._read()
        self._current: dict[str, tuple[str, float]] = {}  # engine id -> (phase, since: monotonic)

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return {e: {p: float(s) for p, s in d.items()} for e, d in data.items() if isinstance(d, dict)}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def last(self, engine: str, phase: str) -> float | None:
        """Seconds the phase took last time (None: never measured here)."""
        with self._lock:
            return self._times.get(engine, {}).get(phase)

    def current(self, engine: str) -> tuple[str, float] | None:
        """The engine's phase now and the seconds it has lasted (None: not watched)."""
        with self._lock:
            entry = self._current.get(engine)
        return (entry[0], time.monotonic() - entry[1]) if entry else None

    def record(self, engine: str, phase: str, seconds: float) -> None:
        with self._lock:
            self._times.setdefault(engine, {})[phase] = round(seconds, 1)
            data = json.dumps(self._times, indent=1)
        try:
            tmp = self.path.with_suffix(".tmp")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            logger.warning("Couldn't save %s", self.path, exc_info=True)

    @contextmanager
    def watching(self, engine) -> Iterator[None]:
        """Follow one request of ``engine``; on success, remember its one-off phases' durations."""
        stop = threading.Event()
        spans: dict[str, float] = {}
        downloaded = threading.Event()

        def poll() -> None:
            phase, since = None, time.monotonic()
            while True:
                try:
                    now_phase = engine.progress().get("phase")
                except Exception:  # (never let the watcher break a request)
                    now_phase = phase
                if models.active_download() is not None:
                    downloaded.set()
                now = time.monotonic()
                if now_phase != phase:
                    if phase in ONE_OFF:
                        spans[phase] = spans.get(phase, 0.0) + now - since
                    phase, since = now_phase, now
                    with self._lock:
                        self._current[engine.id] = (phase, since)
                if stop.wait(POLL_S):
                    if phase in ONE_OFF:
                        spans[phase] = spans.get(phase, 0.0) + time.monotonic() - since
                    return

        thread = threading.Thread(target=poll, daemon=True, name=f"phases-{engine.id}")
        thread.start()
        finished = False
        try:
            yield
            finished = True
        finally:
            stop.set()
            thread.join()
            with self._lock:
                self._current.pop(engine.id, None)
            if finished:
                for phase, seconds in spans.items():
                    if seconds >= MIN_S and not (phase == "loading" and downloaded.is_set()):
                        self.record(engine.id, phase, seconds)
