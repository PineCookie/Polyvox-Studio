"""Where a long job spends its time: seconds per stage (decoding, separation,
ASR, ...), for the import's time breakdown.

``with collect() as times:`` gathers what ``with stage("asr"):`` blocks run in
the same thread report; outside ``collect`` a stage costs nothing. A stage's
time excludes the stages nested in it (e.g. loading a model during the first
ASR call counts as "loading models", not "asr").
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class Timings:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._nested: list[float] = []  # time of finished child stages, per open stage
        self._start = time.perf_counter()

    def summary(self) -> dict[str, float]:
        """Seconds per stage, longest first, plus "other" (time in no stage)
        when that is noticeable."""
        total = time.perf_counter() - self._start
        out = dict(sorted(self.seconds.items(), key=lambda kv: -kv[1]))
        if (other := total - sum(out.values())) >= 0.5:
            out["other"] = other
        return {k: round(v, 1) for k, v in out.items()}


_current: ContextVar[Timings | None] = ContextVar("timings", default=None)


@contextmanager
def collect() -> Iterator[Timings]:
    times = Timings()
    token = _current.set(times)
    try:
        yield times
    finally:
        _current.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    times = _current.get()
    if times is None:
        yield
        return
    start = time.perf_counter()
    times._nested.append(0.0)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        nested = times._nested.pop()
        times.seconds[name] = times.seconds.get(name, 0.0) + elapsed - nested
        if times._nested:
            times._nested[-1] += elapsed
