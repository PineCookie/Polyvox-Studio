"""In-process background jobs with progress and cancellation.

For work that runs in the server process but takes too long for one request
(dataset import with transcription, later preprocessing stages). Training has
its own subprocess manager (training.py).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_FINISHED = 20  # finished jobs kept for the UI to read


class JobCancelled(Exception):
    """Raised inside a job when cancellation was requested."""


@dataclass
class Job:
    id: str
    kind: str
    title: str
    state: str = "running"  # running | done | failed | cancelled
    phase: str = ""  # e.g. "transcribing"
    done: int = 0
    total: int = 0
    error: str | None = None
    result: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    seconds_left: float | None = None  # of the current phase, from its rate so far
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _phase_start: tuple[float, int] = (0.0, 0)  # (time, done) when the phase began

    def progress(self, phase: str, done: int, total: int) -> None:
        """Report progress; raises JobCancelled if cancellation was requested."""
        now = time.time()
        if phase != self.phase:
            self._phase_start, self.seconds_left = (now, done), None
        else:
            since, done0 = self._phase_start
            if done > done0 and total > done:
                self.seconds_left = (now - since) / (done - done0) * (total - done)
            elif total <= done:
                self.seconds_left = 0.0
        self.phase, self.done, self.total = phase, done, total
        if self._cancel.is_set():
            raise JobCancelled()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def public(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class JobRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def start(self, kind: str, title: str, fn: Callable[[Job], dict]) -> Job:
        """Run ``fn(job)`` in a thread; its return value becomes ``job.result``."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, title=title)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        threading.Thread(target=self._run, args=(job, fn), daemon=True, name=f"job-{kind}-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"Unknown job: {job_id}") from None

    def running(self, kind: str) -> Job | None:
        return next((j for j in self._jobs.values() if j.kind == kind and j.state == "running"), None)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job._cancel.set()
        return job

    def _run(self, job: Job, fn: Callable[[Job], dict]) -> None:
        try:
            job.result = fn(job) or {}
            job.state = "done"
        except JobCancelled:
            job.state = "cancelled"
        except Exception as e:
            logger.exception("Job %s (%s) failed", job.id, job.kind)
            job.state, job.error = "failed", str(e) or type(e).__name__
        finally:
            job.finished = time.time()

    def _prune(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.state != "running"), key=lambda j: j.finished or 0)
        for job in finished[: max(0, len(finished) - MAX_FINISHED)]:
            del self._jobs[job.id]
