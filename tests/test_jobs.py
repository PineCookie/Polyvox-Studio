import threading
import time

import pytest

from voxcpm_studio.jobs import JobRunner


def wait(job, timeout=10):
    deadline = time.monotonic() + timeout
    while job.state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return job


def test_job_result_and_progress():
    runner = JobRunner()

    def work(job):
        for i in range(3):
            job.progress("step", i + 1, 3)
        return {"answer": 42}

    job = wait(runner.start("demo", "Demo", work))

    assert (job.state, job.result, job.phase, job.done, job.total) == ("done", {"answer": 42}, "step", 3, 3)
    assert job.finished is not None and "_cancel" not in job.public()


def test_job_failure_and_cancel():
    runner = JobRunner()
    failed = wait(runner.start("demo", "Fails", lambda job: 1 / 0))
    assert failed.state == "failed" and "division" in failed.error

    release = threading.Event()

    def slow(job):
        release.wait(5)
        job.progress("step", 1, 2)  # raises: cancel was requested
        return {}

    job = runner.start("demo", "Slow", slow)
    assert runner.running("demo") is job
    runner.cancel(job.id)
    release.set()
    assert wait(job).state == "cancelled"
    assert runner.running("demo") is None
    with pytest.raises(KeyError):
        runner.get("nope")
