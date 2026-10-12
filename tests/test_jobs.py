import threading
import time

import pytest

from polyvox.jobs import JobRunner


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


def test_seconds_left_from_the_phase_rate(monkeypatch):
    from polyvox import jobs

    now = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: now[0])
    job = jobs.Job(id="x", kind="demo", title="Demo")
    job.progress("slicing", 0, 100)
    assert job.seconds_left is None  # no rate yet
    now[0] += 10
    job.progress("slicing", 25, 100)
    assert job.seconds_left == pytest.approx(30)  # 25 per 10 s, 75 to go
    job.progress("aligning", 0, 8)  # a new phase starts over
    assert job.seconds_left is None
    now[0] += 4
    job.progress("aligning", 8, 8)
    assert job.seconds_left == 0


def test_rate_over_the_last_seconds(monkeypatch):
    from polyvox import jobs

    now = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: now[0])
    job = jobs.Job(id="x", kind="download", title="Download")
    job.progress("model", 0, 1000)
    assert job.rate is None  # under a second of samples
    for _ in range(10):  # 10 MB/s for 10 s
        now[0] += 1
        job.progress("model", job.done + 10, 1000)
    assert job.rate == pytest.approx(10)
    for _ in range(6):  # then 50 MB/s: the rate follows within the window
        now[0] += 1
        job.progress("model", job.done + 50, 1000)
    assert job.rate == pytest.approx(50)
    job.progress("other", 0, 10)  # a new phase starts over
    assert job.rate is None


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
