import time

import pytest

from polyvox import models
from polyvox.server.phase_times import PhaseTimes


class FakeEngine:
    id = "speaker_engine"

    def __init__(self):
        self.phase = "idle"

    def progress(self):
        return {"phase": self.phase}


def walk(engine, phases):
    for phase, seconds in phases:
        engine.phase = phase
        time.sleep(seconds)
    engine.phase = "idle"


def test_one_off_phases_are_remembered_across_restarts(tmp_path):
    times, engine = PhaseTimes(tmp_path / ".phase-times.json"), FakeEngine()
    with times.watching(engine):
        engine.phase = "loading"
        time.sleep(0.3)
        phase, seconds = times.current(engine.id)
        assert phase == "loading" and 0.1 < seconds < 1.0
        walk(engine, [("loading", 0.6), ("compiling", 0.9), ("generating", 0.3)])
    assert times.current(engine.id) is None
    assert times.last(engine.id, "loading") == pytest.approx(0.9, abs=0.25)
    assert times.last(engine.id, "compiling") == pytest.approx(0.9, abs=0.25)
    assert times.last(engine.id, "generating") is None  # depends on the text, not the computer
    again = PhaseTimes(tmp_path / ".phase-times.json")  # the next server start
    assert again.last(engine.id, "compiling") == times.last(engine.id, "compiling")


def test_failed_requests_and_loads_that_downloaded_are_not_remembered(tmp_path, monkeypatch):
    times, engine = PhaseTimes(tmp_path / ".phase-times.json"), FakeEngine()
    # (phases of 0.9 s: the watcher polls every 0.1 s and ignores phases under 0.5 s)
    with pytest.raises(RuntimeError), times.watching(engine):
        walk(engine, [("loading", 0.9)])
        raise RuntimeError("out of memory")
    assert times.last(engine.id, "loading") is None

    monkeypatch.setattr(models, "active_download", lambda: {"name": "a model", "done": 1, "total": 2})
    with times.watching(engine):
        walk(engine, [("loading", 0.9)])
    monkeypatch.setattr(models, "active_download", lambda: None)
    with times.watching(engine):
        walk(engine, [("compiling", 0.9)])
    assert times.last(engine.id, "loading") is None and times.last(engine.id, "compiling")


def test_an_unreadable_file_starts_empty(tmp_path):
    (tmp_path / ".phase-times.json").write_text("not json", encoding="utf-8")
    assert PhaseTimes(tmp_path / ".phase-times.json").last("x", "loading") is None
