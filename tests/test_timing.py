import pytest

from polyvox import timing


def test_stages_count_their_own_time_without_nested_stages(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])

    def wait(seconds):
        clock[0] += seconds

    with timing.collect() as times:
        with timing.stage("asr"):
            wait(5)
            with timing.stage("loading models"):
                wait(10)
        with timing.stage("asr"):
            wait(2)
        wait(1)  # in no stage
    assert times.seconds == {"asr": pytest.approx(7), "loading models": pytest.approx(10)}
    assert times.summary() == {"loading models": 10.0, "asr": 7.0, "other": 1.0}  # longest first


def test_stages_outside_collect_cost_nothing():
    with timing.stage("asr"):
        pass
    with timing.collect() as times:
        pass
    assert times.seconds == {}


def test_import_job_reports_its_timings(tmp_path, data_dir, service, training):
    import soundfile as sf
    from fastapi.testclient import TestClient
    from test_importer import wait_job, write_list

    from polyvox.server import create_app

    clip = tmp_path / "a.wav"
    sf.write(clip, [0.0] * 16000, 16000)
    lst = write_list(tmp_path / "speaker_a.list", [f"{clip}|speaker_a|ZH|示例"])
    client = TestClient(create_app(service.settings, service, training))
    job = wait_job(client, client.post("/api/datasets/import", json={"source": str(lst), "name": "a"}).json()["id"])
    assert job["state"] == "done", job
    assert "scanning" in job["result"]["timings"] and "writing clips" in job["result"]["timings"]
