import time

import pytest

from voxcpm_studio import timing


def test_stages_count_their_own_time_without_nested_stages():
    with timing.collect() as times:
        with timing.stage("asr"):
            time.sleep(0.05)
            with timing.stage("loading models"):
                time.sleep(0.1)
        with timing.stage("asr"):
            time.sleep(0.05)
    assert times.seconds["asr"] == pytest.approx(0.1, abs=0.04)
    assert times.seconds["loading models"] == pytest.approx(0.1, abs=0.04)
    assert list(times.summary()) == ["asr", "loading models"]  # longest first; no "other" below 0.5 s


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

    from voxcpm_studio.api import create_app

    clip = tmp_path / "a.wav"
    sf.write(clip, [0.0] * 16000, 16000)
    lst = write_list(tmp_path / "speaker_a.list", [f"{clip}|speaker_a|ZH|示例"])
    client = TestClient(create_app(service.settings, service, training))
    job = wait_job(client, client.post("/api/datasets/import", json={"source": str(lst), "name": "a"}).json()["id"])
    assert job["state"] == "done", job
    assert "scanning" in job["result"]["timings"] and "writing clips" in job["result"]["timings"]
