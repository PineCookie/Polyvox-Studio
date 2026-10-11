import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from test_importer import wait_job, write_list

from polyvox.prep import fillers
from polyvox.server import create_app


def test_tidy_shortens_long_runs_and_keeps_fillers_distinct():
    assert fillers.tidy("哈哈哈哈，你好呵呵呵。嗯嗯嗯，好。") == "哈哈，你好呵呵。嗯嗯，好。"
    # Short runs, mixed laughs and ordinary words stay as they are.
    assert fillers.tidy("嗯，嗯嗯，哈哈呵呵，啊啊") == "嗯，嗯嗯，哈哈呵呵，啊啊"
    assert fillers.tidy("好好好，谢谢谢谢") == "好好好，谢谢谢谢"


def test_tidy_manifest_rewrites_only_texts_and_keeps_a_backup(tmp_path):
    manifest = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"audio": "wavs/a.wav", "text": "呵呵呵呵，示例", "ref_audio": "wavs/r.wav"}, ensure_ascii=False),
        json.dumps({"audio": "wavs/b.wav", "text": "没有变化"}, ensure_ascii=False),
        "not json",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert fillers.tidy_manifest(manifest) == 1
    after = manifest.read_text(encoding="utf-8").splitlines()
    assert json.loads(after[0]) == {"audio": "wavs/a.wav", "text": "呵呵，示例", "ref_audio": "wavs/r.wav"}
    assert after[1:] == lines[1:]
    assert (tmp_path / "train.jsonl.orig").read_text(encoding="utf-8").splitlines() == lines
    assert fillers.tidy_manifest(manifest) == 0  # nothing left to tidy


def test_import_tidies_by_default_and_datasets_can_be_tidied_later(service, training, data_dir, tmp_path):
    clip = tmp_path / "a.wav"
    sf.write(clip, np.zeros(16000, dtype=np.float32), 16000)
    lst = write_list(tmp_path / "speaker_a.list", [f"{clip}|speaker_a|ZH|哈哈哈哈，示例"])
    client = TestClient(create_app(service.settings, service, training))

    job = wait_job(client, client.post("/api/datasets/import", json={"source": str(lst), "name": "a"}).json()["id"])
    assert job["state"] == "done", job
    manifest = data_dir / "a" / "train.jsonl"
    assert json.loads(manifest.read_text(encoding="utf-8"))["text"] == "哈哈，示例"

    body = {"source": str(lst), "name": "b", "tidy_fillers": False}
    wait_job(client, client.post("/api/datasets/import", json=body).json()["id"])
    assert client.post("/api/datasets/tidy-fillers", json={"id": "b/train.jsonl"}).json() == {"changed": 1}
    assert json.loads((data_dir / "b" / "train.jsonl").read_text(encoding="utf-8"))["text"] == "哈哈，示例"
    assert client.post("/api/datasets/tidy-fillers", json={"id": "nope/train.jsonl"}).status_code == 404


def test_is_filler():
    assert fillers.is_filler("嗯") and fillers.is_filler("呵")
    assert not fillers.is_filler("好") and not fillers.is_filler("hello")
