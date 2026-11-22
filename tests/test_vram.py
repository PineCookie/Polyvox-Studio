import pytest
import torch
from fastapi.testclient import TestClient

from polyvox import vram
from polyvox.server import create_app

from conftest import make_dataset


def test_profiles_by_card_size():
    assert [vram.profile_for(gb).name for gb in (24, 15.9, 11.6, 7.6, 5.8)] == [
        "16 GB or more",
        "16 GB or more",
        "12 GB",
        "8 GB",
        "under 8 GB",
    ]
    small = vram.profile_for(7.6)
    assert small.exclusive and small.denoise_on_cpu
    assert vram.profile_for(5.8).note  # tells what doesn't fit


def test_override_and_cpu(monkeypatch):
    monkeypatch.setenv("POLYVOX_VRAM_GB", "12")
    assert vram.detect("cpu") == (12.0, vram.profile_for(12))
    monkeypatch.delenv("POLYVOX_VRAM_GB")
    assert vram.detect("cpu") == (None, vram.CPU)
    # A CUDA device name without the GPU (CPU-only PyTorch, as in CI): no profile, no crash.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert vram.detect("cuda:0") == (None, vram.CPU)
    vram.apply_limit("cuda:0")  # nothing to limit


def test_is_oom_follows_wrapped_errors():
    try:
        try:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        except torch.OutOfMemoryError as e:
            raise RuntimeError("Audio denoising processing failed") from e
    except RuntimeError as wrapped:
        assert vram.is_oom(wrapped)
    assert not vram.is_oom(ValueError("no"))


def test_in_batches_halves_the_limit_when_out_of_memory():
    done, tried = [], []

    def plan(rest, limit):
        return rest[: max(1, int(limit))]

    def run(batch):
        tried.append(len(batch))
        if len(batch) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory")
        done.extend(batch)

    limit = vram.in_batches(list(range(7)), plan, run, 8, "test")
    assert done == list(range(7)) and limit == 2
    assert tried[:3] == [7, 4, 2]  # 8 -> 4 -> 2

    def always_oom(batch):
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.OutOfMemoryError):  # a single item can't shrink further
        vram.in_batches([0], plan, always_oom, 1, "test")


def test_status_reports_the_profile_and_services_use_it(monkeypatch, service, training):
    monkeypatch.setenv("POLYVOX_VRAM_GB", "8")
    client = TestClient(create_app(service.settings, service, training))
    gpu = client.get("/api/status").json()["gpu"]
    assert (gpu["total_gb"], gpu["name"]) == (8.0, "8 GB")
    assert service.denoise_on_cpu


def test_small_cards_keep_speech_and_dataset_steps_apart(monkeypatch, service, training):
    import threading
    import time

    from polyvox.jobs import JobRunner

    monkeypatch.setenv("POLYVOX_VRAM_GB", "8")
    jobs = JobRunner()
    client = TestClient(create_app(service.settings, service, training, jobs=jobs))
    release = threading.Event()
    step = jobs.start("step", "Steps of a dataset", lambda job: release.wait(5) and {}, "a_dataset")
    try:
        r = client.post("/api/tts", data={"text": "hello"})
        assert r.status_code == 409 and "Steps of a dataset" in r.json()["detail"]
    finally:
        release.set()
    while step.state == "running":
        time.sleep(0.01)
    assert client.post("/api/tts", data={"text": "hello"}).status_code == 200
    # The other way round: a job that holds a speech engine keeps GPU dataset work waiting.
    make_dataset(service.settings.data_dir, "a_dataset")
    release.clear()
    jobs.start("reference_search", "Reference clips", lambda job: release.wait(5) and {})
    try:
        r = client.post("/api/dataset-clip/check", json={"id": "a_dataset/train.jsonl", "index": 0})
        assert r.status_code == 409 and "Reference clips" in r.json()["detail"]
    finally:
        release.set()


def test_dots_checkpoints_batches_too_large_for_the_card(monkeypatch):
    from polyvox.engines.dots.worker import use_checkpointing

    monkeypatch.delenv("POLYVOX_TRAIN_CHECKPOINT", raising=False)
    from polyvox.engines.dots.engine import TRAIN_BATCHES

    big, mid, small = vram.profile_for(16), vram.profile_for(12), vram.profile_for(8)
    batch = lambda profile: vram.per_gpu(profile, TRAIN_BATCHES, 1)  # noqa: E731
    assert not use_checkpointing(batch(big), big)
    assert use_checkpointing(8, big)  # a larger batch set by hand
    assert use_checkpointing(batch(mid), mid) and use_checkpointing(batch(small), small)
    assert not use_checkpointing(16, vram.CPU)
    monkeypatch.setenv("POLYVOX_TRAIN_CHECKPOINT", "0")
    assert not use_checkpointing(8, small)
    monkeypatch.setenv("POLYVOX_TRAIN_CHECKPOINT", "1")
    assert use_checkpointing(1, big)


def test_each_engine_trains_at_its_own_batch_for_the_gpu():
    from polyvox import engines

    names = (vram.GB16, vram.GB12, vram.GB8, vram.SMALL, vram.ON_CPU)
    profiles = [vram.profile_for(gb) for gb in (16, 12, 8, 6)] + [vram.CPU]
    assert [p.name for p in profiles] == list(names)
    batches = {
        cls.id: [next(p.default for p in cls.train_form(profile) if p.name == "batch_size") for profile in profiles]
        for cls in engines.classes().values()
        if cls.capabilities.training
    }
    assert batches == {
        "voxcpm": [4, 2, 1, 1, 1],
        "dots_tts": [4, 8, 4, 1, 1],
        "gpt_sovits": [8, 8, 4, 1, 1],
        "fish_s2": [4, 4, 4, 4, 4],
        "rvc": [16, 16, 8, 4, 4],
    }
    offload = [
        next(p.default for p in engines.classes()["fish_s2"].train_form(f) if p.name == "s2_offload") for f in profiles
    ]
    assert offload == [False, True, True, True, False]  # where the whole model and training don't fit
