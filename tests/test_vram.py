import pytest
import torch
from fastapi.testclient import TestClient

from voxcpm_studio import vram
from voxcpm_studio.server import create_app


def test_profiles_by_card_size():
    assert [vram.profile_for(gb).name for gb in (24, 15.9, 11.6, 7.6, 5.8)] == [
        "16 GB or more",
        "16 GB or more",
        "12 GB",
        "8 GB",
        "under 8 GB",
    ]
    small = vram.profile_for(7.6)
    assert small.exclusive and small.denoise_on_cpu and small.train_batch == 1
    assert vram.profile_for(5.8).note  # tells what doesn't fit


def test_override_and_cpu(monkeypatch):
    monkeypatch.setenv("VOXCPM_STUDIO_VRAM_GB", "12")
    assert vram.detect("cpu") == (12.0, vram.profile_for(12))
    monkeypatch.delenv("VOXCPM_STUDIO_VRAM_GB")
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
    monkeypatch.setenv("VOXCPM_STUDIO_VRAM_GB", "8")
    client = TestClient(create_app(service.settings, service, training))
    gpu = client.get("/api/status").json()["gpu"]
    assert (gpu["total_gb"], gpu["name"], gpu["train_batch"]) == (8.0, "8 GB", 1)
    assert gpu["train_batches"] == {"voxcpm": 1, "dots_tts": 4}
    assert service.denoise_on_cpu


def test_dots_checkpoints_batches_too_large_for_the_card(monkeypatch):
    from voxcpm_studio.engines.dots.worker import use_checkpointing

    monkeypatch.delenv("VOXCPM_STUDIO_TRAIN_CHECKPOINT", raising=False)
    big, mid, small = vram.profile_for(16), vram.profile_for(12), vram.profile_for(8)
    assert not use_checkpointing(big.dots_train_batch, big)
    assert use_checkpointing(8, big)  # a larger batch set by hand
    assert use_checkpointing(mid.dots_train_batch, mid) and use_checkpointing(small.dots_train_batch, small)
    assert not use_checkpointing(16, vram.CPU)
    monkeypatch.setenv("VOXCPM_STUDIO_TRAIN_CHECKPOINT", "0")
    assert not use_checkpointing(8, small)
    monkeypatch.setenv("VOXCPM_STUDIO_TRAIN_CHECKPOINT", "1")
    assert use_checkpointing(1, big)
