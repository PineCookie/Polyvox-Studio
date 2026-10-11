import os
import subprocess
import sys

import pytest
import torch
import transformers
from voxcpm.model import voxcpm as v1
from voxcpm.model import voxcpm2 as v2
from voxcpm.training import TrainingTracker
from voxcpm.training import config as config_module

from voxcpm_studio.engines.voxcpm import worker as train_worker


@pytest.fixture
def restore_patches(monkeypatch):
    """use_optimizer / optimizer_from_config patch these globally: undo after the test."""
    for owner, attr in [
        (torch.optim, "AdamW"),
        (transformers, "get_cosine_schedule_with_warmup"),
        (TrainingTracker, "log_metrics"),
        (config_module, "load_yaml_config"),
        (v2.VoxCPM2Model, "train"),
        (v2.VoxCPM2Model, "state_dict"),
        (v1.VoxCPMModel, "train"),
        (v1.VoxCPMModel, "state_dict"),
    ]:
        monkeypatch.setattr(owner, attr, getattr(owner, attr))


def test_samples_keep_the_models_level_and_references_are_the_original_clips(tmp_path, restore_patches, monkeypatch):
    import json

    import numpy as np
    import soundfile as sf
    from tensorboardX import SummaryWriter

    monkeypatch.setattr(SummaryWriter, "add_audio", SummaryWriter.add_audio)
    t = np.arange(44_100) / 44_100
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1), 44_100)  # stereo, 44.1 kHz
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps({"audio": str(clip), "text": "示例"}) + "\n", encoding="utf-8")
    model_output = torch.full((1, 48_000), 0.05)
    calls = []
    monkeypatch.setattr(v2.VoxCPM2Model, "generate", lambda self, *a, **k: calls.append(k) or model_output)

    samples = tmp_path / "samples"
    train_worker.save_samples_to(samples, val)
    v2.VoxCPM2Model.generate(object(), target_text="示例", cfg_value=2.0)  # as the script's validation does
    assert calls[-1]["cfg_value"] == train_worker.SAMPLE_GUIDANCE
    writer = SummaryWriter(str(tmp_path / "tb"))
    boosted = np.full(48_000, 0.9, dtype=np.float32)  # the script's peak normalization
    writer.add_audio("val_sample_0/generated_audio", boosted, global_step=5, sample_rate=48_000)
    writer.add_audio("val_sample_0/reference_audio", boosted[:16_000], global_step=5, sample_rate=16_000)
    writer.close()

    generated, rate = sf.read(samples / "step_0000005_0.wav")
    assert rate == 48_000 and np.abs(generated).max() == pytest.approx(0.05, abs=1e-3)
    reference, rate = sf.read(samples / "reference_0.wav", always_2d=True)
    assert rate == 44_100 and reference.shape == (44_100, 2) and np.abs(reference).max() == pytest.approx(0.2, abs=1e-3)
    heard, rate = sf.read(samples / "reference_0_16k.wav", always_2d=True)
    assert rate == 16_000 and heard.shape[1] == 1 and abs(len(heard) - 16_000) <= 1


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lora_A = torch.nn.Linear(4, 4)


def train_steps(model, opt, steps=5):
    torch.manual_seed(1)
    for _ in range(steps):
        opt.zero_grad()
        model.lora_A(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()


def test_optimizer_key_is_read_and_hidden_from_the_script(tmp_path, restore_patches):
    config = tmp_path / "train_config.yaml"
    config.write_text("optimizer: prodigy_sf\nlearning_rate: 1.0\n", encoding="utf-8")
    assert train_worker.optimizer_from_config(config) == ("prodigy_sf", 0)
    assert config_module.load_yaml_config(str(config)) == {"learning_rate": 1.0}

    config.write_text("learning_rate: 0.0001\nprodigy_steps: 40\n", encoding="utf-8")
    assert train_worker.optimizer_from_config(config) == ("adamw", 40)
    assert config_module.load_yaml_config(str(config)) == {"learning_rate": 0.0001}


@pytest.mark.parametrize("name", ["prodigy", "prodigy_sf"])
def test_prodigy_replaces_adamw_and_trains(name, restore_patches):
    train_worker.use_optimizer(name)
    model = Toy()
    before = model.lora_A.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.01)
    assert type(opt).__name__ != "AdamW"
    train_steps(model, opt)
    assert not torch.equal(model.lora_A.weight, before)


def test_schedule_free_checkpoints_hold_the_average_and_resume_exactly(restore_patches):
    train_worker.use_optimizer("prodigy_sf")
    model = Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    train_steps(model, opt, steps=200)  # Prodigy's step size starts tiny (d0 1e-6)
    training = model.lora_A.weight.detach().clone()
    opt.eval()
    average = model.lora_A.weight.detach().clone()
    opt.train()
    gap = (training - average).abs().max().item()
    assert gap > 1e-6

    saved = v2.VoxCPM2Model.state_dict(model)  # as save_checkpoint calls it
    assert torch.equal(saved["lora_A.weight"], average)
    assert torch.equal(model.lora_A.weight, training)  # training goes on unchanged

    # Resume: the script loads the weights, then the optimizer state.
    resumed = Toy()
    resumed.load_state_dict(saved)
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=1.0)
    opt2.load_state_dict(opt.state_dict())
    assert (resumed.lora_A.weight - training).abs().max().item() < gap / 100


def test_parameter_groups_per_network():
    model = torch.nn.Module()
    model.base_lm = torch.nn.Linear(2, 2)
    model.feat_decoder = torch.nn.Linear(2, 2)
    model.feat_decoder.bias.requires_grad_(False)
    stray = torch.nn.Parameter(torch.zeros(1))
    trainable = [p for p in model.parameters() if p.requires_grad] + [stray]

    groups = train_worker.by_network(model, iter(trainable))

    assert {g["name"]: len(g["params"]) for g in groups} == {"base_lm": 2, "feat_decoder": 1, "other": 1}


def test_schedule_free_trains_one_group_per_network(restore_patches, monkeypatch):
    train_worker.use_optimizer("prodigy_sf")
    model = Toy()
    model.feat_decoder = torch.nn.Linear(4, 4)
    monkeypatch.setattr(model, "train", v2.VoxCPM2Model.train.__get__(model))
    model.train()  # the script calls this before creating the optimizer
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1.0)
    assert [g["name"] for g in opt.param_groups] == ["lora_A", "feat_decoder"]
    train_steps(model, opt)


def test_schedule_free_logs_step_sizes_per_network_without_rounding(restore_patches, monkeypatch, tmp_path):
    from voxcpm_studio import training

    train_worker.use_optimizer("prodigy_sf")
    model = Toy()
    model.feat_decoder = torch.nn.Linear(4, 4)
    monkeypatch.setattr(model, "train", v2.VoxCPM2Model.train.__get__(model))
    model.train()
    opt = torch.optim.AdamW(list(model.parameters()), lr=1.0)
    train_steps(model, opt, steps=3)

    log = tmp_path / "train.log"
    tracker = TrainingTracker(writer=None, log_file=str(log), rank=0)
    tracker.step = 3
    tracker.log_metrics({"loss/diff": 0.5, "lr": 1.0}, split="train")

    (point,) = training.parse_metrics(log)["train"]
    group = opt.param_groups[0]
    assert point["lr_lora_A"] == pytest.approx(group["d"] * group["lr"], rel=1e-3)  # d is ~1e-6: not rounded to 0
    assert 0 < point["elr_lora_A"] <= point["lr_lora_A"]
    assert {"lr_feat_decoder", "elr_feat_decoder", "loss/diff"} <= set(point)


def test_schedule_free_resumes_a_single_group_checkpoint_with_fresh_state(restore_patches):
    train_worker.use_optimizer("prodigy_sf")
    old = Toy()
    old_opt = torch.optim.AdamW(old.parameters(), lr=1.0)  # no train() seen: one group
    train_steps(old, old_opt)
    saved = v2.VoxCPM2Model.state_dict(old)

    model = Toy()
    model.feat_decoder = torch.nn.Linear(4, 4)
    model.load_state_dict(saved, strict=False)
    opt = torch.optim.AdamW(train_worker.by_network(model, model.parameters()), lr=1.0)
    opt.load_state_dict(old_opt.state_dict())
    assert torch.equal(model.lora_A.weight, saved["lora_A.weight"])
    train_steps(model, opt)


def test_schedule_free_follows_the_models_train_and_eval(restore_patches, monkeypatch):
    train_worker.use_optimizer("prodigy_sf")
    model = Toy()
    monkeypatch.setattr(model, "train", v2.VoxCPM2Model.train.__get__(model))
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    train_steps(model, opt)
    model.eval()  # validation
    assert not any(g["train_mode"] for g in opt.param_groups)
    model.train()
    assert all(g["train_mode"] for g in opt.param_groups)


def test_each_clip_is_encoded_once(monkeypatch):
    from voxcpm.training import packers

    encoded = []

    def encode(self, wav):
        encoded.append(wav)
        return wav.sum().reshape(1, 1, 1)

    monkeypatch.setattr(packers.AudioFeatureProcessingPacker, "encode_audio", encode)  # restored afterwards
    train_worker.cache_audio_latents()
    packer = object.__new__(packers.AudioFeatureProcessingPacker)
    a, b = torch.rand(1600), torch.rand(1600)
    first = packer.encode_audio(a)
    assert packer.encode_audio(a.clone()) is first  # the same clip in a later epoch
    assert packer.encode_audio(b).item() == pytest.approx(b.sum().item())
    assert len(encoded) == 2


def test_sample_paths_for_tensorboard_tags(tmp_path):
    assert train_worker.sample_path(tmp_path, "val_sample_1/generated_audio", 500) == tmp_path / "step_0000500_1.wav"
    assert train_worker.sample_path(tmp_path, "val_sample_0/reference_audio", 500) == tmp_path / "reference_0.wav"
    assert train_worker.sample_path(tmp_path, "val_sample_0/mel_spectrogram", 500) is None


def test_process_alive():
    assert train_worker.process_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not train_worker.process_alive(proc.pid)


def test_stop_file_runs_the_sigint_handler(tmp_path):
    # Mirrors how the training script registers its checkpoint-saving handler.
    script = tmp_path / "job.py"
    script.write_text(
        "import signal, sys, threading, time\n"
        "from pathlib import Path\n"
        "from voxcpm_studio.engines.voxcpm import worker as train_worker\n"
        "train_worker.POLL_S = 0.02\n"
        "signal.signal(signal.SIGINT, lambda *a: (print('saved', flush=True), sys.exit(0)))\n"
        f"threading.Thread(target=train_worker.watch_for_stop, args=(Path(r'{tmp_path / 'stop'}'), None), daemon=True).start()\n"
        "print('ready', flush=True)\n"
        "while True: time.sleep(0.01)\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    (tmp_path / "stop").touch()

    assert proc.wait(timeout=20) == 0
    assert "saved" in proc.stdout.read()
