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

from voxcpm_studio import train_worker


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
    assert train_worker.optimizer_from_config(config) == "prodigy_sf"
    assert config_module.load_yaml_config(str(config)) == {"learning_rate": 1.0}

    config.write_text("learning_rate: 0.0001\n", encoding="utf-8")
    assert train_worker.optimizer_from_config(config) == "adamw"


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
        "from voxcpm_studio import train_worker\n"
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
