"""Training subprocess: runs the vendored VoxCPM training script unmodified.

``python -m voxcpm_studio.train_worker <train_config.yaml> --samples-dir <dir>
--stop-file <path> [--parent-pid <pid>]``

On top of the script this adds:

- Graceful stop. When the stop file appears, or the parent process exits,
  the script's own SIGINT handler runs: it saves a checkpoint and exits. On
  Windows a parent cannot deliver SIGINT/SIGTERM to a child, and
  ``Popen.terminate()`` kills it without saving.

  This is not done by reading stdin: on Windows, DataLoader worker processes
  query the inherited std handles at startup, and that query blocks forever
  behind a pending synchronous read on the same pipe.
- Validation samples. Audio the script logs to TensorBoard is also written as
  WAV files that the UI can play.
- Frozen base weights in bfloat16 for LoRA runs (``frozen_weights_in_bf16``).
- ``torchaudio.load`` / ``save`` through audio_io, as in the server
  (torchcodec, which they would use, is not installed).
- On small GPUs (vram.Profile.train_checkpointing), gradient checkpointing
  (``checkpoint_layers``).
- Prodigy / Prodigy Schedule-Free instead of AdamW (``use_optimizer``).
"""

from __future__ import annotations

import _thread
import argparse
import os
import re
import runpy
import signal
import sys
import threading
import time
from pathlib import Path

SCRIPT = Path(__file__).with_name("vendor") / "train_voxcpm_finetune.py"
SAMPLE_TAG = re.compile(r"val_sample_(\d+)/(generated|reference)_audio")
POLL_S = 1.0


def sample_path(samples_dir: Path, tag: str, step: int) -> Path | None:
    m = SAMPLE_TAG.fullmatch(tag)
    if not m:
        return None
    index, kind = m.groups()
    name = f"step_{step:07d}_{index}.wav" if kind == "generated" else f"reference_{index}.wav"
    return samples_dir / name


def save_samples_to(samples_dir: Path) -> None:
    import numpy as np
    import soundfile as sf
    from tensorboardX import SummaryWriter

    add_audio = SummaryWriter.add_audio

    def add_audio_and_save(self, tag, snd_tensor, global_step=None, sample_rate=44100, *args, **kwargs):
        add_audio(self, tag, snd_tensor, global_step, sample_rate, *args, **kwargs)
        path = sample_path(samples_dir, tag, int(global_step or 0))
        if path is None:
            return
        try:
            samples_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            sf.write(tmp, np.asarray(snd_tensor, dtype=np.float32).reshape(-1), int(sample_rate), format="WAV")
            tmp.replace(path)  # the UI never sees a partial file
        except Exception as e:
            print(f"[studio] Failed to save sample {path.name}: {e}", file=sys.stderr, flush=True)

    SummaryWriter.add_audio = add_audio_and_save


def frozen_weights_in_bf16() -> None:
    """For LoRA runs, keep the frozen base weights in bfloat16 instead of float32.

    voxcpm's training mode leaves the whole model in float32, which fills a
    16 GB card at batch size 1. Training steps run under bfloat16 autocast,
    which casts those weights to bfloat16 for every matrix product anyway, so
    storing the frozen matrices that way changes little but halves their memory.
    LoRA weights (trained), norms and biases (1-D) and the audio VAE stay float32.
    Set VOXCPM_STUDIO_TRAIN_FP32=1 to keep everything float32.
    """
    if os.environ.get("VOXCPM_STUDIO_TRAIN_FP32") == "1":
        return
    import torch
    from voxcpm.model import voxcpm as v1
    from voxcpm.model import voxcpm2 as v2

    for cls in (v2.VoxCPM2Model, v1.VoxCPMModel):
        original = cls.from_local.__func__

        def from_local(klass, *args, _original=original, **kwargs):
            model = _original(klass, *args, **kwargs)
            if kwargs.get("training") and kwargs.get("lora_config") is not None:
                converted = 0
                for name, param in model.named_parameters():
                    frozen_matrix = not param.requires_grad and param.dim() >= 2 and param.dtype == torch.float32
                    if frozen_matrix and "audio_vae" not in name:
                        param.data = param.data.to(torch.bfloat16)
                        converted += param.numel()
                print(f"[studio] Frozen weights in bfloat16: {converted / 1e9:.2f}B parameters.", file=sys.stderr)
            return model

        cls.from_local = classmethod(from_local)


def checkpoint_layers() -> None:
    """Gradient checkpointing for VoxCPM's transformer layers (both language
    models and the diffusion decoder use MiniCPMDecoderLayer): their activations
    are recomputed in the backward pass instead of kept, so a LoRA run fits a
    smaller GPU, at some speed. Only while training with gradients (validation
    and sample generation run as before)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from voxcpm.modules.minicpm4.model import MiniCPMDecoderLayer

    forward = MiniCPMDecoderLayer.forward

    def checkpointed(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(forward, self, *args, use_reentrant=False, **kwargs)
        return forward(self, *args, **kwargs)

    MiniCPMDecoderLayer.forward = checkpointed
    print("[studio] Gradient checkpointing on (transformer layers).", file=sys.stderr)


def use_optimizer(name: str) -> None:
    """Swap the script's AdamW (and its schedule) for a Prodigy variant.

    - ``prodigy`` (prodigyopt): estimates its own step size; ``lr`` is a multiplier
      (the run's learning_rate, normally 1.0). The script's cosine schedule with
      warmup stays; safeguard_warmup is the authors' advice for warmup.
    - ``prodigy_sf`` (prodigy-plus-schedule-free): Prodigy with Schedule-Free
      averaging, so a constant schedule. It trains on interpolated weights and
      keeps the average elsewhere: the model's train() / eval() switch the
      optimizer too (validation and samples use the average), checkpoints hold the
      average, and loading a checkpoint switches back to training weights.

    The logged learning rate is the effective one (Prodigy's d x lr).
    """
    if name == "adamw":
        return
    import torch
    import transformers
    from voxcpm.model import voxcpm as v1
    from voxcpm.model import voxcpm2 as v2
    from voxcpm.training import TrainingTracker

    active: list[torch.optim.Optimizer] = []

    if name == "prodigy":
        from prodigyopt import Prodigy

        def make(params, lr, weight_decay=0.0, **_):
            opt = Prodigy(params, lr=lr, weight_decay=weight_decay, safeguard_warmup=True, use_bias_correction=True)
            active.append(opt)
            return opt

    elif name == "prodigy_sf":
        from prodigyplus import ProdigyPlusScheduleFree

        class ScheduleFree(ProdigyPlusScheduleFree):
            def load_state_dict(self, state_dict):
                super().load_state_dict(state_dict)
                # The model was just loaded with the checkpoint's averaged weights.
                for group in self.param_groups:
                    group["train_mode"] = False
                self.train()

        def make(params, lr, weight_decay=0.0, **_):
            # A shorter averaging window than the package default (schedulefree_c 0)
            # and beta1 0.95, as its docs suggest for Schedule-Free.
            opt = ScheduleFree(params, lr=lr, weight_decay=weight_decay, betas=(0.95, 0.99), schedulefree_c=20)
            active.append(opt)
            return opt

        def constant(optimizer, **_):
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        transformers.get_cosine_schedule_with_warmup = constant

        for cls in (v2.VoxCPM2Model, v1.VoxCPMModel):

            def train(self, mode=True, _train=cls.train):
                result = _train(self, mode)
                for opt in active:
                    opt.train() if mode else opt.eval()
                return result

            def state_dict(self, *args, _state_dict=cls.state_dict, **kwargs):
                if not active:
                    return _state_dict(self, *args, **kwargs)
                opt = active[0]
                training = any(g["train_mode"] for g in opt.param_groups)
                opt.eval()
                try:  # the LoRA tensors change back below: copy them
                    return {k: v.clone() if "lora_" in k else v for k, v in _state_dict(self, *args, **kwargs).items()}
                finally:
                    if training:
                        opt.train()

            cls.train, cls.state_dict = train, state_dict
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    torch.optim.AdamW = make

    log_metrics = TrainingTracker.log_metrics

    def log_with_effective_lr(self, metrics, *args, **kwargs):
        if active and "lr" in metrics:
            group = active[0].param_groups[0]
            metrics = {**metrics, "lr": float(group.get("d", 1.0)) * float(group["lr"])}
        return log_metrics(self, metrics, *args, **kwargs)

    TrainingTracker.log_metrics = log_with_effective_lr
    print(f"[studio] Optimizer: {name}.", file=sys.stderr)


def optimizer_from_config(config_path: Path) -> str:
    """The run config's ``optimizer`` (adamw if absent). The script never sees the
    key: its train() takes no such argument."""
    import yaml
    from voxcpm.training import config as config_module

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    load_yaml_config = config_module.load_yaml_config

    def without_optimizer(*args, **kwargs):
        data = load_yaml_config(*args, **kwargs)
        data.pop("optimizer", None)
        return data

    config_module.load_yaml_config = without_optimizer
    return str(config.get("optimizer") or "adamw")


def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import _winapi

        try:
            handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid)
        except OSError:
            return False
        try:
            return _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_TIMEOUT
        finally:
            _winapi.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # on Windows this would terminate the process
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def watch_for_stop(stop_file: Path, parent_pid: int | None) -> None:
    while not stop_file.exists():
        if parent_pid is not None and not process_alive(parent_pid):
            print("[studio] Server exited.", file=sys.stderr, flush=True)
            break
        time.sleep(POLL_S)
    print("[studio] Stop requested; saving a checkpoint.", file=sys.stderr, flush=True)
    _thread.interrupt_main(signal.SIGINT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()

    save_samples_to(args.samples_dir)
    frozen_weights_in_bf16()
    from voxcpm_studio import audio_io, vram

    audio_io.patch_torchaudio()
    vram.apply_limit("cuda")  # VOXCPM_STUDIO_VRAM_GB, as in the server
    if os.environ.get("VOXCPM_STUDIO_TRAIN_CHECKPOINT") == "1" or vram.detect("cuda")[1].train_checkpointing:
        checkpoint_layers()
    use_optimizer(optimizer_from_config(args.config_path))
    threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid), daemon=True).start()
    sys.argv = [str(SCRIPT), "--config_path", str(args.config_path)]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
