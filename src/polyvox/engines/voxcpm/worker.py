"""Training subprocess: runs the vendored VoxCPM training script unmodified.

``python -m polyvox.engines.voxcpm.worker <train_config.yaml> --samples-dir <dir>
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
  WAV files that the UI can play: samples at the model's own level, references
  as the original clips (``save_samples_to``).
- Frozen base weights in bfloat16 for LoRA runs (``frozen_weights_in_bf16``).
- Each clip's audio-VAE latents computed once (``cache_audio_latents``).
- ``torchaudio.load`` / ``save`` through audio_io, as in the server
  (torchcodec, which they would use, is not installed).
- On small GPUs (vram.Profile.train_checkpointing), gradient checkpointing
  (``checkpoint_layers``); otherwise, on CUDA unless POLYVOX_OPTIMIZE is off,
  the transformer layers fused by torch.compile (``fuse_layers``).
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

SCRIPT = Path(__file__).resolve().parents[2] / "vendor" / "train_voxcpm_finetune.py"
SAMPLE_TAG = re.compile(r"val_sample_(\d+)/(generated|reference)_audio")
POLL_S = 1.0
MODEL_INPUT_RATE = 16_000  # VoxCPM2's audio encoder: what training and cloning hear
SAMPLE_GUIDANCE = 1.6  # validation samples, as the Synthesize tab's default (not the script's 2.0)


def sample_path(samples_dir: Path, tag: str, step: int) -> Path | None:
    m = SAMPLE_TAG.fullmatch(tag)
    if not m:
        return None
    index, kind = m.groups()
    name = f"step_{step:07d}_{index}.wav" if kind == "generated" else f"reference_{index}.wav"
    return samples_dir / name


def model_input_path(reference: Path) -> Path:
    """Where a reference clip's copy at the model's input rate goes."""
    return reference.with_name(f"{reference.stem}_16k.wav")


def save_samples_to(samples_dir: Path, val_manifest: Path | None = None) -> None:
    """Also write the audio the script logs to TensorBoard as WAV files, but not
    as the script logs it: it peak-normalizes every clip to 0.9 of full scale
    (samples played several dB louder than synthesis, noise and all) and logs
    the reference at the encoder's 16 kHz. Samples are saved as the model made
    them; a reference as the original validation clip, plus the 16 kHz mono
    copy the model hears (``model_input_path``). Samples are generated with
    guidance SAMPLE_GUIDANCE, as synthesis is by default."""
    import numpy as np
    import soundfile as sf
    from tensorboardX import SummaryWriter
    from voxcpm.model import voxcpm as v1
    from voxcpm.model import voxcpm2 as v2

    from polyvox.data import audio_io, manifests

    raw: list[np.ndarray] = []  # the model's last generated audio
    for cls in (v2.VoxCPM2Model, v1.VoxCPMModel):

        def generate(self, *args, _generate=cls.generate, **kwargs):
            if "cfg_value" in kwargs:  # the script's 2.0; samples should sound like synthesis
                kwargs["cfg_value"] = SAMPLE_GUIDANCE
            audio = _generate(self, *args, **kwargs)
            raw[:] = [audio.detach().float().cpu().numpy().reshape(-1)] if hasattr(audio, "detach") else []
            return audio

        cls.generate = generate

    val_clips: list[str] = []
    if val_manifest is not None and val_manifest.is_file():
        val_clips = [e["audio"] for e in manifests.read_manifest(val_manifest)[0]]

    def write(path: Path, audio: np.ndarray, rate: int) -> None:
        tmp = path.with_suffix(".tmp")
        sf.write(tmp, audio, rate, format="WAV")
        tmp.replace(path)  # the UI never sees a partial file

    add_audio = SummaryWriter.add_audio

    def add_audio_and_save(self, tag, snd_tensor, global_step=None, sample_rate=44100, *args, **kwargs):
        add_audio(self, tag, snd_tensor, global_step, sample_rate, *args, **kwargs)
        path = sample_path(samples_dir, tag, int(global_step or 0))
        m = SAMPLE_TAG.fullmatch(tag)
        if path is None or m is None:
            return
        index, kind = m.groups()
        try:
            samples_dir.mkdir(parents=True, exist_ok=True)
            if kind == "generated":
                audio = raw.pop() if raw else np.asarray(snd_tensor, dtype=np.float32).reshape(-1)
                write(path, audio, int(sample_rate))
            elif int(index) < len(val_clips) and not path.exists():  # logged at every validation
                audio, rate = audio_io.read(val_clips[int(index)])
                write(path, audio.T, rate)
                heard = audio_io.resample(audio.mean(axis=0, keepdims=True), rate, MODEL_INPUT_RATE)[0]
                write(model_input_path(path), heard, MODEL_INPUT_RATE)
            elif not val_clips:  # no manifest to read: what the script logged
                write(path, np.asarray(snd_tensor, dtype=np.float32).reshape(-1), int(sample_rate))
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
    Set POLYVOX_TRAIN_FP32=1 to keep everything float32.
    """
    if os.environ.get("POLYVOX_TRAIN_FP32") == "1":
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


def cache_audio_latents() -> None:
    """Encode each clip with the frozen audio VAE once, not at every step (about
    a fifth of the step time). The script's batch packer encodes every clip of
    every batch; the encoder is deterministic (its mean, no sampling) and a clip
    is read the same way each epoch, so its latents are kept on the GPU, keyed by
    a hash of the waveform. Same results; a few MB per hundred clips."""
    import hashlib

    from voxcpm.training import packers

    encode = packers.AudioFeatureProcessingPacker.encode_audio
    latents: dict[bytes, object] = {}

    def encode_audio(self, wav):
        key = hashlib.blake2b(wav.detach().float().cpu().numpy().tobytes(), digest_size=16).digest()
        if key not in latents:
            latents[key] = encode(self, wav)
        return latents[key]

    packers.AudioFeatureProcessingPacker.encode_audio = encode_audio


def fuse_layers() -> None:
    """torch.compile of the transformer layers' training forward
    (MiniCPMDecoderLayer: both language models and the diffusion decoder): one
    compiled function for every layer (their weights are its inputs), any batch
    length (dynamic shapes). Only while training with gradients: validation and
    sample generation run as before. If compiling fails, the layers run as
    they are for the rest of the run."""
    import torch
    from voxcpm.modules.minicpm4.model import MiniCPMDecoderLayer

    forward = MiniCPMDecoderLayer.forward
    fused = [torch.compile(forward, dynamic=True)]
    worked = []  # once a fused call has succeeded, later errors are not about compiling

    def maybe_fused(self, *args, **kwargs):
        if fused and self.training and torch.is_grad_enabled():
            try:
                out = fused[0](self, *args, **kwargs)
                if not worked:
                    worked.append(True)
                return out
            except Exception as e:
                if worked or isinstance(e, torch.OutOfMemoryError):
                    raise
                print(f"[studio] Fusing the transformer layers failed ({e!r}); running them unfused.", file=sys.stderr)
                fused.clear()
        return forward(self, *args, **kwargs)

    MiniCPMDecoderLayer.forward = maybe_fused
    print("[studio] Transformer layers fused for training (torch.compile).", file=sys.stderr)


def optimize_enabled() -> bool:
    """As the server's Settings.optimize: POLYVOX_OPTIMIZE auto (default) / 1 / 0; auto = on CUDA."""
    import torch

    value = os.environ.get("POLYVOX_OPTIMIZE", "auto").strip().lower()
    return torch.cuda.is_available() if value == "auto" else value in ("1", "true", "yes", "on")


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


def by_network(model, params) -> list[dict]:
    """``params`` as parameter groups, one per top-level submodule of ``model``
    (VoxCPM2: base_lm, residual_lm, feat_decoder), each named after it."""
    params = list(params)
    names = {id(p): name.split(".")[0] for name, p in model.named_parameters()}
    groups: dict[str, list] = {}
    for p in params:
        groups.setdefault(names.get(id(p), "other"), []).append(p)
    return [{"params": ps, "name": name} for name, ps in groups.items()]


def use_optimizer(name: str, prodigy_steps: int = 0) -> None:
    """Swap the script's AdamW (and its schedule) for a Prodigy variant.

    - ``prodigy`` (prodigyopt): estimates its own step size; ``lr`` is a multiplier
      (the run's learning_rate, normally 1.0). The script's cosine schedule with
      warmup stays; safeguard_warmup is the authors' advice for warmup.
    - ``prodigy_sf`` (prodigy-plus-schedule-free): Prodigy with Schedule-Free
      averaging, so a constant schedule. It trains on interpolated weights and
      keeps the average elsewhere: the model's train() / eval() switch the
      optimizer too (validation and samples use the average), checkpoints hold the
      average, and loading a checkpoint switches back to training weights.
      ``prodigy_steps`` > 0 keeps its step size fixed from that step on. Each
      network (text LM, residual LM, diffusion decoder) is its own parameter
      group, so Prodigy estimates a step size for each (``by_network``): they
      settle apart (text LM highest, decoder lowest), and results held up
      better across epochs than with one shared step size.

    The logged learning rate is Prodigy's d x lr, also per group as
    ``lr_<network>``. Schedule-Free moves the training weights by less: d x lr x
    (1 - beta1 (1 - c)), c the averaging weight, which shrinks as the average
    builds up; logged as ``elr_<network>`` (the package's ``effective_lr`` x d).
    """
    if name == "adamw":
        return
    import torch
    import transformers
    from voxcpm.model import voxcpm as v1
    from voxcpm.model import voxcpm2 as v2
    from voxcpm.training import TrainingTracker

    active: list[torch.optim.Optimizer] = []
    models: list[torch.nn.Module] = []  # the model, seen in its first train() call

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
                try:
                    super().load_state_dict(state_dict)
                except ValueError as e:  # saved with other groups (before by_network)
                    print(f"[studio] Optimizer state not resumed ({e}); starting it afresh.", file=sys.stderr)
                    return  # the averaged weights train on; the state builds up again
                # The model was just loaded with the checkpoint's averaged weights.
                for group in self.param_groups:
                    group["train_mode"] = False
                self.train()

        def make(params, lr, weight_decay=0.0, **_):
            if models:
                params = by_network(models[-1], params)
                sizes = {g["name"]: len(g["params"]) for g in params}
                print(f"[studio] Parameter groups: {sizes}", file=sys.stderr)
            # A shorter averaging window than the package default (schedulefree_c 0)
            # and beta1 0.95, as its docs suggest for Schedule-Free.
            opt = ScheduleFree(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.95, 0.99),
                schedulefree_c=20,
                prodigy_steps=prodigy_steps,
            )
            active.append(opt)
            return opt

        class Constant(torch.optim.lr_scheduler.LambdaLR):
            def load_state_dict(self, state_dict):
                pass  # nothing to resume (a checkpoint's may hold other groups)

        def constant(optimizer, **_):
            return Constant(optimizer, lambda _: 1.0)

        transformers.get_cosine_schedule_with_warmup = constant

        for cls in (v2.VoxCPM2Model, v1.VoxCPMModel):

            def train(self, mode=True, _train=cls.train):
                if self not in models:
                    models.append(self)
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

    class Rate(float):
        """Printed in scientific notation: the tracker's fixed 6 decimals would
        round Prodigy's small early rates to 0."""

        def __format__(self, spec):
            return format(float(self), ".4e")

    def log_with_effective_lr(self, metrics, *args, **kwargs):
        if active and "lr" in metrics:
            group = active[0].param_groups[0]
            metrics = {**metrics, "lr": float(group.get("d", 1.0)) * float(group["lr"])}
            for g in active[0].param_groups:
                if "name" in g:
                    d = float(g.get("d", 1.0))
                    metrics[f"lr_{g['name']}"] = d * float(g["lr"])
                    if "effective_lr" in g:  # Schedule-Free: the step of the training weights
                        metrics[f"elr_{g['name']}"] = d * float(g["effective_lr"])
            rates: dict[str, float] = {
                k: Rate(v) if k == "lr" or k.startswith(("lr_", "elr_")) else v for k, v in metrics.items()
            }
            metrics = rates
        return log_metrics(self, metrics, *args, **kwargs)

    TrainingTracker.log_metrics = log_with_effective_lr
    print(f"[studio] Optimizer: {name}.", file=sys.stderr)


# Config keys for use_optimizer and the studio (the run's engine), not the script.
OPTIMIZER_KEYS = ("optimizer", "prodigy_steps", "engine")


def optimizer_from_config(config_path: Path) -> tuple[str, int]:
    """The run config's ``optimizer`` (adamw if absent) and ``prodigy_steps``. The
    script never sees these keys: its train() takes no such arguments."""
    import yaml
    from voxcpm.training import config as config_module

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    load_yaml_config = config_module.load_yaml_config

    def without_optimizer(*args, **kwargs):
        data = load_yaml_config(*args, **kwargs)
        for key in OPTIMIZER_KEYS:
            data.pop(key, None)
        return data

    config_module.load_yaml_config = without_optimizer
    return str(config.get("optimizer") or "adamw"), int(config.get("prodigy_steps") or 0)


def val_manifest_from_config(config_path: Path) -> Path | None:
    """The run's validation manifest (studio runs write absolute paths), if any."""
    import yaml

    value = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("val_manifest")
    return Path(value) if value and Path(value).is_absolute() else None


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

    save_samples_to(args.samples_dir, val_manifest_from_config(args.config_path))
    frozen_weights_in_bf16()
    cache_audio_latents()
    from polyvox import vram
    from polyvox.data import audio_io

    audio_io.patch_torchaudio()
    vram.apply_limit("cuda")  # POLYVOX_VRAM_GB, as in the server
    if os.environ.get("POLYVOX_TRAIN_CHECKPOINT") == "1" or vram.detect("cuda")[1].train_checkpointing:
        # Not fused too: recomputing compiled layers mismatches, and compiling the
        # checkpointed ones needs a C++ compiler on Windows (checked 2026-10-12).
        checkpoint_layers()
    elif optimize_enabled():
        fuse_layers()
    use_optimizer(*optimizer_from_config(args.config_path))
    threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid), daemon=True).start()
    sys.argv = [str(SCRIPT), "--config_path", str(args.config_path)]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
