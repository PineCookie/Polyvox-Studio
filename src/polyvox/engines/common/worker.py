"""What the training workers (subprocesses the studio starts; see training.py) share: a run's
config, folders, log lines, resume state and samples (Run); the update loop of LoRA runs
(LoraTraining); watching for the stop request; optimizers as VoxCPM's worker chooses them, the
learning rates and metrics they log; shuffled batches."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

POLL_S = 1.0


def log(message: str) -> None:
    print(f"[studio] {message}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def replacing(path: Path):
    """A temporary path beside ``path`` to write to; it becomes ``path`` once written, so a stop or
    crash midway leaves the old file (a run stays resumable), never half of a new one."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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


def wait_for_stop(stop_file: Path, parent_pid: int | None) -> None:
    """Return once ``stop_file`` exists or the server process is gone."""
    while not stop_file.exists():
        if parent_pid is not None and not process_alive(parent_pid):
            log("Server exited.")
            break
        time.sleep(POLL_S)
    log("Stop requested; saving a checkpoint.")


def watch_for_stop(stop_file: Path, parent_pid: int | None, stop: threading.Event) -> None:
    """wait_for_stop, then set ``stop`` (the training loop saves and ends)."""
    wait_for_stop(stop_file, parent_pid)
    stop.set()


def make_optimizer(name: str, groups: list[dict], config: dict, num_iters: int):
    """(optimizer, scheduler or None) as engines.voxcpm.worker.use_optimizer chooses them."""
    import torch

    lr, wd = float(config["learning_rate"]), float(config["weight_decay"])
    if name == "prodigy_sf":
        from prodigyplus import ProdigyPlusScheduleFree

        opt = ProdigyPlusScheduleFree(
            groups,
            lr=lr,
            weight_decay=wd,
            betas=(0.95, 0.99),
            schedulefree_c=20,
            prodigy_steps=int(config.get("prodigy_steps") or 0),
        )
        return opt, None
    if name == "prodigy":
        from prodigyopt import Prodigy

        opt = Prodigy(groups, lr=lr, weight_decay=wd, safeguard_warmup=True, use_bias_correction=True)
    elif name == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    warmup = int(config.get("warmup_steps") or 0)

    def cosine(step: int) -> float:  # transformers' cosine schedule with warmup
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(1, num_iters - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, cosine)


def rates(opt) -> dict[str, float]:
    """lr = d x lr for Prodigy (per group lr_<net>, elr_<net> for Schedule-Free), as VoxCPM's worker logs."""
    out: dict[str, float] = {}
    for i, g in enumerate(opt.param_groups):
        d = float(g.get("d", 1.0))
        if i == 0:
            out["lr"] = d * float(g["lr"])
        if "d" in g:
            out[f"lr_{g['name']}"] = d * float(g["lr"])
            if "effective_lr" in g:
                out[f"elr_{g['name']}"] = d * float(g["effective_lr"])
    return out


def format_metrics(values: dict[str, float]) -> str:
    def fmt(k: str, v: float) -> str:
        if k == "lr" or k.startswith(("lr_", "elr_")):
            return f"{k}: {v:.4e}"
        if k == "log interval":
            return f"{k}: {v:.2f}s"
        return f"{k}: {v:.6f}"

    return ", ".join(fmt(k, v) for k, v in values.items())


class Run:
    """A training run as its worker sees it (``python -m <worker> <train_config.yaml> --samples-dir
    <dir> --stop-file <path> [--parent-pid <pid>]``): the config, the stop request, the folders,
    the ``[train]`` / ``[val]`` lines of checkpoints/train.log, the resume state in latest/, the
    samples' WAVs. Every worker keeps this contract, so the studio shows runs alike."""

    def __init__(self, description: str):
        import argparse

        import yaml

        from ... import vram

        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("config_path", type=Path)
        parser.add_argument("--samples-dir", type=Path, required=True)
        parser.add_argument("--stop-file", type=Path, required=True)
        parser.add_argument("--parent-pid", type=int)
        self.args = parser.parse_args()
        self.config_path, self.samples_dir = self.args.config_path, self.args.samples_dir
        self.config: dict = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        vram.apply_limit("cuda")
        self.stop = threading.Event()
        watch = (self.args.stop_file, self.args.parent_pid, self.stop)
        threading.Thread(target=watch_for_stop, args=watch, daemon=True).start()
        self.save_dir = Path(self.config["save_path"])  # <run>/checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.save_dir.parent
        self.latest = self.save_dir / "latest"
        self.log_interval = int(self.config.get("log_interval") or 10)
        self._window: dict[str, list[tuple]] = {}
        self._t_log = time.time()

    @property
    def num_iters(self) -> int:
        return int(self.config["num_iters"])

    def records(self, manifest: str) -> list[dict]:
        """The clips of the config's ``manifest`` ("train_manifest" / "val_manifest"; none: [])."""
        from ...data import manifests

        path = Path(self.config[manifest]) if self.config.get(manifest) else None
        return [dict(e) for e in manifests.read_manifest(path)[0]] if path and path.is_file() else []

    def write_config(self, **values) -> None:
        """``values`` into the run's config (e.g. the exact steps once the clips are known: the
        server shows them)."""
        import yaml

        self.config.update(values)
        with replacing(self.config_path) as tmp:
            tmp.write_text(yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # ------------------------------------------------------------ log lines
    def record(self, step: int, values: dict, extra: dict[str, float], last: bool) -> None:
        """A step's values: numbers, GPU tensors (read back only when a line is written) or
        (sum, count) pairs. Every log_interval steps and the last one, a ``[train]`` line with
        their means over the steps since (values that aren't finite, from steps a gradient scaler
        skipped, left out), then ``extra`` as is."""
        for k, v in values.items():
            self._window.setdefault(k, []).append(v if isinstance(v, tuple) else (v, 1))
        if step % self.log_interval and not last:
            return
        now = time.time()
        means = {}
        for k, pairs in self._window.items():
            kept = [(float(s), float(n)) for s, n in pairs]
            kept = [(s, n) for s, n in kept if math.isfinite(s)]
            if kept:
                means[k] = sum(s for s, _ in kept) / max(sum(n for _, n in kept), 1.0)
        means |= extra
        means["log interval"] = now - self._t_log  # since the last line (training.seconds_per_step)
        line = f"[train] step {step}: {format_metrics(means)}"
        self._append(line)
        print(line, flush=True)
        self._t_log, self._window = now, {}

    def reset_log(self) -> None:
        self._window, self._t_log = {}, time.time()

    def validation(self, step: int, values: dict[str, float]) -> None:
        self._append(f"[val] step {step}: {format_metrics(values)}")

    def _append(self, line: str) -> None:
        with (self.save_dir / "train.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------ resume state and samples
    def save_state(self, step: int, state: dict) -> None:
        """latest/training_state.pt (what resuming needs) and .json (its step), atomically."""
        import torch

        self.latest.mkdir(parents=True, exist_ok=True)
        with replacing(self.latest / "training_state.pt") as tmp:
            torch.save({"step": step, **state}, tmp)
        (self.latest / "training_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
        log(f"Saved step {step}.")

    def load_state(self, weights_only: bool = True) -> dict | None:
        """The state a stopped run saved (None: a new run). ``weights_only`` False for optimizer
        states holding more than tensors (Prodigy's)."""
        import torch

        path = self.latest / "training_state.pt"
        return torch.load(path, map_location="cpu", weights_only=weights_only) if path.is_file() else None

    def write_wav(self, name: str, audio, rate: int) -> None:
        """samples/<name>, atomically."""
        import soundfile as sf

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        with replacing(self.samples_dir / name) as tmp:
            sf.write(tmp, audio, rate, format="WAV")

    def reference(self, i: int, audio: Path | str) -> None:
        """samples/reference_<i>.wav: the validation clip sample ``i`` speaks (once)."""
        from ...data import audio_io

        if not (self.samples_dir / f"reference_{i}.wav").exists():
            samples, rate = audio_io.read(audio)
            self.write_wav(f"reference_{i}.wav", samples.T, rate)

    def finish(self, step: int, peaks: dict[str, float] | None = None) -> None:
        """The last lines: peak GPU memory (``peaks`` by phase, else the whole run's), and how it ended."""
        import torch

        if peaks:
            detail = ", ".join(f"{phase} {gb:.1f}" for phase, gb in peaks.items() if gb)
            log(f"Peak GPU memory {max(peaks.values()):.1f} GB ({detail}).")
        else:
            log(f"Peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GB.")
        log("Stopped." if self.stop.is_set() and step < self.num_iters else "Training complete.")


class LoraTraining:
    """The update loop of a LoRA run (dots.tts, S2-Pro): the optimizer of its parameter groups,
    resuming, Schedule-Free's averaged weights for checkpoints and validation, checkpoints
    (step_N/ and latest/, written by ``write_checkpoint(folder, weights)``) and the log /
    validation / checkpoint intervals. Steps count optimizer updates, 1..num_iters."""

    def __init__(self, run: Run, loras: dict, groups: list[dict], epoch_steps: float, write_checkpoint):
        config = run.config
        self.run, self.loras, self.epoch_steps, self.write_checkpoint = run, loras, epoch_steps, write_checkpoint
        self.optimizer = str(config.get("optimizer") or "prodigy_sf")
        self.opt, self.scheduler = make_optimizer(self.optimizer, groups, config, run.num_iters)
        self.schedule_free = hasattr(self.opt, "train") and hasattr(self.opt, "eval")
        self.params = [p for g in groups for p in g["params"]]
        self.max_grad_norm = float(config.get("max_grad_norm") or 1e9)
        self.valid_interval = int(config.get("valid_interval") or 0)
        self.save_interval = int(config.get("save_interval") or 0)

    def weights(self) -> dict:
        from .lora import state

        return state(self.loras)

    def resume(self) -> int:
        """The step to go on from (0: a new run), its weights and optimizer state loaded."""
        import torch

        step = 0
        if state := self.run.load_state(weights_only=False):
            with torch.no_grad():
                for name, lora in self.loras.items():
                    lora.A.copy_(state["train_weights"][f"{name}.lora_A"])
                    lora.B.copy_(state["train_weights"][f"{name}.lora_B"])
            self.opt.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler"):
                self.scheduler.load_state_dict(state["scheduler"])
            step = int(state["step"])
            log(f"Resumed at step {step}.")
        if self.schedule_free:
            self.opt.train()
        return step

    def averaged(self, fn):
        """``fn()`` with the weights to use (Schedule-Free: the average)."""
        if self.schedule_free:
            self.opt.eval()
        try:
            return fn()
        finally:
            if self.schedule_free:
                self.opt.train()

    def save(self, step: int) -> None:
        weights = self.averaged(self.weights)
        for folder in (self.run.save_dir / f"step_{step:07d}", self.run.latest):
            folder.mkdir(parents=True, exist_ok=True)
            self.write_checkpoint(folder, weights)
        scheduler = self.scheduler.state_dict() if self.scheduler is not None else None
        state = {"train_weights": self.weights(), "optimizer": self.opt.state_dict(), "scheduler": scheduler}
        self.run.save_state(step, state)

    def train(self, step: int, update, validate) -> int:
        """Updates until num_iters or a stop request; returns the last step. ``update()`` runs one
        update's forward and backward passes and returns its losses {name: (sum, count)};
        ``validate(step)`` at the intervals and at the end, then the last checkpoint."""
        import torch

        run, num_iters = self.run, self.run.num_iters
        while step < num_iters and not run.stop.is_set():
            losses = update()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm))
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.opt.zero_grad(set_to_none=True)
            step += 1
            extra = rates(self.opt) | {"epoch": step / self.epoch_steps, "grad_norm": grad_norm}
            run.record(step, losses, extra, step == num_iters)
            if self.valid_interval and step % self.valid_interval == 0 and step < num_iters:
                validate(step)
            if self.save_interval and step % self.save_interval == 0 and step < num_iters:
                self.save(step)
        if step >= num_iters:
            validate(step)
        if step > 0:
            self.save(step)
        return step


class Batches:
    """Shuffled batches of encoded clips; one pass = one epoch."""

    def __init__(self, clips: list, batch_size: int, seed: int):
        self.clips, self.batch_size = clips, batch_size
        self.per_epoch = max(1, len(clips) // batch_size)
        self._rng = random.Random(seed)
        self._order: list[int] = []

    def next(self) -> list:
        if len(self._order) < self.batch_size:
            self._order = list(range(len(self.clips)))
            self._rng.shuffle(self._order)
        picked, self._order = self._order[: self.batch_size], self._order[self.batch_size :]
        return [self.clips[i] for i in picked]
