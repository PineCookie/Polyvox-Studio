"""Helpers of the training workers (subprocesses the studio starts; see training.py): the
``[studio]`` log lines, watching for the stop request, optimizers as VoxCPM's worker chooses
them, the learning rates and metrics they log, and shuffled batches."""

from __future__ import annotations

import contextlib
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
