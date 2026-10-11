"""Settings that depend on how much memory the GPU has.

Batch sizes and which models may share the GPU are picked from its total
memory (a profile), so a 8 or 12 GB card doesn't run out where a 16 GB one
doesn't. Batched work also backs off by itself when it runs out anyway
(``is_oom``). POLYVOX_VRAM_GB overrides the detected size (a GPU shared
with other programs, or testing the profiles).

Measured peaks (VoxCPM2, Qwen3-ASR 1.7B; CUDA context ~0.5 GB extra): ASR model
3.8 GB, +~0.9 GB per 100 s of audio in a batch; TTS 5.4 GB resident, ~7.4 GB
while generating, ~10.5 GB with the reference denoiser on the GPU; separation
3.9 GB (2 chunks); aligner 3.1 GB (160 s); training batch 1 / 2 / 4: ~7 / 8.2 /
10.9 GB. dots.tts training (clips up to ~17 s): batch 4 13.5 GB; with gradient
checkpointing 4 / 8: 6.9 / 7.7 GB, ~1.9x / 1.3x the time per clip.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    name: str
    asr_batch_seconds: float  # padded audio per ASR batch
    align_batch_seconds: float  # padded audio per aligner batch
    separation_batch: int  # chunks per separation batch
    exclusive: bool  # one model group on the GPU at a time: TTS, or the preparation models
    swap_prep_models: bool  # also one preparation model at a time (the others wait in CPU memory)
    denoise_on_cpu: bool  # the reference denoiser (ZipEnhancer) runs on the CPU
    train_batch: int  # training micro-batch; accumulation makes up the rest of 16
    train_checkpointing: bool = False  # recompute layer activations in the backward pass (less memory, slower)
    note: str = ""  # shown to the user: what doesn't fit
    # dots.tts training: the micro-batch to use, and the largest one that trains
    # without gradient checkpointing (larger ones checkpoint: engines.dots.worker)
    dots_train_batch: int = 1
    dots_plain_batch: int = 0

    def public(self) -> dict:
        return {**asdict(self), "train_batches": {"voxcpm": self.train_batch, "dots_tts": self.dots_train_batch}}


PROFILES = (  # (at least this many GB, profile), largest first
    (15.0, Profile("16 GB or more", 400, 160, 2, False, False, False, 4, dots_train_batch=4, dots_plain_batch=4)),
    (11.0, Profile("12 GB", 200, 160, 2, True, False, False, 2, dots_train_batch=8, dots_plain_batch=2)),
    (7.0, Profile("8 GB", 60, 80, 1, True, True, True, 1, True, dots_train_batch=4)),
    (
        0.0,
        Profile(
            "under 8 GB",
            60,
            60,
            1,
            True,
            True,
            True,
            1,
            True,
            "Speech synthesis and training need about 8 GB of GPU memory; preparing datasets works.",
        ),
    ),
)
CPU = Profile(
    "CPU",
    400,
    160,
    2,
    False,
    False,
    False,
    1,
    note="No CUDA GPU: everything runs on the CPU (slow).",
    dots_plain_batch=64,
)


def profile_for(total_gb: float) -> Profile:
    return next(p for at_least, p in PROFILES if total_gb >= at_least)


def total_gb(device: str) -> float | None:
    """Memory of a CUDA device in GB (POLYVOX_VRAM_GB if set), None for
    other devices, or when that GPU isn't there (a CPU-only PyTorch, a wrong index)."""
    if override := os.environ.get("POLYVOX_VRAM_GB"):
        return float(override)
    if not device.startswith("cuda"):
        return None
    import torch

    index = int(device.split(":")[1]) if ":" in device else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return None
    return torch.cuda.get_device_properties(index).total_memory / 2**30


CONTEXT_GB = 0.5  # CUDA context and libraries, outside PyTorch's allocator


def apply_limit(device: str) -> None:
    """With POLYVOX_VRAM_GB set, keep this process within that much of the
    GPU (minus the CUDA context): for a GPU shared with other programs, and to
    try the smaller profiles on a larger card."""
    override = os.environ.get("POLYVOX_VRAM_GB")
    if not override or not device.startswith("cuda"):
        return
    import torch

    index = int(device.split(":")[1]) if ":" in device else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return
    total = torch.cuda.get_device_properties(index).total_memory / 2**30
    fraction = min(1.0, max(0.05, (float(override) - CONTEXT_GB) / total))
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    logger.info("GPU %d limited to %.1f of %.1f GB (POLYVOX_VRAM_GB)", index, fraction * total, total)


def detect(device: str) -> tuple[float | None, Profile]:
    gb = total_gb(device)
    return gb, CPU if gb is None else profile_for(gb)


def is_oom(error: BaseException) -> bool:
    """A CUDA out-of-memory error (also when a library wrapped it)."""
    import torch

    for _ in range(10):  # the chain of causes, not forever
        if error is None:
            return False
        if isinstance(error, torch.OutOfMemoryError) or "CUDA out of memory" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def in_batches(
    order: list[int],
    plan: Callable[[list[int], float], list[int]],
    run: Callable[[list[int]], None],
    limit: float,
    what: str,
) -> float:
    """Run the items in ``order`` through ``run`` a batch at a time, each batch
    being ``plan(remaining items, limit)`` (at least one item). When a batch of
    several items runs out of GPU memory, the limit is halved and the batch
    retried. Returns the limit that worked, for the caller to start from next time."""
    pending = list(order)
    while pending:
        batch = plan(pending, limit)
        try:
            run(batch)
        except Exception as e:
            if len(batch) == 1 or not is_oom(e):
                raise
            free_cache()
            limit /= 2
            logger.warning("%s ran out of GPU memory; batches of up to %.0f from now on", what, limit)
            continue
        pending = pending[len(batch) :]
    return limit


def free_cache() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
