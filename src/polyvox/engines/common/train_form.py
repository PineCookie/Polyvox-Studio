"""Training form fields shared by engines (Engine.train_form), and the sizing of LoRA runs.

A LoRA run is sized in epochs: steps, warmup and the checkpoint / validation / log intervals
follow from them and the dataset (epoch_plan). The defaults are an effective batch of 16 (the
GPU profile's micro-batch, accumulated) and Prodigy Schedule-Free at a multiplier of 1: VoxCPM's
fine-tuning guide uses batch 16 and max_batch_tokens 8192; its AdamW at 1e-4 for 1-3 epochs left
voices close to the base model, Prodigy-SF finds a larger step size itself.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

from ..base import Param

EFFECTIVE_BATCH = 16
WARMUP_SHARE = 0.05  # of the steps
CHECKPOINTS_PER_EPOCH = 2
DEFAULT_EPOCHS = 4
# Prodigy finds its own step size: learning_rate is a multiplier, 1 by its authors' advice.
OPTIMIZER_LR = {"adamw": 1e-4, "prodigy": 1.0, "prodigy_sf": 1.0}
OPTIMIZERS = tuple(OPTIMIZER_LR)
SCHEDULE = ("num_iters", "warmup_steps", "save_interval", "valid_interval", "log_interval")
WITHOUT_GPU_BATCH = 4  # the micro-batch before the GPU is known (a 16 GB card's)


def js_round(x: float) -> int:
    """Math.round (halves up), as the form computed these before."""
    return math.floor(x + 0.5)


def gpu_batch(profile, engine_id: str) -> int:
    """The micro-batch that fits ``profile``'s GPU for an engine (at most the effective batch)."""
    if profile is None:
        return WITHOUT_GPU_BATCH
    return max(1, min(profile.public()["train_batches"].get(engine_id, profile.train_batch), EFFECTIVE_BATCH))


def replace(form: tuple[Param, ...], name: str, **changes) -> tuple[Param, ...]:
    """``form`` with the field ``name`` changed."""
    return tuple(dataclasses.replace(p, **changes) if p.name == name else p for p in form)


def insert_after(form: tuple[Param, ...], name: str, *params: Param) -> tuple[Param, ...]:
    i = next(i for i, p in enumerate(form) if p.name == name)
    return (*form[: i + 1], *params, *form[i + 1 :])


def optimizer_rates(adamw_lr: float) -> dict[str, dict]:
    """Param.implies of the optimizer: the learning rate each starts at."""
    return {o: {"learning_rate": adamw_lr if o == "adamw" else lr} for o, lr in OPTIMIZER_LR.items()}


def lora_form(
    engine_id: str,
    profile=None,
    *,
    adamw_lr: float = OPTIMIZER_LR["adamw"],
    layers: tuple[str, ...] = (),
    token_budget: bool = False,
    workers: bool = False,
) -> tuple[Param, ...]:
    """The fields of a LoRA run sized in epochs. ``layers``: the networks LoRA can go on
    (TrainParams.enable_*); ``token_budget``: max_batch_tokens; ``workers``: data loader workers."""
    batch = gpu_batch(profile, engine_id)
    accum = max(1, js_round(EFFECTIVE_BATCH / batch))
    adv = "advanced"
    return (
        Param("epochs", "float", DEFAULT_EPOCHS, min=0.5, step=0.5, scope="plan"),
        Param("lora_rank", "int", 32, min=1, max=512, config="lora.r"),
        Param("lora_alpha", "float", 32, min=0.1, config="lora.alpha"),
        Param("optimizer", "choice", "prodigy_sf", choices=OPTIMIZERS, implies=optimizer_rates(adamw_lr),
              legacy="adamw"),  # fmt: skip
        Param("learning_rate", "float", 1.0, min=0, legacy=OPTIMIZER_LR["adamw"],
              labels=(({"optimizer": ("prodigy", "prodigy_sf")}, "lr_multiplier"),)),  # fmt: skip
        Param("num_iters", "int", 200, min=1, group=adv),
        Param("batch_size", "int", batch, min=1, max=64, group=adv, legacy=1),
        Param("grad_accum_steps", "int", accum, min=1, max=64, group=adv, legacy=1),
        Param("warmup_steps", "int", 10, min=0, group=adv, shown_if={"optimizer": ("adamw", "prodigy")}),
        Param("prodigy_steps", "int", 0, min=0, group=adv, shown_if={"optimizer": ("prodigy_sf",)}, legacy=0),
        *([Param("max_batch_tokens", "int", 8192, min=0, step=1024, group=adv, legacy=0)] if token_budget else []),
        Param("weight_decay", "float", 0.01, min=0, group=adv),
        Param("max_grad_norm", "float", 1.0, min=0, group=adv),
        Param("lora_dropout", "float", 0.0, min=0, max=0.99, group=adv, config="lora.dropout"),
        Param("save_interval", "int", 50, min=1, group=adv),
        Param("valid_interval", "int", 50, min=1, group=adv),
        Param("log_interval", "int", 5, min=1, group=adv),
        *([Param("num_workers", "int", 2, min=0, max=16, group=adv)] if workers else []),
        *(Param(f"enable_{layer}", "bool", layer != "proj", group=adv, config=f"lora.enable_{layer}") for layer in layers),
    )  # fmt: skip


def full_form(
    profile, engine_id: str, *fields: Param, extra: tuple[Param, ...] = (), learning_rate: float = 1e-4
) -> tuple[Param, ...]:
    """The fields of a full fine-tuning run: its sizing ``fields``, the learning rate and batch
    (no accumulation), then ``extra``. Its steps follow from its own epochs (Engine.plan), shown
    but not set by hand."""
    return (
        *fields,
        Param("learning_rate", "float", learning_rate, min=0, legacy=OPTIMIZER_LR["adamw"]),
        Param("batch_size", "int", gpu_batch(profile, engine_id), min=1, max=64, legacy=1),
        *extra,
        Param("num_iters", "int", 200, min=1, group="advanced", disabled="stepsFollowEpochs"),
        Param("log_interval", "int", 5, min=1, group="advanced"),
    )


def derived(values: dict, schedule: dict, overrides: Iterable[str]) -> tuple[dict, dict]:
    """(the schedule fields not set by hand, the values with them)."""
    keep = set(overrides)
    own = {k: v for k, v in schedule.items() if k not in keep}
    return own, {**values, **own}


def epoch_plan(values: dict, clips: int, overrides: Iterable[str], tag: str) -> dict:
    """Steps for ``values["epochs"]`` passes over ``clips`` at the effective batch, and the
    warmup and intervals that go with them (Engine.plan of a LoRA engine)."""
    batch, accum = int(values.get("batch_size") or 1), int(values.get("grad_accum_steps") or 1)
    per_step = max(1, batch * accum)
    per_epoch = max(1.0, clips / per_step)
    epochs = values.get("epochs")
    schedule = {}
    if clips > 0 and isinstance(epochs, int | float) and math.isfinite(epochs):
        steps = max(1, math.ceil(epochs * per_epoch))
        interval = max(1, js_round(per_epoch / CHECKPOINTS_PER_EPOCH))
        schedule = {
            "num_iters": steps,
            "warmup_steps": max(1, js_round(steps * WARMUP_SHARE)),
            "save_interval": interval,
            "valid_interval": interval,
            "log_interval": max(1, min(10, js_round(per_epoch / 10))),
        }
    own, final = derived(values, schedule, overrides)
    return {
        "values": own,
        "summary": {
            "kind": "epochs",
            "steps": final.get("num_iters"),
            "per_step": per_step,
            "batch": batch,
            "accum": accum,
            # The passes the steps make (≠ the target when the steps were set by hand).
            "epochs": final["num_iters"] * per_step / clips if clips > 0 and final.get("num_iters") else None,
            "save_interval": final.get("save_interval"),
            "warmup_steps": final.get("warmup_steps"),
            "model": "lora",
            "rank": values.get("lora_rank"),
            "alpha": values.get("lora_alpha"),
        },
        "name_tag": f"{tag}-r{values.get('lora_rank')}",
        "samples": {"every": final.get("valid_interval"), "unit": "steps"},
    }
