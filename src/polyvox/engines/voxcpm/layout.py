"""VoxCPM's LoRA layouts (voxcpm's LoRAConfig): which layers a LoRA adapts and at what rank. The
model is built once with a layout covering every checkpoint, so a smaller LoRA loads zero-padded
(see inference)."""

from __future__ import annotations

from voxcpm.model.voxcpm import LoRAConfig

from ...lora_library import LoraCheckpoint


def default_config() -> LoRAConfig:
    """Layout used when a checkpoint has no saved one."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def config_of(ckpt: LoraCheckpoint) -> LoRAConfig:
    """A checkpoint's layout (its lora_config.json), else the default."""
    return LoRAConfig(**ckpt.raw_config) if ckpt.raw_config else default_config()


def layers(raw: dict) -> list[str]:
    """The parts a saved layout adapts (Engine.lora_layers)."""
    config = LoRAConfig(**raw)
    return [
        name for name, on in (("lm", config.enable_lm), ("dit", config.enable_dit), ("proj", config.enable_proj)) if on
    ]


def _union(lists) -> list[str]:
    out: list[str] = []
    for names in lists:
        out += [n for n in names if n not in out]
    return out


def covering(configs: list[LoRAConfig]) -> LoRAConfig:
    """The smallest LoRA layout any of ``configs`` fits into: highest rank,
    union of target layers. A smaller LoRA loads zero-padded (see inference)."""
    configs = configs or [default_config()]
    r = max(c.r for c in configs)
    return LoRAConfig(
        enable_lm=any(c.enable_lm for c in configs),
        enable_dit=any(c.enable_dit for c in configs),
        enable_proj=any(c.enable_proj for c in configs),
        r=r,
        alpha=r,  # unused: each loaded LoRA sets its own alpha / r
        dropout=0.0,
        target_modules_lm=_union(c.target_modules_lm for c in configs if c.enable_lm),
        target_modules_dit=_union(c.target_modules_dit for c in configs if c.enable_dit),
        target_proj_modules=_union(c.target_proj_modules for c in configs if c.enable_proj),
    )


def covers(layout: LoRAConfig, config: LoRAConfig) -> bool:
    """Whether a model built with ``layout`` can hold a LoRA with ``config``."""
    return (
        layout.r >= config.r
        and (layout.enable_lm or not config.enable_lm)
        and (layout.enable_dit or not config.enable_dit)
        and (layout.enable_proj or not config.enable_proj)
        and (not config.enable_lm or set(config.target_modules_lm) <= set(layout.target_modules_lm))
        and (not config.enable_dit or set(config.target_modules_dit) <= set(layout.target_modules_dit))
        and (not config.enable_proj or set(config.target_proj_modules) <= set(layout.target_proj_modules))
    )
