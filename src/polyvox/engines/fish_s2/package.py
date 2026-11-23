"""An S2-Pro LoRA as a shareable package (common/lora_package.py), with the voice's default
reference (reference.wav + its transcript)."""

from __future__ import annotations

import json
from pathlib import Path

from ...lora_library import LoraCheckpoint
from ..common import lora_package
from ..common.lora_package import number
from ..common.reference import REFERENCE_JSON, REFERENCE_WAV, read_reference
from . import lora as s2lora


def config(raw: dict | None) -> dict:
    """The lora_config fields its loader reads, checked."""
    r, alpha = lora_package.rank_alpha(raw)
    raw = raw or {}
    targets = raw.get("targets") or list(s2lora.DEFAULT_TARGETS)
    if not isinstance(targets, list) or not set(targets) <= set(s2lora.TARGETS):
        raise ValueError("unknown S2-Pro LoRA targets.")
    return {"r": r, "alpha": alpha, "dropout": float(raw.get("dropout") or 0.0), "targets": targets}


def files(ckpt: LoraCheckpoint, name: str, base_model: str, half: bool) -> dict[str, bytes]:
    lora = config(ckpt.raw_config)
    extra = {}
    if reference := read_reference(ckpt.path):  # the transcript only: where it came from stays here
        extra[REFERENCE_WAV] = reference[0].read_bytes()
        extra[REFERENCE_JSON] = (json.dumps({"text": reference[1]}, ensure_ascii=False, indent=1) + "\n").encode()
    text = readme(name, base_model, lora, ckpt.step, half, bool(extra))
    return lora_package.files(ckpt, base_model, half, lora, text, extra)


def unpack(folder: Path, name: str, files: dict[str, bytes]) -> str:
    return lora_package.unpack(folder, name, files, s2lora.ENGINE, config, keep=(REFERENCE_WAV, REFERENCE_JSON))


def readme(name: str, base_model: str, config: dict, step: int | None, half: bool, has_reference: bool) -> str:
    reference = (
        "`reference.wav` is the voice's default reference (a few of its clips joined); `reference.json` holds its\n"
        "transcript. S2-Pro speaks with it unless you give another clip: the LoRA refines the sound, the\n"
        "reference still sets much of the voice."
        if has_reference
        else "It comes without a reference clip: give one of the voice, with its transcript, when synthesizing."
    )
    return f"""# {name}

A LoRA for [Fish Audio S2-Pro](https://huggingface.co/{base_model}): its fast transformer
(the 4 layers that predict each audio frame's residual codes), trained on top of
`{base_model}`. It only works with that base model, whose weights are under the Fish Audio
Research License (research and non-commercial use).

| | |
|---|---|
| Base model | `{base_model}` |
| Rank / alpha | {config["r"]} / {number(config["alpha"])} |
| Layers | attention (wqkv, wo) and feed-forward (w1, w2, w3) of the fast transformer |
| Training steps | {step if step is not None else "unknown"} |
| Weights | {lora_package.dtype(half)} |

{reference}

## Use

**Polyvox Studio:** Voice panel -> Import, then choose this zip. It appears as the
S2-Pro voice "{name}", with its reference.

**Other code:** each layer `<name>` (e.g. `audio_decoder.layers.0.attention.wqkv` in the
checkpoint's naming is `fast_layers.0.attention.wqkv` here) has `<name>.lora_A`
(rank x in) and `<name>.lora_B` (out x rank); add `alpha / rank * B @ A` to its weight.
Training heard every clip with a faint noise floor (white noise at -72 dBFS); encode
references the same way.
"""
