"""Runtime settings, read from ``VOXCPM_STUDIO_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import models

ENV_PREFIX = "VOXCPM_STUDIO_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Local model directory or Hugging Face repo id.
    model: str = "openbmb/VoxCPM2"
    lora_dir: Path = Path("lora")
    # Where JSONL training manifests are found.
    data_dir: Path = Path("data")
    device: str = "auto"
    # None = compile with Triton whenever running on CUDA.
    optimize: bool | None = None
    load_denoiser: bool = True
    # VoxCPM's reference-audio denoiser (ZipEnhancer): local dir or ModelScope id.
    denoiser_model: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"
    # CUDA device index for training jobs; None = the default device.
    train_gpu: int | None = None
    # Speech recognition (transformers-native Qwen3-ASR): local dir or Hugging Face id.
    asr_model: str = "Qwen/Qwen3-ASR-1.7B-hf"
    # "auto" = the inference device; forced aligners run there too.
    asr_device: str = "auto"
    # Forced aligners (aligner.py): Qwen3-ForcedAligner, and a wav2vec2 CTC model
    # with a character vocabulary for the transcripts' language.
    align_model: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    ctc_align_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    # RoFormer checkpoints (separation.py) for the import's optional clean-up:
    # "model.ckpt" (config: same-stem or only .yaml beside it) or "model.ckpt|config.yaml".
    # Empty = that option is unavailable.
    # AudioSet tagger for singing detection (singing.py): local dir or Hugging Face id.
    singing_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    vocals_model: str = ""
    dereverb_model: str = ""
    denoise_model: str = ""
    # Downloaded models (models.py) live here; a model setting left at its
    # default uses the copy here once it is complete.
    models_dir: Path = Path("models")
    # Hugging Face or a mirror of it (e.g. https://hf-mirror.com) for downloads;
    # "" = huggingface.co.
    hf_endpoint: str = ""

    def configured(self, setting: str) -> bool:
        """The setting was changed from its default (e.g. by an environment variable)."""
        return getattr(self, setting) != Settings.__dataclass_fields__[setting].default

    def model_path(self, setting: str) -> str:
        """The model a setting names: as configured when set explicitly, else the
        downloaded copy in models_dir, else the default (a Hugging Face id that
        loading downloads to its cache; "" = not available)."""
        value = getattr(self, setting)
        spec = models.BY_SETTING.get(setting)
        if spec is None or self.configured(setting):
            return value
        return str(spec.path(self.models_dir).resolve()) if spec.downloaded(self.models_dir) else value

    @classmethod
    def from_env(cls) -> Settings:
        optimize = _env("OPTIMIZE", "auto")
        train_gpu = _env("TRAIN_GPU", "").strip()
        return cls(
            model=_env("MODEL", cls.model),
            lora_dir=Path(_env("LORA_DIR", str(cls.lora_dir))),
            data_dir=Path(_env("DATA_DIR", str(cls.data_dir))),
            device=_env("DEVICE", cls.device),
            optimize=None if optimize == "auto" else _parse_bool(optimize),
            load_denoiser=_parse_bool(_env("DENOISER", "1")),
            denoiser_model=_env("DENOISER_MODEL", cls.denoiser_model),
            train_gpu=int(train_gpu) if train_gpu else None,
            asr_model=_env("ASR_MODEL", cls.asr_model),
            asr_device=_env("ASR_DEVICE", cls.asr_device),
            align_model=_env("ALIGN_MODEL", cls.align_model),
            ctc_align_model=_env("CTC_ALIGN_MODEL", cls.ctc_align_model),
            singing_model=_env("SINGING_MODEL", cls.singing_model),
            vocals_model=_env("VOCALS_MODEL", cls.vocals_model),
            dereverb_model=_env("DEREVERB_MODEL", cls.dereverb_model),
            denoise_model=_env("DENOISE_MODEL", cls.denoise_model),
            models_dir=Path(_env("MODELS_DIR", str(cls.models_dir))),
            hf_endpoint=_env("HF_ENDPOINT", cls.hf_endpoint).rstrip("/"),
        )
