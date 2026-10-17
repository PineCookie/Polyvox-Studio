"""VoxCPM (2 and 1.5) as an engine: inference.py's service and worker.py's
hooks around the vendored training script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ...config import Settings
from ...data import manifests
from ..base import Capabilities, Engine, Param, ReferenceInput, SynthesisRequest, SynthesisResult
from .inference import InferenceService
from .inference import SynthesisRequest as VoxRequest

if TYPE_CHECKING:
    from ...training import TrainParams

LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]


def resolve_model(model: str) -> tuple[str, int, int]:
    """Return (local path, encoder sample rate, decoder output rate) for a model."""
    from voxcpm.core import resolve_model_path

    path = Path(resolve_model_path(model)).resolve()
    vae = json.loads((path / "config.json").read_text(encoding="utf-8")).get("audio_vae_config") or {}
    return str(path), int(vae.get("sample_rate", 16_000)), int(vae.get("out_sample_rate") or 0)


def build_config(
    params: TrainParams,
    run_dir: Path,
    model: tuple[str, int, int],
    train_manifest: Path,
    val_manifest: Path | None,
) -> dict:
    pretrained_path, sample_rate, out_sample_rate = model
    return {
        "pretrained_path": pretrained_path,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_manifest else "",
        "sample_rate": sample_rate,
        "out_sample_rate": out_sample_rate,
        "batch_size": params.batch_size,
        "max_batch_tokens": params.max_batch_tokens,
        "grad_accum_steps": params.grad_accum_steps,
        "num_workers": params.num_workers,
        "num_iters": params.num_iters,
        "log_interval": params.log_interval,
        "valid_interval": params.valid_interval,
        "save_interval": params.save_interval,
        "learning_rate": params.learning_rate,
        # Read (and removed) by worker.py.
        "optimizer": params.optimizer,
        "prodigy_steps": params.prodigy_steps,
        "weight_decay": params.weight_decay,
        "warmup_steps": params.warmup_steps,
        "max_steps": params.num_iters,
        "max_grad_norm": params.max_grad_norm,
        "save_path": str(run_dir / "checkpoints"),
        "tensorboard": str(run_dir / "logs"),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": params.enable_lm,
            "enable_dit": params.enable_dit,
            "enable_proj": params.enable_proj,
            "r": params.lora_rank,
            "alpha": params.lora_alpha,
            "dropout": params.lora_dropout,
            "target_modules_lm": LORA_TARGETS,
            "target_modules_dit": LORA_TARGETS,
        },
    }


class VoxCPMEngine(Engine):
    id = "voxcpm"
    name = "VoxCPM"
    capabilities = Capabilities(
        sample_rate=48_000,
        training_rate=16_000,
        languages=("zh", "en", "ja", "ko"),
        reference_audio=True,
        reference_transcript=True,
        style=True,
        tags=True,
        lora_hot_swap=True,
        reference_inputs=(
            # reference_wav_path: who it sounds like.
            ReferenceInput("reference", "reference", "none"),
            # prompt_wav_path + prompt_text: speech it continues (with the base model,
            # the reference's own audio here too clones best).
            ReferenceInput("prompt", "prompt", "required", same_as="reference"),
        ),
    )
    params = (
        Param("cfg_value", "float", 2.0, 0.5, 5.0, 0.1),
        Param("inference_timesteps", "int", 10, 1, 50, 1),
        Param("normalize", "bool", True),
        Param("denoise", "bool", True),
    )
    model_setting = "model"
    worker_module = "polyvox.engines.voxcpm.worker"

    def __init__(self, settings: Settings, service: InferenceService | None = None):
        self.settings = settings
        self._service = service

    @property
    def service(self) -> InferenceService:
        if self._service is None:
            self._service = InferenceService(self.settings)
        return self._service

    @property
    def device(self) -> str:
        return self.service.device

    @property
    def loaded(self) -> bool:
        return self._service is not None and self._service.status()["loaded"]

    def status(self) -> dict:
        return self.service.status()

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        options = self.options(req.options)
        return self.service.synthesize(
            VoxRequest(
                text=req.text,
                control=req.style,
                lora_id=req.lora_id,
                reference_wav=req.reference_wav,
                prompt_text=req.prompt_text,
                prompt_wav=req.prompt_wav,
                seed=req.seed,
                **options,
            )
        )

    def progress(self) -> dict:
        return self.service.progress_public()

    def unload(self) -> None:
        if self._service is not None:
            self._service.unload()

    def training_config(
        self, params: TrainParams, run_dir: Path, train_manifest: Path, val_manifest: Path | None
    ) -> dict:
        model = resolve_model(self.settings.model_path(self.model_setting))
        return build_config(params, run_dir, model, train_manifest, val_manifest)

    def adopt(self, run_dir: Path, config: dict) -> dict:
        """Make a run's config self-contained: absolute paths, own manifest copies.

        Webui configs hold paths relative to the VoxCPM checkout and manifests
        whose audio paths the current loader would misread.
        """
        root = manifests.legacy_root(self.settings.data_dir)

        def absolute(value) -> Path:
            path = manifests.local_path(str(value))
            return path if path.is_absolute() else root / path

        config["pretrained_path"] = str(absolute(config["pretrained_path"]))
        config["save_path"] = str(run_dir / "checkpoints")
        config["tensorboard"] = (
            str(absolute(config["tensorboard"])) if config.get("tensorboard") else str(run_dir / "logs")
        )
        for key, file in (("train_manifest", "train.jsonl"), ("val_manifest", "val.jsonl")):
            if not config.get(key):
                continue
            source, local = absolute(config[key]), run_dir / "manifests" / file
            if source.resolve() == local.resolve():
                continue
            if not source.is_file():
                raise ValueError(f"Manifest not found: {source}")
            entries, _ = manifests.read_manifest(source, root)
            manifests.write_manifest(entries, local)
            config[key] = str(local)
        return config
