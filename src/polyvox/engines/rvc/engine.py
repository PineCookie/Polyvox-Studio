"""RVC as an engine: converts recorded speech or singing into a voice (Convert tab), no text."""

from __future__ import annotations

import gc
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ... import lora_library, models
from ...config import Settings
from ..base import Capabilities, ConvertRequest, Engine, Param, SynthesisRequest, SynthesisResult
from . import voices

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """The models every voice shares: ContentVec, RMVPE and the conversion pipeline on top."""

    contentvec: object
    rmvpe: object
    converter: object


RuntimeLoader = Callable[[str, str, str], Runtime]


def load_runtime(rvc_dir: str, contentvec_dir: str, device: str) -> Runtime:
    from .convert import ContentVec, Converter
    from .pitch import RMVPE

    contentvec = ContentVec(Path(contentvec_dir), device)
    rmvpe = RMVPE(Path(rvc_dir) / "rmvpe.pt", device)
    return Runtime(contentvec, rmvpe, Converter(contentvec, rmvpe, device))


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | preparing | generating
    step: int = 0  # pieces converted (about 38 s each)
    expected: int | None = None
    attempt: int = 0
    started: float = 0.0


class RvcEngine(Engine):
    id = voices.ENGINE
    name = "RVC"
    capabilities = Capabilities(
        sample_rate=40_000,  # each voice's own (32, 40 or 48 kHz); 40 kHz is RVC's usual
        training_rate=40_000,
        languages=(),  # any: it converts sound, not text
        reference_audio=False,
        reference_transcript=False,
        style=False,
        tags=False,
        lora_hot_swap=True,
        training=True,
        full_finetune=True,  # from RVC's base models: no LoRA or optimizer choice
        conversion=True,
        package="voice",  # the voice file + index (sharing.py)
    )
    params = (
        Param("shift", "int", 0, -24, 24, 1),  # semitones
        Param("index_rate", "float", 0.75, 0.0, 1.0, 0.05),
        Param("protect", "float", 0.33, 0.0, 0.5, 0.01),
        Param("clean_pitch", "bool", True),
        Param("rms_mix", "float", 1.0, 0.0, 1.0, 0.05),
    )
    model_setting = "rvc_model"
    worker_module = "polyvox.engines.rvc.worker"

    def __init__(self, settings: Settings, loader: RuntimeLoader = load_runtime):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._runtime: Runtime | None = None
        self._voice = None  # (key, model.Voice), key = (checkpoint id, weights mtime)
        self._index = None  # (key, convert.Index | None), key = (index path, mtime)
        self._progress = Progress()

    @property
    def device(self) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def unavailable(self) -> str | None:
        if not self.settings.model_path("rvc_model") or not self.settings.model_path("contentvec_model"):
            return "download its models first (Models: RVC v2 and ContentVec)."
        return None

    def status(self) -> dict:
        return {
            "device": self.device,
            "model": self.settings.model_path(self.model_setting),
            "loaded": self.loaded,
            "active_lora": self._voice[0][0] if self._voice else None,
        }

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        raise ValueError("RVC converts recordings (the Convert tab); it doesn't read text.")

    def _ensure_runtime(self) -> Runtime:
        if self._runtime is None:
            with models.loading(self.settings, "rvc_model", "contentvec_model"):
                rvc = self.settings.model_path("rvc_model")
                contentvec = self.settings.model_path("contentvec_model")
                logger.info("Loading RVC (RMVPE from %s, ContentVec from %s)", rvc, contentvec)
                self._runtime = self._loader(rvc, contentvec, self.device)
        return self._runtime

    def _voice_for(self, ckpt: lora_library.LoraCheckpoint):
        from . import model

        key = (ckpt.id, (ckpt.path / voices.WEIGHTS).stat().st_mtime)
        if not self._voice or self._voice[0] != key:
            self._voice = None
            self._voice = (key, model.load_voice(ckpt.path / voices.WEIGHTS, self.device))
        return self._voice[1]

    def _index_for(self, ckpt: lora_library.LoraCheckpoint):
        import torch

        from .convert import Index

        path = voices.index_file(ckpt.path, self.settings.lora_dir / ckpt.voice_path)
        key = (str(path), path.stat().st_mtime) if path else None
        if not self._index or self._index[0] != key:
            features = torch.load(path, map_location="cpu", weights_only=True) if path else None
            self._index = (key, Index(features.to(self.device)) if features is not None else None)
        return self._index[1]

    def checkpoint(self, lora_id: str) -> lora_library.LoraCheckpoint:
        ckpt = lora_library.resolve(self.settings.lora_dir, lora_id)
        if ckpt.engine != self.id or not (ckpt.path / voices.WEIGHTS).is_file():
            raise ValueError(f"'{lora_id}' isn't an RVC voice.")
        return ckpt

    def convert(self, req: ConvertRequest) -> SynthesisResult:
        import torch

        from .convert import Options

        if reason := self.unavailable():
            raise ValueError(f"RVC can't run: {reason}")
        if req.audio.ndim != 1 or not len(req.audio):
            raise ValueError("Give mono audio to convert.")
        ckpt = self.checkpoint(req.lora_id)
        o = self.options(req.options)
        with self._lock:
            self._progress = Progress(phase="loading" if self._runtime is None else "preparing", started=time.time())
            try:
                rt = self._ensure_runtime()
                voice = self._voice_for(ckpt)
                index = self._index_for(ckpt)
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                torch.manual_seed(seed)
                opts = Options(
                    shift=int(o["shift"]),
                    index_rate=float(o["index_rate"]) if index is not None else 0.0,
                    protect=float(o["protect"]),
                    rms_mix=float(o["rms_mix"]),
                    clean_pitch=bool(o["clean_pitch"]),
                )
                self._progress.phase = "generating"

                def on_piece(done: int, total: int) -> None:
                    self._progress.step, self._progress.expected = done, total
                    if req.on_progress:
                        req.on_progress(done, total)

                audio = rt.converter.convert(voice, req.audio, req.rate, index, opts, on_piece)
                return SynthesisResult(voice.sample_rate, audio, seed)
            finally:
                self._progress.phase = "idle"

    def build_index(
        self, lora_id: str, clips: list[Path], on_progress: Callable[[int, int], None] | None = None
    ) -> int:
        """The voice's index from a dataset's clips (ContentVec of each at 16 kHz), saved in its
        voice folder; rows kept."""
        import librosa
        import soundfile as sf
        import torch

        ckpt = self.checkpoint(lora_id)
        with self._lock:
            rt = self._ensure_runtime()
            rows = []
            for i, clip in enumerate(clips):
                y, sr = sf.read(clip, dtype="float32", always_2d=True)
                y = y.mean(1)
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
                if len(y) >= 1600:  # 0.1 s
                    rows.append(rt.contentvec(y)[0].half().cpu())
                if on_progress:
                    on_progress(i + 1, len(clips))
            if not rows:
                raise ValueError("None of the clips is long enough for an index.")
            kept = voices.save_index(self.settings.lora_dir / ckpt.voice_path, torch.cat(rows).to(self.device))
            self._index = None
            return kept

    def save_index_from(self, lora_id: str, features) -> int:
        """The voice's index from given features (e.g. an RVC WebUI run's total_fea.npy)."""
        ckpt = self.checkpoint(lora_id)
        with self._lock:
            kept = voices.save_index(self.settings.lora_dir / ckpt.voice_path, features.to(self.device))
            self._index = None
            return kept

    def unload(self) -> None:
        with self._lock:
            if self._runtime is None:
                return
            self._runtime = self._voice = self._index = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def training_unavailable(self) -> str | None:
        import torch

        if not torch.cuda.is_available():
            return "it needs a CUDA GPU."
        if not self.settings.model_path("rvc_train_model"):
            return "download its base models first (Models: RVC v2 base models)."
        return self.unavailable()

    def training_config(self, params, run_dir: Path, train_manifest: Path, val_manifest: Path | None) -> dict:
        """RVC's training (worker.py) from its base models at rvc_sample_rate: rvc_epochs over the
        dataset's 3.7 s pieces at batch_size; num_iters estimated from the clips' lengths here,
        exact once the worker has cut them; a voice saved every rvc_save_every epochs."""
        if reason := self.training_unavailable():
            raise ValueError(f"RVC can't train here: {reason}")
        from ...data import audio_io, manifests
        from .train import piece_count
        from .worker import plan

        records = manifests.read_manifest(train_manifest)[0]
        pieces = sum(piece_count(audio_io.info(r["audio"])[0]) for r in records)
        _, num_iters = plan(pieces, params.batch_size, params.rvc_epochs)
        rate, k = params.rvc_sample_rate, params.rvc_sample_rate // 1000
        bases = Path(self.settings.model_path("rvc_train_model"))
        dataset = manifests.resolve(self.settings.data_dir, params.dataset)
        return {
            "version": "v2",
            "sample_rate": rate,
            "pretrained_g": str(bases / f"f0G{k}k.pth"),
            "pretrained_d": str(bases / f"f0D{k}k.pth"),
            "rmvpe": str(Path(self.settings.model_path("rvc_model")) / "rmvpe.pt"),
            "contentvec": self.settings.model_path("contentvec_model"),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest) if val_manifest else "",
            "save_path": str(run_dir / "checkpoints"),
            "num_iters": num_iters,
            "epochs": params.rvc_epochs,
            "save_every": params.rvc_save_every,
            "batch_size": params.batch_size,
            "learning_rate": params.learning_rate,
            "log_interval": params.log_interval,
            # Pieces' features, shared by the dataset's runs at this rate (hidden folders aren't datasets).
            "feature_cache": str(dataset.parent / ".cache" / f"rvc_v2_{k}k.pt"),
        }
