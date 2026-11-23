"""RVC as an engine: converts recorded speech or singing into a voice (Convert tab), no text."""

from __future__ import annotations

import gc
import logging
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ... import lora_library, models, vram
from ...config import Settings
from ...data import audio_io
from ..base import Capabilities, ConvertRequest, Engine, Param, SynthesisRequest, SynthesisResult
from ..common import train_form
from . import voices

logger = logging.getLogger(__name__)

RATES = (32000, 40000, 48000)  # its base models' rates (train.RATES)
USUAL_RATE = 40000  # RVC's usual
# Training batch per GPU (vram.per_gpu); a GAN: no accumulation. Batch 16 peaks near
# 6.4 GB plus ~1 GB of feature / pitch models the worker keeps for samples.
TRAIN_BATCHES = {vram.GB16: 16, vram.GB12: 16, vram.GB8: 8, vram.SMALL: 4, vram.ON_CPU: 4}


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
    checkpoint_files = (voices.INFO, voices.WEIGHTS)

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
        import torch

        ckpt = self.checkpoint(lora_id)
        with self._lock:
            rt = self._ensure_runtime()
            rows = []
            for i, clip in enumerate(clips):
                y, sr = audio_io.read(clip)  # m4a / mp3 clips too (PyAV)
                y = y.mean(0)
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

    def index_from(
        self, lora_id: str, dataset: str | None, logs: str | None, make_room: Callable[[], None]
    ) -> int | None:
        """The voice's index built from a dataset's clips (``make_room`` first: ContentVec on the
        GPU), or an RVC WebUI logs folder's features; neither: None."""
        from ...data import manifests
        from ...prep.importer import clean_path

        if dataset:
            manifest = manifests.resolve(self.settings.data_dir, dataset)
            entries = manifests.training_entries(manifest)[0]
            if not entries:
                raise ValueError(f"'{dataset}' has no clips to build an index from.")
            make_room()
            return self.build_index(lora_id, [Path(e["audio"]) for e in entries])
        if logs:
            path = Path(clean_path(logs))
            if not (path / "total_fea.npy").is_file() and not (path.is_file() and path.suffix == ".npy"):
                raise ValueError("That folder has no total_fea.npy (an RVC WebUI logs/<run> folder has one).")
            return self.save_index_from(lora_id, voices.webui_features(path))
        return None

    def import_voice(self, body, make_room):
        """An RVC WebUI voice file (voices.WebuiImport: a server path), its index built from a dataset
        or a WebUI logs folder (the WebUI's faiss .index isn't read). An index that fails leaves the
        voice imported (ValueError saying so)."""
        from ...prep.importer import clean_path

        data = voices.WebuiImport.model_validate(body)
        name = data.name.strip()
        if not re.fullmatch(voices.VOICE_NAME, name):
            raise ValueError("Use letters, digits, '.', '-' or '_' for the name (up to 80).")
        folder = lora_library.engine_dir(self.settings.lora_dir, voices.ENGINE) / name
        if folder.exists():
            raise FileExistsError(f"A voice named '{name}' already exists.")
        voices.import_webui(folder, Path(clean_path(data.path)))
        lora_id = f"{voices.ENGINE}/{name}"
        try:
            rows = self.index_from(lora_id, data.index_dataset, data.index_logs, make_room)
        except (ValueError, OSError, KeyError) as e:  # the voice stays; its index can be built later
            raise ValueError(f"Imported '{name}', but its index failed: {e}") from e
        return {"id": lora_id, "engine": voices.ENGINE, "index_rows": rows}

    @classmethod
    def package_files(cls, ckpt, name, base_model, half):
        """The voice file as RVC saves it and its index (base_model and half don't apply)."""
        files = voices.package_files(ckpt.path)
        header = voices.check_voice_file(ckpt.path / voices.WEIGHTS)
        index = files.get(voices.PACKAGE_INDEX)
        rows = voices.package_index(index).shape[0] if index else None
        files["README.md"] = voices.readme(name, header["sample_rate"], rows, header["info"], ckpt.step).encode()
        return files

    @classmethod
    def import_package(cls, folder, name, files):
        if voices.WEIGHTS not in files:
            raise ValueError(f"An RVC package needs {voices.WEIGHTS}.")
        return voices.import_package(folder, name, {f: d for f, d in files.items() if f in voices.PACKAGE_FILES})

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

    @classmethod
    def train_form(cls, profile=None):
        """Its whole model from the base models at a rate, in epochs over 3.7 s pieces, as its
        WebUI trains (a GAN: the batch as is); a voice saved every so many epochs."""
        return train_form.full_form(
            profile,
            TRAIN_BATCHES,
            Param("rvc_epochs", "int", 100, min=1, max=1000, scope="option", config="epochs"),
            Param("rvc_save_every", "int", 25, min=1, max=1000, scope="option", config="save_every"),
            Param("rvc_sample_rate", "choice", USUAL_RATE, choices=RATES, scope="option", config="sample_rate",
                  hint="rvcRateHint"),
        )  # fmt: skip

    @classmethod
    def plan(cls, values, clips, seconds, overrides=()):
        from .worker import plan

        # RVC cuts clips into pieces of up to 3.7 s, one every 3.4 s (train.pieces): about this many
        # (~0.37 a clip are remainders on a real dataset; the worker counts them exactly).
        pieces = max(clips, train_form.js_round(seconds / 3.4 + clips * 0.37)) if clips > 0 else 0
        batch, epochs = int(values.get("batch_size") or 1), int(values.get("rvc_epochs") or 1)
        own, final = train_form.derived(
            values, {"num_iters": plan(pieces, batch, epochs)[1]} if pieces else {}, overrides
        )
        rate = int(values.get("rvc_sample_rate") or USUAL_RATE)
        return {
            "values": own,
            "summary": {
                "kind": "pieces",
                "steps": final.get("num_iters"),
                "pieces": pieces,
                "epochs": epochs,
                "batch": batch,
                "save_every": values.get("rvc_save_every"),
                "model": "full",
            },
            "name_tag": "rvc" if rate == USUAL_RATE else f"rvc{rate // 1000}k",
            "samples": {"every": values.get("rvc_save_every"), "unit": "epochs"},
        }

    def training_config(self, params, run_dir: Path, train_manifest: Path, val_manifest: Path | None) -> dict:
        """RVC's training (worker.py) from its base models at rvc_sample_rate: rvc_epochs over the
        dataset's 3.7 s pieces at batch_size; num_iters estimated from the clips' lengths here,
        exact once the worker has cut them; a voice saved every rvc_save_every epochs."""
        if reason := self.training_unavailable():
            raise ValueError(f"RVC can't train here: {reason}")
        from ...data import manifests
        from .train import piece_count
        from .worker import plan

        opts = self.train_options(params.options)
        records = manifests.read_manifest(train_manifest)[0]
        pieces = sum(piece_count(audio_io.info(r["audio"])[0]) for r in records)
        _, num_iters = plan(pieces, params.batch_size, opts["rvc_epochs"])
        rate, k = opts["rvc_sample_rate"], opts["rvc_sample_rate"] // 1000
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
            "epochs": opts["rvc_epochs"],
            "save_every": opts["rvc_save_every"],
            "batch_size": params.batch_size,
            "learning_rate": params.learning_rate,
            "log_interval": params.log_interval,
            # Pieces' features, shared by the dataset's runs at this rate (hidden folders aren't datasets).
            "feature_cache": str(dataset.parent / ".cache" / f"rvc_v2_{k}k.pt"),
        }
