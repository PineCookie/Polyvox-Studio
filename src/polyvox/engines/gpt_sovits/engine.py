"""GPT-SoVITS v2Pro / v2ProPlus / v4 (RVC-Boss/GPT-SoVITS, vendor/gpt_sovits) as an engine.

A GPT turns text into semantic tokens, continuing a reference clip's; SoVITS
decodes them to 32 kHz audio in the reference's timbre (v4: a mel DiT and a
vocoder, 48 kHz). It always clones a clip: the request's, else the voice's own
default (voices.py). Voices are full weights (~300 MB; v4: its trained parts
over the v4 base), loaded when chosen; the base is v2ProPlus.
"""

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
from typing import TYPE_CHECKING

from ... import lora_library, vram
from ...config import Settings
from ...models import ModelSpec
from ..base import Capabilities, Engine, Param, ReferenceInput, SynthesisRequest, SynthesisResult
from ..common import train_form
from . import voices
from .synth import BASE_VERSIONS, REF_MAX_S, V4, VERSIONS, Options, split_text

TIMBRE_VERSIONS = tuple(v for v in VERSIONS if v != V4)

if TYPE_CHECKING:
    from ...training import TrainParams

logger = logging.getLogger(__name__)

TOKENS_PER_CHAR = 5.5  # first guess of semantic tokens per character; refined from finished generations
MIN_TOKENS = 12
LEARN_MIN_CHARS = 10
BASE_VERSION = "v2ProPlus"
TRAIN_VERSIONS = BASE_VERSIONS  # base models a run trains from, the default first
NAME_TAGS = {"v2ProPlus": "v2pp", "v2Pro": "v2pro", "v4": "v4"}  # in run names, as imported webui voices
# Both stages' batch per GPU (vram.per_gpu; 1 elsewhere): SoVITS batch 4 / 6 / 8 ~ 6.6 / 7.2 / 8.3 GB
# incl. its front ends.
TRAIN_BATCHES = {vram.GB16: 8, vram.GB12: 8, vram.GB8: 4}
V4_BATCH = 4  # v4's SoVITS stage (a LoRA on its mel DiT) at the batch it was tuned at

# (base models folder, G2PW folder, device) -> synth.Pipeline
PipelineLoader = Callable[[Path, Path, str], object]


def load_pipeline(base_dir: Path, g2pw_dir: Path, device: str):
    from .synth import Pipeline

    return Pipeline(base_dir, g2pw_dir, device)


@dataclass
class Progress:
    phase: str = "idle"  # idle | loading | preparing | compiling | generating
    step: int = 0  # semantic tokens so far (~25 per second of audio)
    expected: int | None = None
    attempt: int = 0
    started: float = 0.0


# The models it uses (models.all_specs: the Models tab, downloads, Settings.model_path).
MODEL_SPECS = (
    ModelSpec(
        "gsv_model",
        "GPT-SoVITS v2Pro / v2ProPlus",
        "Text to speech (32 kHz), voice training: base models with their HuBERT, RoBERTa and speaker encoders",
        "lj1995/GPT-SoVITS",
        "lj1995__GPT-SoVITS",
        (
            "s1v3.ckpt",
            "v2Pro/s2Gv2ProPlus.pth",
            "v2Pro/s2Gv2Pro.pth",
            "v2Pro/s2Dv2ProPlus.pth",
            "v2Pro/s2Dv2Pro.pth",
            "chinese-hubert-base/config.json",
            "chinese-hubert-base/preprocessor_config.json",
            "chinese-hubert-base/pytorch_model.bin",
            "chinese-roberta-wwm-ext-large/config.json",
            "chinese-roberta-wwm-ext-large/pytorch_model.bin",
            "chinese-roberta-wwm-ext-large/tokenizer.json",
            "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        ),
        1660,
        license="MIT",
    ),
    ModelSpec(
        "gsv_v4_model",
        "GPT-SoVITS v4",
        "Text to speech (48 kHz), voice training: v4's SoVITS base (a mel DiT) and vocoder",
        "lj1995/GPT-SoVITS",
        "lj1995__GPT-SoVITS",
        ("gsv-v4-pretrained/s2Gv4.pth", "gsv-v4-pretrained/vocoder.pth"),
        830,
        entry="gsv-v4-pretrained",
        license="MIT",
    ),
    ModelSpec(
        "gsv_g2pw_model",
        "G2PW",
        "GPT-SoVITS: pronunciation of Chinese characters with several readings",
        "kamiorinn/g2pw",
        "G2PWModel",
        (
            "g2pW.onnx",
            "config.py",
            "char_bopomofo_dict.json",
            "bopomofo_to_pinyin_wo_tune_dict.json",
            "MONOPHONIC_CHARS.txt",
            "POLYPHONIC_CHARS.txt",
        ),
        590,
        license="Apache-2.0",
        hub="modelscope",
        archive="G2PWModel_1.1.zip",
    ),
)


class GptSovitsEngine(Engine):
    id = voices.ENGINE
    name = "GPT-SoVITS"
    capabilities = Capabilities(
        sample_rate=32_000,
        training_rate=32_000,
        languages=("zh", "en"),
        reference_audio=True,
        reference_transcript=True,
        style=False,
        tags=False,
        lora_hot_swap=False,
        reference_required=True,
        voice_reference=True,
        package="voice",
        full_finetune=True,
        reference_inputs=(
            # The GPT's prompt (3-10 s); without a transcript only its timbre is used (not v4).
            ReferenceInput("prompt", "prompt", "optional", max_seconds=REF_MAX_S),
            # SoVITS averages the timbre of several clips (upstream's inp_refs; v4 has none).
            ReferenceInput(
                "timbre", "reference", "none", several=True, max_seconds=REF_MAX_S, versions=TIMBRE_VERSIONS
            ),
        ),
    )
    params = (
        Param("temperature", "float", 1.0, 0.05, 2.0, 0.05),
        Param("top_k", "int", 15, 1, 100, 1),
        Param("top_p", "float", 1.0, 0.05, 1.0, 0.05),
        # Against loops and stutters: penalizes semantic tokens the GPT just made.
        Param("repetition_penalty", "float", 1.35, 1.0, 2.0, 0.05),
        Param("speed", "float", 1.0, 0.6, 1.65, 0.05),
        # v4 only: flow-matching steps per mel window.
        Param("sample_steps", "int", 8, 4, 32, 4, versions=(V4,)),
        # The base model's version (a voice has its own, from its weights).
        Param("base_version", "choice", BASE_VERSION, choices=BASE_VERSIONS, base_only=True),
    )
    model_setting = "gsv_model"

    model_specs = MODEL_SPECS
    worker_module = "polyvox.engines.gpt_sovits.worker"
    checkpoint_files = (voices.SOVITS_FILE, voices.GPT_FILE)

    @classmethod
    def package_files(cls, ckpt, name, base_model, half):
        """Whole models (base_model and half don't apply), as the webui saves them."""
        import json

        files = voices.package_files(ckpt.path, name)
        version = json.loads(files[voices.CONFIG_FILE])["version"]
        has_gpt, has_reference = voices.GPT_FILE in files, voices.REFERENCE_WAV in files
        files["README.md"] = voices.readme(name, version, has_gpt, has_reference, ckpt.step).encode()
        return files

    @classmethod
    def import_package(cls, folder, name, files):
        if voices.SOVITS_FILE not in files:
            raise ValueError(f"A GPT-SoVITS package needs {voices.SOVITS_FILE}.")
        known = (voices.SOVITS_FILE, voices.GPT_FILE, voices.REFERENCE_WAV, voices.REFERENCE_JSON)
        return voices.import_package(folder, name, {f: d for f, d in files.items() if f in known})

    def import_voice(self, body, make_room):
        """Weights the GPT-SoVITS webui trained (voices.WebuiImport: server paths), with a default
        reference picked from a dataset, or a given clip, or none yet."""
        from ...data import manifests
        from ...prep.importer import clean_path

        data = voices.WebuiImport.model_validate(body)
        if data.dataset:
            manifest = manifests.resolve(self.settings.data_dir, data.dataset)
            reference = voices.pick_reference(manifest, manifests.legacy_root(self.settings.data_dir))
            origin = f"dataset {data.dataset}"
        elif data.reference:
            reference, origin = (clean_path(data.reference), data.reference_text), "file"
        else:
            reference, origin = None, ""  # added later in the Voices panel
        folder = lora_library.engine_dir(self.settings.lora_dir, voices.ENGINE)
        gpt = clean_path(data.gpt) if data.gpt else None
        name = voices.import_webui(folder, data.name.strip(), clean_path(data.sovits), gpt, reference, origin)
        return {"id": f"{voices.ENGINE}/{name}", "engine": voices.ENGINE}

    def __init__(self, settings: Settings, loader: PipelineLoader = load_pipeline):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._pipeline = None
        self._voice = None  # synth.Voice, its key = (checkpoint id or "base", weights mtime)
        self._references: dict[object, object] = {}  # default references of the loaded voice, by key
        self._progress = Progress()
        self._tokens_per_char = TOKENS_PER_CHAR

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def unavailable(self) -> str | None:
        if not self.settings.model_path("gsv_model") or not self.settings.model_path("gsv_g2pw_model"):
            return "download its models first (Models: GPT-SoVITS and G2PW)."
        return None

    def status(self) -> dict:
        return {
            "device": self.device,
            "model": self.settings.model_path(self.model_setting),
            "loaded": self.loaded,
            "active_lora": self._voice.key[0] if self._voice is not None else None,
        }

    def progress(self) -> dict:
        p = self._progress
        return {**p.__dict__, "elapsed": round(time.time() - p.started, 1) if p.started else 0.0}

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        text = req.text.strip()
        if not split_text(text):
            raise ValueError("Text to synthesize is empty.")
        if reason := self.unavailable():
            raise ValueError(f"GPT-SoVITS can't run yet: {reason}")
        values = self.options(req.options)
        base_version = values.pop("base_version")
        options = Options(**values)
        ckpt = lora_library.resolve(self.settings.lora_dir, req.lora_id) if req.lora_id else None
        if ckpt and ckpt.engine != self.id:
            raise ValueError(f"'{ckpt.id}' was trained for another model ({ckpt.engine}), not GPT-SoVITS.")
        default_ref = voices.read_reference(ckpt.path) if ckpt else None
        # The GPT's prompt: the given one, else the voice's own, else the timbre clip
        # without its transcript (upstream's reference-free mode).
        source = req.prompt_wav or (None if default_ref else req.reference_wav)
        if source is None and default_ref is None:
            raise ValueError("GPT-SoVITS clones a reference clip: add one (3-10 s), or choose a voice that has one.")

        with self._lock:
            self._progress = Progress(phase="loading" if self._pipeline is None else "preparing", started=time.time())
            try:
                pipeline = self._ensure_pipeline()
                voice = self._ensure_voice(pipeline, ckpt, base_version)
                if source is not None:
                    ref = pipeline.reference(voice, source, req.prompt_text if req.prompt_wav else None)
                else:
                    ref = self._default_reference(pipeline, voice, *default_ref)
                # The voice's clips give the timbre (as upstream's several references do), else the prompt's
                # clip; their transcripts don't matter there: spectrogram and speaker embedding only.
                voice_clips = [req.reference_wav, *req.extra_references] if req.reference_wav else []
                timbre = [
                    ref if wav == source else pipeline.reference(voice, wav, None)
                    for wav in voice_clips
                    if voice.model_version != V4
                ]
                seed = req.seed if req.seed is not None else random.randrange(2**31)
                chars = len(re.sub(r"\W", "", text))
                self._progress.expected = max(MIN_TOKENS, round(chars * self._tokens_per_char))
                self._progress.phase = "generating"
                audio = pipeline.synthesize(
                    voice, ref, text, options, seed, on_token=self._count, on_phase=self._phase, timbre=timbre
                )
                if chars >= LEARN_MIN_CHARS and options.speed == 1.0:
                    observed = self._progress.step / chars
                    self._tokens_per_char = 0.7 * self._tokens_per_char + 0.3 * observed
                return SynthesisResult(voice.sample_rate, audio, seed)
            finally:
                self._progress.phase = "idle"

    def _count(self) -> None:
        self._progress.step += 1
        self._progress.phase = "generating"  # (after a first call's compiling)

    def _phase(self, phase: str) -> None:
        self._progress.phase = phase

    def _ensure_pipeline(self):
        if self._pipeline is None:
            base = Path(self.settings.model_path("gsv_model"))
            g2pw = Path(self.settings.model_path("gsv_g2pw_model"))
            t0 = time.time()
            self._pipeline = self._loader(base, g2pw, self.device)
            if v4 := self.settings.model_path("gsv_v4_model"):
                self._pipeline.v4_dir = Path(v4)
            if self.settings.optimize is False:  # upstream's decoding loop instead of the CUDA graph
                self._pipeline.fast_decode = False
            logger.info("GPT-SoVITS front ends loaded in %.1f s", time.time() - t0)
        return self._pipeline

    def _ensure_voice(self, pipeline, ckpt: lora_library.LoraCheckpoint | None, base_version: str = BASE_VERSION):
        if (base_version if ckpt is None else ckpt.version) == V4 and not self.settings.model_path("gsv_v4_model"):
            raise ValueError("GPT-SoVITS v4 isn't downloaded yet (Models: GPT-SoVITS v4).")
        if ckpt is None:
            key = ("base", base_version)
            sovits, gpt = pipeline.base_weights(base_version)
        else:
            key = (ckpt.id, ckpt.weights_mtime)
            sovits, gpt = ckpt.path / voices.SOVITS_FILE, ckpt.path / voices.GPT_FILE
            if not gpt.is_file():  # trained SoVITS only so far
                gpt = pipeline.base_weights(ckpt.version or BASE_VERSION)[1]
        if self._voice is None or self._voice.key != key:
            self._voice = self._references = None
            gc.collect()
            self._voice = pipeline.load_voice(sovits, gpt, key=key)
            self._references = {}
        return self._voice

    def _default_reference(self, pipeline, voice, wav: Path, text: str):
        key = (str(wav), wav.stat().st_mtime, text)
        if key not in self._references:
            self._references[key] = pipeline.reference(voice, wav, text, key=key)
        return self._references[key]

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            self._pipeline = self._voice = None
            self._references = {}
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def train_form(cls, profile=None):
        """Both stages as the webui trains them (its epochs), the batch that fits the GPU for both
        (no accumulation: SoVITS is a GAN); v4's SoVITS stage is a LoRA on its mel DiT, at the
        batch it was tuned at (smaller GPUs recompute activations instead)."""
        batch = train_form.gpu_batch(profile, TRAIN_BATCHES)
        v4 = {"gsv_version": (V4,)}
        return train_form.full_form(
            profile,
            TRAIN_BATCHES,
            Param("sovits_epochs", "int", 8, min=0, max=100, scope="option"),
            Param("gpt_epochs", "int", 15, min=0, max=100, scope="option", legacy=0),
            Param("lora_rank", "int", 32, min=1, max=512, shown_if=v4, config="lora_rank"),
            extra=(
                Param("gpt_batch_size", "int", batch if profile else 8, min=1, max=64, scope="option"),
                Param("gsv_version", "choice", "v2ProPlus", choices=(*TRAIN_VERSIONS,), scope="option",
                      config="version", implies={v: {"batch_size": V4_BATCH if v == V4 else batch}
                                                 for v in TRAIN_VERSIONS}),
            ),
        )  # fmt: skip

    @classmethod
    def plan(cls, values, clips, seconds, overrides=()):
        from .worker import gpt_batch_size, plan_steps

        batch, gpt_batch = int(values.get("batch_size") or 1), int(values.get("gpt_batch_size") or 1)
        sovits, gpt = plan_steps(
            clips, batch, values.get("sovits_epochs") or 0, clips, gpt_batch, values.get("gpt_epochs") or 0
        )
        own, final = train_form.derived(values, {"num_iters": sovits + gpt} if clips > 0 else {}, overrides)
        version = str(values.get("gsv_version") or "v2ProPlus")
        rank = values.get("lora_rank")
        return {
            "values": own,
            "summary": {
                "kind": "stages",
                "sovits": sovits,
                "gpt": gpt,
                "batch": batch,
                "gpt_batch": gpt_batch_size(clips, gpt_batch) if clips else gpt_batch,
                "steps": final.get("num_iters"),
                "model": "v4_lora" if version == V4 else "full",
                "rank": rank,
            },
            "name_tag": NAME_TAGS.get(version, version.lower()) + (f"-r{rank}" if version == V4 else ""),
            "samples": {"every": 1, "unit": "epochs"},
        }

    def training_config(
        self, params: TrainParams, run_dir: Path, train_manifest: Path, val_manifest: Path | None
    ) -> dict:
        """Full fine-tuning in two stages (worker.py): sovits_epochs at
        batch_size and learning_rate, then gpt_epochs at gpt_batch_size (the
        GPT's optimizer is upstream's); num_iters follows from them (estimated
        here, exact once the worker has the clips' features). The run's
        reference clip (samples, checkpoints) is the dataset's best one."""
        if reason := self.unavailable():
            raise ValueError(f"GPT-SoVITS can't train yet: {reason}")
        opts = self.train_options(params.options)
        if not opts["sovits_epochs"] and not opts["gpt_epochs"]:
            raise ValueError("Train SoVITS, the GPT or both: set their epochs.")
        v4 = opts["gsv_version"] == V4
        if v4 and not self.settings.model_path("gsv_v4_model"):
            raise ValueError("GPT-SoVITS v4 isn't downloaded yet (Models: GPT-SoVITS v4).")
        from ...data import manifests
        from .features import feature_version
        from .worker import plan_steps

        dataset = manifests.resolve(self.settings.data_dir, params.dataset)
        audio, text = voices.pick_reference(dataset, manifests.legacy_root(self.settings.data_dir))
        clips = len(manifests.read_manifest(train_manifest)[0])
        sovits_steps, gpt_steps = plan_steps(
            clips, params.batch_size, opts["sovits_epochs"], clips, opts["gpt_batch_size"], opts["gpt_epochs"]
        )
        config = {
            "version": opts["gsv_version"],
            "pretrained_path": self.settings.model_path("gsv_model"),
            "g2pw_path": self.settings.model_path("gsv_g2pw_model"),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest) if val_manifest else "",
            "save_path": str(run_dir / "checkpoints"),
            "num_iters": sovits_steps + gpt_steps,
            "sovits_epochs": opts["sovits_epochs"],
            "batch_size": params.batch_size,
            "learning_rate": params.learning_rate,
            "gpt_epochs": opts["gpt_epochs"],
            "gpt_batch_size": opts["gpt_batch_size"],
            "log_interval": params.log_interval,
            "reference": {"audio": str(audio), "text": text, "source": f"{params.dataset}: {audio.name}"},
            # Clip features, shared by the dataset's runs (hidden folders aren't datasets; v4 uses v2ProPlus's).
            "feature_cache": str(dataset.parent / ".cache" / f"gpt_sovits_{feature_version(opts['gsv_version'])}.pt"),
        }
        if v4:  # SoVITS as a LoRA on the mel DiT
            config |= {
                "v4_path": self.settings.model_path("gsv_v4_model"),
                "lora_rank": params.lora_rank,
                "compile": self.settings.optimize is not False,  # fuse the DiT's blocks (CUDA)
            }
        return config
