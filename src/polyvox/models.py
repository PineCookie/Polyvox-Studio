"""The models the studio uses, where they live under the models folder, and
downloading them from Hugging Face (or a mirror of it, e.g. hf-mirror.com).

A model counts as downloaded once every file it needs is in its folder, so an
interrupted download is never used. Settings (config.Settings.model_path) use
a downloaded copy unless a model was configured explicitly.
"""

from __future__ import annotations

import logging
import shutil
import threading
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

HF_ENDPOINT = "https://huggingface.co"
# Speech models (and their helpers), voice conversion, dataset tools, audio clean-up.
GROUPS = ("tts", "conversion", "dataset", "cleanup")
WAYS = ("prebuilt", "convert", "convert_keep")  # ModelSpec.ways' ids
# Drama 3 in 4 bits as fish_drama3/convert.py makes it from the official weights (78e5ab7).
DRAMA3_BUILD_REPO = "PineCookie/drama-3-int4-higgs"
DRAMA3_BUILD_REVISION = "dedb85dba9318b2353cec15eda1142f8586a4e80"
POLL_S = 0.5


@dataclass(frozen=True)
class Source:
    """The files a model is built from (downloaded into their own folder); ``build`` names
    "module:function"(source dir, model dir, progress(fraction)), which writes the model's
    folder complete or not at all. The source folder is removed after a build unless ``keep``.
    ``repo`` / ``revision``: where the files are (None: the model's own repo and revision)."""

    folder: str
    files: tuple[str, ...]
    build: str
    keep: bool = False
    repo: str | None = None
    revision: str | None = None
    size_mb: int = 0  # the download (a fallback's: shown when the user picks it)


@dataclass(frozen=True)
class ModelSpec:
    setting: str  # the config.Settings field it fills
    name: str
    purpose: str
    repo: str  # Hugging Face repo id
    folder: str  # under the models folder
    files: tuple[str, ...]  # what is downloaded (only what is used)
    size_mb: int  # about, for the UI before downloading
    # The file the setting points to (RoFormer checkpoints); None = the folder.
    entry: str | None = None
    license: str = ""
    hub: str = "huggingface"  # or "modelscope" (no endpoint choice there)
    # A zip in the repo that holds ``files`` (one top folder, dropped when unpacking).
    archive: str | None = None
    revision: str | None = None  # a pinned commit of the repo (None: its main branch)
    # Built from other files of the repo (``files`` are then the build's, ``size_mb`` the download's).
    source: Source | None = None
    # The files come built from ``repo``; when that repo / revision / file isn't there (removed,
    # say), the model is built from this instead (or when asked: download's ``way``).
    fallback: Source | None = None
    group: str = "tts"  # the Models tab's section: GROUPS

    def ways(self) -> list[dict]:
        """How it can be downloaded, when there's a choice: the build, or its source converted here
        (then kept or removed); with each download's size."""
        if not self.fallback:
            return []
        convert = self.fallback.size_mb
        return [{"id": "prebuilt", "size_mb": self.size_mb}, {"id": "convert", "size_mb": convert},
                {"id": "convert_keep", "size_mb": convert}]  # fmt: skip

    def path(self, models_dir: Path) -> Path:
        folder = models_dir / self.folder
        return folder / self.entry if self.entry else folder

    @property
    def model_repo(self) -> str:
        """The repo that names the model: the one it's built from when ``repo`` holds a build."""
        return (self.fallback.repo if self.fallback else None) or self.repo

    def downloaded(self, models_dir: Path) -> bool:
        folder = models_dir / self.folder
        return all((folder / f).is_file() for f in self.files)


_HF_ASR_FILES = (
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "model",
        "VoxCPM2",
        "Text to speech, LoRA training",
        "openbmb/VoxCPM2",
        "openbmb__VoxCPM2",
        (
            "config.json",
            "model.safetensors",
            "audiovae.pth",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenization_voxcpm2.py",
        ),
        4960,
        license="Apache-2.0",
    ),
    ModelSpec(
        "dots_model",
        "dots.tts SOAR",
        "Text to speech (48 kHz), LoRA training",
        "rednote-hilab/dots.tts-soar",
        "rednote-hilab__dots.tts-soar",
        (
            "config.json",
            "llm_config.json",
            "latent_stats.pt",
            "model.safetensors",
            "vocoder.safetensors",
            "speaker_encoder.safetensors",
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        4920,
        license="Apache-2.0",
    ),
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
        "s2_model",
        "Fish Audio S2-Pro",
        "Text to speech (44.1 kHz), zero-shot cloning; needs a 16 GB GPU",
        "fishaudio/s2-pro",
        "fishaudio__s2-pro",
        (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "codec.pth",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "LICENSE.md",
        ),
        11_000,
        license="Fish Audio Research License (non-commercial)",
    ),
    ModelSpec(
        "drama3_model",
        "Fish Audio Drama 3",
        "Text to speech (44.1 kHz), zero-shot cloning, run in 4 bits",
        DRAMA3_BUILD_REPO,
        "fishaudio__drama-3__int4",
        (
            "config.json",
            "model.safetensors.index.json",
            *(f"model-{i:05d}-of-00011.safetensors" for i in range(1, 12)),
            "codec.pth",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "LICENSE",
            "NOTICE",
            "quant_stats.json",
        ),
        24_070,
        license="Fish Audio Community License (free below USD 500k yearly revenue; no hosted services); "
        "Built with Fish Audio",
        revision=DRAMA3_BUILD_REVISION,
        fallback=Source(
            "fishaudio__drama-3",
            (
                "config.json",
                "model.safetensors.index.json",
                *(f"model-{i:05d}-of-00015.safetensors" for i in range(1, 16)),
                "codec.pth",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "LICENSE",
            ),
            "polyvox.engines.fish_drama3.convert:build",
            repo="fishaudio/drama-3",
            revision="78e5ab79e49aa9a25abe8799eff2c4d73a1c3216",
            size_mb=39_600,
        ),
    ),
    ModelSpec(
        "rvc_model",
        "RVC v2",
        "Voice conversion (covers): RMVPE pitch extraction",
        "lj1995/VoiceConversionWebUI",
        "lj1995__VoiceConversionWebUI",
        ("rmvpe.pt",),
        181,
        license="MIT",
        group="conversion",
    ),
    ModelSpec(
        "contentvec_model",
        "ContentVec",
        "RVC: content features of the audio to convert",
        "IAHispano/Applio",
        "IAHispano__Applio",
        ("Resources/embedders/contentvec/config.json", "Resources/embedders/contentvec/pytorch_model.bin"),
        378,
        entry="Resources/embedders/contentvec",
        license="MIT",
        group="conversion",
    ),
    ModelSpec(
        "rvc_train_model",
        "RVC v2 base models",
        "Training RVC voices: the 32, 40 and 48 kHz base models (generator and discriminator)",
        "lj1995/VoiceConversionWebUI",
        "lj1995__VoiceConversionWebUI",
        tuple(f"pretrained_v2/f0{part}{rate}k.pth" for rate in (32, 40, 48) for part in "GD"),
        651,
        entry="pretrained_v2",
        license="MIT",
        group="conversion",
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
    ModelSpec(
        "denoiser_model",
        "ZipEnhancer",
        "Synthesis: clean up reference audio before cloning",
        "iic/speech_zipenhancer_ans_multiloss_16k_base",
        "iic__speech_zipenhancer_ans_multiloss_16k_base",
        ("configuration.json", "pytorch_model.bin"),
        8,
        license="Apache-2.0",
        hub="modelscope",
    ),
    ModelSpec(
        "asr_model",
        "Qwen3-ASR 1.7B",
        "Speech recognition: transcribe clips, compare with transcripts",
        "Qwen/Qwen3-ASR-1.7B-hf",
        "Qwen__Qwen3-ASR-1.7B-hf",
        (*_HF_ASR_FILES, "generation_config.json"),
        4090,
        license="Apache-2.0",
        group="dataset",
    ),
    ModelSpec(
        "align_model",
        "Qwen3-ForcedAligner 0.6B",
        "Check clips against their text (11 languages)",
        "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        "Qwen__Qwen3-ForcedAligner-0.6B-hf",
        _HF_ASR_FILES,
        1850,
        license="Apache-2.0",
        group="dataset",
    ),
    ModelSpec(
        "ctc_align_model",
        "wav2vec2 Chinese (CTC)",
        "Second aligner, sharper on single wrong words (Chinese)",
        "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "jonatasgrosman__wav2vec2-large-xlsr-53-chinese-zh-cn",
        ("config.json", "preprocessor_config.json", "pytorch_model.bin", "special_tokens_map.json", "vocab.json"),
        1280,
        license="Apache-2.0",
        group="dataset",
    ),
    ModelSpec(
        "singing_model",
        "AST AudioSet tagger",
        "Detect sung clips",
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        "MIT__ast-finetuned-audioset-10-10-0.4593",
        ("config.json", "model.safetensors", "preprocessor_config.json"),
        350,
        license="BSD-3-Clause",
        group="dataset",
    ),
    ModelSpec(
        "speaker_model",
        "CAM++ speaker embeddings",
        "Speaker similarity: training samples vs the dataset's voice, and the Other voice? check",
        "iic/speech_campplus_sv_zh-cn_16k-common",
        "iic__speech_campplus_sv_zh-cn_16k-common",
        ("campplus_cn_common.bin",),
        28,
        license="Apache-2.0",
        hub="modelscope",
        group="dataset",
    ),
    ModelSpec(
        "vocals_model",
        "Mel-Band RoFormer vocals (Kim FT2 bleedless)",
        "Clean up: keep only the voice (remove music)",
        "pcunwa/Kim-Mel-Band-Roformer-FT",
        "pcunwa__Kim-Mel-Band-Roformer-FT",
        ("kimmel_unwa_ft2_bleedless.ckpt", "config_kimmel_unwa_ft.yaml"),
        910,
        entry="kimmel_unwa_ft2_bleedless.ckpt",
        group="cleanup",
    ),
    ModelSpec(
        "dereverb_model",
        "Mel-Band RoFormer de-reverb",
        "Clean up: remove room reverb",
        "anvuew/dereverb_mel_band_roformer",
        "anvuew__dereverb_mel_band_roformer",
        ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "dereverb_mel_band_roformer_anvuew.yaml"),
        910,
        entry="dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        license="GPL-3.0",
        group="cleanup",
    ),
    ModelSpec(
        "denoise_model",
        "Mel-Band RoFormer denoise",
        "Clean up: remove background noise",
        "poiqazwsx/melband-roformer-denoise",
        "aufr33__melband-roformer-denoise",
        ("denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "model_mel_band_roformer_denoise.yaml"),
        910,
        entry="denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        group="cleanup",
    ),
    ModelSpec(
        "karaoke_model",
        "UVR MDX-Net Karaoke 2",
        "Covers: split the vocals into the lead and the backing vocals (harmonies)",
        "Acapellas/uvr5-models",  # UVR's release file (same bytes) with its licence
        "Acapellas__uvr5-models",
        ("UVR_MDXNET_KARA_2.onnx", "UVR_MDXNET_KARA_2.LICENSE"),
        53,
        entry="UVR_MDXNET_KARA_2.onnx",
        license="MIT (Ultimate Vocal Remover)",
        revision="be57392c6b82821fe8a8b6f411ff1c38717d1b47",
        group="cleanup",
    ),
)

BY_SETTING = {m.setting: m for m in MODELS}


SOURCE_FILE = ".download-source"  # in the models folder: the download source chosen in the Models tab


def check_endpoint(url: str) -> str:
    """A download source as given ("" = huggingface.co), cleaned; ValueError when it isn't one."""
    url = url.strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        raise ValueError("The download source must be an http(s) URL.")
    return url


def use_endpoint(url: str) -> None:
    """Make ``url`` ("" = huggingface.co) where models that aren't downloaded come from when first
    used, in this process (huggingface_hub reads its endpoint per call) and in the training
    processes it starts (HF_ENDPOINT). ModelScope models are unaffected."""
    import os

    from huggingface_hub import constants

    endpoint = check_endpoint(url) or constants._HF_DEFAULT_ENDPOINT
    constants.ENDPOINT = endpoint
    constants.HUGGINGFACE_CO_URL_TEMPLATE = endpoint + "/{repo_id}/resolve/{revision}/{filename}"
    if url:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)


def current_endpoint() -> str:
    """The download source in use ("" = huggingface.co)."""
    from huggingface_hub import constants

    return "" if constants.ENDPOINT == constants._HF_DEFAULT_ENDPOINT else constants.ENDPOINT


def saved_endpoint(models_dir: Path) -> str | None:
    """The download source saved in the models folder, or None when none was chosen there."""
    try:
        return check_endpoint((models_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_endpoint(models_dir: Path, url: str) -> str:
    """Remember ``url`` as the download source and use it now; returns it cleaned."""
    url = check_endpoint(url)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / SOURCE_FILE).write_text(url, encoding="utf-8")
    use_endpoint(url)
    return url


class ModelUnavailable(RuntimeError):
    """A model that isn't downloaded couldn't be fetched either (no network, the hub blocked, an
    offline setting): the message says which and where to get it. The server answers 503 with it."""


def _fetch_failure(error: BaseException) -> BaseException | None:
    """The error in ``error``'s chain that says fetching from the hub failed, if any."""
    import socket

    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

    seen = set()
    e: BaseException | None = error
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        name = type(e).__name__
        if isinstance(e, HfHubHTTPError | LocalEntryNotFoundError | OfflineModeIsEnabled | socket.error) or name in (
            "ConnectError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "SSLError",
        ):  # fmt: skip
            return e
        text = str(e)
        if isinstance(e, OSError) and ("couldn't connect" in text or "huggingface.co" in text or "offline" in text):
            return e
        e = e.__cause__ or e.__context__
    return None


_FOLDER_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_ACTIVE: dict[str, dict] = {}  # setting -> {"setting", "name", "done", "total"} (MB): automatic downloads


def _folder_lock(folder: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _FOLDER_LOCKS.setdefault(folder, threading.Lock())


def active_download() -> dict | None:
    """A download started because a feature needed its model (ensure_downloaded), while it runs."""
    return next(iter(_ACTIVE.values()), None)


def ensure_downloaded(settings, setting: str) -> None:
    """A model not configured and not downloaded is downloaded into the models folder now (from
    the download source in use), its progress in active_download(), before it is loaded."""
    spec = BY_SETTING[setting]
    if settings.configured(setting) or spec.downloaded(settings.models_dir):
        return
    from . import jobs

    job = jobs.current()  # in a job: its Cancel stops the download (after the file in flight)
    entry = {"setting": setting, "name": spec.name, "done": 0, "total": spec.size_mb}

    def report(done: int, total: int) -> None:
        entry.update(done=done, total=total or entry["total"])
        if job is not None and job.cancel_requested:
            raise jobs.JobCancelled()

    _ACTIVE[setting] = entry
    try:
        download(spec, settings.models_dir, current_endpoint() or None, report)
    finally:
        _ACTIVE.pop(setting, None)


@contextmanager
def loading(settings, *setting_names: str, fetch: tuple[str, ...] | None = None):
    """Around loading these models. Those in ``fetch`` (default: all) that aren't configured or
    downloaded are downloaded first (ensure_downloaded), so look their paths up inside. A failed
    fetch of one that isn't downloaded becomes a ModelUnavailable naming it; other errors pass."""
    try:
        for name in setting_names if fetch is None else fetch:
            if name in BY_SETTING:
                ensure_downloaded(settings, name)
        yield
    except ModelUnavailable:
        raise
    except Exception as e:
        cause = _fetch_failure(e)
        missing = [
            BY_SETTING[n]
            for n in setting_names
            if n in BY_SETTING and not settings.configured(n) and not BY_SETTING[n].downloaded(settings.models_dir)
        ]
        if cause is None or not missing:
            raise
        chain = f"{e} {cause}"
        spec = next((s for s in missing if s.repo in chain), missing[0])
        source = "ModelScope" if spec.hub == "modelscope" else "Hugging Face"
        reason = str(cause).strip().splitlines()[0][:160] if str(cause).strip() else type(cause).__name__
        raise ModelUnavailable(
            f"{spec.name} isn't downloaded, and fetching it from {source} failed ({reason}). "
            "Download it in the Models tab (a mirror can be chosen there)."
        ) from e


def any_available(settings, setting_names) -> bool:
    """Whether any of these models is configured explicitly or downloaded (config.Settings)."""
    return any(
        settings.configured(name) or BY_SETTING[name].downloaded(settings.models_dir)
        for name in setting_names
        if name in BY_SETTING
    )


# (megabytes done, megabytes in all); may raise to cancel.
Progress = Callable[[int, int], None]


def _partial_bytes(folder: Path) -> int:
    """Bytes of files being downloaded into ``folder`` (huggingface_hub's .incomplete files)."""
    cache = folder / ".cache" / "huggingface" / "download"
    return sum(p.stat().st_size for p in cache.rglob("*.incomplete")) if cache.is_dir() else 0


class _Received:
    """Bytes the hubs report as they download a file (their progress hooks count here)."""

    def __init__(self) -> None:
        self.written = 0
        self.transferred = 0
        self._lock = threading.Lock()

    @property
    def bytes(self) -> int:
        return max(self.written, self.transferred)

    def reset(self) -> None:
        with self._lock:
            self.written = self.transferred = 0

    def add(self, n: int, transfer: bool = False) -> None:
        with self._lock:
            if transfer:
                self.transferred += n
            else:
                self.written += n

    def hf_bar(self) -> type:
        """A progress bar class for huggingface_hub that counts into this. A Xet download
        (also through a mirror) reports through one such bar the bytes received
        (``update_transfer``, steadily) and written (``update``, in a few large steps
        as the file is put together); progress follows whichever is further."""
        from huggingface_hub.utils import tqdm

        received = self

        class Bar(tqdm):
            def update(self, n=1):
                if n:
                    received.add(int(n))
                return super().update(n)

            def update_transfer(self, n=1) -> None:
                if n:
                    received.add(int(n), transfer=True)

            def set_transfer_postfix_str(self, *args, **kwargs) -> None:
                pass

        return Bar

    def modelscope_callback(self) -> type:
        """A ModelScope download callback class that counts into this."""
        from modelscope.hub.callback import ProgressCallback

        received = self

        class Callback(ProgressCallback):
            def update(self, size: int) -> None:
                received.add(int(size))

        return Callback


def _polled(fn: Callable[[], None], report: Callable[[], None], name: str) -> None:
    """Run ``fn`` on a thread, calling ``report`` meanwhile (it may raise to cancel:
    ``fn`` then finishes in the background); re-raise what ``fn`` raised."""
    failure: list[BaseException] = []

    def run() -> None:
        try:
            fn()
        except BaseException as e:  # re-raised on the caller's thread
            failure.append(e)

    worker = threading.Thread(target=run, daemon=True, name=name)
    worker.start()
    while worker.is_alive():
        report()
        worker.join(POLL_S)
    if failure:
        raise failure[0]


def unpack(archive: Path, folder: Path, files: tuple[str, ...]) -> None:
    """Extract ``files`` from a zip whose entries sit in one top folder, into
    ``folder`` without it. Each lands complete or not at all (written aside, then renamed)."""
    with zipfile.ZipFile(archive) as zf:
        by_name = {n.split("/", 1)[1]: n for n in zf.namelist() if "/" in n and not n.endswith("/")}
        missing = [f for f in files if f not in by_name]
        if missing:
            raise FileNotFoundError(f"{archive.name} has no {', '.join(missing)}.")
        for name in files:
            out = folder / name
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            with zf.open(by_name[name]) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(out)


def download(
    spec: ModelSpec,
    models_dir: Path,
    endpoint: str | None = None,
    progress: Progress | None = None,
    way: str = "prebuilt",
) -> None:
    """Download the files of ``spec`` that are missing into its folder, from
    ``endpoint`` (default: huggingface.co). Interrupted files resume. When
    ``progress`` raises (cancel), the file in flight finishes in the background.
    ``way`` (a spec with a fallback, ModelSpec.ways): "prebuilt", or "convert" / "convert_keep"
    to build it here from its source (keeping the source's files after)."""
    if way not in WAYS or (way != "prebuilt" and not spec.fallback):
        raise ValueError(f"{spec.name} can't be downloaded as {way!r}.")
    # One download of a model at a time (the Models tab and a feature that needs it may both ask):
    # the second waits, then fetches only what is still missing.
    with _folder_lock(spec.folder):
        _download(spec, models_dir, endpoint, progress or (lambda done, total: None), way)


def _download(spec: ModelSpec, models_dir: Path, endpoint: str | None, progress: Progress, way: str) -> None:
    endpoint = endpoint or None
    folder = models_dir / spec.folder
    mb = 1_000_000  # decimal MB, as size_mb and the hub's sizes
    if spec.source:
        _download_and_build(spec, models_dir, endpoint, progress, spec.source)
        return
    if spec.fallback and way != "prebuilt":
        _download_and_build(spec, models_dir, endpoint, progress, spec.fallback, keep=way == "convert_keep")
        return
    if spec.fallback:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

        folder.mkdir(parents=True, exist_ok=True)
        try:
            _fetch_hf(spec.repo, spec.revision, folder, spec.files, endpoint,
                      lambda done, total: progress(done // mb, total // mb))  # fmt: skip
            return
        except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, FileNotFoundError) as e:
            logger.warning("%s isn't available (%s); building %s from its source instead", spec.repo, e, spec.name)
        _download_and_build(spec, models_dir, endpoint, progress, spec.fallback)
        return
    folder.mkdir(parents=True, exist_ok=True)
    received = _Received()
    if spec.hub == "modelscope":  # one call; its callback counts the bytes (total: the listed size)
        from modelscope import snapshot_download

        callbacks = [received.modelscope_callback()]
        target = folder / ".download" if spec.archive else folder
        patterns = [spec.archive] if spec.archive else list(spec.files)

        def fetch_all() -> None:
            snapshot_download(spec.repo, local_dir=str(target), allow_patterns=patterns, progress_callbacks=callbacks)

        _polled(
            fetch_all,
            lambda: progress(min(received.bytes // mb, spec.size_mb), spec.size_mb),
            f"download-{spec.folder}",
        )
        if spec.archive:
            unpack(target / spec.archive, folder, spec.files)
            shutil.rmtree(target, ignore_errors=True)
        progress(spec.size_mb, spec.size_mb)
        return
    _fetch_hf(
        spec.repo, spec.revision, folder, spec.files, endpoint, lambda done, total: progress(done // mb, total // mb)
    )


def _fetch_hf(
    repo: str,
    revision: str | None,
    folder: Path,
    files: tuple[str, ...],
    endpoint: str | None,
    report: Callable[[int, int], None],
) -> None:
    """Download the ``files`` of ``repo`` missing in ``folder``; ``report(bytes done, bytes in all)``
    may raise to cancel (the file in flight finishes in the background)."""
    from huggingface_hub import HfApi, hf_hub_download

    if all((folder / f).is_file() for f in files):  # nothing to fetch: no need to ask the hub
        total = sum((folder / f).stat().st_size for f in files)
        report(total, total)
        return
    info = HfApi(endpoint=endpoint).model_info(repo, revision=revision, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    missing = [f for f in files if f not in sizes]
    if missing:
        raise FileNotFoundError(f"{repo} has no {', '.join(missing)}.")
    total = sum(sizes[f] for f in files)
    done = sum(sizes[f] for f in files if (folder / f).is_file())
    received = _Received()
    bar = received.hf_bar()
    for name in files:
        if (folder / name).is_file():
            continue
        # Downloaded on a thread so progress (and cancelling) can be polled meanwhile:
        # the bytes the hub reports, or what is on disk (a resumed file starts there).
        received.reset()

        def fetch(name: str = name) -> None:
            hf_hub_download(repo, name, local_dir=folder, endpoint=endpoint, revision=revision, tqdm_class=bar)

        def poll(name: str = name, done: int = done) -> None:
            report(done + min(max(received.bytes, _partial_bytes(folder)), sizes[name]), total)

        _polled(fetch, poll, f"download-{folder.name}")
        done += sizes[name]
        report(done, total)


BUILD_SHARE = 0.25  # of the download, how much of the progress bar a build takes


def _download_and_build(
    spec: ModelSpec, models_dir: Path, endpoint: str | None, progress: Progress, source: Source, keep: bool = False
) -> None:
    """Download ``source`` (resuming), build the model's folder from it, then remove it (unless
    ``keep`` or ``source.keep``)."""
    import importlib

    repo, revision = (source.repo, source.revision) if source.repo else (spec.repo, spec.revision)
    src = models_dir / source.folder
    src.mkdir(parents=True, exist_ok=True)
    mb = 1_000_000  # decimal MB, as size_mb and the hub's sizes
    downloaded = [spec.size_mb * mb]

    def report(done: int, total: int) -> None:
        downloaded[0] = total
        progress(done // mb, int(total * (1 + BUILD_SHARE)) // mb)

    _fetch_hf(repo, revision, src, source.files, endpoint, report)
    total = downloaded[0]
    module, name = source.build.split(":")
    build = getattr(importlib.import_module(module), name)
    build(src, models_dir / spec.folder, lambda f: report(total + int(total * BUILD_SHARE * f), total))
    if not (source.keep or keep):
        shutil.rmtree(src, ignore_errors=True)
