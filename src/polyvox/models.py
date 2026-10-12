"""The models the studio uses, where they live under the models folder, and
downloading them from Hugging Face (or a mirror of it, e.g. hf-mirror.com).

A model counts as downloaded once every file it needs is in its folder, so an
interrupted download is never used. Settings (config.Settings.model_path) use
a downloaded copy unless a model was configured explicitly.
"""

from __future__ import annotations

import shutil
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HF_ENDPOINT = "https://huggingface.co"
POLL_S = 0.5


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
    required: bool = False
    hub: str = "huggingface"  # or "modelscope" (no endpoint choice there)
    # A zip in the repo that holds ``files`` (one top folder, dropped when unpacking).
    archive: str | None = None

    def path(self, models_dir: Path) -> Path:
        folder = models_dir / self.folder
        return folder / self.entry if self.entry else folder

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
        required=True,
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
    ),
)

BY_SETTING = {m.setting: m for m in MODELS}

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


def download(spec: ModelSpec, models_dir: Path, endpoint: str | None = None, progress: Progress | None = None) -> None:
    """Download the files of ``spec`` that are missing into its folder, from
    ``endpoint`` (default: huggingface.co). Interrupted files resume. When
    ``progress`` raises (cancel), the file in flight finishes in the background."""
    from huggingface_hub import HfApi, hf_hub_download

    progress = progress or (lambda done, total: None)
    endpoint = endpoint or None
    folder = models_dir / spec.folder
    folder.mkdir(parents=True, exist_ok=True)
    mb = 1 << 20
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
    info = HfApi(endpoint=endpoint).model_info(spec.repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    missing = [f for f in spec.files if f not in sizes]
    if missing:
        raise FileNotFoundError(f"{spec.repo} has no {', '.join(missing)}.")
    total = sum(sizes[f] for f in spec.files)
    done = sum(sizes[f] for f in spec.files if (folder / f).is_file())
    bar = received.hf_bar()
    for name in spec.files:
        if (folder / name).is_file():
            continue
        # Downloaded on a thread so progress (and cancelling) can be polled meanwhile:
        # the bytes the hub reports, or what is on disk (a resumed file starts there).
        received.reset()

        def fetch(name: str = name) -> None:
            hf_hub_download(spec.repo, name, local_dir=folder, endpoint=endpoint, tqdm_class=bar)

        def report(name: str = name, done: int = done) -> None:
            in_flight = min(max(received.bytes, _partial_bytes(folder)), sizes[name])
            progress((done + in_flight) // mb, total // mb)

        _polled(fetch, report, f"download-{spec.folder}")
        done += sizes[name]
        progress(done // mb, total // mb)
