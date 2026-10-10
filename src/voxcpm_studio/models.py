"""The models the studio uses, where they live under the models folder, and
downloading them from Hugging Face (or a mirror of it, e.g. hf-mirror.com).

A model counts as downloaded once every file it needs is in its folder, so an
interrupted download is never used. Settings (config.Settings.model_path) use
a downloaded copy unless a model was configured explicitly.
"""

from __future__ import annotations

import threading
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
        "Training: how much samples sound like the dataset's speaker",
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
        "Import: keep only the voice (remove music)",
        "pcunwa/Kim-Mel-Band-Roformer-FT",
        "pcunwa__Kim-Mel-Band-Roformer-FT",
        ("kimmel_unwa_ft2_bleedless.ckpt", "config_kimmel_unwa_ft.yaml"),
        910,
        entry="kimmel_unwa_ft2_bleedless.ckpt",
    ),
    ModelSpec(
        "dereverb_model",
        "Mel-Band RoFormer de-reverb",
        "Import: remove room reverb",
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
        "Import: remove background noise",
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


def download(spec: ModelSpec, models_dir: Path, endpoint: str | None = None, progress: Progress | None = None) -> None:
    """Download the files of ``spec`` that are missing into its folder, from
    ``endpoint`` (default: huggingface.co). Interrupted files resume. When
    ``progress`` raises (cancel), the file in flight finishes in the background."""
    from huggingface_hub import HfApi, hf_hub_download

    progress = progress or (lambda done, total: None)
    endpoint = endpoint or None
    folder = models_dir / spec.folder
    folder.mkdir(parents=True, exist_ok=True)
    if spec.hub == "modelscope":  # small; one call, no byte progress
        from modelscope import snapshot_download

        progress(0, spec.size_mb)
        snapshot_download(spec.repo, local_dir=str(folder), allow_patterns=list(spec.files))
        progress(spec.size_mb, spec.size_mb)
        return
    info = HfApi(endpoint=endpoint).model_info(spec.repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    missing = [f for f in spec.files if f not in sizes]
    if missing:
        raise FileNotFoundError(f"{spec.repo} has no {', '.join(missing)}.")
    total = sum(sizes[f] for f in spec.files)
    done = sum(sizes[f] for f in spec.files if (folder / f).is_file())
    mb = 1 << 20
    for name in spec.files:
        if (folder / name).is_file():
            continue
        # Downloaded on a thread so progress (and cancelling) can be polled meanwhile.
        failure: list[BaseException] = []

        def fetch(name: str = name, failure: list[BaseException] = failure) -> None:
            try:
                hf_hub_download(spec.repo, name, local_dir=folder, endpoint=endpoint)
            except BaseException as e:  # re-raised on the job's thread
                failure.append(e)

        worker = threading.Thread(target=fetch, daemon=True, name=f"download-{spec.folder}")
        worker.start()
        while worker.is_alive():
            progress((done + _partial_bytes(folder)) // mb, total // mb)
            worker.join(POLL_S)
        if failure:
            raise failure[0]
        done += sizes[name]
        progress(done // mb, total // mb)
