# VoxCPM Studio

LoRA training and fast inference for [VoxCPM](https://github.com/OpenBMB/VoxCPM), with a web UI.

VoxCPM is used as a library, pinned to an upstream commit in `pyproject.toml`
(PyPI 2.0.3 predates `generate(seed=...)`; switch to PyPI once a newer release
ships). This project owns the UI, the API, model lifecycle and LoRA management.

To develop against a local VoxCPM checkout instead, temporarily set
`voxcpm = { path = "../VoxCPM", editable = true }` under `[tool.uv.sources]`.

## Layout

```
models/            base model weights, e.g. models/openbmb__VoxCPM2 (git-ignored)
data/              training datasets: <name>/train.jsonl + audio (git-ignored)
lora/              training runs and LoRA checkpoints (git-ignored)
src/voxcpm_studio/
  config.py        settings from VOXCPM_STUDIO_* env vars
  lora_library.py  checkpoint discovery and metadata
  inference.py     model cache, LoRA hot-swap, synthesis
  manifests.py     training manifests: discovery, path fixes, inspection
  importer.py      dataset import from GPT-SoVITS lists, CSV, JSONL, folders
  slicer.py        long recordings -> clips at pauses (Silero VAD)
  asr.py           speech recognition (Qwen3-ASR, transformers-native)
  aligner.py       forced alignment (Qwen3-ForcedAligner or a wav2vec2 CTC model)
  checks.py        per-clip checks from alignment: mismatch, cut edges, silence
  jobs.py          in-process background jobs with progress and cancel
  training.py      training runs: config, job control, metrics
  train_worker.py  training subprocess around the vendored script
  sharing.py       export / import of shareable LoRA packages
  vendor/          train_voxcpm_finetune.py, unmodified from the pinned commit
  api.py           FastAPI routes; serves web/dist when built
web/               Vite + React + TypeScript frontend
tests/             run without a GPU (fake model loader and training worker)
```

## Setup

Requires Windows with an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and Node 20+.

```powershell
uv sync
cd web; npm install; npm run build; cd ..
.\run.bat
```

Open http://127.0.0.1:8808. The first generation after a model load takes about
30 s while Triton compiles; later ones are fast.

## Configuration

| Variable | Default | |
|---|---|---|
| `VOXCPM_STUDIO_MODEL` | `openbmb/VoxCPM2` | local model dir or Hugging Face id |
| `VOXCPM_STUDIO_LORA_DIR` | `lora` | LoRA checkpoints and training runs |
| `VOXCPM_STUDIO_DATA_DIR` | `data` | JSONL training manifests |
| `VOXCPM_STUDIO_DEVICE` | `auto` | inference device: `cuda`, `cuda:1`, `cpu`, ... |
| `VOXCPM_STUDIO_TRAIN_GPU` | | CUDA index for training jobs (default: the first GPU) |
| `VOXCPM_STUDIO_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B-hf` | speech recognition model (dir or Hugging Face id) |
| `VOXCPM_STUDIO_ASR_DEVICE` | `auto` | ASR and aligner device; `auto` = the inference device |
| `VOXCPM_STUDIO_ALIGN_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | forced aligner |
| `VOXCPM_STUDIO_CTC_ALIGN_MODEL` | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | wav2vec2 CTC aligner (character vocabulary, one language) |
| `VOXCPM_STUDIO_OPTIMIZE` | `auto` | Triton `torch.compile`; `auto` = on for CUDA |
| `VOXCPM_STUDIO_DENOISER` | `1` | load the ZipEnhancer denoiser |

`run.bat` defaults to the project's own `models/openbmb__VoxCPM2`, `lora/` and
`data/`; set the variables to use other paths. To fetch the models:

```powershell
hf download openbmb/VoxCPM2 --local-dir models/openbmb__VoxCPM2
hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir models/Qwen__Qwen3-ASR-1.7B-hf  # optional
hf download Qwen/Qwen3-ForcedAligner-0.6B-hf --local-dir models/Qwen__Qwen3-ForcedAligner-0.6B-hf  # optional
hf download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn --local-dir models/jonatasgrosman__wav2vec2-large-xlsr-53-chinese-zh-cn  # optional
```

Speech recognition (the **Transcribe** button for reference audio) uses
Qwen3-ASR through transformers itself, so it needs no extra packages. It loads
on first use (about 4 GB of GPU memory); Unload frees it with the TTS model.

Starting a training job unloads the inference model if both use the same GPU.
To keep synthesizing while training, put them on different GPUs, e.g.
`VOXCPM_STUDIO_DEVICE=cuda:1`.

## Training

The Train tab starts, stops and resumes LoRA runs, one at a time. A new run
starts from a dataset: its stats, warnings and a browsable list of its clips
(play, transcript, length, search; ASR-made transcripts are marked), then the
training parameters. A run lives in `<lora dir>/<name>/` in the layout
`lora_ft_webui.py` uses, so its runs appear too and can be resumed or
continued with more steps.

- Stop saves a checkpoint first; Resume continues from it.
- Validation clips (held out from the training set by default) produce loss
  points and audio samples every "Validate every" steps.
- Each run gets copies of its manifests with absolute audio paths. Webui
  manifests hold paths relative to the folder holding `data/` and `lora/`
  (`data/<name>/wavs/...`), which the current
  `voxcpm` loader would resolve against the manifest's folder instead.

## Preparing datasets

The Prepare tab turns raw recordings and transcript lists into datasets in the
data dir, from a path on this computer:

- a GPT-SoVITS `.list` (`path|speaker|lang|text`),
- a `file|text` CSV (LJSpeech style),
- a `.jsonl` manifest, or
- a folder of audio files with same-name `.txt` / `.lab` transcripts.

Lists often hold stale paths (another machine, a moved folder). Each clip is
looked up as written, by file name next to the list and in its `wavs/` folder,
and by file name in an optional audio folder. A preview shows what was found
before anything is written; rows without audio are skipped. The import writes
`data/<name>/train.jsonl` with paths relative to it and, by default, copies the
audio into `data/<name>/wavs/` so the folder is self-contained.

Long recordings (over 20 s) can be sliced into clips during the import: voice
activity detection (Silero VAD) finds the speech, and cuts are chosen at the
longest pauses so clips stay within a length range (2-15 s by default) and end
between sentences rather than mid-word.

Clips that have audio but no transcript (a folder of bare recordings, slices,
or empty text in a list) can be transcribed with Qwen3-ASR during the import, optionally
with a fixed language and names / terms to spell right. Imports run in the
background with progress and can be cancelled. Transcribed clips are listed in
`data/<name>/asr.json` so they can be reviewed.

### Checking clips against their text

Forced alignment places each word of a transcript in its audio. The import (and
**Check clips** on any dataset) uses it to flag clips for review:

| Flag | Meaning |
|---|---|
| Mismatch | part of the text does not match the audio (low confidence over a few words) |
| Untranscribed | voice activity outside the aligned words: the text may miss something |
| Cut edge | the sound is still loud at the start or end of the file: a word may be cut |
| Long silence | over 1 s of silence before or after the speech |

Two aligners, both running in this environment: **Qwen3-ForcedAligner** (11
languages, mixed scripts) and a **wav2vec2 CTC** model with a character
vocabulary (sharper on single wrong words; Chinese by default, set
`VOXCPM_STUDIO_CTC_ALIGN_MODEL` for another language). The import can also trim
long silent edges from the copied clips, keeping 0.15 s and never cutting into
anything voice activity detection hears. Results are saved in
`data/<name>/align.json` and shown in the clips table, which can show only
flagged clips and sort them worst match first. A check made for a transcript
that was edited since is ignored.

Each row shows the clip's waveform with the span the text covers shaded; click
it to play from that point. Expanding a row (its length) shows a larger
waveform with every word where the aligner placed it; words it is unsure of,
usually not what was said, are marked in orange on both.

Mostly not caught: a transcript that leaves out a few words (about 1 in 7 is
flagged). The aligner spreads the remaining words over the audio; comparing with
ASR output (planned) will find those.

## Sharing LoRAs

Select a checkpoint in the Voice panel and choose **Export for sharing**. The zip
holds only `lora_weights.safetensors` (bfloat16 by default: half the size, same
output), `lora_config.json` with the base model as a Hugging Face id, and a
README with usage for VoxCPM Studio, the `voxcpm` Python API and its CLI. Local
paths, optimizer state, logs, transcripts and dataset audio are left out.

**Import** in the Voice panel adds such a zip to the LoRA dir.

## Development

```powershell
uv run pytest
uv run ruff check; uv run ruff format   # Python lint / format
uv run voxcpm-studio            # API on :8808
cd web; npm run dev             # UI on :5173, proxies /api to :8808
cd web; npm run lint            # ESLint + Prettier check; `npm run format` fixes
```

`src/voxcpm_studio/vendor/` is excluded from Ruff: it stays byte-identical to upstream.

## Roadmap

- Preprocessing: a per-clip audit table (ASR disagreement, speaker outliers,
  edit / drop clips), loudness normalization, optional vocal separation.
- Training speed: batch size / token budget, cuDNN attention.
- VoxCPM 1.5 as a second base model.
- Streaming synthesis via `VoxCPM.generate_streaming`.
