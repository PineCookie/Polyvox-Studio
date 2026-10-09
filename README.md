# VoxCPM Studio

LoRA training and fast inference for [VoxCPM](https://github.com/OpenBMB/VoxCPM), with a web UI.

VoxCPM is used as a library, pinned to an upstream commit in `pyproject.toml`
(PyPI 2.0.3 predates `generate(seed=...)`; switch to PyPI once a newer release
ships). This project owns the UI, the API, model lifecycle and LoRA management.

To develop against a local VoxCPM checkout instead, temporarily set
`voxcpm = { path = "../VoxCPM", editable = true }` under `[tool.uv.sources]`.

## Layout

```
models/            downloaded models, e.g. models/openbmb__VoxCPM2 (git-ignored)
data/              training datasets: <name>/train.jsonl + audio (git-ignored)
lora/              training runs and LoRA checkpoints (git-ignored)
src/voxcpm_studio/
  config.py        settings from VOXCPM_STUDIO_* env vars
  models.py        the models used, and downloading them (Hugging Face / mirror)
  lora_library.py  checkpoint discovery and metadata
  inference.py     model cache, LoRA hot-swap, synthesis
  manifests.py     training manifests: discovery, path fixes, inspection
  importer.py      dataset import from GPT-SoVITS lists, CSV, JSONL, folders
  slicer.py        long recordings -> clips at pauses (Silero VAD)
  asr.py           speech recognition (Qwen3-ASR, transformers-native)
  aligner.py       forced alignment (Qwen3-ForcedAligner or a wav2vec2 CTC model)
  checks.py        per-clip checks from alignment: mismatch, cut edges, silence
  separation.py    optional vocal separation / de-reverb / denoise (RoFormer)
  jobs.py          in-process background jobs with progress and cancel
  training.py      training runs: config, job control, metrics
  train_worker.py  training subprocess around the vendored script
  sharing.py       export / import of shareable LoRA packages
  vendor/          train_voxcpm_finetune.py, unmodified from the pinned commit;
                   roformer/ from Music-Source-Separation-Training (MIT)
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

Open http://127.0.0.1:8808 and download the models in the **Models** tab
(VoxCPM2 is required, about 5 GB; the rest only for preparing datasets). The
first generation after a model load takes about 30 s while Triton compiles;
later ones are fast.

## Models

The **Models** tab lists every model the studio uses, with its size, license and
whether it is downloaded, and downloads the missing ones with progress into the
models folder (`models/`; `VOXCPM_STUDIO_MODELS_DIR`). Downloads come from
Hugging Face or a mirror of it (hf-mirror.com, or any URL; the server default is
`VOXCPM_STUDIO_HF_ENDPOINT`), only the files that are used, and resume when
interrupted. A model downloaded while the server runs is used from its next
load, no restart needed.

A Hugging Face model that is not downloaded still works: it is fetched into the
Hugging Face cache the first time it loads (with no progress shown; the mirror
setting applies). The import's clean-up models are only used from the models
folder. A model variable set explicitly (below) always wins.

## Configuration

| Variable | Default | |
|---|---|---|
| `VOXCPM_STUDIO_MODELS_DIR` | `models` | downloaded models (see Models) |
| `VOXCPM_STUDIO_HF_ENDPOINT` | | Hugging Face mirror for downloads, e.g. `https://hf-mirror.com` |
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
| `VOXCPM_STUDIO_DENOISER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | its dir or ModelScope id |

`run.bat` uses the project's own `models/`, `lora/` and `data/`; set the
variables to use other paths.

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

Long recordings (over 20 s, any format FFmpeg reads: WAV, FLAC, m4a, mp3, ...)
can be sliced into clips during the import. They are read 10 minutes at a time,
so hours of audio are fine. Voice activity detection (Silero VAD) groups the
speech into pieces of up to 28 s, Qwen3-ASR transcribes them, and the forced
aligner places every word. Cuts then fall only between words, at pauses
measured on the waveform, preferring sentence ends and keeping clips within a
length range (2-15 s by default); each clip keeps its exact part of the
transcript, punctuation included.

Clips that have audio but no transcript (a folder of bare recordings, slices,
or empty text in a list) can be transcribed with Qwen3-ASR during the import, optionally
with a fixed language and names / terms to spell right. Imports run in the
background with progress and can be cancelled. Transcribed clips are listed in
`data/<name>/asr.json` so they can be reviewed.

### Optional clean-up: separation, de-reverb, denoise

The import can first clean the audio with the RoFormer models UVR5 and
MSST-WebUI use (architectures vendored from ZFTurbo's Music-Source-Separation-
Training, MIT): **separate** the voice from music and background, **remove
reverb**, **remove noise**. Each is an import option, off by default, and
available once its model is downloaded (Models tab). The de-reverb weights are
GPL-3.0.

Any MSST-style checkpoint works: set `VOXCPM_STUDIO_VOCALS_MODEL`,
`VOXCPM_STUDIO_DEREVERB_MODEL` or `VOXCPM_STUDIO_DENOISE_MODEL` to `model.ckpt`
(its config: the same-name or only `.yaml` beside it) or `model.ckpt|config.yaml`.
The default vocal model is a "bleedless" one: for training data, music left in the
voice is worse than a slightly thinner voice. It adds time to an import and
leaves small artifacts (alignment scores on clean speech drop a little), so use it
for recordings with music or background under the voice, not for clean ones.
Separation does not remove singing.

### Checking clips against their text

Forced alignment places each word of a transcript in its audio. The import (and
**Check clips** on any dataset) uses it to flag clips for review:

| Flag | Meaning |
|---|---|
| Singing | a sung clip (karaoke, a cover): not wanted in training data |
| Mismatch | part of the text does not match the audio (low confidence over a few words; fillers not counted) |
| Filler? | a filler or laugh in the text (嗯, 哈哈, 呵呵, ...) placed on quiet audio: maybe not said, e.g. one 呵 too many |
| Edge sound | sound before or after the words, a pause apart (a sigh, 嗯, a laugh, or words the text misses) |
| Untranscribed | voice activity right next to the aligned words: the text may miss something |
| Cut edge | the sound is still loud at the start or end of the file: a word may be cut |
| Long silence | over 0.5 s of silence after the speech (VoxCPM's guide: the model learns when to stop from it), or over 1 s before it |

Two aligners, both running in this environment: **Qwen3-ForcedAligner** (11
languages, mixed scripts) and a **wav2vec2 CTC** model with a character
vocabulary (sharper on single wrong words; Chinese by default, set
`VOXCPM_STUDIO_CTC_ALIGN_MODEL` for another language). The import can also trim
the copied clips' edges: long silence, and sound before the first or after the
last transcribed word (a laugh, 嗯) when a pause separates it, keeping 0.15 s.

For an existing dataset, the clip view trims one clip (**Trim…**: drag the handles on the waveform; letting go plays the new edge) or
transcribes just its edge sound (**Add to text**), and **Trim all** trims every
clip flagged Edge sound or Long silence, leaving for review those also flagged
Mismatch or Missing words, or starting / ending with a filler the aligner could not
place. The table greys what trimming would cut, and every waveform has a time ruler. Clips are rewritten in place (WAV / FLAC / OGG inside
the dataset folder); the originals are kept in `originals/` next to the manifest.

**Fillers and laughs.** ASR and aligners are unsure of them (is it 呵呵呵 or
呵呵呵呵?), and for TTS training the count does not matter, only consistency. By
default the import shortens runs of one filler to two (哈哈哈哈 → 哈哈, 嗯嗯嗯 →
嗯嗯); **Tidy fillers** does the same for an existing dataset (keeping the
original manifest as `train.jsonl.orig`). Different fillers stay different: 哈哈
is a laugh, 呵呵 a "he he". VoxCPM2's non-verbal tags (`[laughing]`, `[Uhm]`, ...)
sound much the same in Chinese text but can add stray fillers, so transcripts
keep plain characters. Results are saved in
`data/<name>/align.json` and shown in the clips table, which can show only
flagged clips and sort them worst match first. A check made for a transcript
that was edited since is ignored.

Each row shows the clip's waveform with the span the text covers shaded; click
it to play from that point. Expanding a row (its length) shows a larger
waveform with every word where the aligner placed it; words it is unsure of,
usually not what was said, are marked in orange on both.

A transcript that leaves out a few words is mostly not caught by alignment (the
aligner spreads the remaining words over the audio). **Compare with ASR**, an
option of Check clips, transcribes each clip again and flags **Missing words?**
where ASR hears words the transcript lacks; the clip view shows what ASR heard,
with the differences marked. Texts made by ASR are not compared with themselves.

### Reviewing clips

Filter the clips table by flag, worst match first, and open a clip:

- **Edit** its transcript (or **Use ASR text**); saving checks that clip again
  (and so does trimming it).
- **Check again** aligns it now; **Ask ASR** transcribes it and compares with the
  text, also in a dataset checked without ASR.
- **Drop** it: it stays in the table, dimmed, but training runs started from then
  on leave it out (listed in `data/<name>/excluded.json`); **Restore** brings it
  back. The manifest itself keeps every clip.

In a filtered list, a clip you edit, trim or drop stays listed (with its new flags)
until the filter or search changes. With one flag shown, **Drop all** drops every
clip that has it, e.g. all sung clips.

Singing is detected with an AudioSet audio tagger (AST, run through
transformers; `VOXCPM_STUDIO_SINGING_MODEL`) by its singing scores together
with a low speech score, so talk over background music is not taken for
singing. It also works on separated vocals; very short sing-song exclamations
can be flagged too. It is off by default (it slows checks down, and most
datasets have no songs).

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

- Preprocessing: speaker outliers, loudness normalization.
- Training speed: batch size / token budget, cuDNN attention.
- VoxCPM 1.5 as a second base model.
- Streaming synthesis via `VoxCPM.generate_streaming`.
