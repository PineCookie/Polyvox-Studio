# Polyvox Studio

LoRA training and fast inference for several TTS models, with dataset
preparation and a web UI. Models (engines): [VoxCPM](https://github.com/OpenBMB/VoxCPM)
2 and [dots.tts](https://github.com/rednote-hilab/dots.tts); datasets, runs and
tools are shared between them.

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
src/polyvox/
  config.py        settings from POLYVOX_* env vars
  models.py        the models used, and downloading them (Hugging Face / mirror)
  vram.py          settings per GPU size (batches, which models share the GPU)
  engines/         TTS engines behind one interface (base.py: synthesis settings,
                   capabilities, training config; __init__.py: the registry)
    voxcpm/        engine.py; inference.py (model cache, LoRA hot-swap, synthesis);
                   worker.py (training subprocess around the vendored script)
    dots/          engine.py (dots.tts, 48 kHz); lora.py (LoRA layers, merging);
                   worker.py (training subprocess)
    gpt_sovits/    engine.py (GPT-SoVITS v2Pro / v2ProPlus, 32 kHz); synth.py (the
                   synthesis pipeline); voices.py (voice folders with a default
                   reference clip, import of weights the GPT-SoVITS webui trained);
                   features.py + worker.py (training: full fine-tuning, SoVITS stage)
  data/            manifests.py (training manifests: discovery, path fixes,
                   inspection); audio_io.py (audio without a system FFmpeg)
  prep/            dataset preparation:
    importer.py    import from GPT-SoVITS lists, CSV, JSONL, folders
    slicer.py      long recordings -> clips at pauses (Silero VAD)
    asr.py         speech recognition (Qwen3-ASR, transformers-native)
    aligner.py     forced alignment (Qwen3-ForcedAligner or a wav2vec2 CTC model)
    checks.py      per-clip checks from alignment: mismatch, cut edges, silence
    separation.py  optional vocal separation / de-reverb / denoise (RoFormer)
    singing.py, fillers.py, loudness.py, exporter.py
  lora_library.py  checkpoint discovery and metadata (each records its engine)
  jobs.py          in-process background jobs with progress and cancel
  training.py      training runs: config, job control, metrics
  speaker.py       speaker similarity of samples (CAM++)
  sharing.py       export / import of shareable LoRA packages
  vendor/          train_voxcpm_finetune.py, unmodified from the pinned commit;
                   roformer/ from Music-Source-Separation-Training (MIT);
                   dots_tts/ from the dots.tts 0.3.1 wheel (Apache-2.0, changes listed
                   in its __init__.py); gpt_sovits/ from GPT-SoVITS (MIT): the v2Pro
                   synthesis modules and SoVITS losses, Chinese + English text (changes in its __init__.py)
  server/          the FastAPI app (app.py; serves web/dist when built), what the
                   routes share (context.py: services, GPU rules), request / response
                   bodies (schemas.py) and one module of routes per area: status,
                   tts, loras, datasets, clips, training
web/               Vite + React + TypeScript frontend
tests/             run without a GPU (fake model loader and training worker)
```

## Setup

Runs on Windows and Linux with [uv](https://docs.astral.sh/uv/), which installs
Python 3.13, PyTorch and everything else on the first run. `run.bat` / `run.sh`
pick the PyTorch build for the machine (`POLYVOX_TORCH` overrides):

| Build | For |
|---|---|
| `cu130` | NVIDIA RTX 20 series or newer (incl. RTX 50) with a driver for CUDA 13 |
| `cu126` | older NVIDIA GPUs (e.g. GTX 10 series) or drivers (CUDA 12.6) |
| `rocm` | AMD GPUs on Linux (ROCm 7.2: recent Radeon and Instinct; untested) |
| `cpu` | no GPU: datasets can be prepared, but synthesis is slow and training impractical |

Install uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh                                       # Linux
```

Download the `.zip` (Windows) or `.tar.gz` (Linux) of the latest release and run
`run.bat` / `./run.sh` in its folder. Release archives include the built web UI.
From a git checkout instead, the first run also builds the UI, which needs
Node.js 20+.

No FFmpeg install is needed: audio in other formats (m4a, video files) is
decoded with PyAV, which brings its own. On Linux, Triton (used to compile the
model) needs a C compiler (`apt install build-essential`). To reach a server on another
machine, start it with `./run.sh --host 0.0.0.0` (it has no login: use it on a
trusted network or through an SSH tunnel).

Open http://127.0.0.1:8808 and download the models in the **Models** tab
(VoxCPM2 is required, about 5 GB; the rest only for preparing datasets). The
first generation after a model load takes about 30 s while Triton compiles;
later ones are fast.

## Models

The **Models** tab lists every model the studio uses, with its size, license and
whether it is downloaded, and downloads the missing ones with progress into the
models folder (`models/`; `POLYVOX_MODELS_DIR`). Downloads come from
Hugging Face or a mirror of it (hf-mirror.com, or any URL; the server default is
`POLYVOX_HF_ENDPOINT`), only the files that are used, and resume when
interrupted. A model downloaded while the server runs is used from its next
load, no restart needed.

A Hugging Face model that is not downloaded still works: it is fetched into the
Hugging Face cache the first time it loads (with no progress shown; the mirror
setting applies). The import's clean-up models are only used from the models
folder. A model variable set explicitly (below) always wins.

## Configuration

| Variable | Default | |
|---|---|---|
| `POLYVOX_MODELS_DIR` | `models` | downloaded models (see Models) |
| `POLYVOX_HF_ENDPOINT` | | Hugging Face mirror for downloads, e.g. `https://hf-mirror.com` |
| `POLYVOX_MODEL` | `openbmb/VoxCPM2` | local model dir or Hugging Face id |
| `POLYVOX_DOTS_MODEL` | `rednote-hilab/dots.tts-soar` | dots.tts: local model dir or Hugging Face id |
| `POLYVOX_GSV_MODEL` | (downloaded copy) | GPT-SoVITS: a folder laid out like `lj1995/GPT-SoVITS` (v2Pro, s1v3.ckpt, encoders) |
| `POLYVOX_GSV_G2PW_MODEL` | (downloaded copy) | GPT-SoVITS: the G2PW folder (Chinese pronunciation) |
| `POLYVOX_LORA_DIR` | `lora` | LoRA checkpoints and training runs |
| `POLYVOX_DATA_DIR` | `data` | JSONL training manifests |
| `POLYVOX_DEVICE` | `auto` | inference device: `cuda`, `cuda:1`, `cpu`, ... |
| `POLYVOX_TRAIN_GPU` | | CUDA index for training jobs (default: the first GPU) |
| `POLYVOX_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B-hf` | speech recognition model (dir or Hugging Face id) |
| `POLYVOX_ASR_DEVICE` | `auto` | ASR and aligner device; `auto` = the inference device |
| `POLYVOX_ALIGN_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | forced aligner |
| `POLYVOX_CTC_ALIGN_MODEL` | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | wav2vec2 CTC aligner (character vocabulary, one language) |
| `POLYVOX_OPTIMIZE` | `auto` | Triton `torch.compile`; `auto` = on for CUDA |
| `POLYVOX_VRAM_GB` | | use at most this much GPU memory (a shared GPU), and pick the settings for it |
| `POLYVOX_DENOISER` | `1` | load the ZipEnhancer denoiser |
| `POLYVOX_DENOISER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | its dir or ModelScope id |

`run.bat` / `run.sh` use the project's own `models/`, `lora/` and `data/`; set the
variables to use other paths.

Speech recognition (the **Transcribe** button for reference audio) uses
Qwen3-ASR through transformers itself, so it needs no extra packages. It loads
on first use (about 4 GB of GPU memory); Unload frees it with the TTS model.

Starting a training job unloads the inference model if both use the same GPU.
To keep synthesizing while training, put them on different GPUs, e.g.
`POLYVOX_DEVICE=cuda:1`.

### GPU memory

Batch sizes and which models share the GPU follow its memory:

| GPU memory | Synthesis | Training (batch x accumulation) | Preparing datasets |
|---|---|---|---|
| 16 GB or more | yes | 4 x 4 | all models stay loaded |
| 12 GB | yes; unloaded while preparing | 2 x 8 | smaller ASR batches |
| 8 GB | yes; reference denoising on the CPU | 1 x 16, gradient checkpointing (slow) | one model on the GPU at a time |
| under 8 GB | no (needs about 8 GB) | no | as for 8 GB |

The effective batch stays 16, so smaller cards train slower, not differently. A
batch that still runs out of memory is retried in halves.

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
- Optimizer: new runs use Prodigy Schedule-Free
  ([prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free)),
  which finds its own step size for each network (text LM, residual LM,
  diffusion decoder); the learning rate is then a multiplier (1). AdamW with a
  cosine schedule (the training script's own) and Prodigy are the alternatives.
- Speaker similarity: each validation sample is scored against the training
  set's voice with CAM++ speaker embeddings (`POLYVOX_SPEAKER_MODEL`, a
  28 MB ModelScope model, on the CPU), next to the score real clips of the set
  reach. Scores are cached in the run folder (`similarity.json`).
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

Long recordings (over 20 s; WAV, FLAC, mp3, m4a, video files, ...)
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

Any MSST-style checkpoint works: set `POLYVOX_VOCALS_MODEL`,
`POLYVOX_DEREVERB_MODEL` or `POLYVOX_DENOISE_MODEL` to `model.ckpt`
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
`POLYVOX_CTC_ALIGN_MODEL` for another language). The import can also trim
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
transformers; `POLYVOX_SINGING_MODEL`) by its singing scores together
with a low speech score, so talk over background music is not taken for
singing. It also works on separated vocals; very short sing-song exclamations
can be flagged too. It is off by default (it slows checks down, and most
datasets have no songs).

### Exporting for other TTS tools

**Export for other TTS** on a dataset's page writes its training clips (dropped
ones left out) in another tool's layout, by default to `exports/<dataset>_<format>`
next to `data/`:

| Format | Files |
|---|---|
| GPT-SoVITS | `<speaker>.list`: `path\|speaker\|language\|text`, language `zh` / `ja` / `en` / `ko` from each text's script (or one forced for all) |
| LJSpeech | `metadata.csv`: `id\|text\|text` |
| CSV (F5-TTS) | `metadata.csv` with an `audio_file\|text` header |
| Kaldi | `wav.scp`, `text`, `utt2spk`, `spk2utt` |
| Folder | `wavs/<id>.wav` with `wavs/<id>.lab` |

Audio goes to `wavs/` as WAV with plain numbered names, copied or resampled; an
optional clip length range leaves out clips outside it (GPT-SoVITS suggests 1-15 s).

## Sharing LoRAs

Select a checkpoint in the Voice panel and choose **Export for sharing**. The zip
holds only `lora_weights.safetensors` (bfloat16 by default: half the size, same
output), `lora_config.json` with the base model as a Hugging Face id, and a
README with usage for Polyvox Studio, the `voxcpm` Python API and its CLI. Local
paths, optimizer state, logs, transcripts and dataset audio are left out.

**Import** in the Voice panel adds such a zip to the LoRA dir.

## Development

```powershell
uv run pytest
uv run ruff check; uv run ruff format   # Python lint / format
uv run polyvox            # API on :8808
cd web; npm run dev             # UI on :5173, proxies /api to :8808
cd web; npm run lint            # ESLint + Prettier check; `npm run format` fixes
```

`src/polyvox/vendor/` is excluded from Ruff: it stays byte-identical to upstream.

GitHub Actions run the tests, linters and web build on Ubuntu and Windows for
every push. Pushing a tag like `v0.2.0` publishes a release whose `.zip` and
`.tar.gz` include the built web UI.

## Roadmap

- Preprocessing: speaker outliers, loudness normalization.
- Training speed: batch size / token budget, cuDNN attention.
- VoxCPM 1.5 as a second base model.
- Streaming synthesis via `VoxCPM.generate_streaming`.
