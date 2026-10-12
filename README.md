# Polyvox Studio

English | [简体中文](README_zh.md)

Train text-to-speech voices from your own recordings, and use them, in one web
app on your own computer. Polyvox Studio turns raw audio into a clean dataset,
fine-tunes a voice on [VoxCPM2](https://github.com/OpenBMB/VoxCPM),
[dots.tts](https://github.com/rednote-hilab/dots.tts) or
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), and synthesizes with it.

The interface is in English and Simplified Chinese (简体中文).

## Highlights

- **Three TTS models, one workflow.** VoxCPM2 and dots.tts (LoRA training, 48 kHz)
  and GPT-SoVITS v2Pro, v2ProPlus and v4 (voice training). A dataset prepared
  once trains any of them, and voices of every model sit side by side.
- **From hours of raw audio to a dataset.** Point it at recordings (WAV, FLAC,
  mp3, m4a, even video files) or at existing transcript lists. Long recordings
  are cut between words with speech recognition and forced alignment, clips
  without text are transcribed, and music, reverb or noise can be removed first.
- **Every clip checked.** Clips are flagged when the text doesn't match the
  audio, a word is cut off, someone else is speaking, the audio is clipped or
  sung, or there is stray sound at the edges. Review them worst first: listen,
  edit the text, trim, or drop.
- **Training you can follow.** Loss charts, audio samples at each checkpoint,
  and a speaker-similarity score that shows how close each checkpoint sounds to
  the voice. Stop, resume and continue any run; try a checkpoint in one click.
- **Good defaults.** New runs pick their length from the dataset size and use
  Prodigy Schedule-Free, which finds its own learning rate. Batch sizes follow
  your GPU's memory (8, 12 or 16 GB and up).
- **Faster than the models' own code, with the same results.** dots.tts
  synthesizes about 4x faster and GPT-SoVITS about 2.5x; switching between
  VoxCPM2 or dots.tts voices takes about a second instead of a reload. Training
  steps run about 1.6x (VoxCPM2) to 3x (dots.tts) faster, and VoxCPM2 needs half
  the GPU memory. See [Speed](#speed).
- **Works with GPT-SoVITS.** Import voices you trained in the GPT-SoVITS webui;
  voices trained here load in the webui too.
- **Share voices** as small zip packages, free of local paths, logs and dataset
  audio, and import them with one click.
- **Easy to install.** One command; [uv](https://docs.astral.sh/uv/) brings
  Python, PyTorch and everything else. No FFmpeg needed. Windows and Linux.

## Quick start

1. Install uv:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
   curl -LsSf https://astral.sh/uv/install.sh | sh                                       # Linux
   ```

2. Download the `.zip` (Windows) or `.tar.gz` (Linux) of the
   [latest release](../../releases/latest), unpack it, and run `run.bat`
   (Windows) or `./run.sh` (Linux) in its folder. The first start installs
   everything, which takes a few minutes.
3. Open http://127.0.0.1:8808, go to the **Models** tab and download **VoxCPM2**
   (about 5 GB). Add the others when you need them (see [Models](#models)).
4. Type something in **Synthesize** and press **Generate**. The first generation
   after a model loads takes about 30 s while it is compiled; later ones are fast.

An NVIDIA GPU with 8 GB or more is recommended; see [Requirements](#requirements).

## How it works

1. **Prepare** a dataset: **New dataset**, then **Add source** with a folder or a
   transcript list, and **Run the remaining steps**. Look through the flagged
   clips and fix or drop the bad ones.
2. **Train** a voice: **Train on it** on the dataset's page, pick the model and
   start. Watch the charts and listen to the samples as checkpoints arrive.
3. **Synthesize** with it: the voice appears in the Voice list. Pick it (and a
   checkpoint, if you like), type, and generate.

## Synthesizing

The **Voice** panel lists the base models and every trained voice, grouped by
model; a voice with several checkpoints lets you pick one. Then:

- **Text**, with VoxCPM2's non-verbal tags (laugh, sigh, ...) a click away.
- **Style** (VoxCPM2): describe the voice, e.g. "a calm, warm voice, speaking
  slowly".
- **Reference audio** clones a voice from a clip; adding its transcript
  (**Transcribe** fills it in) usually makes the result closer.
- **Advanced**: guidance, steps, seed and the model's own settings.
- **History** keeps this session's clips to compare and download.

GPT-SoVITS always speaks from a reference clip: each of its voices has a default
one, picked from the training data, that you can change in the Voice panel.

## Preparing datasets

The Prepare tab turns raw recordings and transcript lists into datasets in the
data folder. **New dataset** makes an empty one; on its page, **Add source** takes
a path on this computer:

- a GPT-SoVITS `.list` (`path|speaker|lang|text`),
- a `file|text` CSV (LJSpeech style),
- a `.jsonl` manifest, or
- a folder of audio files, with same-name `.txt` / `.lab` transcripts or without.

Lists often hold stale paths (another machine, a moved folder). Each clip is
looked up as written, by file name next to the list and in its `wavs/` folder,
and by file name in an optional audio folder. **Scan** shows what was found
before anything is written; rows without audio are skipped. Clips go into
`data/<name>/train.jsonl` and, by default, are copied into `data/<name>/wavs/` so
the folder is self-contained. Long recordings (over 20 s) are listed in
`data/<name>/dataset.json` and stay where they are until they are sliced.
Sources can be added at any time; audio already in the dataset is skipped.

The page lists the dataset's steps in order: **Sources**, **Clean up**
(optional), **Slice recordings**, **Transcribe**, **Check clips** and **Tidy**.
Each step shows what still needs it (a dataset may mix recordings and ready
clips, or have stopped halfway), keeps its settings for next time, works only on
what is left, and saves as it goes, so a stopped step loses little. **Run the
remaining steps** runs the needed ones in order; Add source can start that right
after adding. Steps run in the background with progress and can be cancelled.

**Slicing.** Long recordings are read 10 minutes at a time, so hours of audio
are fine. Voice activity detection (Silero VAD) finds the speech, Qwen3-ASR
transcribes it and a forced aligner places every word. Cuts fall only between
words, at real pauses, preferring sentence ends and keeping clips within a length
range (2-15 s by default); each clip keeps its exact part of the transcript,
punctuation included.

**Transcribing.** Clips without text are kept, and the Transcribe step fills
them in with Qwen3-ASR, optionally with a fixed language and names or terms to
spell right; until then they are left out of training. Transcribed clips are
listed in `data/<name>/asr.json` so they can be reviewed.

**Deleting.** **Delete…** on a dataset's page removes its folder after showing
what goes with it (clips, size, training runs that used it) and asking to
confirm.

### Optional clean-up: separation, de-reverb, denoise

The Clean up step cleans the audio with the RoFormer models that UVR5 and
MSST-WebUI use (architectures vendored from ZFTurbo's Music-Source-Separation-
Training, MIT): **separate** the voice from music and background, **remove
reverb**, **remove noise**. Each is off by default and available once its model
is downloaded (Models tab). Clips are cleaned in place; recordings are cleaned
while they are sliced. The de-reverb weights are GPL-3.0.

The default vocal model is a "bleedless" one: for training data, music left in
the voice is worse than a slightly thinner voice. Separation adds time and
leaves small artifacts, so use it for recordings with music or background under
the voice, not for clean ones. It does not remove singing.

Any MSST-style checkpoint works: set `POLYVOX_VOCALS_MODEL`,
`POLYVOX_DEREVERB_MODEL` or `POLYVOX_DENOISE_MODEL` to `model.ckpt` (its config:
the same-name or only `.yaml` beside it) or `model.ckpt|config.yaml`.

### Checking clips

Forced alignment places each word of a transcript in its audio. The **Check
clips** step uses it, and a few other measures, to flag clips for review (new
clips and edited texts only; **Check all again** redoes every clip):

| Flag | Meaning |
|---|---|
| Singing | a sung clip (karaoke, a cover): not wanted in training data |
| Other voice? | the voice is unlike the rest of the dataset: maybe a guest or played-back audio |
| Clipped | the waveform is cut flat at its peak (recorded or saved too loud) |
| Mismatch | part of the text does not match the audio |
| Missing words? | speech recognition hears words the transcript lacks (with **Compare with ASR**) |
| Filler? | a filler or laugh in the text (嗯, 哈哈, 呵呵, ...) placed on quiet audio: maybe not said |
| Edge sound | sound before or after the words, a pause apart (a sigh, 嗯, a laugh, or words the text misses) |
| Untranscribed | speech right next to the aligned words: the text may miss something |
| Cut edge | the sound is still loud at the start or end of the file: a word may be cut |
| Long silence | over 0.5 s of silence after the speech (the model learns when to stop from it), or over 1 s before it |

Two aligners are available: **Qwen3-ForcedAligner** (11 languages, mixed
scripts) and a **wav2vec2 CTC** model (sharper on single wrong words; Chinese by
default, `POLYVOX_CTC_ALIGN_MODEL` for another language). A transcript that
leaves out a few words is mostly not caught by alignment, so **Compare with
ASR** transcribes each clip again and flags Missing words. Singing detection
(an AudioSet tagger) is off by default: it slows checks down and most datasets
have no songs. Results are saved in `data/<name>/align.json`.

The **Tidy** step trims edges (long silence, and a sigh or 嗯 a pause away from
the words, keeping 0.15 s) and can bring every clip to one loudness. Clips
flagged Mismatch or Missing words are left for review. Clips are rewritten in
place; originals are kept in `originals/` next to the manifest.

### Reviewing clips

The clips table can show only flagged clips, worst first. Each row shows the
clip's waveform with the span the text covers shaded; click it to play from that
point. Open a clip to see every word where the aligner placed it (words it is
unsure of are marked) and:

- **Edit** its transcript (or **Use ASR text**); saving checks that clip again.
- **Trim…**: drag the handles on the waveform; letting go plays the new edge.
- **Check again**, or **Ask ASR** to compare with what speech recognition hears.
- **Drop** it: it stays in the table, dimmed, but training leaves it out
  (listed in `data/<name>/excluded.json`); **Restore** brings it back.

**Trim all** trims every clip flagged Edge sound or Long silence (the table greys
what it would cut), and with one flag shown, **Drop all** drops every clip that
has it, e.g. all sung clips.

A note on fillers and laughs: for training, their count matters less than
consistency, so the Transcribe step shortens runs of one filler in every text
(哈哈哈哈 → 哈哈, 嗯嗯嗯 → 嗯嗯; on by default). Different fillers stay different
(哈哈 is a laugh, 呵呵 a "he he"), and transcripts keep plain characters rather
than tags like `[laughing]`, which can add stray sounds.

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

## Training

The Train tab starts, stops and resumes runs, one at a time. **New run** starts
from a dataset (its stats, warnings and clips to browse), then the model and the
training settings; the name is suggested from the dataset and model.

| Model | What is trained | Defaults |
|---|---|---|
| VoxCPM2 | a LoRA (rank 32) on the language models and the diffusion decoder | 4 epochs, Prodigy Schedule-Free |
| dots.tts | a LoRA (rank 32) on the language model and the diffusion head | 4 epochs, Prodigy Schedule-Free |
| GPT-SoVITS v2Pro / v2ProPlus | both models in full, SoVITS then GPT, as the webui does | 8 + 15 epochs |
| GPT-SoVITS v4 | a LoRA on the SoVITS diffusion model, then the GPT in full | 8 + 15 epochs |

- **Stop** saves a checkpoint first; **Resume** continues from it, and
  **Continue to** adds more steps to a finished run.
- **Progress** shows the loss charts (each folds away), the learning rate,
  and, from clips held out of the training set, validation loss and audio samples
  at each checkpoint. **Try** opens a checkpoint in Synthesize.
- **Speaker similarity** scores each sample against the training set's voice
  (CAM++ speaker embeddings) next to the score the set's own clips reach, so you
  can pick the checkpoint that sounds most like the speaker.
- **Optimizer:** Prodigy Schedule-Free
  ([prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free))
  finds its own step size for each part of the model, so the learning rate is a
  multiplier (1). AdamW with a cosine schedule and Prodigy are the alternatives.

A run lives in `lora/<model>/<name>/` (`voxcpm`, `dots_tts` or `gpt_sovits`). Runs
from VoxCPM's `lora_ft_webui.py` use the same layout, so they appear too and can be
resumed or continued. Runs and voices lying directly in `lora/` (from older
versions, or copied in) are moved into their model's folder when the server
starts.

## Speed

Synthesis and training run on optimized paths written for this studio. Each was
checked against the original code: the same loss curves in training, and in
synthesis the same output up to floating-point rounding (with the same speaker
similarity and transcripts), so the speed costs nothing in quality.
Measured on a 16 GB NVIDIA GPU; the baseline is the model's own code where it
has one, else a plain PyTorch loop:

| | What is done | Gain |
|---|---|---|
| Switching voices (VoxCPM2, dots.tts) | LoRAs of any rank are written into the running, compiled model | about 1 s instead of a 30-40 s reload |
| dots.tts synthesis | compiled; every decoding step replayed as a CUDA graph | about 4x faster |
| GPT-SoVITS synthesis | the GPT's decoding step as one CUDA graph with a static cache and fused layer math; v4's diffusion model compiled | about 2.5x faster (v2Pro / v2ProPlus), 1.3x (v4) |
| VoxCPM2 training | frozen weights in bfloat16 (half the memory, so batches of 4 fit 16 GB); each clip's audio encoded once; transformer layers fused | about 1.6x faster per step; 2.3x on 8 GB cards |
| dots.tts training | each clip encoded once and cached on disk; no vocabulary logits; fused layers | about 3x faster per step |
| GPT-SoVITS training | no per-parameter GPU syncs; features cached per clip in the dataset; v4's diffusion blocks compiled | about 1.2x faster per step; starts in seconds |
| Preparing datasets | compiled vocal separation; voice detection and speaker embeddings batched on the GPU | separation 2x, voice detection 3.5x, the other-voice check about 10x |

Compiling takes a little time once per model load (and longer on the very first
run, while caches fill); `POLYVOX_OPTIMIZE=0` turns it off.

## Sharing voices

Select a voice in the Voice panel and choose **Export**. A LoRA zip
holds only the weights (bfloat16 by default: half the size, same output), its
configuration with the base model as a Hugging Face id, and a README with usage.
A GPT-SoVITS voice exports both models in the GPT-SoVITS webui's format, with
its reference clip and transcript. Local paths, optimizer state, logs and
dataset audio are left out.

**Import** in the Voice panel adds such a zip. **GPT-SoVITS…** imports a pair
of weights trained in the GPT-SoVITS webui (v2, v2Pro, v2ProPlus or v4), with a
reference clip from a dataset or a file.

## Requirements

Windows or Linux. [uv](https://docs.astral.sh/uv/) installs Python 3.13,
PyTorch and everything else on the first run, and `run.bat` / `run.sh` pick the
PyTorch build for the machine (`POLYVOX_TORCH` overrides):

| Build | For |
|---|---|
| `cu130` | NVIDIA RTX 20 series or newer (incl. RTX 50) with a driver for CUDA 13 |
| `cu126` | older NVIDIA GPUs (e.g. GTX 10 series) or drivers (CUDA 12.6) |
| `rocm` | AMD GPUs on Linux (ROCm 7.2: recent Radeon and Instinct; untested) |
| `cpu` | no GPU: datasets can be prepared, but synthesis is slow and training impractical |

What a GPU can do depends on its memory; the studio picks batch sizes and which
models share the GPU accordingly:

| GPU memory | Synthesis | Training | Preparing datasets |
|---|---|---|---|
| 16 GB or more | yes | yes | all models stay loaded |
| 12 GB | yes; unloaded while preparing | yes; some models with gradient checkpointing (slower) | smaller batches |
| 8 GB | yes; reference denoising on the CPU | yes, with gradient checkpointing (slower) | one model on the GPU at a time |
| under 8 GB | no (needs about 8 GB) | no | as for 8 GB |

Smaller cards keep the effective batch, so they train slower, not differently. A
batch that still runs out of memory is retried in halves. Starting a training run
unloads the synthesis model if both use the same GPU; to keep synthesizing while
training, put them on different GPUs (e.g. `POLYVOX_DEVICE=cuda:1`).

No FFmpeg install is needed: audio in other formats (m4a, video files) is
decoded with PyAV, which brings its own. On Linux, Triton (used to compile the
models) needs a C compiler (`apt install build-essential`).

To reach the studio from another machine, start it with
`./run.sh --host 0.0.0.0`. It has no login: use it on a trusted network or
through an SSH tunnel.

## Models

The **Models** tab lists every model the studio uses, with its size, license and
whether it is downloaded, and downloads the missing ones with progress into
`models/`. Only VoxCPM2 is required; the rest are needed only for what they do:

| Model | For | Size |
|---|---|---|
| VoxCPM2 | synthesis and LoRA training | 5 GB |
| dots.tts SOAR | synthesis and LoRA training | 4.9 GB |
| GPT-SoVITS v2Pro / v2ProPlus, G2PW | synthesis and voice training (G2PW: Chinese pronunciation) | 1.7 + 0.6 GB |
| GPT-SoVITS v4 | synthesis and voice training at 48 kHz | 0.8 GB |
| Qwen3-ASR 1.7B | transcribing, slicing, Compare with ASR | 4.1 GB |
| Qwen3-ForcedAligner 0.6B | slicing and checking clips | 1.9 GB |
| wav2vec2 Chinese | second aligner for checks | 1.3 GB |
| CAM++, ZipEnhancer, AST | speaker similarity and Other voice; denoising reference clips; singing detection | 0.4 GB |
| RoFormer vocals / de-reverb / denoise | the optional Clean up step | 0.9 GB each |

Downloads come from Hugging Face, a mirror of it (hf-mirror.com, or any URL;
`POLYVOX_HF_ENDPOINT` sets the server's default) or ModelScope, fetch only the
files that are used, and resume when interrupted. A model downloaded while the
server runs is used from its next load, no restart needed. A Hugging Face model
that is not downloaded still works: it is fetched into the Hugging Face cache
the first time it loads (without progress).

## Configuration

Environment variables, all optional:

| Variable | Default | |
|---|---|---|
| `POLYVOX_MODELS_DIR` | `models` | downloaded models |
| `POLYVOX_HF_ENDPOINT` | | Hugging Face mirror for downloads, e.g. `https://hf-mirror.com` |
| `POLYVOX_LORA_DIR` | `lora` | training runs, LoRAs and voices |
| `POLYVOX_DATA_DIR` | `data` | datasets |
| `POLYVOX_DEVICE` | `auto` | synthesis device: `cuda`, `cuda:1`, `cpu`, ... |
| `POLYVOX_TRAIN_GPU` | | CUDA index for training (default: the first GPU) |
| `POLYVOX_ASR_DEVICE` | `auto` | speech recognition and aligner device; `auto` = the synthesis device |
| `POLYVOX_VRAM_GB` | | use at most this much GPU memory (a shared GPU), and pick the settings for it |
| `POLYVOX_OPTIMIZE` | `auto` | compile models with Triton (`torch.compile`); `auto` = on for CUDA |
| `POLYVOX_DENOISER` | `1` | load the ZipEnhancer denoiser for reference clips |
| `POLYVOX_MODEL` | `openbmb/VoxCPM2` | VoxCPM2: a local folder or Hugging Face id |
| `POLYVOX_DOTS_MODEL` | `rednote-hilab/dots.tts-soar` | dots.tts: a local folder or Hugging Face id |
| `POLYVOX_GSV_MODEL` | (downloaded copy) | GPT-SoVITS: a folder laid out like `lj1995/GPT-SoVITS` (v2Pro, s1v3.ckpt, encoders) |
| `POLYVOX_GSV_G2PW_MODEL` | (downloaded copy) | GPT-SoVITS: the G2PW folder |
| `POLYVOX_GSV_V4_MODEL` | (downloaded copy) | GPT-SoVITS v4: a folder with `s2Gv4.pth` and `vocoder.pth` |
| `POLYVOX_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B-hf` | speech recognition model |
| `POLYVOX_ALIGN_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | forced aligner |
| `POLYVOX_CTC_ALIGN_MODEL` | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | wav2vec2 CTC aligner (one language) |
| `POLYVOX_DENOISER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | its folder or ModelScope id |

`run.bat` / `run.sh` use the project's own `models/`, `lora/` and `data/`; set
the variables to use other folders.

## Development

```powershell
uv run pytest                            # no GPU needed (fake model loader and training worker)
uv run ruff check; uv run ruff format    # Python lint / format
uv run polyvox                           # API on :8808
cd web; npm run dev                      # UI on :5173, proxies /api to :8808
cd web; npm run lint                     # ESLint + Prettier check; `npm run format` fixes
```

From a git checkout, the first `run.bat` / `run.sh` also builds the web UI, which
needs Node.js 20+. GitHub Actions run the tests, linters and web build on Ubuntu
and Windows for every push; pushing a tag like `v0.4.0` publishes a release whose
archives include the built web UI.

VoxCPM is used as a library, pinned to an upstream commit in `pyproject.toml`
(PyPI 2.0.3 predates `generate(seed=...)`). To develop against a local VoxCPM
checkout, temporarily set `voxcpm = { path = "../VoxCPM", editable = true }`
under `[tool.uv.sources]`. `src/polyvox/vendor/` is excluded from Ruff: it stays
as upstream has it, with any changes listed in each package's `__init__.py`.

```
models/            downloaded models (git-ignored)
data/              datasets: <name>/train.jsonl, audio, dataset.json (git-ignored)
lora/              training runs, LoRAs and voices: voxcpm/, dots_tts/, gpt_sovits/ (git-ignored)
src/polyvox/
  config.py        settings from POLYVOX_* env vars
  models.py        the models used, and downloading them
  vram.py          settings per GPU size (batches, which models share the GPU)
  engines/         TTS engines behind one interface (base.py; __init__.py: the registry)
    voxcpm/        engine.py; inference.py (model cache, LoRA hot-swap); worker.py (training)
    dots/          engine.py; lora.py (LoRA layers, merging); worker.py (training)
    gpt_sovits/    engine.py; synth.py (synthesis); voices.py (voice folders, webui import);
                   features.py + worker.py (training: SoVITS then GPT)
  data/            manifests.py (training manifests); record.py (dataset.json);
                   audio_io.py (audio without a system FFmpeg)
  prep/            dataset preparation: steps.py (the dataset page's steps), importer.py,
                   slicer.py, asr.py, aligner.py, checks.py, separation.py, singing.py,
                   fillers.py, loudness.py, exporter.py
  lora_library.py  checkpoint discovery and metadata
  jobs.py          background jobs with progress and cancel
  training.py      training runs: config, job control, metrics
  speaker.py       speaker similarity (CAM++)
  sharing.py       export / import of voice packages
  vendor/          train_voxcpm_finetune.py (unmodified); roformer/ (MSST, MIT);
                   dots_tts/ (Apache-2.0); gpt_sovits/ (MIT; rebuilt by
                   tools/vendor_gpt_sovits.py from a pinned upstream commit)
  server/          the FastAPI app: app.py, context.py (services, GPU rules),
                   schemas.py, one module of routes per area
web/               Vite + React + TypeScript frontend
tests/             backend tests
```

## Roadmap

- Streaming synthesis: playback starts before generation ends.
- VoxCPM 1.5 as another base model.
- Keeping a long recording's own transcript when it is sliced.
