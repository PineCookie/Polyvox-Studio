# Polyvox Studio

English | [简体中文](README_zh.md)

Train text-to-speech voices from your own recordings and use them, all in one
web app on your own computer. Works with [VoxCPM2](https://github.com/OpenBMB/VoxCPM),
[dots.tts](https://github.com/rednote-hilab/dots.tts) and
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).

## Highlights

- **Three TTS models, one workflow.** VoxCPM2, dots.tts and GPT-SoVITS (v2Pro,
  v2ProPlus, v4). Prepare a dataset once and train any of them.
- **Raw recordings in, clean dataset out.** Hours of audio, even m4a or video
  files, are cut into clips between words and transcribed automatically. Music,
  reverb and noise can be removed.
- **Bad clips found for you.** Wrong transcripts, cut-off words, other speakers,
  distortion and singing are flagged, so you only review what needs it.
- **See how training goes.** Loss charts, audio samples and a "sounds like the
  speaker" score at every checkpoint.
- **Faster than the original code, same quality.** Synthesis up to 4x faster,
  training steps up to 3x, and switching voices takes about a second.
- **Fits your GPU.** Settings for 8, 12 and 16 GB cards are picked for you.
- **GPT-SoVITS compatible.** Bring the voices you trained in its webui; voices
  trained here work there too.
- **Easy to install.** One command, no Python or FFmpeg setup. Windows and
  Linux; interface in English and 简体中文.

## Quick start

1. Install [uv](https://docs.astral.sh/uv/):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
   curl -LsSf https://astral.sh/uv/install.sh | sh                                       # Linux
   ```

2. Download the [latest release](../../releases/latest) (`.zip` for Windows,
   `.tar.gz` for Linux), unpack it and run `run.bat` or `./run.sh`. The first
   start installs everything, which takes a few minutes.
3. Open http://127.0.0.1:8808 and download **VoxCPM2** in the **Models** tab.
4. Type something in **Synthesize** and press **Generate**.

## How it works

1. **Prepare:** create a dataset, add a folder of recordings or a transcript
   list, and run the steps. Fix or drop the clips it flags.
2. **Train:** pick a model and start. Listen to the samples as checkpoints
   arrive, and keep the one that sounds best.
3. **Synthesize:** choose your voice, type, and generate.

## Requirements

- Windows or Linux.
- An NVIDIA GPU with 8 GB of memory or more; 16 GB trains at full speed. AMD
  GPUs on Linux should work but are untested. Without a GPU you can only
  prepare datasets.
- Disk space for models: about 5 GB per TTS model, plus about 6 GB for the
  dataset tools (speech recognition and alignment).

## Learn more

The [guide](docs/GUIDE.md) covers each step in depth: dataset preparation and
the clip checks, training options, the speed optimizations and their
measurements, models, configuration and development.
