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
| `VOXCPM_STUDIO_OPTIMIZE` | `auto` | Triton `torch.compile`; `auto` = on for CUDA |
| `VOXCPM_STUDIO_DENOISER` | `1` | load the ZipEnhancer denoiser |

`run.bat` defaults to the project's own `models/openbmb__VoxCPM2`, `lora/` and
`data/`; set the variables to use other paths. To fetch the model:
`hf download openbmb/VoxCPM2 --local-dir models/openbmb__VoxCPM2`.

Starting a training job unloads the inference model if both use the same GPU.
To keep synthesizing while training, put them on different GPUs, e.g.
`VOXCPM_STUDIO_DEVICE=cuda:1`.

## Training

The Train tab starts, stops and resumes LoRA runs, one at a time. A run lives in
`<lora dir>/<name>/` in the layout `lora_ft_webui.py` uses, so its runs appear
too and can be resumed or continued with more steps.

- Stop saves a checkpoint first; Resume continues from it.
- Validation clips (held out from the training set by default) produce loss
  points and audio samples every "Validate every" steps.
- Each run gets copies of its manifests with absolute audio paths. Webui
  manifests hold paths relative to the folder holding `data/` and `lora/`
  (`data/<name>/wavs/...`), which the current
  `voxcpm` loader would resolve against the manifest's folder instead.

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
uv run voxcpm-studio            # API on :8808
cd web; npm run dev             # UI on :5173, proxies /api to :8808
```

## Roadmap

- Dataset tools: import GPT-SoVITS `.list` / CSV datasets, ASR transcripts.
- Streaming synthesis via `VoxCPM.generate_streaming`.
