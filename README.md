# VoxCPM Studio

LoRA training and fast inference for [VoxCPM](https://github.com/OpenBMB/VoxCPM), with a web UI.

VoxCPM is used as a library, pinned to an upstream commit in `pyproject.toml`
(PyPI 2.0.3 predates `generate(seed=...)`; switch to PyPI once a newer release
ships). This project owns the UI, the API, model lifecycle and LoRA management.

To develop against a local VoxCPM checkout instead, temporarily set
`voxcpm = { path = "../VoxCPM", editable = true }` under `[tool.uv.sources]`.

## Layout

```
src/voxcpm_studio/
  config.py        settings from VOXCPM_STUDIO_* env vars
  lora_library.py  checkpoint discovery and metadata
  inference.py     model cache, LoRA hot-swap, synthesis
  api.py           FastAPI routes; serves web/dist when built
web/               Vite + React + TypeScript frontend
tests/             run without a GPU (fake model loader)
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
| `VOXCPM_STUDIO_LORA_DIR` | `lora` | where LoRA checkpoints are found |
| `VOXCPM_STUDIO_DEVICE` | `auto` | `cuda`, `cuda:1`, `cpu`, ... |
| `VOXCPM_STUDIO_OPTIMIZE` | `auto` | Triton `torch.compile`; `auto` = on for CUDA |
| `VOXCPM_STUDIO_DENOISER` | `1` | load the ZipEnhancer denoiser |

`run.bat` defaults the model and LoRA dirs to the data in an existing
`../VoxCPM` checkout (data only, no code); set the variables to use other paths.

## Development

```powershell
uv run pytest
uv run voxcpm-studio            # API on :8808
cd web; npm run dev             # UI on :5173, proxies /api to :8808
```

## Roadmap

- Training: job runner around VoxCPM's `scripts/train_voxcpm_finetune.py`
  (start / stop / resume, live loss and step progress, sample audio). The script
  is not part of the `voxcpm` package, so it will be vendored from the pinned commit.
- Dataset tools: import, validation (`voxcpm.training.validate`), ASR transcripts.
- Streaming synthesis via `VoxCPM.generate_streaming`.
