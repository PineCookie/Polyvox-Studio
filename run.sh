#!/usr/bin/env bash
# Start the VoxCPM Studio server on Linux (the counterpart of run.bat).
# Arguments go to the server, e.g. ./run.sh --host 0.0.0.0 --port 8808
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# UTF-8 mode, as on Windows (Torch Inductor writes generated sources as text).
export PYTHONUTF8=1
# Models downloaded in the Models tab live here; settings left at their
# defaults use them (see README: Models).
export VOXCPM_STUDIO_MODELS_DIR="${VOXCPM_STUDIO_MODELS_DIR:-$here/models}"
export VOXCPM_STUDIO_LORA_DIR="${VOXCPM_STUDIO_LORA_DIR:-$here/lora}"
export VOXCPM_STUDIO_DATA_DIR="${VOXCPM_STUDIO_DATA_DIR:-$here/data}"

exec uv run --project "$here" voxcpm-studio "$@"