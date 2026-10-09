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

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it, then run this again:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Release archives include the built web UI; a git checkout builds it once (Node.js 20+).
if [ ! -f "$here/web/dist/index.html" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "The web UI is not built. Download a release archive, which includes it," >&2
    echo "or install Node.js 20+ and run this again." >&2
    exit 1
  fi
  echo "Building the web UI, first run only..."
  (cd "$here/web" && npm ci && npm run build)
fi

# uv installs Python and the dependencies on the first run.
exec uv run --project "$here" voxcpm-studio "$@"