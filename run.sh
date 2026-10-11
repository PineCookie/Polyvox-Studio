#!/usr/bin/env bash
# Start the Polyvox Studio server on Linux (the counterpart of run.bat).
# Arguments go to the server, e.g. ./run.sh --host 0.0.0.0 --port 8808
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# UTF-8 mode, as on Windows (Torch Inductor writes generated sources as text).
export PYTHONUTF8=1
# Models downloaded in the Models tab live here; settings left at their
# defaults use them (see README: Models).
export POLYVOX_MODELS_DIR="${POLYVOX_MODELS_DIR:-$here/models}"
export POLYVOX_LORA_DIR="${POLYVOX_LORA_DIR:-$here/lora}"
export POLYVOX_DATA_DIR="${POLYVOX_DATA_DIR:-$here/data}"

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

# The PyTorch build for this machine: cu130, cu126 or cpu (POLYVOX_TORCH overrides).
variant="$(uv run --no-project --python 3.13 "$here/src/polyvox/torch_variant.py")"
echo "PyTorch build: $variant"
# uv installs Python and the dependencies on the first run (and swaps PyTorch builds).
exec uv run --project "$here" --no-default-groups --group "$variant" polyvox "$@"