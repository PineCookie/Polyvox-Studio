#!/usr/bin/env bash
# Start the Polyvox Studio server on Linux (the counterpart of run.bat).
# Arguments go to the server, e.g. ./run.sh --host 0.0.0.0 --port 8808
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# UTF-8 mode, as on Windows (Torch Inductor writes generated sources as text).
export PYTHONUTF8=1
# Models downloaded in the Models tab live here; settings left at their
# defaults use them (see docs/GUIDE.md: Models).
export POLYVOX_MODELS_DIR="${POLYVOX_MODELS_DIR:-$here/models}"
export POLYVOX_LORA_DIR="${POLYVOX_LORA_DIR:-$here/lora}"
export POLYVOX_DATA_DIR="${POLYVOX_DATA_DIR:-$here/data}"

# uv installs Python and the dependencies; offer to install it when it's missing.
no_uv() {
  echo "Install uv, then run this again:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. It installs Python and everything else the studio needs" >&2
  echo "(https://docs.astral.sh/uv/, a single program)." >&2
  printf "Install uv now? [Y/n] " >&2
  # No answer to read (not run from a terminal): don't install.
  read -r answer || answer=n
  case "$answer" in [nN]*) no_uv ;; esac
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || no_uv
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh || no_uv
  else
    echo "Neither curl nor wget is installed." >&2
    no_uv
  fi
  # The installer puts uv on PATH for new shells; this one needs it too.
  export PATH="${UV_INSTALL_DIR:+$UV_INSTALL_DIR:}${XDG_BIN_HOME:+$XDG_BIN_HOME:}$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv was installed, but this shell can't find it yet: open a new one and run this again." >&2
    exit 1
  fi
fi

# Mirrors in mainland China: asked once before the first install, the answer kept in .mirror
# (cn or off; POLYVOX_MIRROR overrides it).
mirror="${POLYVOX_MIRROR:-}"
if [ -z "$mirror" ] && [ -f "$here/.mirror" ]; then
  mirror="$(head -n 1 "$here/.mirror" | tr -d '[:space:]')"
fi
if [ -z "$mirror" ] && [ ! -d "$here/.venv" ]; then
  # Suggested when the system is set to China's time zone or a zh_CN locale.
  suggest=n
  zone="${TZ:-$(cat /etc/timezone 2>/dev/null || readlink /etc/localtime 2>/dev/null || true)}"
  case "$zone" in *Asia/Shanghai* | *Asia/Chongqing* | *Asia/Harbin* | *Asia/Urumqi* | *PRC*) suggest=y ;; esac
  case "${LC_ALL:-${LANG:-}}" in zh_CN*) suggest=y ;; esac
  choices="[y/N]"
  [ "$suggest" = y ] && choices="[Y/n]"
  echo
  echo "The first start downloads Python, PyTorch and other packages (several GB). In mainland"
  echo "China, mirrors are usually much faster: Tsinghua University for packages, Nanjing"
  echo "University for PyTorch and Python, hf-mirror.com for models."
  printf "Download from these mirrors? %s " "$choices"
  read -r answer || answer=""
  case "${answer:-$suggest}" in [yY]*) mirror=cn ;; *) mirror=off ;; esac
  echo "$mirror" >"$here/.mirror"
  echo "Saved in .mirror (cn or off): change or delete it to choose again."
fi
# Rewrites uv.lock's download addresses (tools/lock_urls.ps1 does the same on Windows): the
# mirrors keep the original paths, so only the host changes; versions and hashes stay as locked.
lock_urls() {
  local lock="$here/uv.lock" script
  [ -f "$lock" ] || return 0
  if [ "$1" = mirror ]; then
    grep -qE 'files\.pythonhosted\.org/packages/|download-r2\.pytorch\.org/whl/(cu[0-9]+|cpu)/' "$lock" || return 0
    script='s#https://files\.pythonhosted\.org/packages/#https://pypi.tuna.tsinghua.edu.cn/packages/#g
s#https://download-r2\.pytorch\.org/whl/(cu[0-9]+|cpu)/#https://mirrors.nju.edu.cn/pytorch/whl/\1/#g'
  else
    grep -qE 'pypi\.tuna\.tsinghua\.edu\.cn/packages/|mirrors\.nju\.edu\.cn/pytorch/whl/' "$lock" || return 0
    script='s#https://pypi\.tuna\.tsinghua\.edu\.cn/packages/#https://files.pythonhosted.org/packages/#g
s#https://mirrors\.nju\.edu\.cn/pytorch/whl/#https://download-r2.pytorch.org/whl/#g'
  fi
  sed -E "$script" "$lock" >"$lock.tmp" && mv "$lock.tmp" "$lock"
}
frozen=()
if [ "$mirror" = cn ]; then
  echo "Downloading from mirrors in mainland China (.mirror)."
  export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"
  # The Models tab's default download source (one chosen there still wins).
  export POLYVOX_HF_ENDPOINT="${POLYVOX_HF_ENDPOINT:-https://hf-mirror.com}"
  # --frozen installs exactly what the lock lists, without resolving against other indexes.
  frozen=(--frozen)
  lock_urls mirror
else
  lock_urls official
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
exec uv run "${frozen[@]}" --project "$here" --no-default-groups --group "$variant" polyvox "$@"