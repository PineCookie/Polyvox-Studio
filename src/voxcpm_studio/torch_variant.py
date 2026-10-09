"""Pick the PyTorch build for this machine: cu130, cu126 or cpu (dependency
groups in pyproject.toml). Prints the choice; notes go to stderr.

Standard library only: run.bat / run.sh run this file before the project's
environment exists (``uv run --no-project``). VOXCPM_STUDIO_TORCH overrides.

- cu130: a driver with CUDA 13 support and a Turing (RTX 20) or newer GPU.
  CUDA 13 dropped Maxwell, Pascal and Volta; it is the only build here for
  Blackwell (RTX 50), which CUDA 12.6 predates.
- cu126: other NVIDIA GPUs whose driver supports CUDA 12.6.
- rocm: AMD GPUs on Linux (the amdgpu driver's /dev/kfd; the wheels bring the
  ROCm libraries). ROCm 7.2 officially supports recent Radeon (RDNA3 / RDNA4)
  and Instinct GPUs. PyTorch has no official ROCm builds for Windows.
- cpu: otherwise (or a driver too old for the CUDA builds).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

VARIANTS = ("cu130", "cu126", "rocm", "cpu")
ENV = "VOXCPM_STUDIO_TORCH"
CU130_DRIVER_CUDA = (13, 0)
CU126_DRIVER_CUDA = (12, 6)
CU130_MIN_CAPABILITY = (7, 5)  # Turing
BLACKWELL = (10, 0)  # not in the CUDA 12.6 build

Version = tuple[int, int]


def _version(text: str) -> Version | None:
    m = re.match(r"\s*(\d+)\.(\d+)", text)
    return (int(m[1]), int(m[2])) if m else None


def query_gpus() -> tuple[Version | None, list[Version]] | None:
    """(CUDA version the driver supports, compute capability per GPU), or None
    without nvidia-smi or GPUs. Either part is None / [] when the driver doesn't
    report it."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        summary = subprocess.run([smi], capture_output=True, text=True, timeout=30).stdout
        caps = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # "CUDA Version: 13.0"; newer drivers print "CUDA UMD Version: 13.3".
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", summary)
    capabilities = [v for line in caps.stdout.splitlines() if (v := _version(line))] if caps.returncode == 0 else []
    if not m and not capabilities:
        return None  # no GPU, or the driver isn't working
    return (_version(m[1]) if m else None), capabilities


def has_amd_gpu() -> bool:
    """An AMD GPU usable with ROCm: the amdgpu driver's compute device (Linux)."""
    return sys.platform == "linux" and os.path.exists("/dev/kfd")


def choose(gpus: tuple[Version | None, list[Version]] | None, amd: bool = False) -> tuple[str, str | None]:
    """(variant, a note for the user or None). NVIDIA GPUs come first when both are present."""
    if gpus is None and amd:
        return "rocm", None
    if gpus is None:
        return "cpu", "No NVIDIA or AMD GPU found: running on the CPU (synthesis is slow; training is impractical)."
    driver, capabilities = gpus
    best = max(capabilities) if capabilities else None
    if driver is None:  # version not reported: go by the GPU
        return ("cu126" if best is not None and best < CU130_MIN_CAPABILITY else "cu130"), None
    if driver >= CU130_DRIVER_CUDA and (best is None or best >= CU130_MIN_CAPABILITY):
        return "cu130", None
    if best is not None and best >= BLACKWELL:
        return "cu130", "This GPU needs a newer NVIDIA driver (with CUDA 13 support): update it to use the GPU."
    if driver >= CU126_DRIVER_CUDA:
        return "cu126", None
    return "cpu", "The NVIDIA driver is too old for PyTorch's CUDA builds: update it to use the GPU."


def main() -> None:
    forced = os.environ.get(ENV, "").strip().lower()
    if forced:
        if forced not in VARIANTS:
            sys.exit(f"{ENV} must be one of {', '.join(VARIANTS)}, not '{forced}'.")
        if forced == "rocm" and sys.platform != "linux":
            sys.exit("PyTorch's ROCm builds are for Linux only.")
        print(forced)
        return
    variant, note = choose(query_gpus(), has_amd_gpu())
    if note:
        print(note, file=sys.stderr)
    print(variant)


if __name__ == "__main__":
    main()
