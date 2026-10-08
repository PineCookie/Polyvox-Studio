"""Training subprocess: runs the vendored VoxCPM training script unmodified.

``python -m voxcpm_studio.train_worker <train_config.yaml> --samples-dir <dir>
--stop-file <path> [--parent-pid <pid>]``

On top of the script this adds:

- Graceful stop. When the stop file appears, or the parent process exits,
  the script's own SIGINT handler runs: it saves a checkpoint and exits. On
  Windows a parent cannot deliver SIGINT/SIGTERM to a child, and
  ``Popen.terminate()`` kills it without saving.

  This is not done by reading stdin: on Windows, DataLoader worker processes
  query the inherited std handles at startup, and that query blocks forever
  behind a pending synchronous read on the same pipe.
- Validation samples. Audio the script logs to TensorBoard is also written as
  WAV files that the UI can play.
"""

from __future__ import annotations

import _thread
import argparse
import os
import re
import runpy
import signal
import sys
import threading
import time
from pathlib import Path

SCRIPT = Path(__file__).with_name("vendor") / "train_voxcpm_finetune.py"
SAMPLE_TAG = re.compile(r"val_sample_(\d+)/(generated|reference)_audio")
POLL_S = 1.0


def sample_path(samples_dir: Path, tag: str, step: int) -> Path | None:
    m = SAMPLE_TAG.fullmatch(tag)
    if not m:
        return None
    index, kind = m.groups()
    name = f"step_{step:07d}_{index}.wav" if kind == "generated" else f"reference_{index}.wav"
    return samples_dir / name


def save_samples_to(samples_dir: Path) -> None:
    import numpy as np
    import soundfile as sf
    from tensorboardX import SummaryWriter

    add_audio = SummaryWriter.add_audio

    def add_audio_and_save(self, tag, snd_tensor, global_step=None, sample_rate=44100, *args, **kwargs):
        add_audio(self, tag, snd_tensor, global_step, sample_rate, *args, **kwargs)
        path = sample_path(samples_dir, tag, int(global_step or 0))
        if path is None:
            return
        try:
            samples_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            sf.write(tmp, np.asarray(snd_tensor, dtype=np.float32).reshape(-1), int(sample_rate), format="WAV")
            tmp.replace(path)  # the UI never sees a partial file
        except Exception as e:
            print(f"[studio] Failed to save sample {path.name}: {e}", file=sys.stderr, flush=True)

    SummaryWriter.add_audio = add_audio_and_save


def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import _winapi

        try:
            handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid)
        except OSError:
            return False
        try:
            return _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_TIMEOUT
        finally:
            _winapi.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # on Windows this would terminate the process
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def watch_for_stop(stop_file: Path, parent_pid: int | None) -> None:
    while not stop_file.exists():
        if parent_pid is not None and not process_alive(parent_pid):
            print("[studio] Server exited.", file=sys.stderr, flush=True)
            break
        time.sleep(POLL_S)
    print("[studio] Stop requested; saving a checkpoint.", file=sys.stderr, flush=True)
    _thread.interrupt_main(signal.SIGINT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()

    save_samples_to(args.samples_dir)
    threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid), daemon=True).start()
    sys.argv = [str(SCRIPT), "--config_path", str(args.config_path)]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
