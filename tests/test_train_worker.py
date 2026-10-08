import os
import subprocess
import sys

from voxcpm_studio import train_worker


def test_sample_paths_for_tensorboard_tags(tmp_path):
    assert train_worker.sample_path(tmp_path, "val_sample_1/generated_audio", 500) == tmp_path / "step_0000500_1.wav"
    assert train_worker.sample_path(tmp_path, "val_sample_0/reference_audio", 500) == tmp_path / "reference_0.wav"
    assert train_worker.sample_path(tmp_path, "val_sample_0/mel_spectrogram", 500) is None


def test_process_alive():
    assert train_worker.process_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not train_worker.process_alive(proc.pid)


def test_stop_file_runs_the_sigint_handler(tmp_path):
    # Mirrors how the training script registers its checkpoint-saving handler.
    script = tmp_path / "job.py"
    script.write_text(
        "import signal, sys, threading, time\n"
        "from pathlib import Path\n"
        "from voxcpm_studio import train_worker\n"
        "train_worker.POLL_S = 0.02\n"
        "signal.signal(signal.SIGINT, lambda *a: (print('saved', flush=True), sys.exit(0)))\n"
        f"threading.Thread(target=train_worker.watch_for_stop, args=(Path(r'{tmp_path / 'stop'}'), None), daemon=True).start()\n"
        "print('ready', flush=True)\n"
        "while True: time.sleep(0.01)\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    (tmp_path / "stop").touch()

    assert proc.wait(timeout=20) == 0
    assert "saved" in proc.stdout.read()
