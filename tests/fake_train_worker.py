"""Stands in for train_worker in tests: mimics the training script's output files.

Logs 3 steps, saves a sample, then finishes if num_iters <= 20; otherwise waits
for the stop file and saves a checkpoint like the real SIGINT handler. A
"crash" file in the run dir makes it exit with an error instead.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

config_path, samples_dir, stop_file = map(Path, sys.argv[1:4])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
save_dir = Path(config["save_path"])
save_dir.mkdir(parents=True, exist_ok=True)


def save(step):
    latest = save_dir / "latest"
    latest.mkdir(exist_ok=True)
    (latest / "training_state.json").write_text(json.dumps({"step": step}))


start = 0
state = save_dir / "latest" / "training_state.json"
if state.exists():
    start = json.loads(state.read_text())["step"]

with (save_dir / "train.log").open("a", encoding="utf-8") as log:
    for step in range(start, start + 21, 10):
        line = f"[train] step {step}: loss/diff: 0.5, loss/stop: 0.1, lr: 0.0001, epoch: 0.1, grad_norm: 0.2"
        log.write(line + (", log interval: 2.00s\n" if step > start else "\n"))
    log.write(f"[val] step {start}: loss/total: 0.7, loss/diff: 0.6, loss/stop: 0.1\n")

samples_dir.mkdir(exist_ok=True)
sf.write(samples_dir / f"step_{start:07d}_0.wav", np.zeros(480, dtype=np.float32), 48000)
print("progress 50%\rprogress 100%", flush=True)

if config["num_iters"] <= 20:
    save(config["num_iters"])
    sys.exit(0)

print("waiting for stop", flush=True)
while not stop_file.exists():
    if (config_path.parent / "crash").exists():
        sys.exit(1)
    time.sleep(0.02)
save(start + 20)
