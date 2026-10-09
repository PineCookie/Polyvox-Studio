"""LoRA training runs: configuration, job control and progress.

A run is a directory ``<lora_dir>/<name>/`` holding ``train_config.yaml`` (the
layout ``lora_ft_webui`` uses, so its runs show up and can be resumed):

    train_config.yaml   arguments for the training script
    manifests/          the run's own copies of its manifests, absolute paths
    checkpoints/        step_NNNNNNN/, latest/, train.log (written by the script)
    logs/               TensorBoard events
    samples/            validation audio (written by train_worker)
    console.log         the training process's output

One job runs at a time, as a ``train_worker`` subprocess.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from . import manifests
from .config import Settings

logger = logging.getLogger(__name__)

CONFIG_FILE = "train_config.yaml"
ORIGINAL_CONFIG_FILE = "train_config.orig.yaml"  # a webui config, before paths were made absolute
CONSOLE_FILE = "console.log"
STOP_FILE = "stop.request"  # created to ask the worker to save and exit
RUN_NAME = r"^[^\W_][\w.-]{0,79}$"  # Unicode letters too: voice names are often Chinese
SAMPLE_FILE = re.compile(r"step_(\d{7})_(\d+)\.wav")
REFERENCE_FILE = re.compile(r"reference_(\d+)\.wav")
METRIC_LINE = re.compile(r"\[(train|val)\] step (\d+): (.*)")
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
STOP_TIMEOUT_S = 120.0  # saving a checkpoint takes seconds; kill after this
MAX_POINTS = 1000  # per metric series sent to the UI
LOG_TAIL_LINES = 200
LOG_TAIL_BYTES = 128 * 1024


class TrainingConflict(Exception):
    """The request conflicts with the current job or existing runs."""


class TrainParams(BaseModel):
    name: str = Field(pattern=RUN_NAME)
    # Manifest ids relative to the data dir.
    dataset: str
    val_dataset: str | None = None
    # Without val_dataset, hold out this many training clips for validation
    # loss and audio samples.
    val_holdout: int = Field(2, ge=0, le=50)
    num_iters: int = Field(5000, ge=1, le=1_000_000)
    batch_size: int = Field(1, ge=1, le=64)
    # Per-batch token budget: clips too long for it are left out (0 = no limit).
    max_batch_tokens: int = Field(8192, ge=0)
    grad_accum_steps: int = Field(1, ge=1, le=64)
    learning_rate: float = Field(1e-4, gt=0, le=1)
    weight_decay: float = Field(0.01, ge=0)
    warmup_steps: int = Field(100, ge=0)
    max_grad_norm: float = Field(1.0, ge=0)
    log_interval: int = Field(10, ge=1)
    valid_interval: int = Field(500, ge=1)
    save_interval: int = Field(1000, ge=1)
    num_workers: int = Field(2, ge=0, le=16)
    lora_rank: int = Field(32, ge=1, le=512)
    lora_alpha: float = Field(32, gt=0)
    lora_dropout: float = Field(0.0, ge=0, lt=1)
    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = False


class RunSummary(BaseModel):
    name: str
    # running | stopping | complete | incomplete | failed
    status: str
    # Latest logged step while running, else the last saved step.
    step: int | None
    saved_step: int | None
    num_iters: int
    lora_rank: int | None
    lora_alpha: float | None
    train_manifest: str
    val_manifest: str
    updated: float


class Sample(BaseModel):
    step: int
    index: int
    file: str


class RunDetail(RunSummary):
    config: dict
    train: list[dict]  # {"step", "loss/diff", "loss/stop", "lr", "epoch", "grad_norm", "interval"?}
    val: list[dict]  # {"step", "loss/total", "loss/diff", "loss/stop"}
    seconds_per_step: float | None
    samples: list[Sample]
    references: list[str]
    sample_texts: list[str]
    val_clips: int  # clips in the validation manifest (0: none)
    log: list[str]


# --------------------------------------------------------------------------- #
# Reading run files
# --------------------------------------------------------------------------- #


def parse_metrics(log_path: Path) -> dict[str, list[dict]]:
    """Parse the ``[train]`` / ``[val]`` lines the training script logs."""
    series: dict[str, list[dict]] = {"train": [], "val": []}
    if not log_path.is_file():
        return series
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = METRIC_LINE.match(line.strip())
            if not m:
                continue
            split, step, rest = m.group(1), int(m.group(2)), m.group(3)
            point: dict = {"step": step}
            try:
                for part in rest.split(", "):
                    key, _, value = part.partition(": ")
                    if key == "log interval":
                        point["interval"] = float(value.rstrip("s"))
                    else:
                        point[key] = float(value)
            except ValueError:
                continue
            points = series[split]
            # A resumed run restarts at its last checkpoint and logs those steps again.
            while points and points[-1]["step"] >= step:
                points.pop()
            points.append(point)
    return series


def seconds_per_step(train: list[dict], window: int = 20) -> float | None:
    recent = train[-window - 1 :]
    rates = [
        cur["interval"] / (cur["step"] - prev["step"])
        for prev, cur in zip(recent, recent[1:], strict=False)  # consecutive pairs
        if "interval" in cur and cur["step"] > prev["step"]
    ]
    # Median: the interval after a validation also includes sample generation.
    return statistics.median(rates) if rates else None


def downsample(points: list[dict], limit: int = MAX_POINTS) -> list[dict]:
    if len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    kept = points[::stride]
    return kept if kept[-1] is points[-1] else [*kept, points[-1]]


def saved_step(run_dir: Path) -> int | None:
    ckpt_dir = run_dir / "checkpoints"
    try:
        return int(json.loads((ckpt_dir / "latest" / "training_state.json").read_text(encoding="utf-8"))["step"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Checkpoints from before training_state.json existed.
    steps = [int(d.name[5:]) for d in ckpt_dir.glob("step_*") if d.name[5:].isdigit()] if ckpt_dir.is_dir() else []
    if steps:
        return max(steps)
    # Only latest/ kept: it holds the last step the log shows.
    if (ckpt_dir / "latest").is_dir():
        train = parse_metrics(ckpt_dir / "train.log")["train"]
        return train[-1]["step"] if train else None
    return None


def is_complete(saved: int | None, num_iters: int) -> bool:
    # The loop's last step is num_iters - 1; the script saves it before the final save.
    return saved is not None and saved >= num_iters - 1


def tail_lines(path: Path, limit: int = LOG_TAIL_LINES) -> list[str]:
    if not path.is_file():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - LOG_TAIL_BYTES))
        text = f.read().decode("utf-8", errors="replace")
    lines = text.split("\n")
    if size > LOG_TAIL_BYTES:
        lines = lines[1:]  # partial
    # Progress bars redraw with "\r"; keep the final state of each line.
    lines = [line.rstrip("\r").rsplit("\r", 1)[-1] for line in lines]
    while lines and not lines[-1]:
        lines.pop()
    return lines[-limit:]


def read_config(run_dir: Path) -> dict:
    data = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{run_dir / CONFIG_FILE} is not a mapping")
    return data


def write_config(run_dir: Path, config: dict) -> None:
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    (run_dir / CONFIG_FILE).write_text(text, encoding="utf-8")


def resolve_model(model: str) -> tuple[str, int, int]:
    """Return (local path, encoder sample rate, decoder output rate) for a model."""
    from voxcpm.core import resolve_model_path

    path = Path(resolve_model_path(model)).resolve()
    vae = json.loads((path / "config.json").read_text(encoding="utf-8")).get("audio_vae_config") or {}
    return str(path), int(vae.get("sample_rate", 16_000)), int(vae.get("out_sample_rate") or 0)


def build_config(
    params: TrainParams,
    run_dir: Path,
    model: tuple[str, int, int],
    train_manifest: Path,
    val_manifest: Path | None,
) -> dict:
    pretrained_path, sample_rate, out_sample_rate = model
    return {
        "pretrained_path": pretrained_path,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_manifest else "",
        "sample_rate": sample_rate,
        "out_sample_rate": out_sample_rate,
        "batch_size": params.batch_size,
        "max_batch_tokens": params.max_batch_tokens,
        "grad_accum_steps": params.grad_accum_steps,
        "num_workers": params.num_workers,
        "num_iters": params.num_iters,
        "log_interval": params.log_interval,
        "valid_interval": params.valid_interval,
        "save_interval": params.save_interval,
        "learning_rate": params.learning_rate,
        "weight_decay": params.weight_decay,
        "warmup_steps": params.warmup_steps,
        "max_steps": params.num_iters,
        "max_grad_norm": params.max_grad_norm,
        "save_path": str(run_dir / "checkpoints"),
        "tensorboard": str(run_dir / "logs"),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": params.enable_lm,
            "enable_dit": params.enable_dit,
            "enable_proj": params.enable_proj,
            "r": params.lora_rank,
            "alpha": params.lora_alpha,
            "dropout": params.lora_dropout,
            "target_modules_lm": LORA_TARGETS,
            "target_modules_dit": LORA_TARGETS,
        },
    }


def _count_lines(manifest: str) -> int:
    if not manifest or not Path(manifest).is_file():
        return 0
    with open(manifest, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _first_texts(manifest: str, count: int = 2) -> list[str]:
    """Texts of the validation clips the training script generates samples for."""
    if not manifest or not Path(manifest).is_file():
        return []
    texts = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(str(json.loads(line).get("text", "")))
                if len(texts) == count:
                    break
    return texts


# --------------------------------------------------------------------------- #
# Job control
# --------------------------------------------------------------------------- #

# (config path, samples dir, stop file) -> argv
CommandFactory = Callable[[Path, Path, Path], list[str]]


def worker_command(config_path: Path, samples_dir: Path, stop_file: Path) -> list[str]:
    return [
        sys.executable, "-m", "voxcpm_studio.train_worker", str(config_path),
        "--samples-dir", str(samples_dir),
        "--stop-file", str(stop_file),
        "--parent-pid", str(os.getpid()),
    ]  # fmt: skip


@dataclass
class Job:
    run: str
    process: subprocess.Popen
    stop_file: Path
    stop_requested: bool = False


class TrainingManager:
    def __init__(
        self,
        settings: Settings,
        command: CommandFactory = worker_command,
        before_start: Callable[[], None] | None = None,
    ):
        self.settings = settings
        self._command = command
        # Called before a job launches, e.g. to free the GPU it will use.
        self._before_start = before_start
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._failed: set[str] = set()  # runs whose last job exited with an error

    @property
    def lora_dir(self) -> Path:
        return self.settings.lora_dir.resolve()

    @property
    def active_run(self) -> str | None:
        job = self._job
        return job.run if job else None

    def start(self, params: TrainParams) -> RunDetail:
        data_dir = self.settings.data_dir
        root = manifests.legacy_root(data_dir)
        train_path = manifests.resolve(data_dir, params.dataset)
        val_path = manifests.resolve(data_dir, params.val_dataset) if params.val_dataset else None

        with self._lock:
            self._ensure_idle()
            run_dir = self.lora_dir / params.name
            if run_dir.exists():
                raise TrainingConflict(f"A run named '{params.name}' already exists.")

            # Clips dropped in review stay out of the run.
            train, errors = manifests.training_entries(train_path, root)
            if not train:
                raise ValueError(f"Dataset '{params.dataset}' has no usable entries. {' '.join(errors[:3])}")
            if val_path:
                val, _ = manifests.training_entries(val_path, root)
            else:
                train, val = manifests.split_holdout(train, params.val_holdout)
            model = resolve_model(self.settings.model_path("model"))

            run_dir.mkdir(parents=True)
            train_manifest = run_dir / "manifests" / "train.jsonl"
            val_manifest = run_dir / "manifests" / "val.jsonl" if val else None
            manifests.write_manifest(train, train_manifest)
            if val_manifest:
                manifests.write_manifest(val, val_manifest)
            write_config(run_dir, build_config(params, run_dir, model, train_manifest, val_manifest))
            self._launch(params.name, run_dir)
        return self.detail(params.name)

    def resume(self, name: str, num_iters: int | None = None) -> RunDetail:
        with self._lock:
            self._ensure_idle()
            run_dir = self._run_dir(name)
            original = read_config(run_dir)
            config = self._adopt(run_dir, dict(original))

            saved = saved_step(run_dir)
            if num_iters is not None:
                if num_iters <= (saved or 0) + 1:
                    raise ValueError(f"Step count must exceed the saved step ({saved}).")
                # The cosine schedule restarts its decay over the new length.
                config["num_iters"] = config["max_steps"] = num_iters
            if is_complete(saved, int(config.get("num_iters", 0))):
                raise ValueError("This run is complete; raise the step count to continue training.")

            # Without this the script cannot tell where a latest/-only checkpoint
            # left off, and restarts at step 0.
            state = run_dir / "checkpoints" / "latest" / "training_state.json"
            if saved is not None and state.parent.is_dir() and not state.exists():
                state.write_text(json.dumps({"step": saved}), encoding="utf-8")

            if config != original:
                backup = run_dir / ORIGINAL_CONFIG_FILE
                if not backup.exists():
                    shutil.copy2(run_dir / CONFIG_FILE, backup)
                write_config(run_dir, config)
            self._launch(name, run_dir)
        return self.detail(name)

    def stop(self) -> None:
        with self._lock:
            job = self._job
            if job is None:
                raise TrainingConflict("No training job is running.")
            if job.stop_requested:
                return
            job.stop_requested = True
        job.stop_file.touch()
        timer = threading.Timer(STOP_TIMEOUT_S, self._kill, args=(job,))
        timer.daemon = True
        timer.start()

    def shutdown(self, timeout: float = STOP_TIMEOUT_S) -> None:
        job = self._job
        if job is None:
            return
        try:
            self.stop()
        except TrainingConflict:
            return
        try:
            job.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self._kill(job)

    def list_runs(self) -> list[RunSummary]:
        lora_dir = self.lora_dir
        if not lora_dir.is_dir():
            return []
        runs = []
        for run_dir in lora_dir.iterdir():
            if not (run_dir / CONFIG_FILE).is_file():
                continue
            try:
                runs.append(self._summary(run_dir, read_config(run_dir)))
            except (OSError, ValueError, yaml.YAMLError) as e:
                logger.warning("Skipping run %s: %s", run_dir.name, e)
        return sorted(runs, key=lambda r: r.updated, reverse=True)

    def detail(self, name: str) -> RunDetail:
        run_dir = self._run_dir(name)
        config = read_config(run_dir)
        metrics = parse_metrics(run_dir / "checkpoints" / "train.log")
        summary = self._summary(run_dir, config, metrics["train"])

        samples, references = [], []
        samples_dir = run_dir / "samples"
        if samples_dir.is_dir():
            for f in sorted(samples_dir.iterdir()):
                if m := SAMPLE_FILE.fullmatch(f.name):
                    samples.append(Sample(step=int(m.group(1)), index=int(m.group(2)), file=f.name))
                elif REFERENCE_FILE.fullmatch(f.name):
                    references.append(f.name)

        return RunDetail(
            **summary.model_dump(),
            config=config,
            train=downsample(metrics["train"]),
            val=downsample(metrics["val"]),
            seconds_per_step=seconds_per_step(metrics["train"]),
            samples=samples,
            references=references,
            sample_texts=_first_texts(str(config.get("val_manifest") or "")),
            val_clips=_count_lines(str(config.get("val_manifest") or "")),
            log=tail_lines(run_dir / CONSOLE_FILE),
        )

    def train_manifest(self, name: str) -> Path:
        """The manifest a run trains on (its own copy, or a webui run's original)."""
        value = read_config(self._run_dir(name)).get("train_manifest")
        path = Path(str(value)) if value else None
        if path is not None and not path.is_absolute():
            path = manifests.legacy_root(self.settings.data_dir) / path
        if path is None or not path.is_file():
            raise KeyError(f"Run '{name}' has no readable training manifest.")
        return path

    def sample_file(self, name: str, file: str) -> Path:
        path = self._run_dir(name) / "samples" / file
        if not (SAMPLE_FILE.fullmatch(file) or REFERENCE_FILE.fullmatch(file)) or not path.is_file():
            raise KeyError(f"Unknown sample: {file}")
        return path

    # ------------------------------------------------------------------ #

    def _ensure_idle(self) -> None:
        if self._job is not None:
            raise TrainingConflict(f"Run '{self._job.run}' is already training.")

    def _run_dir(self, name: str) -> Path:
        run_dir = self.lora_dir / name
        if not re.fullmatch(RUN_NAME, name) or not (run_dir / CONFIG_FILE).is_file():
            raise KeyError(f"Unknown training run: {name}")
        return run_dir

    def _summary(self, run_dir: Path, config: dict, train: list[dict] | None = None) -> RunSummary:
        name = run_dir.name
        num_iters = int(config.get("num_iters", 0))
        saved = saved_step(run_dir)
        job = self._job
        step = saved

        if job is not None and job.run == name:
            status = "stopping" if job.stop_requested else "running"
            if train is None:
                train = parse_metrics(run_dir / "checkpoints" / "train.log")["train"]
            if train:
                step = train[-1]["step"]
        elif is_complete(saved, num_iters):
            status = "complete"
        elif name in self._failed:
            status = "failed"
        else:
            status = "incomplete"

        log = run_dir / "checkpoints" / "train.log"
        lora = config.get("lora") or {}
        return RunSummary(
            name=name,
            status=status,
            step=step,
            saved_step=saved,
            num_iters=num_iters,
            lora_rank=lora.get("r"),
            lora_alpha=lora.get("alpha"),
            train_manifest=str(config.get("train_manifest") or ""),
            val_manifest=str(config.get("val_manifest") or ""),
            updated=(log if log.is_file() else run_dir / CONFIG_FILE).stat().st_mtime,
        )

    def _adopt(self, run_dir: Path, config: dict) -> dict:
        """Make a run's config self-contained: absolute paths, own manifest copies.

        Webui configs hold paths relative to the VoxCPM checkout and manifests
        whose audio paths the current loader would misread.
        """
        root = manifests.legacy_root(self.settings.data_dir)

        def absolute(value) -> Path:
            path = Path(str(value))
            return path if path.is_absolute() else root / path

        config["pretrained_path"] = str(absolute(config["pretrained_path"]))
        config["save_path"] = str(run_dir / "checkpoints")
        config["tensorboard"] = (
            str(absolute(config["tensorboard"])) if config.get("tensorboard") else str(run_dir / "logs")
        )
        for key, file in (("train_manifest", "train.jsonl"), ("val_manifest", "val.jsonl")):
            if not config.get(key):
                continue
            source, local = absolute(config[key]), run_dir / "manifests" / file
            if source.resolve() == local.resolve():
                continue
            if not source.is_file():
                raise ValueError(f"Manifest not found: {source}")
            entries, _ = manifests.read_manifest(source, root)
            manifests.write_manifest(entries, local)
            config[key] = str(local)
        return config

    def _launch(self, name: str, run_dir: Path) -> None:
        if self._before_start:
            self._before_start()
        stop_file = run_dir / STOP_FILE
        stop_file.unlink(missing_ok=True)
        cmd = self._command(run_dir / CONFIG_FILE, run_dir / "samples", stop_file)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        if self.settings.train_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.settings.train_gpu)

        console = run_dir / CONSOLE_FILE
        with console.open("a", encoding="utf-8") as f:
            f.write(f"[studio] {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(cmd)}\n")
        # No stdin pipe: see train_worker's docstring.
        process = subprocess.Popen(
            cmd, cwd=run_dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        job = Job(name, process, stop_file)
        self._job = job
        self._failed.discard(name)
        logger.info("Training %s (pid %d)", name, process.pid)
        threading.Thread(target=self._pump, args=(job, console), daemon=True, name=f"train-{name}").start()

    def _pump(self, job: Job, console: Path) -> None:
        with console.open("ab") as f:
            for line in job.process.stdout:
                f.write(line)
                f.flush()
        code = job.process.wait()
        with console.open("a", encoding="utf-8") as f:
            f.write(f"[studio] {time.strftime('%Y-%m-%d %H:%M:%S')} exited with code {code}\n")
        logger.info("Training %s exited with code %d", job.run, code)
        job.stop_file.unlink(missing_ok=True)
        with self._lock:
            if code != 0 and not job.stop_requested:
                self._failed.add(job.run)
            if self._job is job:
                self._job = None

    @staticmethod
    def _kill(job: Job) -> None:
        if job.process.poll() is None:
            logger.warning("Training %s did not stop in time; killing it", job.run)
            job.process.kill()
