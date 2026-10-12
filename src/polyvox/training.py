"""LoRA training runs: configuration, job control and progress.

A run is a directory ``<lora_dir>/<engine>/<name>/`` (runs from before engine
folders: ``<lora_dir>/<name>/`` until lora_library.migrate moves them) holding ``train_config.yaml`` (the
layout ``lora_ft_webui`` uses, so its runs show up and can be resumed):

    train_config.yaml   arguments for the training script
    manifests/          the run's own copies of its manifests, absolute paths
    checkpoints/        step_NNNNNNN/, latest/, train.log (written by the script)
    logs/               TensorBoard events
    samples/            validation audio (written by the engine's worker)
    console.log         the training process's output

One job runs at a time, as the engine's worker subprocess (engines/<engine>/worker.py).
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from . import engines, lora_library
from .config import Settings
from .data import manifests
from .engines import Engine

logger = logging.getLogger(__name__)

CONFIG_FILE = "train_config.yaml"
ORIGINAL_CONFIG_FILE = "train_config.orig.yaml"  # a webui config, before paths were made absolute
CONSOLE_FILE = "console.log"
STOP_FILE = "stop.request"  # created to ask the worker to save and exit
RUN_NAME = r"^[^\W_][\w.-]{0,79}$"  # Unicode letters too: voice names are often Chinese
SAMPLE_FILE = re.compile(r"step_(\d{7})_(\d+)\.wav")
REFERENCE_FILE = re.compile(r"reference_(\d+)\.wav")
MODEL_INPUT_FILE = re.compile(r"reference_(\d+)_16k\.wav")  # a reference as the model hears it
METRIC_LINE = re.compile(r"\[(train|val)\] step (\d+): (.*)")
STOP_TIMEOUT_S = 120.0  # saving a checkpoint takes seconds; kill after this
MAX_POINTS = 1000  # per metric series sent to the UI
LOG_TAIL_LINES = 200
LOG_TAIL_BYTES = 128 * 1024


class TrainingConflict(Exception):
    """The request conflicts with the current job or existing runs."""


class TrainParams(BaseModel):
    name: str = Field(pattern=RUN_NAME)
    engine: str = engines.DEFAULT
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
    # adamw: the script's own (AdamW, cosine with warmup). prodigy / prodigy_sf find
    # their own step size; learning_rate is then a multiplier, 1.0 as their authors
    # recommend (see engines.voxcpm.worker.use_optimizer).
    optimizer: Literal["adamw", "prodigy", "prodigy_sf"] = "adamw"
    # prodigy_sf: keep its step size fixed after this many steps (0 = adapt throughout).
    prodigy_steps: int = Field(0, ge=0)
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
    # GPT-SoVITS (full fine-tuning, two stages; the webui's defaults): epochs of
    # each stage (num_iters then follows from them) and the GPT stage's batch.
    sovits_epochs: int = Field(8, ge=0, le=100)
    gpt_epochs: int = Field(15, ge=0, le=100)
    gpt_batch_size: int = Field(8, ge=1, le=64)
    gsv_version: Literal["v2ProPlus", "v2Pro", "v4"] = "v2ProPlus"  # the base model trained from (v4: LoRA rank)


class RunSummary(BaseModel):
    name: str
    engine: str
    # The run's folder from the LoRA folder ("<engine>/<name>"): its checkpoints' ids start with it.
    path: str
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
    model_inputs: list[str] = []  # reference_<i>_16k.wav: the references as the model hears them
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


def engine_of(config: dict) -> str:
    """The engine a run trains (runs from before engines: VoxCPM)."""
    return str(config.get("engine") or engines.DEFAULT)


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
    """The training process of the run's engine."""
    engine = engine_of(read_config(config_path.parent))
    module = engines.classes()[engine].worker_module
    return [
        sys.executable, "-m", module, str(config_path),
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
        engines_: Mapping[str, Engine] | None = None,
    ):
        self.settings = settings
        self.engines = engines_ if engines_ is not None else engines.create(settings)
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
        engine = self._engine(params.engine)
        data_dir = self.settings.data_dir
        root = manifests.legacy_root(data_dir)
        train_path = manifests.resolve(data_dir, params.dataset)
        val_path = manifests.resolve(data_dir, params.val_dataset) if params.val_dataset else None

        with self._lock:
            self._ensure_idle()
            run_dir = lora_library.engine_dir(self.lora_dir, engine.id) / params.name
            if run_dir.exists() or self._find_run(params.name) is not None:
                raise TrainingConflict(f"A run named '{params.name}' already exists.")

            # Clips dropped in review stay out of the run.
            train, errors = manifests.training_entries(train_path, root)
            if not train:
                raise ValueError(f"Dataset '{params.dataset}' has no usable entries. {' '.join(errors[:3])}")
            if val_path:
                val, _ = manifests.training_entries(val_path, root)
            else:
                train, val = manifests.split_holdout(train, params.val_holdout)

            train_manifest = run_dir / "manifests" / "train.jsonl"
            val_manifest = run_dir / "manifests" / "val.jsonl" if val else None
            run_dir.mkdir(parents=True)
            try:  # the manifests first: an engine may size the run from them
                manifests.write_manifest(train, train_manifest)
                if val_manifest:
                    manifests.write_manifest(val, val_manifest)
                config = {"engine": engine.id, **engine.training_config(params, run_dir, train_manifest, val_manifest)}
            except BaseException:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise
            write_config(run_dir, config)
            self._launch(params.name, run_dir)
        return self.detail(params.name)

    def resume(self, name: str, num_iters: int | None = None) -> RunDetail:
        with self._lock:
            self._ensure_idle()
            run_dir = self._run_dir(name)
            original = read_config(run_dir)
            config = self._engine(engine_of(original)).adopt(run_dir, dict(original))

            saved = saved_step(run_dir)
            if num_iters is not None:
                if num_iters <= (saved or 0) + 1:
                    raise ValueError(f"Step count must exceed the saved step ({saved}).")
                # The cosine schedule restarts its decay over the new length.
                config["num_iters"] = num_iters
                if "max_steps" in config:
                    config["max_steps"] = num_iters
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
        runs = []
        for run_dir in self._run_dirs():
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

        samples, references, model_inputs = [], [], []
        samples_dir = run_dir / "samples"
        if samples_dir.is_dir():
            for f in sorted(samples_dir.iterdir()):
                if m := SAMPLE_FILE.fullmatch(f.name):
                    samples.append(Sample(step=int(m.group(1)), index=int(m.group(2)), file=f.name))
                elif REFERENCE_FILE.fullmatch(f.name):
                    references.append(f.name)
                elif MODEL_INPUT_FILE.fullmatch(f.name):
                    model_inputs.append(f.name)

        return RunDetail(
            **summary.model_dump(),
            config=config,
            train=downsample(metrics["train"]),
            val=downsample(metrics["val"]),
            seconds_per_step=seconds_per_step(metrics["train"]),
            samples=samples,
            references=references,
            model_inputs=model_inputs,
            sample_texts=_first_texts(str(config.get("val_manifest") or "")),
            val_clips=_count_lines(str(config.get("val_manifest") or "")),
            log=tail_lines(run_dir / CONSOLE_FILE),
        )

    def runs_using(self, folder: Path) -> list[str]:
        """Runs whose training clips are in ``folder`` (a dataset's): their
        manifest copies point at its audio, so resuming / continuing them needs it."""
        folder = folder.resolve()
        root = manifests.legacy_root(self.settings.data_dir)
        out = []
        for run_dir in self._run_dirs():
            try:
                path = self.train_manifest(run_dir.name)
                with path.open(encoding="utf-8") as f:
                    line = next((ln for ln in f if ln.strip()), "")
                entry = json.loads(line) if line else {}
                audio = manifests.local_path(str(entry.get("audio", "")))
                audio = (
                    audio
                    if audio.is_absolute()
                    else (path.parent / audio if (path.parent / audio).exists() else root / audio)
                )
            except (KeyError, OSError, ValueError, yaml.YAMLError):
                continue
            if audio.resolve().is_relative_to(folder):
                out.append(run_dir.name)
        return sorted(out)

    def train_manifest(self, name: str) -> Path:
        """The manifest a run trains on (its own copy, or a webui run's original)."""
        value = read_config(self._run_dir(name)).get("train_manifest")
        path = Path(str(value)) if value else None
        if path is not None and not path.is_absolute():
            path = manifests.legacy_root(self.settings.data_dir) / path
        if path is None or not path.is_file():
            raise KeyError(f"Run '{name}' has no readable training manifest.")
        return path

    def run_dir(self, name: str) -> Path:
        return self._run_dir(name)

    def sample_file(self, name: str, file: str) -> Path:
        path = self._run_dir(name) / "samples" / file
        known = SAMPLE_FILE.fullmatch(file) or REFERENCE_FILE.fullmatch(file) or MODEL_INPUT_FILE.fullmatch(file)
        if not known or not path.is_file():
            raise KeyError(f"Unknown sample: {file}")
        return path

    # ------------------------------------------------------------------ #

    def _engine(self, id: str) -> Engine:
        engine = self.engines.get(id)
        if engine is None:
            raise ValueError(f"Unknown engine '{id}'.")
        if reason := engine.unavailable():
            raise ValueError(f"{engine.name} can't train here: {reason}")
        return engine

    def _ensure_idle(self) -> None:
        if self._job is not None:
            raise TrainingConflict(f"Run '{self._job.run}' is already training.")

    def _engine_folders(self) -> set[str]:
        return lora_library.engine_folders() | set(self.engines)

    def _places(self) -> list[Path]:
        """Where runs are: the engine folders, then the LoRA folder itself (runs from before them)."""
        return [*(self.lora_dir / f for f in sorted(self._engine_folders())), self.lora_dir]

    def _run_dirs(self) -> list[Path]:
        folders = self._engine_folders()
        return [
            d
            for place in self._places()
            if place.is_dir()
            for d in place.iterdir()
            if (d / CONFIG_FILE).is_file() and not (place == self.lora_dir and d.name in folders)
        ]

    def _find_run(self, name: str) -> Path | None:
        if not re.fullmatch(RUN_NAME, name):
            return None
        return next((d for d in self._run_dirs() if d.name == name), None)

    def _run_dir(self, name: str) -> Path:
        run_dir = self._find_run(name)
        if run_dir is None:
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
            engine=engine_of(config),
            path=run_dir.relative_to(self.lora_dir).as_posix(),
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

    def _launch(self, name: str, run_dir: Path) -> None:
        if self._before_start:
            self._before_start()
        stop_file = run_dir / STOP_FILE
        stop_file.unlink(missing_ok=True)
        cmd = self._command(run_dir / CONFIG_FILE, run_dir / "samples", stop_file)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        # Less fragmentation: a run that generates validation samples between steps
        # otherwise filled the GPU's cache and hit allocation retries (same speed).
        # Not on Windows: there growing a segment can fail in the driver ("CUDA
        # driver error: device not ready", "Failed to create GPU mapping") on the
        # first backward pass of some datasets; the retries are only warnings.
        if sys.platform != "win32":
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.settings.train_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.settings.train_gpu)

        console = run_dir / CONSOLE_FILE
        with console.open("a", encoding="utf-8") as f:
            f.write(f"[studio] {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(cmd)}\n")
        # No stdin pipe: see engines.voxcpm.worker's docstring.
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
