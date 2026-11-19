"""Training subprocess for RVC v2 voices (40 kHz, with pitch).

``python -m polyvox.engines.rvc.worker <train_config.yaml> --samples-dir <dir> --stop-file <path>
[--parent-pid <pid>]``

Same contract as the other engines' workers: ``[train]`` lines in checkpoints/train.log,
checkpoints/step_NNNNNNN/ (each a voice: voice.json + rvc_voice.pth) and latest/training_state.pt,
samples/ with step_NNNNNNN_<i>.wav (held-out clips converted with the voice so far) and
reference_<i>.wav (those clips); the stop file (or the server exiting) saves and ends the run.
Steps count updates, 1..num_iters.

As RVC's WebUI trains (train.py): each clip resampled to 40 kHz, high-passed, cut into 3.7 s
pieces, each normalized; features per piece (ContentVec, RMVPE pitch) cached in the dataset
folder for its runs; the GAN from the 40 kHz base models; the run's index (the pieces' content
features) in the run folder, used by all its checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import yaml

from ..common.worker import format_metrics, log

SAMPLES = 2  # held-out clips converted at each checkpoint
CACHE_VERSION = 1  # bump when piece_features' output changes
NAMES = ("loss/disc", "loss/gen", "loss/fm", "loss/mel", "loss/kl")  # train.LOSSES, as logged


def plan(pieces: int, batch: int, epochs: int) -> tuple[int, int]:
    """(steps per epoch, num_iters)."""
    per_epoch = max(1, math.ceil(pieces / batch))
    return per_epoch, per_epoch * epochs


class Run:
    def __init__(self, args, config: dict):
        import torch

        from ... import vram
        from ..common.worker import watch_for_stop

        vram.apply_limit("cuda")
        self.args, self.config = args, config
        self.stop = threading.Event()
        threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid, self.stop), daemon=True).start()
        self.save_dir = Path(config["save_path"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.save_dir.parent
        self.latest = self.save_dir / "latest"
        self.log_interval = int(config.get("log_interval") or 10)
        self.dev = torch.device("cuda")
        self._window: list = []
        self._t_log = time.time()

    def record(self, step: int, losses, extra: dict[str, float], last: bool) -> None:
        """A step's losses (a GPU tensor, read back only here); every log_interval steps and the
        last one a ``[train]`` line with their means and ``extra``."""
        import torch

        self._window.append(losses)
        if step % self.log_interval and not last:
            return
        now = time.time()
        means = dict(zip(NAMES, torch.stack(self._window).mean(0).tolist(), strict=True))
        means |= extra
        means["log interval"] = now - self._t_log
        line = f"[train] step {step}: {format_metrics(means)}"
        with (self.save_dir / "train.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
        self._t_log, self._window = now, []

    def write_plan(self, pieces: int, num_iters: int) -> None:
        """The exact step count into the run's config (the server shows it)."""
        self.config["pieces"], self.config["num_iters"] = pieces, num_iters
        tmp = self.args.config_path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.args.config_path)

    def save_state(self, step: int, epoch: int, state: dict) -> None:
        import torch

        self.latest.mkdir(parents=True, exist_ok=True)
        tmp = self.latest / "training_state.tmp"
        torch.save({"step": step, "epoch": epoch, **state}, tmp)
        tmp.replace(self.latest / "training_state.pt")
        (self.latest / "training_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
        log(f"Saved step {step}.")

    def load_state(self) -> dict | None:
        import torch

        path = self.latest / "training_state.pt"
        return torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None


def features(run: Run, records: list[dict], contentvec, rmvpe, cache_file: Path) -> list[dict] | None:
    """Every clip's pieces, from the dataset's cache where it has them (None: stopped)."""
    from ..common.cache import clip_key, load_cache, save_cache
    from .train import clip_features

    meta = {"version": CACHE_VERSION, "contentvec": "contentvec-768", "pitch": "rmvpe"}
    cache = load_cache(cache_file, meta)
    keys = [clip_key(r) for r in records]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        log(f"Extracting features of {len(todo)} clips ({len(records) - len(todo)} cached)...")
    t0 = time.time()
    for n, i in enumerate(todo):
        if run.stop.is_set():
            save_cache(cache_file, meta, cache, records, keys)
            return None
        try:
            # (the shared cache helpers read an entry's "audio" to prune clips that are gone)
            cache[keys[i]] = {
                "audio": str(records[i]["audio"]),
                "pieces": clip_features(records[i]["audio"], contentvec, rmvpe),
            }
        except Exception as e:  # an unreadable clip is left out, not cached
            log(f"Skipping {Path(records[i]['audio']).name}: {e}")
        if (n + 1) % 100 == 0:
            log(f"Features: {n + 1} / {len(todo)} clips")
    if todo:
        save_cache(cache_file, meta, cache, records, keys)
        log(f"Features of {len(todo)} clips in {time.time() - t0:.0f} s.")
    return [p for k in keys for p in cache.get(k, {}).get("pieces", [])]


def write_index(run: Run, pieces: list[dict]) -> int:
    """The run's index: its pieces' content features (k-means centers above 200k rows)."""
    import torch

    from .voices import INDEX, save_index

    if (run.run_dir / INDEX).is_file():
        return 0
    rows = save_index(run.run_dir, torch.cat([p["feats"] for p in pieces]).to(run.dev))
    log(f"Index: {rows} rows.")
    return rows


def samples(run: Run, trainer, converter, index, step: int, val_records: list[dict]) -> None:
    """The first held-out clips converted with the voice as trained so far (the run's index)."""
    import soundfile as sf

    from ...data import audio_io
    from .convert import Options
    from .model import Voice

    if not val_records:
        return
    out = run.args.samples_dir
    out.mkdir(parents=True, exist_ok=True)

    def write(path: Path, audio, rate: int) -> None:
        tmp = path.with_suffix(".tmp")
        sf.write(tmp, audio, rate, format="WAV")
        tmp.replace(path)

    trainer.g.eval()
    try:
        voice = Voice(trainer.g, 40000, "v2", "")
        for i, record in enumerate(val_records[:SAMPLES]):
            audio, rate = audio_io.read(record["audio"])
            if not (out / f"reference_{i}.wav").exists():
                write(out / f"reference_{i}.wav", audio.T, rate)
            converted = converter.convert(voice, audio.mean(0), rate, index, Options())
            write(out / f"step_{step:07d}_{i}.wav", converted, 40000)
    except Exception as e:  # a failed sample never stops training
        log(f"Sample conversion failed: {e}")
    finally:
        trainer.g.train()


def checkpoint(run: Run, trainer, step: int, epoch: int) -> None:
    from .voices import WEIGHTS, write_info

    folder = run.save_dir / f"step_{step:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_voice(folder / WEIGHTS, f"{epoch}epoch")
    write_info(folder, 40000, "trained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))

    import torch

    from ...data import manifests
    from .convert import ContentVec, Converter, Index
    from .pitch import RMVPE
    from .train import Trainer, collate, epochs_batches

    run = Run(args, config)
    t0 = time.time()
    contentvec = ContentVec(Path(config["contentvec"]), "cuda")
    rmvpe = RMVPE(Path(config["rmvpe"]), "cuda")
    converter = Converter(contentvec, rmvpe, "cuda")
    log(f"ContentVec and RMVPE loaded in {time.time() - t0:.0f} s.")

    records = [dict(e) for e in manifests.read_manifest(Path(config["train_manifest"]))[0]]
    val_path = Path(config["val_manifest"]) if config.get("val_manifest") else None
    val_records = [dict(e) for e in manifests.read_manifest(val_path)[0]] if val_path and val_path.is_file() else []
    pieces = features(run, records, contentvec, rmvpe, Path(config["feature_cache"]))
    if pieces is None:
        log("Stopped.")
        return
    if not pieces:
        raise SystemExit("No clip of this dataset can be trained on (each needs at least 0.32 s of audio).")
    write_index(run, pieces)

    batch = int(config["batch_size"])
    per_epoch, num_iters = plan(len(pieces), batch, int(config["epochs"]))
    if "pieces" not in config:  # a new run: the exact steps, now that the pieces are known
        run.write_plan(len(pieces), num_iters)
    num_iters = int(config["num_iters"])  # continuing a run raises it (more epochs)
    epochs = math.ceil(num_iters / per_epoch)
    save_every = int(config.get("save_every") or 25)

    trainer = Trainer(
        Path(config["pretrained_g"]), Path(config["pretrained_d"]), "cuda", float(config["learning_rate"])
    )
    state = run.load_state()
    step, epoch = (int(state["step"]), int(state["epoch"])) if state else (0, 0)
    if state:
        trainer.load(state)
        log(f"Resumed at step {step} (epoch {epoch}).")
    index = Index(torch.load(run.run_dir / "rvc_index.pt", weights_only=True).to(run.dev))
    log(f"{len(pieces)} pieces, batch {batch}: {per_epoch} steps per epoch, {num_iters} steps ({epochs} epochs).")

    def save(at_epoch: int) -> None:
        checkpoint(run, trainer, step, at_epoch)
        samples(run, trainer, converter, index, step, val_records)
        run.save_state(step, at_epoch, trainer.state())

    torch.cuda.reset_peak_memory_stats()
    saved = step
    while step < num_iters and not run.stop.is_set():
        # Batches of this epoch already done (a run resumed mid-epoch): skipped, same seeded order.
        done = step - epoch * per_epoch
        for group in epochs_batches(pieces, batch, epoch + 1)[done:]:
            if run.stop.is_set() or step >= num_iters:
                break
            losses = trainer.step(collate([pieces[i] for i in group], run.dev, trainer.spectro))
            step += 1
            progress = epoch + (step - epoch * per_epoch) / per_epoch
            lr = trainer.opt_g.param_groups[0]["lr"]
            run.record(step, losses, {"lr": lr, "epoch": progress}, step == num_iters)
        if step - epoch * per_epoch >= per_epoch:  # a whole epoch
            epoch += 1
            trainer.end_epoch()
            if epoch % save_every == 0:
                save(epoch)
                saved = step
    if step > saved:  # stopped, or the last epoch off the save interval
        save(epoch)
    log(f"Peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GB.")
    log("Stopped." if run.stop.is_set() and step < num_iters else "Training complete.")


if __name__ == "__main__":
    main()
