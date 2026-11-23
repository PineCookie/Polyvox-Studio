"""Training subprocess for RVC v2 voices (32, 40 or 48 kHz, with pitch).

``python -m polyvox.engines.rvc.worker <train_config.yaml> --samples-dir <dir> --stop-file <path>
[--parent-pid <pid>]``

Same contract as the other engines' workers: ``[train]`` lines in checkpoints/train.log,
checkpoints/step_NNNNNNN/ (each a voice: voice.json + rvc_voice.pth) and latest/training_state.pt,
samples/ with step_NNNNNNN_<i>.wav (held-out clips converted with the voice so far) and
reference_<i>.wav (those clips); the stop file (or the server exiting) saves and ends the run.
Steps count updates, 1..num_iters.

As RVC's WebUI trains (train.py): each clip resampled to the voice's rate (config sample_rate),
high-passed, cut into 3.7 s pieces, each normalized; features per piece (ContentVec, RMVPE pitch)
cached in the dataset folder for its runs at that rate; the GAN from that rate's base models; the
run's index (the pieces' content features) in the run folder, used by all its checkpoints.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from ..common.worker import Run, log

SAMPLES = 2  # held-out clips converted at each checkpoint
CACHE_VERSION = 1  # bump when piece_features' output changes
NAMES = ("loss/disc", "loss/gen", "loss/fm", "loss/mel", "loss/kl")  # train.LOSSES, as logged


def plan(pieces: int, batch: int, epochs: int) -> tuple[int, int]:
    """(steps per epoch, num_iters)."""
    per_epoch = max(1, math.ceil(pieces / batch))
    return per_epoch, per_epoch * epochs


def features(
    run: Run, records: list[dict], contentvec, rmvpe, cache_file: Path, rate: int = 40000
) -> list[dict] | None:
    """Every clip's pieces at ``rate``, from the dataset's cache where it has them (None: stopped)."""
    from ..common.cache import clip_key, load_cache, save_cache
    from .train import clip_features

    # (the rate is in the cache file's name: rvc_v2_<k>k.pt)
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
                "pieces": clip_features(records[i]["audio"], contentvec, rmvpe, rate),
            }
        except Exception as e:  # an unreadable clip is left out, not cached
            log(f"Skipping {Path(records[i]['audio']).name}: {e}")
        if (n + 1) % 100 == 0:
            log(f"Features: {n + 1} / {len(todo)} clips")
    if todo:
        save_cache(cache_file, meta, cache, records, keys)
        log(f"Features of {len(todo)} clips in {time.time() - t0:.0f} s.")
    return [p for k in keys for p in cache.get(k, {}).get("pieces", [])]


def write_index(run: Run, pieces: list[dict], dev) -> int:
    """The run's index: its pieces' content features (k-means centers above 200k rows)."""
    import torch

    from .voices import INDEX, save_index

    if (run.run_dir / INDEX).is_file():
        return 0
    rows = save_index(run.run_dir, torch.cat([p["feats"] for p in pieces]).to(dev))
    log(f"Index: {rows} rows.")
    return rows


def samples(run: Run, trainer, converter, index, step: int, val_records: list[dict]) -> None:
    """The first held-out clips converted with the voice as trained so far (the run's index)."""
    from ...data import audio_io
    from .convert import Options
    from .model import Voice

    if not val_records:
        return
    trainer.g.eval()
    try:
        voice = Voice(trainer.g, trainer.rate, "v2", "")
        for i, record in enumerate(val_records[:SAMPLES]):
            run.reference(i, record["audio"])
            audio, rate = audio_io.read(record["audio"])
            converted = converter.convert(voice, audio.mean(0), rate, index, Options())
            run.write_wav(f"step_{step:07d}_{i}.wav", converted, trainer.rate)
    except Exception as e:  # a failed sample never stops training
        log(f"Sample conversion failed: {e}")
    finally:
        trainer.g.train()


def checkpoint(run: Run, trainer, step: int, epoch: int) -> None:
    from .voices import WEIGHTS, write_info

    folder = run.save_dir / f"step_{step:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_voice(folder / WEIGHTS, f"{epoch}epoch")
    write_info(folder, trainer.rate, "trained")


def main() -> None:
    run = Run(__doc__.splitlines()[0])
    config = run.config

    import torch

    from .convert import ContentVec, Converter, Index
    from .pitch import RMVPE
    from .train import FRAMES_PER_S, RATES, Trainer, collate, epochs_batches

    dev = torch.device("cuda")
    t0 = time.time()
    contentvec = ContentVec(Path(config["contentvec"]), "cuda")
    rmvpe = RMVPE(Path(config["rmvpe"]), "cuda")
    converter = Converter(contentvec, rmvpe, "cuda")
    log(f"ContentVec and RMVPE loaded in {time.time() - t0:.0f} s.")

    records, val_records = run.records("train_manifest"), run.records("val_manifest")
    rate = int(config.get("sample_rate") or 40000)
    pieces = features(run, records, contentvec, rmvpe, Path(config["feature_cache"]), rate)
    if pieces is None:
        log("Stopped.")
        return
    if not pieces:
        least = RATES[rate].segment / FRAMES_PER_S
        raise SystemExit(f"No clip of this dataset can be trained on (each needs at least {least:.2f} s of audio).")
    write_index(run, pieces, dev)

    batch = int(config["batch_size"])
    per_epoch, num_iters = plan(len(pieces), batch, int(config["epochs"]))
    if "pieces" not in config:  # a new run: the exact steps, now that the pieces are known
        run.write_config(pieces=len(pieces), num_iters=num_iters)
    num_iters = run.num_iters  # continuing a run raises it (more epochs)
    epochs = math.ceil(num_iters / per_epoch)
    save_every = int(config.get("save_every") or 25)

    trainer = Trainer(
        Path(config["pretrained_g"]), Path(config["pretrained_d"]), "cuda", float(config["learning_rate"]), rate=rate
    )
    state = run.load_state()
    step, epoch = (int(state["step"]), int(state["epoch"])) if state else (0, 0)
    if state:
        trainer.load(state)
        log(f"Resumed at step {step} (epoch {epoch}).")
    index = Index(torch.load(run.run_dir / "rvc_index.pt", weights_only=True).to(dev))
    log(f"{len(pieces)} pieces, batch {batch}: {per_epoch} steps per epoch, {num_iters} steps ({epochs} epochs).")

    def save(at_epoch: int) -> None:
        checkpoint(run, trainer, step, at_epoch)
        samples(run, trainer, converter, index, step, val_records)
        run.save_state(step, {"epoch": at_epoch, **trainer.state()})

    torch.cuda.reset_peak_memory_stats()
    saved = step
    while step < num_iters and not run.stop.is_set():
        # Batches of this epoch already done (a run resumed mid-epoch): skipped, same seeded order.
        done = step - epoch * per_epoch
        for group in epochs_batches(pieces, batch, epoch + 1)[done:]:
            if run.stop.is_set() or step >= num_iters:
                break
            losses = trainer.step(collate([pieces[i] for i in group], dev, trainer.spectro))
            step += 1
            progress = epoch + (step - epoch * per_epoch) / per_epoch
            lr = trainer.opt_g.param_groups[0]["lr"]
            run.record(step, dict(zip(NAMES, losses, strict=True)), {"lr": lr, "epoch": progress}, step == num_iters)
        if step - epoch * per_epoch >= per_epoch:  # a whole epoch
            epoch += 1
            trainer.end_epoch()
            if epoch % save_every == 0:
                save(epoch)
                saved = step
    if step > saved:  # stopped, or the last epoch off the save interval
        save(epoch)
    run.finish(step)


if __name__ == "__main__":
    main()
