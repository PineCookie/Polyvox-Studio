"""Training subprocess for S2-Pro LoRAs (the fast transformer; see lora.py, train.py).

``python -m polyvox.engines.fish_s2.worker <train_config.yaml> --samples-dir
<dir> --stop-file <path> [--parent-pid <pid>]``

Same contract as the other workers: ``[train]`` / ``[val]`` lines in
checkpoints/train.log (loss/codes = the residual codes' cross-entropy;
loss/semantic = the semantic tokens', when the slow transformer trains too;
validation also loss/*_ref, cloning from other clips),
checkpoints/step_NNNNNNN/ and latest/ (training_state.json), samples/ with
step_NNNNNNN_<i>.wav (spoken with the run's reference) and reference_<i>.wav.
The stop file (or the server exiting) saves a checkpoint and ends the run.
Steps count optimizer updates, 1..num_iters.

Clips are encoded once (the codec then leaves the GPU) and cached in the run
folder. Half of the training sequences clone from 1-3 other clips of the voice,
as synthesis usually does; the run's reference (a few good clips joined, picked
when the run was made) goes into every checkpoint as the voice's default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import threading
import time
from pathlib import Path

import yaml

from ..dots.worker import Batches, cached, format_metrics, log, make_optimizer, rates, watch_for_stop
from .lora import REFERENCE_JSON, REFERENCE_WAV

SAMPLES = 2  # validation clips whose texts are spoken at each validation
SAMPLE_SEED = 42
SAMPLE_MAX_FRAMES = 1024
CACHE_VERSION = 1  # bump when the encoding changes


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cache_key(manifest: Path, model_path: str) -> dict:
    from .train import NOISE_FLOOR_DB

    return {
        "version": CACHE_VERSION,
        "manifest": hashlib.sha1(manifest.read_bytes()).hexdigest(),
        "model": str(model_path),
        "noise_floor_db": NOISE_FLOOR_DB,
    }


def encode_audio(codec, path: Path | str, seed: int):
    """A clip's codes [codebooks, frames] (int16, CPU), heard with the noise floor."""
    import numpy as np
    import torch
    import torchaudio

    from ...data import audio_io
    from .train import with_noise_floor

    audio, rate = audio_io.read(path)
    mono = torch.from_numpy(np.asarray(audio.mean(axis=0) if audio.ndim == 2 else audio, dtype=np.float32))
    mono = torchaudio.functional.resample(mono.to(next(codec.parameters()).device), rate, codec.sample_rate)
    return codec.encode(with_noise_floor(mono, seed)).to(torch.int16).cpu()


def encode(codec, records: list[dict], stop: threading.Event, what: str) -> list[dict]:
    import torch

    out, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(records):
            if stop.is_set():
                break
            out.append({"text": clean(r["text"]), "codes": encode_audio(codec, r["audio"], i)})
            if (i + 1) % 200 == 0:
                log(f"Encoding {what}: {i + 1} / {len(records)} clips")
    log(f"Encoded {len(out)} {what} clips in {time.time() - t0:.0f} s.")
    return out


def write_reference(run_dir: Path, config: dict) -> tuple[Path, str]:
    """The run's reference (its clips joined) in the run folder, once."""
    from ...data import references
    from .lora import write_reference_clips

    wav, info = run_dir / REFERENCE_WAV, run_dir / REFERENCE_JSON
    if not (wav.is_file() and info.is_file()):
        ref = config["reference"]
        clips = [references.Clip(i, Path(c["audio"]), c["text"], 0.0) for i, c in enumerate(ref["clips"])]
        write_reference_clips(run_dir, clips, ref.get("source", ""))
    return wav, json.loads(info.read_text(encoding="utf-8"))["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))

    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from polyvox import vram
    from polyvox.data import audio_io, manifests

    from . import codec as s2codec
    from . import decode, generate, prompt
    from . import lora as s2lora
    from . import model as s2model
    from .train import REFERENCE_SHARE, Sequences, Trainer, collate, pick_reference

    vram.apply_limit("cuda")
    stop = threading.Event()
    threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid, stop), daemon=True).start()

    save_dir = Path(config["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir.parent
    num_iters = int(config["num_iters"])
    lora_cfg = config["lora"]
    rank, alpha = int(lora_cfg["r"]), float(lora_cfg["alpha"])
    targets = tuple(lora_cfg.get("targets") or s2lora.DEFAULT_TARGETS)
    torch.manual_seed(0)
    random.seed(0)
    model_dir = Path(config["pretrained_path"])

    t0 = time.time()
    model = s2model.load(model_dir, "cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
    codec = s2codec.load(model_dir / "codec.pth", "cuda")
    loras = s2lora.attach(model, rank, alpha, float(lora_cfg.get("dropout") or 0.0), targets)
    log(f"S2-Pro loaded in {time.time() - t0:.0f} s; LoRA r{rank} on {len(loras)} layers ({', '.join(targets)}).")

    train_records = [dict(e) for e in manifests.read_manifest(Path(config["train_manifest"]))[0]]
    val_path = Path(config["val_manifest"]) if config.get("val_manifest") else None
    val_records = [dict(e) for e in manifests.read_manifest(val_path)[0]] if val_path and val_path.is_file() else []
    cache_dir = run_dir / "cache"
    train_clips = cached(
        cache_dir / "codes_train.pt",
        cache_key(Path(config["train_manifest"]), str(model_dir)),
        lambda: encode(codec, train_records, stop, "training"),
        len(train_records),
    )
    val_clips = (
        cached(
            cache_dir / "codes_val.pt",
            cache_key(val_path, str(model_dir)),
            lambda: encode(codec, val_records, stop, "validation"),
            len(val_records),
        )
        if val_records
        else []
    )
    if len(train_clips) < 2:
        raise SystemExit("S2-Pro training needs at least 2 clips.")
    reference_wav, reference_text = write_reference(run_dir, config)
    with torch.no_grad():
        reference = (encode_audio(codec, reference_wav, 0), reference_text)
    codec.to("cpu")  # back on the GPU only to decode samples
    torch.cuda.empty_cache()

    slow = any(t in s2lora.SLOW_TARGETS for t in targets)
    seq = Sequences(tokenizer, model, end_token)
    trainer = Trainer(model, end_token, slow=slow)
    batch_size, accum = int(config["batch_size"]), int(config["grad_accum_steps"])
    batches = Batches(list(range(len(train_clips))), batch_size, seed=0)  # batches of clip indices
    epoch_steps = max(1.0, batches.per_epoch / accum)
    params = [p for lo in loras.values() for p in (lo.A, lo.B)]
    groups = []
    for net in ("fast", "slow"):
        group = [p for name, lo in loras.items() if name.startswith("fast_") == (net == "fast") for p in (lo.A, lo.B)]
        if group:  # the slow layers at their own, much smaller rate (see lora.py)
            lr = (
                float(config.get("slow_learning_rate") or s2lora.SLOW_LR)
                if net == "slow"
                else float(config["learning_rate"])
            )
            groups.append({"name": net, "params": group, "lr": lr})
    optimizer_name = str(config.get("optimizer") or "prodigy_sf")
    opt, scheduler = make_optimizer(optimizer_name, groups, config, num_iters)
    schedule_free = hasattr(opt, "train") and hasattr(opt, "eval")
    log(f"Optimizer: {optimizer_name}; {len(train_clips)} clips, {epoch_steps:.1f} steps per epoch.")

    # ---------------------------------------------------------------- resume
    step = 0
    latest = save_dir / "latest"
    if (latest / "training_state.pt").is_file():
        state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, lora in loras.items():
                lora.A.copy_(state["train_weights"][f"{name}.lora_A"])
                lora.B.copy_(state["train_weights"][f"{name}.lora_B"])
        opt.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        log(f"Resumed at step {step}.")
    if schedule_free:
        opt.train()

    def averaged(fn):
        """fn() with the weights to use (Schedule-Free: the average)."""
        if schedule_free:
            opt.eval()
        try:
            return fn()
        finally:
            if schedule_free:
                opt.train()

    def save(at: int) -> None:
        weights = averaged(lambda: s2lora.state(loras))
        lora_config = {"r": rank, "alpha": alpha, "dropout": lora_cfg.get("dropout") or 0.0, "targets": list(targets)}
        for folder in (save_dir / f"step_{at:07d}", latest):
            folder.mkdir(parents=True, exist_ok=True)
            save_file(weights, folder / "lora_weights.safetensors")
            s2lora.write_config(folder, lora_config, str(model_dir))
            for name in (REFERENCE_WAV, REFERENCE_JSON):
                shutil.copy2(run_dir / name, folder / name)
        tmp = latest / "training_state.tmp"
        torch.save(
            {
                "step": at,
                "train_weights": s2lora.state(loras),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            },
            tmp,
        )
        tmp.replace(latest / "training_state.pt")
        (latest / "training_state.json").write_text(json.dumps({"step": at}), encoding="utf-8")
        log(f"Saved step {at}.")

    def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
        tmp = path.with_suffix(".tmp")
        sf.write(tmp, audio, rate, format="WAV")
        tmp.replace(path)

    def item(clips: list[dict], i: int, rng: random.Random, with_reference: bool):
        c = clips[i]
        return seq.rows(c["text"], c["codes"], pick_reference(clips, i, rng) if with_reference else None)

    def validate(at: int) -> None:
        if not val_clips:
            return

        def losses() -> dict[str, float]:
            totals: dict[str, list[float]] = {}
            with torch.no_grad():
                for suffix, with_reference in (("", False), ("_ref", True)):
                    rng = random.Random(5)
                    for i in range(0, len(val_clips), batch_size):
                        # References from the training clips: validation clips are often few.
                        pool = val_clips + train_clips
                        items = [
                            item(pool, j, rng, with_reference) for j in range(i, min(i + batch_size, len(val_clips)))
                        ]
                        for k, (s, n) in trainer.losses(*collate(items, "cuda")).items():
                            acc = totals.setdefault(f"loss/{k}{suffix}", [0.0, 0])
                            acc[0] += float(s)
                            acc[1] += n
            return {k: s / max(n, 1) for k, (s, n) in totals.items()}

        values = averaged(losses)
        with (save_dir / "train.log").open("a", encoding="utf-8") as f:
            f.write(f"[val] step {at}: {format_metrics(values)}\n")
        averaged(lambda: samples(at))

    decoder: list = []

    def samples(at: int) -> None:
        """The first validation texts, spoken with the run's reference (LoRA as trained)."""
        args.samples_dir.mkdir(parents=True, exist_ok=True)
        try:
            if not decoder:
                decoder.append(decode.Decoder(model, end_token, graphs=False))
            c = model.config
            spoken = []
            for i, record in enumerate(val_records[:SAMPLES]):
                ref_path = args.samples_dir / f"reference_{i}.wav"
                if not ref_path.exists():
                    audio, rate = audio_io.read(record["audio"])
                    write_wav(ref_path, audio.T, rate)
                tokens = prompt.build(tokenizer, c.codebooks, c.semantic_begin, clean(record["text"]), reference)
                torch.manual_seed(SAMPLE_SEED)
                spoken.append((i, decoder[0].generate(tokens, generate.DEFAULT_SAMPLING, SAMPLE_MAX_FRAMES)))
            codec.to("cuda")
            for i, codes in spoken:
                if codes.shape[1]:
                    audio = codec.decode(codes.cuda()).float().cpu().numpy()
                    write_wav(args.samples_dir / f"step_{at:07d}_{i}.wav", audio, codec.sample_rate)
        except Exception as e:  # a failed sample never stops training
            log(f"Sample generation failed: {e}")
        finally:
            codec.to("cpu")
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- train
    log_interval = int(config.get("log_interval") or 10)
    valid_interval = int(config.get("valid_interval") or 0)
    save_interval = int(config.get("save_interval") or 0)
    max_grad_norm = float(config.get("max_grad_norm") or 1e9)
    rng = random.Random(1)
    if step == 0:
        validate(0)
    model.train()

    t_log, window = time.time(), {}
    while step < num_iters and not stop.is_set():
        for _ in range(accum):
            items = [item(train_clips, i, rng, rng.random() < REFERENCE_SHARE) for i in batches.next()]
            terms = trainer.losses(*collate(items, "cuda"))
            (sum(s / n for s, n in terms.values()) / accum).backward()
            for k, (s, n) in terms.items():
                acc = window.setdefault(f"loss/{k}", [0.0, 0])
                acc[0] += float(s.detach())
                acc[1] += n
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
        opt.step()
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % log_interval == 0 or step == num_iters:
            now = time.time()
            values = {k: s / max(n, 1) for k, (s, n) in window.items()} | rates(opt)
            values |= {"epoch": step / epoch_steps, "grad_norm": grad_norm, "log interval": now - t_log}
            line = f"[train] step {step}: {format_metrics(values)}"
            with (save_dir / "train.log").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)
            t_log, window = now, {}
        if valid_interval and step % valid_interval == 0 and step < num_iters:
            validate(step)
        if save_interval and step % save_interval == 0 and step < num_iters:
            save(step)

    if step >= num_iters:
        validate(step)
    if step > 0:
        save(step)
    log(f"Peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GB.")
    log("Stopped." if stop.is_set() and step < num_iters else "Training complete.")


if __name__ == "__main__":
    main()
