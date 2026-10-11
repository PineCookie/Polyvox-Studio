"""Training subprocess for dots.tts LoRAs.

``python -m voxcpm_studio.engines.dots.worker <train_config.yaml> --samples-dir
<dir> --stop-file <path> [--parent-pid <pid>]``

Same contract as VoxCPM's worker, so runs look alike in the studio:
``[train]`` / ``[val]`` lines in checkpoints/train.log (loss/diff = the
diffusion head's flow-matching loss, loss/stop = the end-of-speech loss),
checkpoints/step_NNNNNNN/ and latest/ (training_state.json), samples/ with
step_NNNNNNN_<i>.wav and reference_<i>.wav. The stop file (or the server
exiting) saves a checkpoint and ends the run.

Training follows the official script (scripts/train_dots_tts.py: the "basic"
text -> speech template, losses averaged over the accumulated batch, gradient
clipping) with LoRA (lora.py) instead of full fine-tuning: frozen weights in
bfloat16 under bfloat16 autocast, the audio VAE and speaker encoder in float32.
Steps count optimizer updates, 1..num_iters.

Two shortcuts, same results: the frozen audio VAE and speaker encoder run once
per clip before training (``encode``: the latent distribution, sampled afresh
each step as before, and the x-vector, a few random 10 s crops of longer clips),
not at every step (a third of the step time); and the language model's
vocabulary logits are not computed, as the basic template puts no loss on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from torch.utils.data import Dataset

POLL_S = 1.0
SAMPLES = 2  # validation clips whose texts are synthesized at each validation
SAMPLE_SEED = 42
XVEC_CROPS = 3  # x-vectors kept per clip longer than the speaker encoder's 10 s window
NAMES = {"fm_loss": "loss/diff", "eos_loss": "loss/stop"}  # the losses trained and logged


def log(message: str) -> None:
    print(f"[studio] {message}", file=sys.stderr, flush=True)


def watch_for_stop(stop_file: Path, parent_pid: int | None, stop: threading.Event) -> None:
    from voxcpm_studio.engines.voxcpm.worker import process_alive

    while not stop_file.exists():
        if parent_pid is not None and not process_alive(parent_pid):
            log("Server exited.")
            break
        time.sleep(POLL_S)
    log("Stop requested; saving a checkpoint.")
    stop.set()


def make_optimizer(name: str, groups: list[dict], config: dict, num_iters: int):
    """(optimizer, scheduler or None) as engines.voxcpm.worker.use_optimizer chooses them."""
    import torch

    lr, wd = float(config["learning_rate"]), float(config["weight_decay"])
    if name == "prodigy_sf":
        from prodigyplus import ProdigyPlusScheduleFree

        opt = ProdigyPlusScheduleFree(
            groups,
            lr=lr,
            weight_decay=wd,
            betas=(0.95, 0.99),
            schedulefree_c=20,
            prodigy_steps=int(config.get("prodigy_steps") or 0),
        )
        return opt, None
    if name == "prodigy":
        from prodigyopt import Prodigy

        opt = Prodigy(groups, lr=lr, weight_decay=wd, safeguard_warmup=True, use_bias_correction=True)
    elif name == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    warmup = int(config.get("warmup_steps") or 0)

    def cosine(step: int) -> float:  # transformers' cosine schedule with warmup
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(1, num_iters - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, cosine)


def rates(opt) -> dict[str, float]:
    """lr = d x lr for Prodigy (per group lr_<net>, elr_<net> for Schedule-Free), as VoxCPM's worker logs."""
    out: dict[str, float] = {}
    for i, g in enumerate(opt.param_groups):
        d = float(g.get("d", 1.0))
        if i == 0:
            out["lr"] = d * float(g["lr"])
        if "d" in g:
            out[f"lr_{g['name']}"] = d * float(g["lr"])
            if "effective_lr" in g:
                out[f"elr_{g['name']}"] = d * float(g["effective_lr"])
    return out


def format_metrics(values: dict[str, float]) -> str:
    def fmt(k: str, v: float) -> str:
        if k == "lr" or k.startswith(("lr_", "elr_")):
            return f"{k}: {v:.4e}"
        if k == "log interval":
            return f"{k}: {v:.2f}s"
        return f"{k}: {v:.6f}"

    return ", ".join(fmt(k, v) for k, v in values.items())


class Clips(Dataset):
    """Manifest entries through dots.tts's sample pipeline (a map-style dataset;
    top-level so DataLoader workers can be spawned)."""

    def __init__(self, records: list[dict], pipeline):
        self.records, self.pipeline = records, pipeline

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        return self.pipeline.process_sample({"fid": str(i), "text": r["text"], "audio": r["audio"]})


def encode(model, pipeline, records: list[dict], workers: int, stop: threading.Event, what: str) -> list[dict]:
    """Each clip's tokens, audio-VAE latent distribution and x-vector(s), on the CPU.

    Audio is read and resampled in DataLoader workers; they end with the pass
    (persistent workers outlive the process and hold the studio's stdout pipe)."""
    import torch
    from torch.utils.data import DataLoader

    dev = next(model.core.parameters()).device
    loader = DataLoader(Clips(records, pipeline), batch_size=None, num_workers=workers, persistent_workers=False)
    window = round(model.xvector_extractor.sample_rate * model.xvector_extractor.max_audio_seconds)
    out = []
    t0 = time.time()
    model.vocoder.eval()
    model.xvector_extractor.eval()
    with torch.no_grad():
        for i, s in enumerate(loader):
            if stop.is_set():
                break
            wav = s["sample"].to(dev)[None]  # [1, 1, samples], padded to whole audio tokens
            n = int(s["sample_length"])
            fbank, fbank_length = s["fbank"][None].to(dev), torch.tensor([s["fbank_length"]], device=dev)
            xvecs = [
                model.xvector_extractor(
                    wav, audio_lengths=torch.tensor([n], device=dev), fbank=fbank, fbank_lengths=fbank_length
                )[0].cpu()
                for _ in range(XVEC_CROPS if n > window else 1)  # each call crops at random
            ]
            out.append(
                {
                    "input_ids": torch.tensor(s["input_ids"]),
                    "labels": torch.tensor(s["labels"]),
                    "loss_mask": torch.tensor(s["loss_mask"], dtype=torch.float32),
                    # [2 x latent dim, frames]; float16 halves the cache, far below the sampling noise
                    "latents": model.vocoder.extract_latents(wav)[0].half().cpu(),
                    "xvectors": xvecs,
                }
            )
            if (i + 1) % 200 == 0:
                log(f"Encoding {what}: {i + 1} / {len(records)} clips")
    log(f"Encoded {len(out)} {what} clips in {time.time() - t0:.0f} s.")
    return out


CACHE_VERSION = 1  # bump when encode's output changes


def cache_key(manifest: Path, model_path: str) -> dict:
    """What encoded clips depend on: the manifest's bytes (edits, dropped clips),
    the base model and the encoding itself."""
    return {
        "version": CACHE_VERSION,
        "manifest": hashlib.sha1(manifest.read_bytes()).hexdigest(),
        "model": str(model_path),
        "xvec_crops": XVEC_CROPS,
    }


def cached(cache_file: Path, key: dict, make, expected: int) -> list[dict]:
    """``make()``'s encoded clips, from ``cache_file`` when it was written for the
    same ``key``; a complete fresh result (``expected`` clips; a stop cuts it
    short) is saved there. A resumed or continued run skips the encode pass."""
    import torch

    if cache_file.is_file():
        try:
            saved = torch.load(cache_file, map_location="cpu", weights_only=True)
            if saved.get("key") == key and len(saved.get("clips", ())) == expected:
                log(f"Encoded clips from {cache_file.name} ({expected}).")
                return saved["clips"]
        except Exception as e:  # unreadable or from another version: encode again
            log(f"Ignoring {cache_file.name}: {e}")
    clips = make()
    if len(clips) == expected:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        torch.save({"key": key, "clips": clips}, tmp)
        tmp.replace(cache_file)
    return clips


def collate(model, items: list[dict], pad_id: int) -> dict:
    """A training batch of encoded clips, on the model's device, as DotsTtsCore.forward reads it."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    dev = next(model.core.parameters()).device
    items = sorted(items, key=lambda c: -c["latents"].shape[-1])
    batch = model.prepare_training_batch(
        {
            "input_ids": pad_sequence([c["input_ids"] for c in items], batch_first=True, padding_value=pad_id),
            "labels": pad_sequence([c["labels"] for c in items], batch_first=True, padding_value=pad_id),
            "loss_mask": pad_sequence([c["loss_mask"] for c in items], batch_first=True),
            "input_ids_lengths": torch.tensor([len(c["input_ids"]) for c in items]),
        }
    )
    out = {k: v.to(dev, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
    out["loss_masks"] = {k: v.to(dev) for k, v in batch["loss_masks"].items()}
    latents = pad_sequence([c["latents"].T for c in items], batch_first=True).transpose(1, 2).to(dev, torch.float32)
    out["latents"] = latents
    out["latent_lengths"] = torch.tensor([c["latents"].shape[-1] for c in items], device=dev)
    out["latents_sampled"] = model.core.io_helper.sample_from_latent(latents)  # a new draw each step
    out["xvector"] = torch.stack([random.choice(c["xvectors"]) for c in items]).to(dev)
    return out


class Batches:
    """Shuffled batches of encoded clips; one pass = one epoch."""

    def __init__(self, clips: list[dict], batch_size: int, seed: int):
        self.clips, self.batch_size = clips, batch_size
        self.per_epoch = max(1, len(clips) // batch_size)
        self._rng = random.Random(seed)
        self._order: list[int] = []

    def next(self) -> list[dict]:
        if len(self._order) < self.batch_size:
            self._order = list(range(len(self.clips)))
            self._rng.shuffle(self._order)
        picked, self._order = self._order[: self.batch_size], self._order[self.batch_size :]
        return [self.clips[i] for i in picked]


def checkpoint_layers(model) -> None:
    """Recompute layer activations in the backward pass instead of keeping them:
    the language model's decoder layers (transformers' own switch) and the
    diffusion head's blocks (whose activations are most of training's memory).
    Inference (no gradients) runs the blocks as before."""
    import torch
    from torch.utils.checkpoint import checkpoint

    model.core.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for block in model.core.velocity_field_predictor.blocks:

        def forward(*args, _forward=block.forward, **kwargs):
            if not torch.is_grad_enabled():
                return _forward(*args, **kwargs)
            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)

        block.forward = forward


def use_checkpointing(batch_size: int, profile) -> bool:
    """Checkpoint layers when the micro-batch is larger than what trains without it
    on this GPU (vram.Profile.dots_plain_batch); VOXCPM_STUDIO_TRAIN_CHECKPOINT
    1 / 0 forces it on / off."""
    forced = os.environ.get("VOXCPM_STUDIO_TRAIN_CHECKPOINT")
    if forced in ("0", "1"):
        return forced == "1"
    return batch_size > profile.dots_plain_batch


def loss_terms(model, batch: dict) -> dict:
    """The trained losses of a collated batch, without vocabulary logits."""
    llm = model.core.llm
    forward = llm.forward
    llm.forward = lambda *args, **kwargs: forward(*args, **{**kwargs, "logits_to_keep": 1})
    try:
        out = model.core(batch)
    finally:
        del llm.forward  # the class's forward again (inference needs every position)
    masks = batch["loss_masks"]
    return {
        "fm_loss": model._compute_fm_loss_term(out.pred, out.target, masks["fm_loss"]),
        "eos_loss": model._compute_eos_loss_term(out.eos_out, masks["eos_loss"]),
    }


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

    from voxcpm_studio import vram
    from voxcpm_studio.data import audio_io, manifests
    from voxcpm_studio.engines.dots import lora as dots_lora
    from voxcpm_studio.vendor.dots_tts.data.pipelines.tts_pipeline import BasicTtsPipeline
    from voxcpm_studio.vendor.dots_tts.models.dots_tts.model import DotsTtsModel
    from voxcpm_studio.vendor.dots_tts.runtime import DotsTtsRuntime

    vram.apply_limit("cuda")
    dots_lora.quiet_logging()
    stop = threading.Event()
    threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid, stop), daemon=True).start()

    save_dir = Path(config["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    num_iters = int(config["num_iters"])
    lora_cfg = config["lora"]
    networks = tuple(
        n for n, on in (("llm", lora_cfg.get("enable_lm", True)), ("dit", lora_cfg.get("enable_dit", True))) if on
    )
    rank, alpha = int(lora_cfg["r"]), float(lora_cfg["alpha"])
    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda")

    t0 = time.time()
    model = DotsTtsModel.from_pretrained(config["pretrained_path"])
    model.core.to(dtype=torch.bfloat16)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    merger = dots_lora.Merger(model)  # before any generation fuses the diffusion head
    loras = dots_lora.attach(model, rank, alpha, float(lora_cfg.get("dropout") or 0.0), networks)
    runtime = DotsTtsRuntime(model, Path(config["pretrained_path"]), precision="bfloat16", optimize=False)
    checkpointing = use_checkpointing(int(config["batch_size"]), vram.detect("cuda")[1])
    if checkpointing:
        checkpoint_layers(model)
    log(f"dots.tts loaded in {time.time() - t0:.0f} s; LoRA r{rank} on {len(loras)} layers ({', '.join(networks)})"
        f"{'; gradient checkpointing' if checkpointing else ''}.")  # fmt: skip

    train_records = [dict(e) for e in manifests.read_manifest(Path(config["train_manifest"]))[0]]
    val_path = Path(config["val_manifest"]) if config.get("val_manifest") else None
    val_records = [dict(e) for e in manifests.read_manifest(val_path)[0]] if val_path and val_path.is_file() else []
    data_cfg = SimpleNamespace(
        train_audio_sample_rate=model.config.vocoder.sample_rate,
        audio_samples_per_llm_token=model.hop_size * model.config.patch_size,
    )
    pipeline = BasicTtsPipeline(model.tokenizer, data_cfg)
    pad_id = model.tokenizer.pad_token_id
    pad_id = pad_id if pad_id is not None else (model.tokenizer.eos_token_id or 0)
    batch_size, accum = int(config["batch_size"]), int(config["grad_accum_steps"])
    workers = int(config.get("num_workers") or 0)
    cache_dir = save_dir.parent / "cache"  # the run folder's
    train_clips = cached(
        cache_dir / "encoded_train.pt",
        cache_key(Path(config["train_manifest"]), config["pretrained_path"]),
        lambda: encode(model, pipeline, train_records, workers, stop, "training"),
        len(train_records),
    )
    val_clips = (
        cached(
            cache_dir / "encoded_val.pt",
            cache_key(val_path, config["pretrained_path"]),
            lambda: encode(model, pipeline, val_records, 0, stop, "validation"),
            len(val_records),
        )
        if val_records
        else []
    )
    batches = Batches(train_clips, batch_size, seed=0)
    epoch_steps = max(1.0, batches.per_epoch / accum)

    groups = [
        {"name": net, "params": [p for n, lo in loras.values() if n == net for p in (lo.A, lo.B)]} for net in networks
    ]
    params = [p for g in groups for p in g["params"]]
    optimizer_name = str(config.get("optimizer") or "prodigy_sf")
    opt, scheduler = make_optimizer(optimizer_name, groups, config, num_iters)
    schedule_free = hasattr(opt, "train") and hasattr(opt, "eval")
    log(f"Optimizer: {optimizer_name}; {len(train_records)} clips, {epoch_steps:.1f} steps per epoch.")

    # ---------------------------------------------------------------- resume
    step = 0
    latest = save_dir / "latest"
    if (latest / "training_state.pt").is_file():
        state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, (_, lora) in loras.items():
                lora.A.copy_(state["train_weights"][f"{name}.lora_A"])
                lora.B.copy_(state["train_weights"][f"{name}.lora_B"])
        opt.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        log(f"Resumed at step {step}.")
    if schedule_free:
        opt.train()

    def lora_tensors() -> dict:
        """The weights to use (Schedule-Free: the average)."""
        if schedule_free:
            opt.eval()
        try:
            return dots_lora.state(loras)
        finally:
            if schedule_free:
                opt.train()

    def save(at: int) -> None:
        weights = lora_tensors()
        lora_config = {"r": rank, "alpha": alpha, "dropout": lora_cfg.get("dropout") or 0.0,
                       "enable_lm": "llm" in networks, "enable_dit": "dit" in networks}  # fmt: skip
        for folder in (save_dir / f"step_{at:07d}", latest):
            folder.mkdir(parents=True, exist_ok=True)
            save_file(weights, folder / "lora_weights.safetensors")
            dots_lora.write_config(folder, lora_config, str(config["pretrained_path"]))
        torch.save(
            {
                "step": at,
                "train_weights": dots_lora.state(loras),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            },
            latest / "training_state.pt",
        )
        (latest / "training_state.json").write_text(json.dumps({"step": at}), encoding="utf-8")
        log(f"Saved step {at}.")

    def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
        tmp = path.with_suffix(".tmp")
        sf.write(tmp, audio, rate, format="WAV")
        tmp.replace(path)

    def batch_loss(batch: dict, norms: dict[str, float] | None = None):
        """(gradient loss, {name: (sum, count)}) of one collated batch."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            terms = loss_terms(model, batch)
        sums, loss = {}, 0.0
        for k, t in terms.items():
            s = (t.loss.float() * t.mask.float()).sum()
            sums[k] = (float(s.detach()), float(t.mask.float().sum()))
            if norms is not None:
                loss = loss + s / max(norms[k], 1.0)
        return loss, sums

    def validate(at: int) -> None:
        if not val_clips:
            return
        model.eval()
        if schedule_free:
            opt.eval()
        try:
            totals: dict[str, list[float]] = {}
            with torch.no_grad():
                for i in range(0, len(val_clips), batch_size):
                    batch = collate(model, val_clips[i : i + batch_size], pad_id)
                    for k, (s, n) in batch_loss(batch)[1].items():
                        acc = totals.setdefault(k, [0.0, 0.0])
                        acc[0] += s
                        acc[1] += n
            values = {NAMES[k]: s / max(n, 1.0) for k, (s, n) in totals.items() if k in NAMES}
            values = {"loss/total": sum(values.values()), **values}
            with (save_dir / "train.log").open("a", encoding="utf-8") as f:
                f.write(f"[val] step {at}: {format_metrics(values)}\n")
            samples(at)
        finally:
            model.train()
            if schedule_free:
                opt.train()

    def samples(at: int) -> None:
        """Synthesize the first validation texts without a reference, LoRA merged."""
        args.samples_dir.mkdir(parents=True, exist_ok=True)
        for lora in (lo for _, lo in loras.values()):
            lora.enabled = False
        try:
            merger.apply(dots_lora.state(loras), alpha / rank)  # opt.eval(): the averaged weights
            for i, record in enumerate(val_records[:SAMPLES]):
                reference = args.samples_dir / f"reference_{i}.wav"
                if not reference.exists():
                    audio, rate = audio_io.read(record["audio"])
                    write_wav(reference, audio.T, rate)
                torch.manual_seed(SAMPLE_SEED)
                from voxcpm_studio.vendor.dots_tts.utils.text import detect

                with torch.no_grad():
                    out = runtime.generate(text=record["text"], language=detect(record["text"]))
                audio = out["audio"].detach().float().cpu().numpy().reshape(-1)
                write_wav(args.samples_dir / f"step_{at:07d}_{i}.wav", audio, runtime.sample_rate)
        except Exception as e:  # a failed sample never stops training
            log(f"Sample generation failed: {e}")
        finally:
            merger.apply(None)
            for lora in (lo for _, lo in loras.values()):
                lora.enabled = True

    # ---------------------------------------------------------------- train
    log_interval = int(config.get("log_interval") or 10)
    valid_interval = int(config.get("valid_interval") or 0)
    save_interval = int(config.get("save_interval") or 0)
    if step == 0:
        validate(0)
    model.train()
    model.vocoder.eval()
    model.xvector_extractor.eval()

    def train(step: int) -> int:
        """Optimizer steps until num_iters or a stop request; returns the last step."""
        t_log = time.time()
        window: dict[str, list[float]] = {}
        while step < num_iters and not stop.is_set():
            micro = [collate(model, batches.next(), pad_id) for _ in range(accum)]
            norms: dict[str, float] = {}  # loss weights over the whole accumulated batch
            for batch in micro:
                for k in NAMES:
                    norms[k] = norms.get(k, 0.0) + float(batch["loss_masks"][k].sum())
            for batch in micro:
                loss, sums = batch_loss(batch, norms)
                loss.backward()
                for k, (s, n) in sums.items():
                    acc = window.setdefault(k, [0.0, 0.0])
                    acc[0] += s
                    acc[1] += n
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(config.get("max_grad_norm") or 1e9)))
            opt.step()
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % log_interval == 0 or step == num_iters:
                now = time.time()
                values = {NAMES[k]: s / max(n, 1.0) for k, (s, n) in window.items() if k in NAMES}
                values |= rates(opt)
                values |= {"epoch": step / epoch_steps, "grad_norm": grad_norm}
                values["log interval"] = now - t_log  # since the last line (training.seconds_per_step)
                line = f"[train] step {step}: {format_metrics(values)}"
                with (save_dir / "train.log").open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                print(line, flush=True)
                t_log, window = now, {}
            if valid_interval and step % valid_interval == 0 and step < num_iters:
                validate(step)
            if save_interval and step % save_interval == 0 and step < num_iters:
                save(step)
        return step

    step = train(step)
    if step >= num_iters:
        validate(step)
    if step > 0:
        save(step)
    log(f"Peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GB.")
    log("Stopped." if stop.is_set() and step < num_iters else "Training complete.")


if __name__ == "__main__":
    main()
