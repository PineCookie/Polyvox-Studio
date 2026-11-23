"""Training subprocess for dots.tts LoRAs.

``python -m polyvox.engines.dots.worker <train_config.yaml> --samples-dir
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

import hashlib
import os
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import Dataset

from ... import vram
from ..common.worker import Batches, LoraTraining, Run, log, replacing

SAMPLES = 2  # validation clips whose texts are synthesized at each validation
SAMPLE_SEED = 42
XVEC_CROPS = 3  # x-vectors kept per clip longer than the speaker encoder's 10 s window
NAMES = {"fm_loss": "loss/diff", "eos_loss": "loss/stop"}  # the losses trained and logged


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


def training_layers(model, checkpointing: bool, fused: bool) -> None:
    """How the language model's decoder layers and the diffusion head's blocks
    run while training with gradients (validation and samples run them as they
    are; sample generation rewires the diffusion head for inference, and Unfuse
    puts it back):

    - ``checkpointing``: activations recomputed in the backward pass instead of
      kept (less memory, slower; the diffusion head's are most of training's memory);
    - ``fused``: torch.compile of that forward, one compiled function per kind
      (a layer's weights are its inputs), any batch length, with the
      checkpointing inside (so torch.compile plans the recomputation). If
      compiling fails before a fused call has worked, the layers run unfused
      for the rest of the run."""
    import functools

    import torch
    from torch.utils.checkpoint import checkpoint

    from polyvox.engines.base import fuse_settings

    if fused:
        fuse_settings()
    layers = {"llm": list(model.core.llm.model.layers), "dit": list(model.core.velocity_field_predictor.blocks)}
    for kind, modules in layers.items():
        forward = type(modules[0]).forward
        train = forward
        if checkpointing:

            def train(self, *args, _forward=forward, **kwargs):
                return checkpoint(_forward, self, *args, use_reentrant=False, **kwargs)

        compiled = [torch.compile(train, dynamic=True)] if fused else []
        worked: list[bool] = []

        def run(self, *args, _forward=forward, _train=train, _compiled=compiled, _worked=worked, _kind=kind, **kwargs):
            if not torch.is_grad_enabled():
                return _forward(self, *args, **kwargs)
            if _compiled:
                try:
                    out = _compiled[0](self, *args, **kwargs)
                    if not _worked:
                        _worked.append(True)
                    return out
                except Exception as e:
                    if _worked or isinstance(e, torch.OutOfMemoryError):
                        raise
                    log(f"Fusing the {_kind} layers failed ({e!r}); running them unfused.")
                    _compiled.clear()
            return _train(self, *args, **kwargs)

        for module in modules:
            module.forward = functools.partial(run, module)


# The largest micro-batch that trains without gradient checkpointing, per GPU (vram.per_gpu; 0
# elsewhere): checkpointing the DiT and LM, batch 4 / 8 take ~6.9 / 7.7 GB instead of 13.5 / 20.6.
PLAIN_BATCHES = {vram.GB16: 4, vram.GB12: 2, vram.ON_CPU: 64}


def use_checkpointing(batch_size: int, profile) -> bool:
    """Checkpoint layers when the micro-batch is larger than what trains without it
    on this GPU (PLAIN_BATCHES); POLYVOX_TRAIN_CHECKPOINT 1 / 0 forces it on / off."""
    forced = os.environ.get("POLYVOX_TRAIN_CHECKPOINT")
    if forced in ("0", "1"):
        return forced == "1"
    return batch_size > vram.per_gpu(profile, PLAIN_BATCHES, 0)


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
    run = Run(__doc__.splitlines()[0])
    config, stop = run.config, run.stop

    import torch
    from safetensors.torch import save_file

    from polyvox.config import optimize_from_env
    from polyvox.engines.dots import lora as dots_lora
    from polyvox.vendor.dots_tts.data.pipelines.tts_pipeline import BasicTtsPipeline
    from polyvox.vendor.dots_tts.models.dots_tts.model import DotsTtsModel
    from polyvox.vendor.dots_tts.runtime import DotsTtsRuntime
    from polyvox.vendor.dots_tts.utils.text import detect

    from ..common.lora import state, write_config

    dots_lora.quiet_logging()
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
    fused = optimize_from_env()
    if checkpointing or fused:
        training_layers(model, checkpointing, fused)
    log(f"dots.tts loaded in {time.time() - t0:.0f} s; LoRA r{rank} on {len(loras)} layers ({', '.join(networks)})"
        f"{'; gradient checkpointing' if checkpointing else ''}{'; fused layers' if fused else ''}.")  # fmt: skip

    train_records, val_records = run.records("train_manifest"), run.records("val_manifest")
    data_cfg = SimpleNamespace(
        train_audio_sample_rate=model.config.vocoder.sample_rate,
        audio_samples_per_llm_token=model.hop_size * model.config.patch_size,
    )
    pipeline = BasicTtsPipeline(model.tokenizer, data_cfg)
    pad_id = model.tokenizer.pad_token_id
    pad_id = pad_id if pad_id is not None else (model.tokenizer.eos_token_id or 0)
    batch_size, accum = int(config["batch_size"]), int(config["grad_accum_steps"])
    workers = int(config.get("num_workers") or 0)
    cache_dir = run.run_dir / "cache"
    train_clips = cached(
        cache_dir / "encoded_train.pt",
        cache_key(Path(config["train_manifest"]), config["pretrained_path"]),
        lambda: encode(model, pipeline, train_records, workers, stop, "training"),
        len(train_records),
    )
    val_clips = (
        cached(
            cache_dir / "encoded_val.pt",
            cache_key(Path(config["val_manifest"]), config["pretrained_path"]),
            lambda: encode(model, pipeline, val_records, 0, stop, "validation"),
            len(val_records),
        )
        if val_records
        else []
    )
    batches = Batches(train_clips, batch_size, seed=0)
    epoch_steps = max(1.0, batches.per_epoch / accum)
    groups = [
        {"name": net, "params": [p for lo in loras.values() if lo.network == net for p in (lo.A, lo.B)]}
        for net in networks
    ]
    lora_config = {"r": rank, "alpha": alpha, "dropout": lora_cfg.get("dropout") or 0.0,
                   "enable_lm": "llm" in networks, "enable_dit": "dit" in networks}  # fmt: skip

    def write_checkpoint(folder: Path, weights: dict) -> None:
        with replacing(folder / "lora_weights.safetensors") as tmp:
            save_file(weights, tmp)
        write_config(folder, dots_lora.ENGINE, lora_config, str(config["pretrained_path"]))

    training = LoraTraining(run, loras, groups, epoch_steps, write_checkpoint)
    log(f"Optimizer: {training.optimizer}; {len(train_records)} clips, {epoch_steps:.1f} steps per epoch.")
    step = training.resume()

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
            run.validation(at, {"loss/total": sum(values.values()), **values})
            samples(at)
        finally:
            model.train()

    def samples(at: int) -> None:
        """Synthesize the first validation texts without a reference, LoRA merged."""
        for lora in loras.values():
            lora.enabled = False
        try:
            merger.apply(state(loras), alpha / rank)  # the averaged weights (validate runs averaged)
            for i, record in enumerate(val_records[:SAMPLES]):
                run.reference(i, record["audio"])
                torch.manual_seed(SAMPLE_SEED)
                with torch.no_grad():
                    out = runtime.generate(text=record["text"], language=detect(record["text"]))
                audio = out["audio"].detach().float().cpu().numpy().reshape(-1)
                run.write_wav(f"step_{at:07d}_{i}.wav", audio, runtime.sample_rate)
        except Exception as e:  # a failed sample never stops training
            log(f"Sample generation failed: {e}")
        finally:
            merger.apply(None)
            for lora in loras.values():
                lora.enabled = True

    def update() -> dict:
        """One optimizer step's micro-batches: losses weighted over the whole accumulated batch."""
        micro = [collate(model, batches.next(), pad_id) for _ in range(accum)]
        norms: dict[str, float] = {}
        for batch in micro:
            for k in NAMES:
                norms[k] = norms.get(k, 0.0) + float(batch["loss_masks"][k].sum())
        totals: dict[str, list[float]] = {}
        for batch in micro:
            loss, sums = batch_loss(batch, norms)
            loss.backward()
            for k, (s, n) in sums.items():
                acc = totals.setdefault(k, [0.0, 0.0])
                acc[0] += s
                acc[1] += n
        return {NAMES[k]: (s, n) for k, (s, n) in totals.items() if k in NAMES}

    if step == 0:
        training.averaged(lambda: validate(0))
    model.train()
    model.vocoder.eval()
    model.xvector_extractor.eval()
    step = training.train(step, update, lambda at: training.averaged(lambda: validate(at)))
    run.finish(step)


if __name__ == "__main__":
    main()
