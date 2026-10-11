"""Training subprocess for GPT-SoVITS v2Pro / v2ProPlus voices.

``python -m polyvox.engines.gpt_sovits.worker <train_config.yaml> --samples-dir
<dir> --stop-file <path> [--parent-pid <pid>]``

Same contract as the other engines' workers: ``[train]`` lines in
checkpoints/train.log, checkpoints/step_NNNNNNN/ and latest/
(training_state.json / .pt), samples/ with step_NNNNNNN_<i>.wav and
reference_<i>.wav; the stop file (or the server exiting) saves a checkpoint and
ends the run. Steps count optimizer updates, 1..num_iters.

Full fine-tuning, as the GPT-SoVITS webui trains voices:

1. Features of every clip (features.py), cached in <run>/cache/.
2. SoVITS, as upstream's s2_train.py: the generator from the pretrained s2G
   and its discriminator from s2D, adversarial + feature-matching + mel L1
   (x45) + KL losses; AdamW (betas 0.8 / 0.99), the text side at 0.4 x lr,
   lr x 0.999875 per epoch; float16 autocast with a gradient scaler.
   Batches are grouped by length like upstream's bucket sampler, but an epoch
   is exactly ceil(clips / batch) batches (so its steps are known up front),
   and datasets under 100 clips are not repeated within an epoch.

Checkpoints hold sovits_weights.pth saved as the webui saves it (float16,
without the training-only posterior encoder, version tag in place of the zip
signature: the webui loads it), the run's reference clip, and no GPT weights
yet: the voice then uses the base GPT (the GPT stage comes later).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import threading
import time
from pathlib import Path

import yaml

SAMPLES = 2  # validation clips whose texts are synthesized at each validation
SAMPLE_SEED = 42
STAGE = "sovits"
# Upstream's SoVITS training settings (configs/s2v2ProPlus.json, the same for v2Pro).
BETAS, EPS = (0.8, 0.99), 1e-9
LR_DECAY = 0.999875  # per epoch
TEXT_LR_RATE = 0.4  # text embedding, text encoder and MRTE train at this share of the lr
C_MEL, C_KL = 45.0, 1.0
# Upstream keeps clips of 0.6-54 s, and its bucket sampler only 32 < frames <= 1900 (50 Hz).
MIN_FRAMES, MAX_FRAMES = 33, 1900
BUCKETS = (32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)
# Logged losses (the quantizer's commitment loss is in the total too, but stays 0 with it frozen).
NAMES = ("loss/mel", "loss/kl", "loss/fm", "loss/gen", "loss/disc")


def log(message: str) -> None:
    print(f"[studio] {message}", file=sys.stderr, flush=True)


class LengthBatches:
    """Batches of clips of similar length (upstream's buckets), shuffled; epoch
    ``e``'s batches are the same on every call, so a resumed run continues
    within its epoch. A short last batch is filled with the clips before it."""

    def __init__(self, frames: list[int], batch_size: int, seed: int = 0):
        self.frames, self.batch_size, self.seed = frames, batch_size, seed
        self.per_epoch = max(1, math.ceil(len(frames) / batch_size))

    def epoch(self, e: int) -> list[list[int]]:
        import bisect

        rng = random.Random(self.seed * 1_000_003 + e)
        keys = {i: (bisect.bisect_left(BUCKETS, f), rng.random()) for i, f in enumerate(self.frames)}
        order = sorted(keys, key=keys.__getitem__)
        bs = self.batch_size
        batches = [order[i : i + bs] for i in range(0, len(order), bs)]
        if len(batches[-1]) < bs:
            batches[-1] = (order * bs)[-bs:] if len(order) < bs else order[-bs:]
        rng.shuffle(batches)
        return batches


def format_metrics(values: dict[str, float]) -> str:
    def fmt(k: str, v: float) -> str:
        if k == "lr":
            return f"{k}: {v:.4e}"
        if k == "log interval":
            return f"{k}: {v:.2f}s"
        return f"{k}: {v:.6f}"

    return ", ".join(fmt(k, v) for k, v in values.items())


def save_sovits(path: Path, weights: dict, config: dict, info: str, version: str) -> None:
    """Weights as the webui saves a voice (process_ckpt.savee): float16, no
    posterior encoder, the version tag over the zip signature."""
    from ...vendor.gpt_sovits.process_ckpt import my_save2

    opt = {"weight": {k: v.half() for k, v in weights.items() if "enc_q" not in k}, "config": config, "info": info}
    tmp = path.with_name(f".{path.name}.tmp")
    my_save2(opt, str(tmp), version)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))

    import soundfile as sf
    import torch
    import torch.nn.functional as F

    from ... import vram
    from ...data import manifests
    from ...vendor.gpt_sovits.module import commons
    from ...vendor.gpt_sovits.module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from ...vendor.gpt_sovits.module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
    from ...vendor.gpt_sovits.module.models import MultiPeriodDiscriminator, SynthesizerTrn
    from ...vendor.gpt_sovits.process_ckpt import load_sovits_new, load_weights
    from ..dots.worker import watch_for_stop
    from . import features, voices
    from .synth import Config, Options, Pipeline

    vram.apply_limit("cuda")
    stop = threading.Event()
    threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid, stop), daemon=True).start()

    save_dir = Path(config["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir.parent
    num_iters = int(config["num_iters"])
    version = str(config.get("version") or "v2ProPlus")
    base_dir = Path(config["pretrained_path"])
    torch.manual_seed(0)
    dev = torch.device("cuda")

    t0 = time.time()
    pipeline = Pipeline(base_dir, Path(config["g2pw_path"]), "cuda")
    sovits_path, _ = pipeline.base_weights(version)
    # The base model in inference form: its quantizer makes the semantic tokens,
    # and samples load the trained weights into it.
    voice = pipeline.load_voice(*pipeline.base_weights(version))
    log(f"GPT-SoVITS {version} front ends loaded in {time.time() - t0:.0f} s.")

    # Reference for the samples and checkpoints, copied into the run once.
    if voices.read_reference(run_dir) is None:
        ref = config["reference"]
        voices.write_reference(run_dir, Path(ref["audio"]), ref["text"], ref.get("source", ""))
    ref_wav, ref_text = voices.read_reference(run_dir)

    # ---------------------------------------------------------------- stage 0
    train_manifest = Path(config["train_manifest"])
    records = [dict(e) for e in manifests.read_manifest(train_manifest)[0]]
    val_path = Path(config["val_manifest"]) if config.get("val_manifest") else None
    val_records = [dict(e) for e in manifests.read_manifest(val_path)[0]] if val_path and val_path.is_file() else []

    def make():
        result = features.extract(pipeline, voice, records, stop)
        return None if stop.is_set() else result

    clips, skipped = features.cached(
        run_dir / "cache" / "features_train.pt", features.cache_key(train_manifest, base_dir, version), make
    )
    if stop.is_set():
        log("Stopped.")
        return
    frames = [len(c["wav"]) // 640 for c in clips]
    kept = [c for c, f in zip(clips, frames, strict=True) if MIN_FRAMES <= f <= MAX_FRAMES]
    if len(kept) < len(clips):
        skipped = {**skipped, "shorter than 0.66 s or longer than 38 s": len(clips) - len(kept)}
    clips = kept
    if skipped:
        log(f"Left out {sum(skipped.values())} clips: " + ", ".join(f"{k} {n}" for k, n in skipped.items()))
    if not clips:
        raise SystemExit("No clip of this dataset can be trained on.")

    # ---------------------------------------------------------------- models
    pretrained = load_sovits_new(sovits_path)
    hps = Config(pretrained["config"])
    data, train_cfg = hps.data, hps.train
    model_cfg = {**pretrained["config"]["model"], "version": version}
    segment = int(train_cfg.segment_size)
    hop = int(data.hop_length)
    net_g = SynthesizerTrn(data.filter_length // 2 + 1, segment // hop, n_speakers=data.n_speakers, **model_cfg).to(dev)
    net_g.load_state_dict(pretrained["weight"], strict=False)
    net_d = MultiPeriodDiscriminator(model_cfg.get("use_spectral_norm", False), version=version).to(dev)
    net_d.load_state_dict(load_weights(base_dir / "v2Pro" / f"s2D{version}.pth")["weight"], strict=False)
    del pretrained

    lr = float(config["learning_rate"])
    enc_p = net_g.enc_p
    text_side = [*enc_p.text_embedding.parameters(), *enc_p.encoder_text.parameters(), *enc_p.mrte.parameters()]
    text_ids = {id(p) for p in text_side}
    optim_g = torch.optim.AdamW(
        [
            {"params": [p for p in net_g.parameters() if id(p) not in text_ids and p.requires_grad], "lr": lr},
            {"params": text_side, "lr": lr * TEXT_LR_RATE},
        ],
        lr,
        betas=BETAS,
        eps=EPS,
    )
    optim_d = torch.optim.AdamW(net_d.parameters(), lr, betas=BETAS, eps=EPS)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in (optim_g, optim_d)]
    scaler = torch.amp.GradScaler("cuda")

    batch_size = int(config["batch_size"])
    batches = LengthBatches([len(c["wav"]) // hop for c in clips], batch_size, seed=int(config.get("seed") or 0))
    log(f"SoVITS: {len(clips)} clips, batch {batch_size}, {batches.per_epoch} steps per epoch.")

    # The config saved with each voice: the base model's, as the webui records it.
    saved_config = {
        **pretrained_config_without_paths(hps),
        "train": {
            **dict(hps["train"]),
            "epochs": math.ceil(num_iters / batches.per_epoch),
            "learning_rate": lr,
            "batch_size": batch_size,
            "text_low_lr_rate": TEXT_LR_RATE,
        },
        "model": model_cfg,
        "version": version,
        "name": run_dir.name,
    }

    # ---------------------------------------------------------------- resume
    step = 0
    latest = save_dir / "latest"
    if (latest / "training_state.pt").is_file():
        state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=True)
        net_g.load_state_dict(state["net_g"])
        net_d.load_state_dict(state["net_d"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        log(f"Resumed at step {step}.")

    def save(at: int) -> None:
        epoch = at / batches.per_epoch
        folder = save_dir / f"step_{at:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        info = f"{math.ceil(epoch)}epoch_{at}iteration"
        save_sovits(folder / voices.SOVITS_FILE, net_g.state_dict(), saved_config, info, version)
        voices.write_config(folder, version, stage=STAGE, epoch=round(epoch, 2))
        for name in (voices.REFERENCE_WAV, voices.REFERENCE_JSON):
            shutil.copy2(run_dir / name, folder / name)
        latest.mkdir(parents=True, exist_ok=True)
        tmp = latest / "training_state.tmp"
        torch.save(
            {
                "step": at,
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "scaler": scaler.state_dict(),
            },
            tmp,
        )
        tmp.replace(latest / "training_state.pt")
        (latest / "training_state.json").write_text(json.dumps({"step": at}), encoding="utf-8")
        log(f"Saved step {at}.")

    def write_wav(path: Path, audio, rate: int) -> None:
        tmp = path.with_suffix(".tmp")
        sf.write(tmp, audio, rate, format="WAV")
        tmp.replace(path)

    def samples(at: int) -> None:
        """The first validation texts in the run's reference voice, with the
        SoVITS weights so far (and the base GPT)."""
        if not val_records:
            return
        args.samples_dir.mkdir(parents=True, exist_ok=True)
        try:
            weights = {k: v for k, v in net_g.state_dict().items() if not k.startswith("enc_q")}
            voice.sovits.load_state_dict(weights)
            ref = pipeline.reference(voice, ref_wav, ref_text)
            from ...data import audio_io

            for i, record in enumerate(val_records[:SAMPLES]):
                reference = args.samples_dir / f"reference_{i}.wav"
                if not reference.exists():
                    audio, rate = audio_io.read(record["audio"])
                    write_wav(reference, audio.T, rate)
                audio = pipeline.synthesize(voice, ref, record["text"], Options(), SAMPLE_SEED)
                write_wav(args.samples_dir / f"step_{at:07d}_{i}.wav", audio, voice.sample_rate)
        except Exception as e:  # a failed sample never stops training
            log(f"Sample generation failed: {e}")

    def collate(items: list[dict]) -> dict:
        """A batch on the GPU as upstream's TextAudioSpeakerCollate pads it, the
        spectrograms computed here (upstream: in its data loader)."""
        specs, ssls, wavs = [], [], []
        for c in items:
            y = c["wav"].to(dev, non_blocking=True).float() / 32768
            spec = spectrogram_torch(y[None], data.filter_length, data.sampling_rate, hop, data.win_length)[0]
            ssl = c["ssl"].to(dev).float()
            n = spec.shape[-1]
            ssl = F.pad(ssl[None], (0, n - ssl.shape[-1]), mode="replicate")[0] if ssl.shape[-1] < n else ssl[:, :n]
            specs.append(spec)
            ssls.append(ssl)
            wavs.append(y)
        b = len(items)
        max_spec = 2 * (max(s.shape[-1] for s in specs) // 2 + 1)
        max_wav = max(len(w) for w in wavs)
        max_text = max(len(c["phones"]) for c in items)
        out = {
            "ssl": torch.zeros(b, ssls[0].shape[0], max_spec, device=dev),
            "spec": torch.zeros(b, specs[0].shape[0], max_spec, device=dev),
            "wav": torch.zeros(b, 1, max_wav, device=dev),
            "text": torch.zeros(b, max_text, dtype=torch.long, device=dev),
            "sv": torch.stack([c["sv"] for c in items]).to(dev).float(),
        }
        for i, c in enumerate(items):
            out["ssl"][i, :, : ssls[i].shape[-1]] = ssls[i]
            out["spec"][i, :, : specs[i].shape[-1]] = specs[i]
            out["wav"][i, 0, : len(wavs[i])] = wavs[i]
            out["text"][i, : len(c["phones"])] = c["phones"].to(dev).long()
        out["spec_lengths"] = torch.tensor([s.shape[-1] for s in specs], device=dev)
        out["text_lengths"] = torch.tensor([len(c["phones"]) for c in items], device=dev)
        return out

    def train_step(batch: dict) -> dict[str, float]:
        """One discriminator and one generator update, as s2_train.train_and_evaluate."""
        autocast = torch.autocast("cuda", dtype=torch.float16)
        spec, spec_lengths = batch["spec"], batch["spec_lengths"]
        with autocast:
            y_hat, kl_ssl, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q), _ = net_g(
                batch["ssl"], spec, spec_lengths, batch["text"], batch["text_lengths"], batch["sv"]
            )
            mel = spec_to_mel_torch(
                spec, data.filter_length, data.n_mel_channels, data.sampling_rate, data.mel_fmin, data.mel_fmax
            )
            y_mel = commons.slice_segments(mel, ids_slice, segment // hop)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                data.filter_length,
                data.n_mel_channels,
                data.sampling_rate,
                hop,
                data.win_length,
                data.mel_fmin,
                data.mel_fmax,
            )
            y = commons.slice_segments(batch["wav"], ids_slice * hop, segment)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with torch.autocast("cuda", enabled=False):
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad(set_to_none=True)
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast:
            _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with torch.autocast("cuda", enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * C_MEL
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * C_KL
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_g = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl
        optim_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        losses = (loss_mel, loss_kl, loss_fm, loss_gen, loss_disc)
        return {
            **{k: float(v) for k, v in zip(NAMES, losses, strict=True)},
            "grad_norm": float(grad_norm_g),
            "grad_norm_d": float(grad_norm_d),
        }

    # ---------------------------------------------------------------- train
    log_interval = int(config.get("log_interval") or 10)
    valid_interval = int(config.get("valid_interval") or 0)
    save_interval = int(config.get("save_interval") or 0)
    if step == 0:
        samples(0)
    net_g.train()
    net_d.train()
    t_log = time.time()
    window: dict[str, list[float]] = {}
    while step < num_iters and not stop.is_set():
        epoch, offset = divmod(step, batches.per_epoch)
        for o, lrs in zip((optim_g, optim_d), base_lrs, strict=True):
            for g, base in zip(o.param_groups, lrs, strict=True):
                g["lr"] = base * LR_DECAY ** (epoch + 1)  # upstream's ExponentialLR, stepped once before epoch 1
        for indices in batches.epoch(epoch)[offset:]:
            if step >= num_iters or stop.is_set():
                break
            for k, v in train_step(collate([clips[i] for i in indices])).items():
                window.setdefault(k, []).append(v)
            step += 1
            if step % log_interval == 0 or step == num_iters:
                now = time.time()
                # Steps the gradient scaler skipped (inf / NaN gradients) left out.
                values = {k: sum(f) / len(f) for k, v in window.items() if (f := [x for x in v if math.isfinite(x)])}
                values |= {"lr": optim_g.param_groups[0]["lr"], "epoch": step / batches.per_epoch}
                values["log interval"] = now - t_log
                line = f"[train] step {step}: {format_metrics(values)}"
                with (save_dir / "train.log").open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                print(line, flush=True)
                t_log, window = now, {}
            if valid_interval and step % valid_interval == 0 and step < num_iters:
                samples(step)
            if save_interval and step % save_interval == 0 and step < num_iters:
                save(step)

    if step >= num_iters:
        samples(step)
    saved = (save_dir / f"step_{step:07d}" / voices.SOVITS_FILE).is_file()  # stopped right after a save
    if step > 0 and not saved:
        save(step)
    log(f"Peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GB.")
    log("Stopped." if stop.is_set() and step < num_iters else "Training complete.")


def pretrained_config_without_paths(hps) -> dict:
    """The base model's saved config minus its makers' paths and file lists."""
    data = {k: v for k, v in dict(hps["data"]).items() if k not in ("training_files", "validation_files", "exp_dir")}
    return {"data": data, "content_module": hps.get("content_module", "cnhubert")}


if __name__ == "__main__":
    main()
