"""Training subprocess for GPT-SoVITS v2Pro / v2ProPlus voices.

``python -m polyvox.engines.gpt_sovits.worker <train_config.yaml> --samples-dir
<dir> --stop-file <path> [--parent-pid <pid>]``

Same contract as the other engines' workers: ``[train]`` lines in
checkpoints/train.log, checkpoints/step_NNNNNNN/ and latest/
(training_state.json / .pt), samples/ with step_NNNNNNN_<i>.wav and
reference_<i>.wav; the stop file (or the server exiting) saves a checkpoint and
ends the run. Steps count optimizer updates, 1..num_iters, across both stages.

Full fine-tuning, as the GPT-SoVITS webui trains voices:

1. Features of every clip (features.py), cached in <run>/cache/.
2. SoVITS (steps 1..sovits_steps), as upstream's s2_train.py: the generator
   from the pretrained s2G and its discriminator from s2D, adversarial +
   feature-matching + mel L1 (x45) + KL losses; AdamW (betas 0.8 / 0.99), the
   text side at 0.4 x lr, lr x 0.999875 per epoch; float16 autocast with a
   gradient scaler.
3. GPT (the remaining steps), as upstream's s1_train.py: the text-to-semantic
   model from s1v3, cross-entropy summed over the semantic tokens (logged as
   the mean per token); ScaledAdam (betas 0.9 / 0.95, clipping scale 2), one
   update per 4 batches, lr 0.01 for the first update and 0.002 after (upstream's
   warmup-cosine schedule is pinned to that in its code); float16 autocast with
   a gradient scaler. Clips as upstream keeps them: up to 54 s, 3-25 phonemes
   per second.

Batches are grouped by length like upstream's bucket samplers, but each epoch
is exactly ceil(clips / batch) batches, so the steps are known once the
features are (the worker then writes sovits_steps and num_iters into the
config); datasets under 100 clips are not repeated within an epoch. Raising
num_iters (continuing a run) lengthens the GPT stage.

Checkpoints are voices: sovits_weights.pth and gpt_weights.ckpt saved as the
webui saves them (float16; SoVITS without the training-only posterior encoder
and with the version tag over the zip signature), and the run's reference clip.
SoVITS-stage checkpoints have no GPT weights: such a voice uses the base GPT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import torch

SAMPLES = 2  # validation clips whose texts are synthesized at each validation
SAMPLE_SEED = 42
SAMPLE_EPOCHS = 1  # samples after every epoch of either stage
SAVE_EPOCHS = {"sovits": 4, "gpt": 5}  # the webui's default checkpoint intervals
# Upstream's SoVITS training settings (configs/s2v2ProPlus.json, the same for v2Pro).
BETAS, EPS = (0.8, 0.99), 1e-9
LR_DECAY = 0.999875  # per epoch
TEXT_LR_RATE = 0.4  # text embedding, text encoder and MRTE train at this share of the lr
C_MEL, C_KL = 45.0, 1.0
# Upstream keeps clips of 0.6-54 s, and its bucket sampler only 32 < frames <= 1900 (50 Hz).
MIN_FRAMES, MAX_FRAMES = 33, 1900
BUCKETS = (32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)
# Logged SoVITS losses (the quantizer's commitment loss is in the total too, but stays 0 with it frozen).
NAMES = ("loss/mel", "loss/kl", "loss/fm", "loss/gen", "loss/disc")
# Upstream's GPT training (s1_train.py, configs/s1longer-v2.yaml).
GPT_ACCUM = 4  # batches per optimizer update
GPT_LR_FIRST, GPT_LR = 0.01, 0.002
GPT_MAX_SEC, SEMANTIC_HZ = 54, 25
PHONES_PER_SEC = (3, 25)
SEMANTIC_PAD = 1024
GPT_BUCKET_S = 2  # upstream's buckets: 2 s of semantic tokens wide


def log(message: str) -> None:
    print(f"[studio] {message}", file=sys.stderr, flush=True)


def sovits_bucket(frames: int) -> int:
    import bisect

    return bisect.bisect_left(BUCKETS, frames)


def gpt_bucket(tokens: int) -> int:
    return tokens // (SEMANTIC_HZ * GPT_BUCKET_S)


class LengthBatches:
    """Batches of clips of similar length (``bucket`` of a length: upstream's
    buckets), shuffled; epoch ``e``'s batches are the same on every call, so a
    resumed run continues within its epoch. A short last batch is filled with
    the clips before it."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 0, bucket=sovits_bucket):
        self.lengths, self.batch_size, self.seed, self.bucket = lengths, batch_size, seed, bucket
        self.per_epoch = max(1, math.ceil(len(lengths) / batch_size))
        self._cached: tuple[int, list[list[int]]] | None = None

    def epoch(self, e: int) -> list[list[int]]:
        if self._cached and self._cached[0] == e:
            return self._cached[1]
        rng = random.Random(self.seed * 1_000_003 + e)
        keys = {i: (self.bucket(n), rng.random()) for i, n in enumerate(self.lengths)}
        order = sorted(keys, key=keys.__getitem__)
        bs = self.batch_size
        batches = [order[i : i + bs] for i in range(0, len(order), bs)]
        if len(batches[-1]) < bs:
            batches[-1] = (order * bs)[-bs:] if len(order) < bs else order[-bs:]
        rng.shuffle(batches)
        self._cached = (e, batches)
        return batches

    def batch(self, n: int) -> list[int]:
        """The ``n``-th batch counted across epochs."""
        e, i = divmod(n, self.per_epoch)
        return self.epoch(e)[i]


def sovits_keeps(clip: dict) -> bool:
    return MIN_FRAMES <= len(clip["wav"]) // 640 <= MAX_FRAMES


def gpt_keeps(clip: dict) -> bool:
    tokens, phones = len(clip["semantic"]), len(clip["phones"])
    if not tokens or tokens > GPT_MAX_SEC * SEMANTIC_HZ or phones > GPT_MAX_SEC * SEMANTIC_HZ / 2.5:
        return False
    return PHONES_PER_SEC[0] <= phones / (tokens / SEMANTIC_HZ) <= PHONES_PER_SEC[1]


def gpt_batch_size(clips: int, batch_size: int) -> int:
    """As upstream: at most a quarter of the clips (so an epoch has updates)."""
    return max(1, min(batch_size, clips // 4))


def plan_steps(
    sovits_clips: int, sovits_batch: int, sovits_epochs: float, gpt_clips: int, gpt_batch: int, gpt_epochs: float
) -> tuple[int, int]:
    """(SoVITS steps, GPT steps) of a run: whole batches per epoch, GPT updates of 4 batches."""
    sovits = math.ceil(sovits_epochs * math.ceil(sovits_clips / sovits_batch)) if sovits_clips else 0
    gpt_batches = math.ceil(gpt_clips / gpt_batch_size(gpt_clips, gpt_batch)) if gpt_clips else 0
    return sovits, math.ceil(gpt_epochs * gpt_batches / GPT_ACCUM)


def grad_norm(params) -> torch.Tensor:
    """The gradients' total L2 norm (for the log), as upstream's
    commons.clip_grad_value_(params, None) computes it, in one fused call
    instead of a GPU sync per parameter."""
    import torch

    grads = [p.grad for p in params if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)).float())


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


def save_gpt(path: Path, weights: dict, config: dict, info: str) -> None:
    """Weights as the webui saves a GPT (s1_train's checkpoint callback): float16,
    under the Lightning module's "model." prefix."""
    import torch

    opt = {"weight": {f"model.{k}": v.half() for k, v in weights.items()}, "config": config, "info": info}
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(opt, tmp)
    tmp.replace(path)


def link_or_copy(src: Path, dst: Path) -> None:
    """``dst`` as a hard link to ``src`` (same file, no extra disk), else a copy."""
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Run:
    """What both stages share: the config, the front ends and the base model in
    inference form (samples load the trained weights into it), the reference,
    logging and checkpoints."""

    def __init__(self, args, config: dict):
        import torch

        from ... import vram
        from ..dots.worker import watch_for_stop
        from . import voices
        from .synth import Pipeline

        vram.apply_limit("cuda")
        self.args, self.config = args, config
        self.stop = threading.Event()
        threading.Thread(target=watch_for_stop, args=(args.stop_file, args.parent_pid, self.stop), daemon=True).start()
        self.save_dir = Path(config["save_path"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.save_dir.parent
        self.latest = self.save_dir / "latest"
        self.version = str(config.get("version") or "v2ProPlus")
        self.base_dir = Path(config["pretrained_path"])
        self.dev = torch.device("cuda")
        torch.manual_seed(0)

        t0 = time.time()
        self.pipeline = Pipeline(self.base_dir, Path(config["g2pw_path"]), "cuda")
        self.voice = self.pipeline.load_voice(*self.pipeline.base_weights(self.version))
        log(f"GPT-SoVITS {self.version} front ends loaded in {time.time() - t0:.0f} s.")
        if voices.read_reference(self.run_dir) is None:  # copied into the run once
            ref = config["reference"]
            voices.write_reference(self.run_dir, Path(ref["audio"]), ref["text"], ref.get("source", ""))
        self.reference = voices.read_reference(self.run_dir)
        self.log_interval = int(config.get("log_interval") or 10)
        self._window: dict[str, list[float]] = {}
        self._t_log = time.time()

    # ------------------------------------------------------------ logging
    def record(self, step: int, values: dict, extra: dict[str, float], last: bool) -> None:
        """Collect a step's values (numbers, or GPU tensors read back only here);
        every log_interval steps (and the last) a ``[train]`` line with their
        means (non-finite ones, from steps the gradient scaler skipped, left
        out) and ``extra`` as is."""
        for k, v in values.items():
            self._window.setdefault(k, []).append(v)
        if step % self.log_interval and not last:
            return
        now = time.time()
        window = {k: [float(x) for x in v] for k, v in self._window.items()}
        means = {k: sum(f) / len(f) for k, v in window.items() if (f := [x for x in v if math.isfinite(x)])}
        means |= extra
        means["log interval"] = now - self._t_log
        line = f"[train] step {step}: {format_metrics(means)}"
        with (self.save_dir / "train.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
        self._t_log, self._window = now, {}

    def reset_log(self) -> None:
        self._window, self._t_log = {}, time.time()

    # ------------------------------------------------------------ samples and checkpoints
    def samples(self, at: int, val_records: list[dict]) -> None:
        """The first validation texts in the run's reference voice, with the
        weights currently in ``self.voice``."""
        import soundfile as sf

        from ...data import audio_io
        from .synth import Options

        if not val_records:
            return
        samples_dir = self.args.samples_dir
        samples_dir.mkdir(parents=True, exist_ok=True)

        def write_wav(path: Path, audio, rate: int) -> None:
            tmp = path.with_suffix(".tmp")
            sf.write(tmp, audio, rate, format="WAV")
            tmp.replace(path)

        try:
            ref = self.pipeline.reference(self.voice, *self.reference)
            for i, record in enumerate(val_records[:SAMPLES]):
                reference = samples_dir / f"reference_{i}.wav"
                if not reference.exists():
                    audio, rate = audio_io.read(record["audio"])
                    write_wav(reference, audio.T, rate)
                audio = self.pipeline.synthesize(self.voice, ref, record["text"], Options(), SAMPLE_SEED)
                write_wav(samples_dir / f"step_{at:07d}_{i}.wav", audio, self.voice.sample_rate)
        except Exception as e:  # a failed sample never stops training
            log(f"Sample generation failed: {e}")

    def checkpoint(self, at: int, stage: str, epoch: float) -> Path:
        """A fresh step folder with the voice's config and reference; the stage adds the weights."""
        from . import voices

        folder = self.save_dir / f"step_{at:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        voices.write_config(folder, self.version, stage=stage, epoch=round(epoch, 2))
        for name in (voices.REFERENCE_WAV, voices.REFERENCE_JSON):
            shutil.copy2(self.run_dir / name, folder / name)
        return folder

    def save_state(self, at: int, state: dict) -> None:
        import torch

        self.latest.mkdir(parents=True, exist_ok=True)
        tmp = self.latest / "training_state.tmp"
        torch.save({"step": at, **state}, tmp)
        tmp.replace(self.latest / "training_state.pt")
        (self.latest / "training_state.json").write_text(json.dumps({"step": at}), encoding="utf-8")
        log(f"Saved step {at}.")

    def load_state(self) -> dict | None:
        import torch

        path = self.latest / "training_state.pt"
        return torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None

    def write_plan(self, sovits_steps: int, num_iters: int) -> None:
        """The exact step counts into the run's config (the server shows them)."""
        self.config["sovits_steps"], self.config["num_iters"] = sovits_steps, num_iters
        text = yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True)
        tmp = self.args.config_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.args.config_path)


def train_sovits(run: Run, clips: list[dict], val_records: list[dict], step: int, end: int, state: dict | None) -> int:
    """SoVITS steps ``step + 1 .. end``; returns the last step done. The
    checkpoint at ``end`` holds the SoVITS weights the GPT stage uses."""
    import torch
    import torch.nn.functional as F

    from ...vendor.gpt_sovits.module import commons
    from ...vendor.gpt_sovits.module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from ...vendor.gpt_sovits.module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
    from ...vendor.gpt_sovits.module.models import MultiPeriodDiscriminator, SynthesizerTrn
    from ...vendor.gpt_sovits.process_ckpt import load_sovits_new, load_weights
    from . import voices
    from .synth import Config

    dev, version, config = run.dev, run.version, run.config
    pretrained = load_sovits_new(run.pipeline.base_weights(version)[0])
    hps = Config(pretrained["config"])
    data = hps.data
    model_cfg = {**pretrained["config"]["model"], "version": version}
    segment = int(hps.train.segment_size)
    hop = int(data.hop_length)
    net_g = SynthesizerTrn(data.filter_length // 2 + 1, segment // hop, n_speakers=data.n_speakers, **model_cfg).to(dev)
    net_g.load_state_dict(pretrained["weight"], strict=False)
    net_d = MultiPeriodDiscriminator(model_cfg.get("use_spectral_norm", False), version=version).to(dev)
    net_d.load_state_dict(load_weights(run.base_dir / "v2Pro" / f"s2D{version}.pth")["weight"], strict=False)
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
    if state is not None:
        net_g.load_state_dict(state["net_g"])
        net_d.load_state_dict(state["net_d"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        scaler.load_state_dict(state["scaler"])

    batch_size = int(config["batch_size"])
    batches = LengthBatches([len(c["wav"]) // hop for c in clips], batch_size, seed=int(config.get("seed") or 0))
    per_epoch = batches.per_epoch
    log(f"SoVITS: {len(clips)} clips, batch {batch_size}, {per_epoch} steps per epoch, steps {step + 1}-{end}.")
    saved_config = {  # saved with each voice: the base model's, as the webui records it
        **pretrained_config_without_paths(hps),
        "train": {
            **dict(hps["train"]),
            "epochs": math.ceil(end / per_epoch),
            "learning_rate": lr,
            "batch_size": batch_size,
            "text_low_lr_rate": TEXT_LR_RATE,
        },
        "model": model_cfg,
        "version": version,
        "name": run.run_dir.name,
    }

    def to_voice() -> None:
        run.voice.sovits.load_state_dict({k: v for k, v in net_g.state_dict().items() if not k.startswith("enc_q")})

    def save(at: int) -> None:
        folder = run.checkpoint(at, "sovits", at / per_epoch)
        info = f"{math.ceil(at / per_epoch)}epoch_{at}iteration"
        save_sovits(folder / voices.SOVITS_FILE, net_g.state_dict(), saved_config, info, version)
        (folder / voices.GPT_FILE).unlink(missing_ok=True)  # this stage's voices speak with the base GPT
        run.save_state(
            at,
            {
                "stage": "sovits",
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "scaler": scaler.state_dict(),
            },
        )

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

    def train_step(batch: dict) -> dict[str, torch.Tensor]:
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
        grad_norm_d = grad_norm(net_d.parameters())
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
        grad_norm_g = grad_norm(net_g.parameters())
        scaler.step(optim_g)
        scaler.update()
        losses = (loss_mel, loss_kl, loss_fm, loss_gen, loss_disc)
        # Tensors: read back when a log line is written, not every step (a GPU sync).
        return {
            **{k: v.detach() for k, v in zip(NAMES, losses, strict=True)},
            "grad_norm": grad_norm_g,
            "grad_norm_d": grad_norm_d,
        }

    if step == 0:
        run.samples(0, val_records)  # the base model, for comparison
    net_g.train()
    net_d.train()
    run.reset_log()
    sample_every, save_every = per_epoch * SAMPLE_EPOCHS, per_epoch * SAVE_EPOCHS["sovits"]
    while step < end and not run.stop.is_set():
        epoch = step // per_epoch
        for o, lrs in zip((optim_g, optim_d), base_lrs, strict=True):
            for g, base in zip(o.param_groups, lrs, strict=True):
                g["lr"] = base * LR_DECAY ** (epoch + 1)  # upstream's ExponentialLR, stepped once before epoch 1
        values = train_step(collate([clips[i] for i in batches.batch(step)]))
        step += 1
        extra = {"lr": optim_g.param_groups[0]["lr"], "epoch": step / per_epoch}
        run.record(step, values, extra, last=step == end)
        if step < end and step % sample_every == 0:
            to_voice()
            run.samples(step, val_records)
        if step < end and step % save_every == 0:
            save(step)
    if step == end:
        to_voice()
        run.samples(step, val_records)
    if step > 0 and not (run.save_dir / f"step_{step:07d}" / voices.SOVITS_FILE).is_file():
        save(step)  # (not twice when stopped right after a save)
    return step


def train_gpt(
    run: Run, clips: list[dict], val_records: list[dict], start: int, step: int, end: int, state: dict | None
) -> int:
    """GPT steps ``step + 1 .. end`` of a stage that began after step ``start``
    (the SoVITS stage's last step, whose checkpoint holds its weights); returns
    the last step done."""
    import torch

    from ...vendor.gpt_sovits.AR.models.t2s_model import Text2SemanticDecoder
    from ...vendor.gpt_sovits.AR.modules.optim import ScaledAdam
    from ...vendor.gpt_sovits.process_ckpt import load_sovits_new, load_weights
    from . import voices

    dev, config = run.dev, run.config
    # The SoVITS weights this stage's voices speak with.
    sovits_file = run.save_dir / f"step_{start:07d}" / voices.SOVITS_FILE if start else None
    if sovits_file is None or not sovits_file.is_file():
        sovits_file = run.pipeline.base_weights(run.version)[0]
    run.voice.sovits.load_state_dict(load_sovits_new(sovits_file)["weight"], strict=False)

    base = load_weights(run.pipeline.base_weights(run.version)[1])
    gpt_config = {**base["config"], "data": {**base["config"]["data"], "max_sec": GPT_MAX_SEC}}
    gpt = Text2SemanticDecoder(gpt_config, top_k=3).to(dev)
    gpt.load_state_dict({k.removeprefix("model."): v for k, v in base["weight"].items()})
    del base
    names = [[n for n, _ in gpt.named_parameters()]]
    opt = ScaledAdam(
        gpt.parameters(),
        lr=GPT_LR_FIRST,
        betas=(0.9, 0.95),
        clipping_scale=2.0,
        parameters_names=names,
        show_dominant_parameters=False,
        clipping_update_period=1000,
    )
    scaler = torch.amp.GradScaler("cuda")
    if state is not None:
        gpt.load_state_dict(state["gpt"])
        opt.load_state_dict(state["optim"])
        scaler.load_state_dict(state["scaler"])

    batch_size = gpt_batch_size(len(clips), int(config.get("gpt_batch_size") or 8))
    batches = LengthBatches([len(c["semantic"]) for c in clips], batch_size, seed=1, bucket=gpt_bucket)
    per_epoch = batches.per_epoch  # batches; an update takes GPT_ACCUM of them
    log(f"GPT: {len(clips)} clips, batch {batch_size} x {GPT_ACCUM}, steps {step + 1}-{end}.")

    def epoch_of(at: int) -> float:
        return (at - start) * GPT_ACCUM / per_epoch

    saved_config = {  # saved with each GPT, as the webui records its training config
        **gpt_config,
        "train": {**gpt_config.get("train", {}), "batch_size": batch_size, "epochs": math.ceil(epoch_of(end))},
    }

    def to_voice() -> None:
        run.voice.gpt.load_state_dict(gpt.state_dict())

    def save(at: int) -> None:
        folder = run.checkpoint(at, "gpt", epoch_of(at))
        link_or_copy(sovits_file, folder / voices.SOVITS_FILE)
        save_gpt(folder / voices.GPT_FILE, gpt.state_dict(), saved_config, f"GPT-e{math.ceil(epoch_of(at))}")
        run.save_state(
            at, {"stage": "gpt", "gpt": gpt.state_dict(), "optim": opt.state_dict(), "scaler": scaler.state_dict()}
        )

    def collate(items: list[dict]) -> dict:
        """As upstream's Text2SemanticDataset.collate: phonemes padded with 0,
        semantic tokens with the end-of-speech id, RoBERTa features with zeros."""
        b = len(items)
        phones = torch.zeros(b, max(len(c["phones"]) for c in items), dtype=torch.long)
        semantic = torch.full((b, max(len(c["semantic"]) for c in items)), SEMANTIC_PAD, dtype=torch.long)
        bert = torch.zeros(b, 1024, phones.shape[1])
        for i, c in enumerate(items):
            phones[i, : len(c["phones"])] = c["phones"].long()
            semantic[i, : len(c["semantic"])] = c["semantic"].long()
            bert[i, :, : c["bert"].shape[-1]] = c["bert"].float()
        lengths = {
            "phone_lengths": torch.tensor([len(c["phones"]) for c in items]),
            "semantic_lengths": torch.tensor([len(c["semantic"]) for c in items]),
        }
        tensors = {"phones": phones, "semantic": semantic, "bert": bert, **lengths}
        return {k: v.to(dev, non_blocking=True) for k, v in tensors.items()}

    def update(at: int) -> dict:
        """One optimizer update over GPT_ACCUM batches (upstream: every 4 batches)."""
        losses, accs = [], []
        for j in range(GPT_ACCUM):
            batch = collate([clips[i] for i in batches.batch((at - start) * GPT_ACCUM + j)])
            with torch.autocast("cuda", dtype=torch.float16):
                loss, acc = gpt.forward_old(
                    batch["phones"], batch["phone_lengths"], batch["semantic"], batch["semantic_lengths"], batch["bert"]
                )
            scaler.scale(loss).backward()
            tokens = batch["semantic"].shape[0] * (batch["semantic"].shape[1] + 1)  # each with its end of speech
            losses.append(loss.detach() / tokens)
            accs.append(acc)
        for g in opt.param_groups:  # upstream: 0.01 until its scheduler first steps, then pinned to 0.002
            g["lr"] = GPT_LR_FIRST if at == start else GPT_LR
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return {"loss/gpt": torch.stack(losses).mean(), "top3_acc": sum(accs) / len(accs)}

    gpt.train()
    run.reset_log()
    sample_every, save_every = SAMPLE_EPOCHS, SAVE_EPOCHS["gpt"]
    while step < end and not run.stop.is_set():
        before = epoch_of(step)
        values = update(step)
        step += 1
        after = epoch_of(step)
        run.record(step, values, {"lr": opt.param_groups[0]["lr"], "epoch": after}, last=step == end)
        if step < end and math.floor(after / sample_every) > math.floor(before / sample_every):
            to_voice()
            run.samples(step, val_records)
        if step < end and math.floor(after / save_every) > math.floor(before / save_every):
            save(step)
    if step == end:
        to_voice()
        run.samples(step, val_records)
    if step > start and not (run.save_dir / f"step_{step:07d}" / voices.GPT_FILE).is_file():
        save(step)
    return step


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
    from . import features

    run = Run(args, config)

    # ---------------------------------------------------------------- features
    train_manifest = Path(config["train_manifest"])
    records = [dict(e) for e in manifests.read_manifest(train_manifest)[0]]
    val_path = Path(config["val_manifest"]) if config.get("val_manifest") else None
    val_records = [dict(e) for e in manifests.read_manifest(val_path)[0]] if val_path and val_path.is_file() else []

    # The dataset's cache (shared by its runs); runs from before that: their own folder.
    cache = config.get("feature_cache") or run.run_dir / "cache" / f"gpt_sovits_{run.version}.pt"
    meta = features.cache_meta(run.base_dir, run.version)
    result = features.extract(run.pipeline, run.voice, records, run.stop, Path(cache), meta)
    if result is None:
        log("Stopped.")
        return
    clips, skipped = result
    if skipped:
        log(f"Left out {sum(skipped.values())} clips: " + ", ".join(f"{k} {n}" for k, n in skipped.items()))
    sovits_clips = [c for c in clips if sovits_keeps(c)]
    gpt_clips = [c for c in clips if gpt_keeps(c)]
    for stage, kept, rule in (
        ("SoVITS", sovits_clips, "0.66-38 s"),
        ("GPT", gpt_clips, f"up to {GPT_MAX_SEC} s and {PHONES_PER_SEC[0]}-{PHONES_PER_SEC[1]} phonemes per second"),
    ):
        if len(kept) < len(clips):
            log(f"{stage} leaves out {len(clips) - len(kept)} clips outside {rule}.")

    # ---------------------------------------------------------------- plan
    if "sovits_steps" not in config:  # a new run: its steps, now that the clips are known
        if "gpt_epochs" in config:
            sovits_steps, gpt_steps = plan_steps(
                len(sovits_clips),
                int(config["batch_size"]),
                float(config.get("sovits_epochs") or 0),
                len(gpt_clips),
                int(config.get("gpt_batch_size") or 8),
                float(config["gpt_epochs"]),
            )
        else:  # from before the GPT stage: SoVITS only
            sovits_steps, gpt_steps = int(config["num_iters"]), 0
        run.write_plan(sovits_steps, sovits_steps + gpt_steps)
    sovits_steps, num_iters = int(config["sovits_steps"]), int(config["num_iters"])
    if (sovits_steps and not sovits_clips) or (num_iters > sovits_steps and not gpt_clips):
        raise SystemExit("No clip of this dataset can be trained on.")

    # ---------------------------------------------------------------- train
    state = run.load_state()
    step = int(state["step"]) if state else 0
    if state:
        log(f"Resumed at step {step} ({state.get('stage', 'sovits')}).")
    if step < sovits_steps:
        sovits_state = state if state and state.get("stage", "sovits") == "sovits" else None
        step = train_sovits(run, sovits_clips, val_records, step, sovits_steps, sovits_state)
        state = None
        torch.cuda.empty_cache()
    if step >= sovits_steps and step < num_iters and not run.stop.is_set():
        gpt_state = state if state and state.get("stage") == "gpt" else None
        step = train_gpt(run, gpt_clips, val_records, sovits_steps, step, num_iters, gpt_state)
    log(f"Peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GB.")
    log("Stopped." if run.stop.is_set() and step < num_iters else "Training complete.")


def pretrained_config_without_paths(hps) -> dict:
    """The base model's saved config minus its makers' paths and file lists."""
    data = {k: v for k, v in dict(hps["data"]).items() if k not in ("training_files", "validation_files", "exp_dir")}
    return {"data": data, "content_module": hps.get("content_module", "cnhubert")}


if __name__ == "__main__":
    main()
