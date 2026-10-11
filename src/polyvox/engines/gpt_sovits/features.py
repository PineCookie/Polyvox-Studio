"""Stage 0 of GPT-SoVITS training: what both models train on, per clip.

The same features upstream's prepare_datasets scripts write as files
(1-get-text, 2-get-hubert-wav32k, 2-get-sv, 3-get-semantic), computed once per
run and cached in ``<run>/cache/``:

    phones    phoneme ids (int16)
    bert      RoBERTa features per phoneme [1024, phones] (zeros for English; float16)
    wav       the 32 kHz audio SoVITS learns to produce (int16)
    ssl       HuBERT features of that audio [768, frames at 50 Hz] (float16)
    sv        speaker embedding (ERes2NetV2) [20480] (float16)
    semantic  the base model's semantic tokens (25 Hz; int16): the GPT stage's targets

Two differences from upstream: text goes through the Chinese / English front
end synthesis uses (upstream's training front end drops English words in a
Chinese clip), and audio is decoded with audio_io and resampled with librosa
instead of ffmpeg.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

CACHE_VERSION = 1  # bump when what extract() makes changes
SAMPLE_RATE = 32_000
MAX_PEAK = 2.2  # upstream skips clips this loud (broken files)
# Upstream's level: half the clip normalized to a 0.95 peak, half as it was.
NORM_PEAK, NORM_MIX = 0.95, 0.5
HUBERT_SCALE = 1145.14  # upstream feeds HuBERT this (odd but trained-on) level


class Skipped(Exception):
    """A clip that can't be trained on; the message is the reason, counted."""


def log(message: str) -> None:
    print(f"[studio] {message}", file=sys.stderr, flush=True)


def clip_features(pipeline, base_voice, audio_path: str, text: str) -> dict:
    """One clip's features. ``pipeline``: synth.Pipeline (front ends);
    ``base_voice``: the base model (its quantizer makes the semantic tokens)."""
    import librosa
    import torch
    import torchaudio

    from ...data import audio_io

    audio, rate = audio_io.read(audio_path)
    mono = np.asarray(audio.mean(axis=0) if audio.ndim == 2 else audio, dtype=np.float32)
    a32 = librosa.resample(mono, orig_sr=rate, target_sr=SAMPLE_RATE) if rate != SAMPLE_RATE else mono
    peak = float(np.abs(a32).max()) if len(a32) else 0.0
    if peak == 0.0:
        raise Skipped("silent")
    if peak > MAX_PEAK:
        raise Skipped("too loud")
    mixed = a32 / peak * (NORM_PEAK * NORM_MIX) + (1 - NORM_MIX) * a32
    wav = np.clip(mixed * 32768, -32768, 32767).astype(np.int16)

    phones, bert = pipeline.phones_and_bert(text.replace("%", "-").replace("￥", ","), lead_in=False)
    if not phones:
        raise Skipped("no phonemes")
    if bert.shape[-1] != len(phones):
        raise Skipped("text features don't match the phonemes")

    dev, dtype = pipeline.device, pipeline.dtype
    with torch.no_grad():
        hubert_in = librosa.resample(mixed * HUBERT_SCALE, orig_sr=SAMPLE_RATE, target_sr=16_000)
        x = torch.from_numpy(hubert_in)[None].to(dev)
        ssl = pipeline.hubert.model(x.to(dtype))["last_hidden_state"].transpose(1, 2)
        if torch.isnan(ssl).any():  # as upstream: half precision overflowed; again in full
            pipeline.hubert.float()
            try:
                ssl = pipeline.hubert.model(x.float())["last_hidden_state"].transpose(1, 2)
            finally:
                pipeline.hubert.to(dtype)
            if torch.isnan(ssl).any():
                raise Skipped("HuBERT gave NaN")
        semantic = base_voice.sovits.extract_latent(ssl.to(dtype))[0, 0]
        wav16 = torchaudio.functional.resample(torch.from_numpy(wav).to(dev).float() / 32768, SAMPLE_RATE, 16_000)
        sv = pipeline.sv.compute_embedding3(wav16[None].to(dtype))[0]
    return {
        "phones": torch.tensor(phones, dtype=torch.int16),
        "bert": bert.to(torch.float16).cpu(),
        "wav": torch.from_numpy(wav),
        "ssl": ssl[0].to(torch.float16).cpu(),
        "sv": sv.to(torch.float16).cpu(),
        "semantic": semantic.to(torch.int16).cpu(),
    }


def extract(pipeline, base_voice, records: list[dict], stop: threading.Event) -> tuple[list[dict], dict[str, int]]:
    """Features of every clip that can be trained on (each with its manifest
    ``index`` and ``text``), and how many were skipped for which reason."""
    clips, skipped = [], Counter()
    t0 = time.time()
    for i, r in enumerate(records):
        if stop.is_set():
            break
        try:
            clips.append({"index": i, "text": r["text"], **clip_features(pipeline, base_voice, r["audio"], r["text"])})
        except Skipped as e:
            skipped[str(e)] += 1
        except Exception as e:  # an unreadable clip or text the front end can't handle
            skipped["unreadable"] += 1
            log(f"Skipping {Path(r['audio']).name}: {e}")
        if (i + 1) % 200 == 0:
            log(f"Features: {i + 1} / {len(records)} clips")
    log(f"Features of {len(clips)} clips in {time.time() - t0:.0f} s.")
    return clips, dict(skipped)


def cache_key(manifest: Path, base_dir: Path, version: str) -> dict:
    """What the features depend on: the manifest's bytes, the base models and this code."""
    return {
        "version": CACHE_VERSION,
        "manifest": hashlib.sha1(manifest.read_bytes()).hexdigest(),
        "models": str(base_dir),
        "model_version": version,
    }


def cached(cache_file: Path, key: dict, make) -> tuple[list[dict], dict[str, int]]:
    """``make()``'s (clips, skipped), from ``cache_file`` when it was written for
    ``key``; a complete fresh result (``make`` returns None when stopped) is
    saved there, so a resumed run skips this stage."""
    import torch

    if cache_file.is_file():
        try:
            saved = torch.load(cache_file, map_location="cpu", weights_only=True)
            if saved.get("key") == key:
                log(f"Features from {cache_file.name} ({len(saved['clips'])} clips).")
                return saved["clips"], saved["skipped"]
        except Exception as e:  # unreadable or from another version: extract again
            log(f"Ignoring {cache_file.name}: {e}")
    result = make()
    if result is None:
        return [], {}
    clips, skipped = result
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    torch.save({"key": key, "clips": clips, "skipped": skipped}, tmp)
    tmp.replace(cache_file)
    return clips, skipped
