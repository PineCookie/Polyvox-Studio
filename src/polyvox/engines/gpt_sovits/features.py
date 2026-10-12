"""Stage 0 of GPT-SoVITS training: what both models train on, per clip.

The same features upstream's prepare_datasets scripts write as files
(1-get-text, 2-get-hubert-wav32k, 2-get-sv, 3-get-semantic), computed once per
run and cached in ``<run>/cache/``:

    phones    phoneme ids (int16)
    bert      RoBERTa features per phoneme [1024, phones] (zeros for English; float16)
    wav       the 32 kHz audio SoVITS learns to produce (int16)
    ssl       HuBERT features of that audio [768, frames at 50 Hz] (float16)
    sv        speaker embedding (ERes2NetV2) [20480] (float16; v2Pro only, but cheap)
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


def clip_key(record: dict) -> str:
    """What a clip's features depend on besides the models: its audio file and text."""
    path = Path(record["audio"])
    try:
        stat = path.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    return hashlib.sha1(f"{path.resolve()}|{stamp}|{record['text']}".encode()).hexdigest()


def feature_version(version: str) -> str:
    """The version whose features a run trains on: v4's quantizer is
    v2ProPlus's (byte-identical) and the front ends are shared, so v4 runs use
    that version's features and cache."""
    return "v2ProPlus" if version == "v4" else version


def cache_meta(base_dir: Path, version: str) -> dict:
    return {"version": CACHE_VERSION, "models": str(base_dir), "model_version": feature_version(version)}


def extract(
    pipeline,
    base_voice,
    records: list[dict],
    stop: threading.Event,
    cache_file: Path | None = None,
    meta: dict | None = None,
) -> tuple[list[dict], dict[str, int]] | None:
    """Features of every clip that can be trained on, in ``records`` order (each
    with its manifest ``index`` and ``text``), and how many were skipped for
    which reason. Clips already in ``cache_file`` (written for ``meta``) are read
    from it; the new ones (and the skipped, with their reason) are added. A stop
    returns None, keeping what was done in the cache."""
    import torch

    cache = load_cache(cache_file, meta) if cache_file else {}
    keys = [clip_key(r) for r in records]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if len(todo) < len(records):
        log(f"Features of {len(records) - len(todo)} clips from {cache_file.name}.")
    t0 = time.time()
    for n, i in enumerate(todo):
        if stop.is_set():
            break
        r = records[i]
        try:
            cache[keys[i]] = {"audio": str(r["audio"]), **clip_features(pipeline, base_voice, r["audio"], r["text"])}
        except Skipped as e:
            cache[keys[i]] = {"audio": str(r["audio"]), "skipped": str(e)}
        except Exception as e:  # an unreadable clip or text the front end can't handle (not cached: may be fixed)
            log(f"Skipping {Path(r['audio']).name}: {e}")
        if (n + 1) % 200 == 0:
            log(f"Features: {n + 1} / {len(todo)} clips")
    if todo:
        log(f"Features of {len(todo)} clips in {time.time() - t0:.0f} s.")
        if cache_file:
            save_cache(cache_file, meta, cache, records, keys)
    if stop.is_set():
        return None
    clips, skipped = [], Counter()
    for i, (r, k) in enumerate(zip(records, keys, strict=True)):
        entry = cache.get(k)
        if entry is None:
            skipped["unreadable"] += 1
        elif "skipped" in entry:
            skipped[entry["skipped"]] += 1
        else:
            clips.append({"index": i, "text": r["text"], **{n: v for n, v in entry.items() if torch.is_tensor(v)}})
    return clips, dict(skipped)


def load_cache(cache_file: Path, meta: dict | None) -> dict:
    import torch

    if not cache_file.is_file():
        return {}
    try:
        saved = torch.load(cache_file, map_location="cpu", weights_only=True)
        if saved.get("meta") == meta:
            return saved["clips"]
        log(f"{cache_file.name} is for other models or an older version; computing features again.")
    except Exception as e:  # unreadable: computed again
        log(f"Ignoring {cache_file.name}: {e}")
    return {}


def save_cache(cache_file: Path, meta: dict | None, cache: dict, records: list[dict], keys: list[str]) -> None:
    """Write the cache, leaving out clips whose file is gone and older versions
    of this run's clips (other runs' clips, e.g. their validation ones, stay)."""
    import torch

    current = {str(r["audio"]): k for r, k in zip(records, keys, strict=True)}
    kept = {
        k: v
        for k, v in cache.items()
        if current.get(v["audio"], k) == k and (v["audio"] in current or Path(v["audio"]).is_file())
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    torch.save({"meta": meta, "clips": kept}, tmp)
    tmp.replace(cache_file)
