"""Per-clip caches kept in a dataset's folder (``<dataset>/.cache/<name>.pt``): what a training
worker computes from each clip once (GPT-SoVITS's features, S2-Pro's codes and states), keyed
by the clip's file and text, so later runs on the same dataset skip that work."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .worker import log


def clip_key(record: dict) -> str:
    """What a clip's entry depends on besides the models: its audio file and text."""
    path = Path(record["audio"])
    try:
        stat = path.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    return hashlib.sha1(f"{path.resolve()}|{stamp}|{record['text']}".encode()).hexdigest()


def load_cache(cache_file: Path, meta: dict | None) -> dict:
    """The cache's entries by clip key; empty when it's missing, unreadable or written for
    another ``meta`` (other models, an older version of what is cached)."""
    import torch

    if not cache_file.is_file():
        return {}
    try:
        saved = torch.load(cache_file, map_location="cpu", weights_only=True)
        if saved.get("meta") == meta:
            return saved["clips"]
        log(f"{cache_file.name} is for other models or an older version; computing it again.")
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
