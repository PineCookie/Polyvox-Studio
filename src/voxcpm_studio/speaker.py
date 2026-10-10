"""Speaker similarity of a run's validation samples to the voice it trains on.

CAM++ (3D-Speaker, trained on Chinese speech; funasr ships the network) turns a
clip into a speaker embedding. A run's voice is the mean embedding of clips from
its training set; a sample's score is its cosine similarity to that voice (1 =
the same direction). Further real clips of the set give the score to aim for.
Runs on the CPU (a small model), so scoring never competes with training.

Real clips of the same speaker score far above other speakers and the base
model; a WavLM x-vector model, tried first, barely separated speakers. With
two samples per checkpoint, single scores are noisy: compare trends.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import audio_io, manifests
from .config import Settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_SECONDS = 20.0  # longer clips are cut: the voice is clear well before that
VOICE_CLIPS = 60  # training clips averaged into the voice
TARGET_CLIPS = 30  # further clips scored like samples: the level to aim for
WEIGHTS = "campplus_cn_common.bin"
CACHE_FILE = "similarity.json"
CACHE_VERSION = 1

# model (local dir or ModelScope id) -> embed(16 kHz mono samples) -> vector
Embedder = Callable[[np.ndarray], np.ndarray]
SpeakerLoader = Callable[[str], Embedder]


def load_campplus(model: str) -> Embedder:
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from funasr.models.campplus.model import CAMPPlus

    folder = Path(model)
    if not (folder / WEIGHTS).is_file():  # a ModelScope id: its cache, downloaded on first use
        from modelscope import snapshot_download

        folder = Path(snapshot_download(model, allow_patterns=[WEIGHTS]))
    net = CAMPPlus()
    net.load_state_dict(torch.load(folder / WEIGHTS, map_location="cpu", weights_only=True))
    net.eval()

    def embed(wav: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            feats = kaldi.fbank(torch.from_numpy(wav).unsqueeze(0), num_mel_bins=80)
            feats = feats - feats.mean(0, keepdim=True)  # as funasr's extract_feature
            return net(feats.unsqueeze(0))[0].numpy()

    return embed


def read_16k(path: Path | str) -> np.ndarray:
    audio, rate = audio_io.read(path, 0.0, MAX_SECONDS)
    mono = audio.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(audio_io.resample(mono, rate, SAMPLE_RATE)[0], dtype=np.float32)


class SpeakerService:
    """Loads CAM++ on first use (CPU)."""

    def __init__(self, settings: Settings, loader: SpeakerLoader | None = None):
        self.settings = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._embed: Embedder | None = None

    def embed(self, paths: list[Path | str]) -> np.ndarray:
        """Unit-length speaker embeddings, one row per clip."""
        with self._lock:
            if self._embed is None:
                model = self.settings.model_path("speaker_model")
                logger.info("Loading speaker model %s", model)
                self._embed = (self._loader or load_campplus)(model)
            vectors = np.stack([self._embed(read_16k(p)) for p in paths]).astype(np.float64)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9)

    def score_run(self, run_dir: Path, train_manifest: Path) -> dict:
        """{"target": score of real clips or None, "scores": {sample file: score}}
        for the files in ``run_dir/samples`` (validation samples and the
        held-out references). Cached in the run folder: the voice until the
        manifest changes, each sample until its file does."""
        cache_path = run_dir / CACHE_FILE
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        manifest_mtime = train_manifest.stat().st_mtime_ns
        voice = cache.get("voice") if cache.get("version") == CACHE_VERSION else None
        if not voice or voice.get("manifest_mtime") != manifest_mtime:
            voice = {"manifest_mtime": manifest_mtime, **self._voice(train_manifest)}
            cache = {"version": CACHE_VERSION, "voice": voice, "samples": {}}
        centroid = np.asarray(voice["centroid"])

        known: dict = cache.setdefault("samples", {})
        samples_dir = run_dir / "samples"
        files = sorted(p for p in samples_dir.glob("*.wav")) if samples_dir.is_dir() else []
        stale = [p for p in files if known.get(p.name, {}).get("mtime") != p.stat().st_mtime_ns]
        if stale:
            for p, vector in zip(stale, self.embed(stale), strict=True):
                known[p.name] = {"mtime": p.stat().st_mtime_ns, "score": round(float(vector @ centroid), 4)}
        names = {p.name for p in files}
        cache["samples"] = {k: v for k, v in known.items() if k in names}
        if stale or len(cache["samples"]) != len(known):
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(cache_path)
        return {"target": voice.get("target"), "scores": {k: v["score"] for k, v in cache["samples"].items()}}

    def _voice(self, train_manifest: Path) -> dict:
        """The mean embedding of up to VOICE_CLIPS training clips, and the mean
        score of up to TARGET_CLIPS others (None for very small sets)."""
        entries, _ = manifests.training_entries(train_manifest)
        paths = [e["audio"] for e in entries if Path(e["audio"]).is_file()]
        if not paths:
            raise ValueError("The run's training clips were not found.")
        random.Random(0).shuffle(paths)
        voice_n = min(VOICE_CLIPS, max(1, len(paths) * 2 // 3))
        vectors = self.embed(paths[: voice_n + TARGET_CLIPS])
        centroid = vectors[:voice_n].mean(axis=0)
        centroid /= np.linalg.norm(centroid).clip(1e-9)
        others = vectors[voice_n:]
        target = round(float((others @ centroid).mean()), 4) if len(others) >= 3 else None
        return {"centroid": centroid.round(6).tolist(), "target": target}
