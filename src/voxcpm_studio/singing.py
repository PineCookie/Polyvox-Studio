"""Singing detection with an AudioSet audio tagger (AST, native in transformers).

Sung clips (karaoke in a stream, a cover) are not wanted in TTS training data.
The tagger's Singing / Song / Humming scores are small in absolute terms but
near zero for speech; its Speech score is high for speech and low for singing,
which keeps talk over background music (that lifts the singing score) from
counting. Works on separated vocals too, i.e. singing without the music.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BATCH_SIZE = 16
# AudioSet classes (AST's id2label): Speech, Singing, Humming, Song.
SPEECH, SINGING, HUMMING, SONG = 0, 27, 37, 266
# Singing: the sung classes above this, and speech below that.
SUNG_ABOVE = 0.01
SPEECH_BELOW = 0.3


def is_sung(sung: float, speech: float) -> bool:
    return sung > SUNG_ABOVE and speech < SPEECH_BELOW


# (model, device) -> (feature extractor, model)
SingingLoader = Callable[[str, str], tuple]


def load_ast(model: str, device: str):
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    return ASTFeatureExtractor.from_pretrained(model), ASTForAudioClassification.from_pretrained(model).to(
        device
    ).eval()


class SingingService:
    """Loads the tagger on first use (on the ASR device)."""

    def __init__(self, settings: Settings, device: Callable[[], str], loader: SingingLoader | None = None):
        self.settings = settings
        self._device = device
        self._loader = loader
        self._lock = threading.Lock()
        self._extractor = self._model = None

    @property
    def device(self) -> str:
        return self._device()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def scores(self, wavs: list[np.ndarray]) -> list[tuple[float, float]]:
        """(sung, speech) per 16 kHz mono clip: the summed Singing / Song /
        Humming probability, and the Speech probability."""
        import torch

        out: list[tuple[float, float]] = []
        with self._lock:
            if self._model is None:
                logger.info("Loading singing detector %s on %s", self.settings.model_path("singing_model"), self.device)
                load = self._loader or load_ast  # looked up when first used
                self._extractor, self._model = load(self.settings.model_path("singing_model"), self.device)
            for start in range(0, len(wavs), BATCH_SIZE):
                batch = wavs[start : start + BATCH_SIZE]
                x = self._extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
                with torch.inference_mode():
                    probs = torch.sigmoid(self._model(x.to(self._model.device)).logits.float()).cpu().numpy()
                out += [(float(p[[SINGING, SONG, HUMMING]].sum()), float(p[SPEECH])) for p in probs]
        return out

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            self._extractor = self._model = None
            gc.collect()
            if self.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
