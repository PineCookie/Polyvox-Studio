"""Pieces several engines share, so no engine reaches into another's modules:

lora.py        LoRA layers added through forward hooks (dots.tts, S2-Pro training)
worker.py      training-worker helpers: logging, the stop watch, optimizers, batches
cache.py       per-clip caches in a dataset's folder (GPT-SoVITS features, S2-Pro codes)
reference.py   a voice's default reference clip (GPT-SoVITS, S2-Pro, Drama 3)
layers.py      small tensor operations (RMS norm, interleaved rotary)
fish_codec.py  Fish Audio's 44.1 kHz codec (S2-Pro, Drama 3)
"""
