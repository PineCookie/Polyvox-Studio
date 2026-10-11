"""dots.tts (studio-dots-ai/dots.tts, Apache-2.0, see LICENSE), the package of the
PyPI wheel dots.tts 0.3.1, vendored because that wheel requires Python < 3.13 and
gradio (its demo only). Used by engines/dots.py.

Changes from the wheel:
- imports of ``dots_tts.`` made absolute under ``polyvox.vendor``;
- left out: cli.py, edit_cli.py, edit_runtime.py, runtime_double_streaming.py,
  models/dots_tts/edit_model.py, data/edit_instruction.py (editing, CLI, streaming);
- this file: the wheel's import-time check that torch and torchaudio share a
  minor version is gone (torchaudio's last release, 2.11, never matches torch 2.14);
- utils/text.py: language detection from the script instead of lingua (~290 MB),
  language tags from a small table instead of langcodes, and text normalization
  through wetext instead of WeTextProcessing (needs pynini: no Windows wheel).
"""
