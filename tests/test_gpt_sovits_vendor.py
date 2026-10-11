import io
import pickle
import sys
import types

import pytest
import torch

from polyvox.vendor.gpt_sovits import process_ckpt
from polyvox.vendor.gpt_sovits.text.g2pw import g2pw, utils
from polyvox.vendor.gpt_sovits.text.LangSegmenter import LangSegmenter


def test_chinese_and_english_are_split_by_script():
    assert LangSegmenter.getTexts("2024年我们用了Model X, 很好!") == [
        {"lang": "zh", "text": "2024年我们用了"},
        {"lang": "en", "text": "Model X, "},
        {"lang": "zh", "text": "很好!"},
    ]
    assert LangSegmenter.getTexts("Hello.") == [{"lang": "en", "text": "Hello."}]
    assert LangSegmenter.getTexts("123") == [{"lang": "zh", "text": "123"}]


def tagged(obj, tag: bytes) -> bytes:
    """As the webui saves SoVITS weights: the zip's "PK" replaced by a version tag."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return tag + buf.getvalue()[2:]


def test_saved_weights_load_with_the_webui_config_class(tmp_path, monkeypatch):
    path = tmp_path / "voice.pth"
    config = process_ckpt.HParams(model={"version": "v2Pro"}, data={"sampling_rate": 32000})
    webui_utils = types.ModuleType("utils")  # where the webui's HParams lives, for saving like it
    webui_utils.HParams = process_ckpt.HParams
    monkeypatch.setitem(sys.modules, "utils", webui_utils)
    path.write_bytes(tagged({"weight": {"w": torch.ones(2)}, "config": config}, b"05"))
    monkeypatch.delitem(sys.modules, "utils")
    saved = process_ckpt.load_sovits_new(path)
    assert saved["config"].model.version == "v2Pro" and saved["config"]["data"]["sampling_rate"] == 32000
    assert torch.equal(saved["weight"]["w"], torch.ones(2))


class Payload:
    def __reduce__(self):
        return (print, ("pickled code ran",))


def test_pickled_code_is_refused(tmp_path):
    path = tmp_path / "bad.pth"
    path.write_bytes(tagged({"config": Payload()}, b"05"))
    with pytest.raises(pickle.UnpicklingError):
        process_ckpt.load_sovits_new(path)


def test_g2pw_config_is_read_not_run(tmp_path):
    marker = tmp_path / "ran"
    config = tmp_path / "config.py"
    config.write_text(f"window_size = 32\nparam_pos = {{'weight': 0.1}}\nopen(r'{marker}', 'w')\n", encoding="utf-8")
    loaded = utils.load_config(config, use_default=True)
    assert loaded.window_size == 32 and loaded.param_pos["weight"] == 0.1
    assert loaded.model_source == "bert-base-chinese"  # a default filled in
    assert not marker.exists()


def test_polyphonic_dictionary_comes_from_the_text_files():
    words = g2pw.get_dict()
    assert len(words) > 1000 and all(isinstance(v, list) for v in list(words.values())[:100])
