import os

import pytest

from voxcpm_studio.inference import SynthesisRequest, compose_text

from conftest import make_lora


def test_compose_text_wraps_control_and_strips_parentheses():
    assert compose_text("  hi ", "") == "hi"
    assert compose_text("hi", " (warm)（soft） ") == "(warmsoft)hi"
    with pytest.raises(ValueError):
        compose_text("   ", "x")


def test_base_request_loads_default_layout_with_lora_disabled(service, loader):
    result = service.synthesize(SynthesisRequest(text="hi"))

    (model,) = loader.models
    assert model.lora_config.r == 32
    assert model.lora_enabled is False
    assert result.sample_rate == 48000 and result.seed == 7


def test_same_layout_hot_swaps_without_reload(service, loader, lora_dir):
    make_lora(lora_dir, "a", r=32)
    make_lora(lora_dir, "b", r=32)

    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))
    service.synthesize(SynthesisRequest(text="hi", lora_id="b"))

    (model,) = loader.models
    assert [p.split(os.sep)[-1] for p in model.loaded_paths] == ["a", "b"]
    assert model.lora_enabled is True


def test_different_rank_reloads_model(service, loader, lora_dir):
    make_lora(lora_dir, "r64", r=64)

    service.synthesize(SynthesisRequest(text="hi"))
    service.synthesize(SynthesisRequest(text="hi", lora_id="r64"))

    assert [m.lora_config.r for m in loader.models] == [32, 64]


def test_switching_to_base_keeps_model_and_lora_weights(service, loader, lora_dir):
    make_lora(lora_dir, "r64", r=64)

    service.synthesize(SynthesisRequest(text="hi", lora_id="r64"))
    service.synthesize(SynthesisRequest(text="hi"))
    assert loader.models[-1].lora_enabled is False

    service.synthesize(SynthesisRequest(text="hi", lora_id="r64"))

    (model,) = loader.models
    assert len(model.loaded_paths) == 1  # weights still resident, not re-read
    assert model.lora_enabled is True


def test_rewritten_weights_are_reloaded(service, loader, lora_dir):
    ckpt = make_lora(lora_dir, "a", r=32)
    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))

    weights = ckpt / "lora_weights.safetensors"
    st = weights.stat()
    os.utime(weights, (st.st_atime, st.st_mtime + 10))
    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))

    assert len(loader.models[0].loaded_paths) == 2


def test_checkpoint_without_config_uses_default_layout(service, loader, lora_dir):
    make_lora(lora_dir, "r64", r=64)
    make_lora(lora_dir, "legacy", r=None)

    service.synthesize(SynthesisRequest(text="hi", lora_id="r64"))
    service.synthesize(SynthesisRequest(text="hi", lora_id="legacy"))

    assert [m.lora_config.r for m in loader.models] == [64, 32]


def test_prompt_text_only_used_with_reference_audio(service, loader, tmp_path):
    ref = tmp_path / "ref.wav"
    service.synthesize(SynthesisRequest(text="hi", prompt_text="ignored"))
    service.synthesize(SynthesisRequest(text="hi", reference_wav=ref, prompt_text=" spoken "))

    first, second = loader.models[0].generate_calls
    assert "prompt_text" not in first
    assert second["prompt_text"] == "spoken"
    assert second["prompt_wav_path"] == second["reference_wav_path"] == str(ref)


def test_unknown_lora_raises_key_error(service):
    with pytest.raises(KeyError):
        service.synthesize(SynthesisRequest(text="hi", lora_id="nope"))
