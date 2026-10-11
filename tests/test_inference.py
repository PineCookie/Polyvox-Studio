import os

import pytest
import torch
from safetensors.torch import load_file
from voxcpm.modules.layers.lora import LoRALinear

from polyvox.engines.voxcpm import inference
from polyvox.engines.voxcpm.inference import SynthesisRequest, compose_text

from conftest import LM_LAYERS, make_lora


def expected_delta(ckpt_dir, layer, alpha, r):
    w = load_file(str(ckpt_dir / "lora_weights.safetensors"))
    return (alpha / r) * w[f"{layer}.lora_B"] @ w[f"{layer}.lora_A"]


def assert_lora_matches(model, ckpt_dir, alpha, r):
    for layer in LM_LAYERS:
        torch.testing.assert_close(model.tts_model.lora_delta(layer), expected_delta(ckpt_dir, layer, alpha, r))


def assert_lora_off(model):
    for layer in LM_LAYERS:
        assert model.tts_model.get_submodule(layer).scaling.item() == 0.0


def test_compose_text_wraps_control_and_strips_parentheses():
    assert compose_text("  hi ", "") == "hi"
    assert compose_text("hi", " (warm)（soft） ") == "(warmsoft)hi"
    with pytest.raises(ValueError):
        compose_text("   ", "x")


def test_padded_lora_equals_its_own_rank():
    """Zero-padding a rank-4 LoRA into a rank-8 layer gives the same output."""
    torch.manual_seed(0)
    small = LoRALinear(torch.nn.Linear(16, 16), r=4, alpha=8)
    big = LoRALinear(torch.nn.Linear(16, 16), r=8, alpha=8)
    big.weight, big.bias = small.weight, small.bias
    with torch.no_grad():
        small.lora_B.normal_()
        big.lora_A.zero_()
        big.lora_B.zero_()
        big.lora_A[:4].copy_(small.lora_A)
        big.lora_B[:, :4].copy_(small.lora_B)
        big.scaling.fill_(small.scaling.item())  # alpha / r of the small LoRA

    x = torch.randn(3, 16)
    torch.testing.assert_close(big(x), small(x))


def test_base_request_builds_a_layout_covering_every_checkpoint(service, loader, lora_dir):
    make_lora(lora_dir, "a", r=32)
    make_lora(lora_dir, "b", r=64)

    result = service.synthesize(SynthesisRequest(text="hi"))

    (model,) = loader.models
    assert model.lora_config.r == 64
    assert_lora_off(model)
    assert result.sample_rate == 48000 and result.seed == 7


def test_switching_between_ranks_never_reloads(service, loader, lora_dir):
    a = make_lora(lora_dir, "a", r=32, alpha=16)
    b = make_lora(lora_dir, "b", r=64, alpha=64)

    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))
    (model,) = loader.models
    assert_lora_matches(model, a, alpha=16, r=32)
    q = model.tts_model.get_submodule(LM_LAYERS[0])
    assert q.lora_A[32:].abs().sum() == 0 and q.lora_B[:, 32:].abs().sum() == 0  # padding

    service.synthesize(SynthesisRequest(text="hi", lora_id="b"))
    assert_lora_matches(model, b, alpha=64, r=64)
    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))
    assert_lora_matches(model, a, alpha=16, r=32)

    assert len(loader.models) == 1


def test_larger_lora_added_later_reloads_once(service, loader, lora_dir):
    make_lora(lora_dir, "small", r=32)
    service.synthesize(SynthesisRequest(text="hi", lora_id="small"))

    big = make_lora(lora_dir, "big", r=128)
    service.synthesize(SynthesisRequest(text="hi", lora_id="big"))
    service.synthesize(SynthesisRequest(text="hi", lora_id="small"))

    assert [m.lora_config.r for m in loader.models] == [32, 128]
    service.synthesize(SynthesisRequest(text="hi", lora_id="big"))
    assert_lora_matches(loader.models[-1], big, alpha=128, r=128)


def test_base_keeps_weights_resident(service, loader, lora_dir, monkeypatch):
    a = make_lora(lora_dir, "a", r=32)
    loads = []
    real = inference.load_lora_padded
    monkeypatch.setattr(inference, "load_lora_padded", lambda m, c: loads.append(c.id) or real(m, c))

    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))
    assert service.status()["active_lora"] == "a"
    service.synthesize(SynthesisRequest(text="hi"))
    assert_lora_off(loader.models[0])
    assert service.status()["active_lora"] is None
    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))

    assert loads == ["a"]  # not re-read
    assert_lora_matches(loader.models[0], a, alpha=32, r=32)


def test_rewritten_weights_are_reloaded(service, lora_dir, monkeypatch):
    ckpt = make_lora(lora_dir, "a", r=32)
    loads = []
    real = inference.load_lora_padded
    monkeypatch.setattr(inference, "load_lora_padded", lambda m, c: loads.append(c.id) or real(m, c))
    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))

    weights = ckpt / "lora_weights.safetensors"
    st = weights.stat()
    os.utime(weights, (st.st_atime, st.st_mtime + 10))
    service.synthesize(SynthesisRequest(text="hi", lora_id="a"))

    assert loads == ["a", "a"]


def test_checkpoint_without_config_uses_default_scaling(service, loader, lora_dir):
    make_lora(lora_dir, "r64", r=64)
    legacy = make_lora(lora_dir, "legacy", r=None)  # default layout: rank 32, alpha 16

    service.synthesize(SynthesisRequest(text="hi", lora_id="r64"))
    service.synthesize(SynthesisRequest(text="hi", lora_id="legacy"))

    assert len(loader.models) == 1
    assert_lora_matches(loader.models[0], legacy, alpha=16, r=32)


def test_lora_for_another_model_is_rejected_and_leaves_lora_off(service, loader, lora_dir):
    make_lora(lora_dir, "ok", r=32)
    make_lora(lora_dir, "foreign", r=32, layers=["some_other_net.layers.0.q_proj"])

    service.synthesize(SynthesisRequest(text="hi", lora_id="ok"))
    with pytest.raises(ValueError, match="does not match"):
        service.synthesize(SynthesisRequest(text="hi", lora_id="foreign"))

    model = loader.models[0]
    for layer in LM_LAYERS:
        assert model.tts_model.lora_delta(layer).abs().sum() == 0
    service.synthesize(SynthesisRequest(text="hi", lora_id="ok"))  # recovers by reloading
    assert len(loader.models) == 1


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


def test_progress_follows_the_generation_loop(service, monkeypatch):
    from voxcpm.model import voxcpm2

    from conftest import FakeModel

    seen = []

    def generate(self, **kwargs):
        # VoxCPM's loop: max_len = text tokens * 6 + 10; stops early.
        for i in voxcpm2.tqdm(range(int(20 * inference.RATIO_LIMIT + 10))):
            seen.append(dict(service.progress_public()))
            if i == 23:
                break
        return torch.zeros(4800).numpy()

    monkeypatch.setattr(FakeModel, "generate", generate)
    service.optimize = True
    service.synthesize(SynthesisRequest(text="示例"))

    assert seen[0]["phase"] == "compiling"  # the first steps after loading compile
    assert seen[-1]["phase"] == "generating" and seen[-1]["step"] == 24 and seen[-1]["attempt"] == 1
    assert seen[-1]["expected"] == round(20 * inference.STEPS_PER_TOKEN)
    assert service.progress_public()["phase"] == "idle"
    # 24 steps for 20 tokens: the next estimate moves towards 1.2 per token.
    service.synthesize(SynthesisRequest(text="示例"))
    assert seen[-1]["phase"] == "generating"  # compiled already
    assert seen[-1]["expected"] == round(20 * (0.7 * inference.STEPS_PER_TOKEN + 0.3 * 1.2))
