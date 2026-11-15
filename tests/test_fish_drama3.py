"""Fish Audio Drama 3: the 4-bit build from the FP8 checkpoint (a tiny made-up one, on the CPU)."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from polyvox import models
from polyvox.engines import SynthesisRequest
from polyvox.engines.fish_drama3 import convert, engine, model, prompt

P = "model.language_model.layers."
EXPERT = P + "0.mlp.experts.1.down_proj.weight"
DENSE = P + "3.self_attn.q_proj.weight"
SHARED = P + "0.mlp.shared_expert.up_proj.weight"


def fake_fp8_checkpoint(folder):
    """Two shards: an FP8 expert (its block scales in the other shard), a dense bf16
    linear, a shared expert and a norm; plus the files copied as they are."""
    torch.manual_seed(0)
    folder.mkdir(parents=True)
    w = torch.randn(256, 384) * 0.02
    block = torch.tensor([[1.0, 2.0, 0.5], [1.5, 1.0, 1.0]])  # [ceil(256/128), ceil(384/128)]
    fp8 = (w / block.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    a = {
        EXPERT: fp8,
        DENSE: (torch.randn(256, 256) * 0.02).bfloat16(),
        P + "0.input_layernorm.weight": torch.ones(256).bfloat16(),
    }
    b = {EXPERT + "_scale_inv": block, SHARED: (torch.randn(64, 256) * 0.02).bfloat16()}
    save_file(a, str(folder / "model-00001-of-00002.safetensors"))
    save_file(b, str(folder / "model-00002-of-00002.safetensors"))
    weight_map = {
        **dict.fromkeys(a, "model-00001-of-00002.safetensors"),
        **dict.fromkeys(b, "model-00002-of-00002.safetensors"),
    }
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({"model_type": "fish_omni"}))
    for name in (
        "codec.pth",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "LICENSE",
    ):
        (folder / name).write_bytes(b"made up " + name.encode())
    return fp8.float() * block.repeat_interleave(128, 0).repeat_interleave(128, 1), a, b


def read_all(folder):
    weight_map = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for key, shard in weight_map.items():
        with safe_open(str(folder / shard), "pt") as f:
            out[key] = f.get_tensor(key)
    return out


def test_convert_quantizes_experts_and_dense_linears_and_copies_the_rest(tmp_path):
    expert, a, b = fake_fp8_checkpoint(tmp_path / "fp8")
    seen = []
    stats = convert.convert(tmp_path / "fp8", tmp_path / "int4", seen.append, device="cpu")
    out = read_all(tmp_path / "int4")
    base = EXPERT.removesuffix(".weight")
    assert out[base + ".qweight"].shape == (384 // 8, 256) and out[base + ".scales"].dtype == torch.float16
    restored = convert.dequant(out[base + ".qweight"], out[base + ".scales"], torch.float32).T
    assert ((restored - expert).norm() / expert.norm()) < 0.12  # 4-bit error on Gaussian weights ~0.10
    dense = DENSE.removesuffix(".weight")
    assert bool((out[dense + ".qzeros"] == convert.ZERO_POINTS).all())
    assert torch.equal(out[dense + ".g_idx"], torch.arange(256, dtype=torch.int32) // 128)
    # The shared expert and the norm are copied unchanged; the expert's scales aren't kept.
    assert torch.equal(out[SHARED], b[SHARED]) and EXPERT + "_scale_inv" not in out and EXPERT not in out
    assert stats["experts"]["tensors"] == 1 and stats["dense"]["tensors"] == 1
    assert json.loads((tmp_path / "int4" / "config.json").read_text())["quantization_config"]["group_size"] == 128
    assert (tmp_path / "int4" / "codec.pth").read_bytes() == b"made up codec.pth"
    assert (tmp_path / "int4" / convert.STATS).is_file() and not (tmp_path / "int4.build").exists()
    assert seen[-1] == 1.0


def test_pack_layout_matches_the_readers():
    """Value k of a column sits at bits 4 * (k % 8) of word k // 8, stored + 8."""
    q = torch.randint(0, 16, (8, 16), dtype=torch.uint8)
    qw = convert.pack(q)
    assert qw.shape == (2, 8)
    assert int(qw[1, 3]) & 0xF == int(q[3, 8]) and (int(qw[0, 5]) >> 28) & 0xF == int(q[5, 7])
    assert torch.equal(convert.unpack(qw).T.to(torch.uint8), q)


def test_a_cancelled_build_leaves_no_model(tmp_path):
    fake_fp8_checkpoint(tmp_path / "fp8")

    def cancel(fraction):
        if fraction > 0.5:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        convert.convert(tmp_path / "fp8", tmp_path / "int4", cancel, device="cpu")
    assert not (tmp_path / "int4").exists()


def test_download_builds_the_model_and_removes_the_source(tmp_path, monkeypatch):
    spec = models.BY_SETTING["drama3_model"]
    small = models.ModelSpec(
        spec.setting, spec.name, spec.purpose, spec.repo, spec.folder, spec.files, 1, revision=spec.revision,
        source=models.Source("fp8", ("config.json",), spec.source.build),
    )  # fmt: skip
    fetched = []

    def fake_fetch(repo, revision, folder, files, endpoint, report):
        fetched.append((repo, revision, files))
        folder.rmdir()
        fake_fp8_checkpoint(folder)
        report(1 << 20, 1 << 20)

    monkeypatch.setattr(models, "_fetch_hf", fake_fetch)
    progress = []
    models.download(small, tmp_path, progress=lambda done, total: progress.append((done, total)))
    assert fetched == [(spec.repo, spec.revision, ("config.json",))]
    assert small.downloaded(tmp_path) and not (tmp_path / "fp8").exists()
    assert progress[-1][0] == progress[-1][1] > 0  # download + build reach the end of one bar


def expert_stacks(experts=6, hidden=128, inner=128):
    """4-bit stacks for ``experts`` made-up experts, as model.load builds them."""
    torch.manual_seed(1)
    stacks = {}
    for short, (k, n) in (("gate", (hidden, inner)), ("up", (hidden, inner)), ("down", (inner, hidden))):
        qs = [convert.quantize(torch.randn(n, k) * 0.05) for _ in range(experts)]
        stacks[short + "_q"] = torch.stack([convert.pack(q) for q, _ in qs])
        stacks[short + "_s"] = torch.stack([s.t().contiguous().half() for _, s in qs])
    return stacks


def dense_moe(stacks, hidden, index, weights):
    w = {k: convert.dequant(stacks[k + "_q"], stacks[k + "_s"], torch.float32) for k in ("gate", "up", "down")}
    out = torch.zeros_like(hidden)
    for t in range(hidden.shape[0]):
        for e, g in zip(index[t].tolist(), weights[t], strict=True):
            x = hidden[t]
            out[t] += g * ((torch.nn.functional.silu(x @ w["gate"][e]) * (x @ w["up"][e])) @ w["down"][e])
    return out


@pytest.mark.parametrize("slots", [0, 2, 6])
def test_cached_and_streamed_experts_compute_the_same(slots):
    """Whatever the cache holds (nothing, fewer than a call needs, all), the layer's output is the
    experts' weighted sum; later calls find what the cache kept."""
    stacks = expert_stacks()
    layer = model.Experts(stacks, slots, "cpu")
    for _ in range(3):
        hidden = torch.randn(5, 128)
        index = torch.stack([torch.randperm(6)[:2] for _ in range(5)])
        weights = torch.rand(5, 2)
        assert torch.allclose(layer(hidden, index, weights), dense_moe(stacks, hidden, index, weights), atol=1e-4)
    assert len(layer.held) == slots and (layer.hits == 0) == (slots == 0)


def test_the_cache_keeps_the_most_recently_used_experts():
    layer = model.Experts(expert_stacks(), 2, "cpu")
    hidden, weights = torch.randn(1, 128), torch.ones(1, 1)
    for e in (0, 1, 0, 2):  # 1 is the least recently used when 2 comes
        layer(hidden, torch.tensor([[e]]), weights)
    assert list(layer.held) == [0, 2] and (layer.hits, layer.misses) == (1, 3)


class Letters:
    """A tokenizer with one id per character (prompt tests)."""

    def encode(self, text):
        from types import SimpleNamespace

        return SimpleNamespace(ids=[ord(c) for c in text])


def test_the_prompt_puts_reference_codes_under_their_semantic_tokens():
    codes = torch.arange(30).view(10, 3)
    begin = 0x110000  # past every character
    rows = prompt.build(Letters(), "你好", [("示例。", codes)], semantic_begin=begin)
    text = "".join(chr(r) for r in rows[:, 0].tolist() if r < begin)
    assert "[Reference 1]\nText: <|speaker:0|> 示例。\nSpeech: " in text and text.endswith("<|voice|>")
    assert "<|speaker:0|> 你好" in text and "reference to the following" in text
    audio = rows[rows[:, 0] >= begin]
    assert (audio[:, 0] - begin).tolist() == [0, 1, 2] and torch.equal(audio[:, 1:].T, codes)
    plain = prompt.build(Letters(), "你好", [], semantic_begin=begin)
    assert "reference to the following" not in "".join(chr(r) for r in plain[:, 0].tolist())


class FakeModel:
    """Takes run to the cap on the first attempt of each part when ``runaway``."""

    tokenizer = Letters()
    semantic_begin = 1000

    def __init__(self, runaway=False):
        self.runaway = runaway
        self.calls = []

    def generate(self, rows, sampling, max_frames, on_frame):
        self.calls.append(max_frames)
        n = max_frames if self.runaway and len(self.calls) % 2 == 1 else 20
        for _ in range(n):
            on_frame()
        return torch.zeros(10, n, dtype=torch.long)


class FakeRuntime:
    device = "cpu"

    def __init__(self, m):
        self.model = m
        self.codec = type(
            "Codec", (), {"sample_rate": 44_100, "decode": lambda self, c: torch.zeros(c.shape[1] * 2048)}
        )()

    def encode(self, wav):
        return torch.zeros(10, 5, dtype=torch.long)


def drama3(settings, monkeypatch, m):
    import dataclasses

    monkeypatch.setattr(engine.FishDrama3Engine, "unavailable", lambda self: None)
    return engine.FishDrama3Engine(
        dataclasses.replace(settings, drama3_model="X:/models/drama3"), lambda *a: FakeRuntime(m)
    )


def test_a_take_that_runs_away_is_drawn_again(settings, monkeypatch, tmp_path):
    m = FakeModel(runaway=True)
    tts = drama3(settings, monkeypatch, m)
    result = tts.synthesize(SynthesisRequest(text="你好，今天天气不错。", seed=7))
    cap = engine.frame_cap(round(8 * engine.FRAMES_PER_CHAR))  # 8 characters (punctuation not counted)
    assert m.calls == [cap, cap]  # ran to the cap, drawn again once
    assert len(result.audio) == 20 * 2048 and tts.progress()["step"] == 20


def test_the_engine_needs_its_model_and_a_transcript(settings, monkeypatch, tmp_path):
    import dataclasses

    plain = engine.FishDrama3Engine(dataclasses.replace(settings, drama3_model=""))
    assert "download" in plain.unavailable()
    tts = drama3(settings, monkeypatch, FakeModel())
    clip = tmp_path / "ref.wav"
    with pytest.raises(ValueError, match="transcript"):
        tts.synthesize(SynthesisRequest(text="你好。", reference_wav=clip))
    with pytest.raises(ValueError, match="trained voices"):
        tts.synthesize(SynthesisRequest(text="你好。", lora_id="voxcpm/someone"))
    assert engine.frame_cap(1000) == engine.MAX_FRAMES and engine.frame_cap(50) == 150
