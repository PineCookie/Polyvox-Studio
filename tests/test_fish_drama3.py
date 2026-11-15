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
    assert torch.equal(out[dense + ".g_idx"], torch.arange(256, dtype=torch.int32) // convert.DENSE_GROUP)
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
    """Whatever the GPU slots hold (nothing, some, all), the layer's output is the experts'
    weighted sum; planning between calls moves the used ones into the slots."""
    stacks = expert_stacks()
    layer = model.Experts(stacks, slots, "cpu")
    for _ in range(3):
        hidden = torch.randn(5, 128)
        index = torch.stack([torch.randperm(6)[:2] for _ in range(5)])
        weights = torch.rand(5, 2)
        assert torch.allclose(layer(hidden, index, weights), dense_moe(stacks, hidden, index, weights), atol=1e-4)
        layer.plan()
    assert len(layer.held) == slots


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

    def generate(self, rows, sampling, max_frames, on_frame, head=0):
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


def tiny_drama3(slots=0):
    """A made-up Drama 3 small enough for the CPU: 3 Gated DeltaNet layers and one attention layer,
    6 experts in 4 bits, a 2-layer tower at the real codebook size."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    from polyvox.engines.fish_drama3.decode import Decoder

    torch.manual_seed(3)
    dim, begin = 128, 300
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=dim, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=6, num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=64,
        vocab_size=begin + 4100, layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25},
    )  # fmt: skip
    cfg._attn_implementation = "sdpa"
    m = model.Drama3()
    m.device, m.text_config = "cpu", cfg
    m.backbone = Qwen3_5MoeTextModel(cfg).eval().requires_grad_(False)
    for p in m.backbone.parameters():  # livelier than the default init, so differences show
        p.data.normal_(0, 0.08)
    m.semantic_begin, m.semantic_end, m.eos = begin, begin + 4095, begin + 4096
    m.candidates = torch.tensor([*range(begin, begin + 4096), m.eos])
    m.head = torch.randn(4097, dim) * 0.3
    m.codebook_embeddings = torch.randn(10 * 4096, dim) * 0.1
    m.offsets = torch.arange(10) * 4096
    shape = model.TowerShape(heads=2, kv_heads=1, head_dim=32, rotary_dim=8)

    def tower_layer():
        r = lambda *s: torch.randn(*s) * 0.08  # noqa: E731
        w = {"emb": r(4096, dim), "enorm": r(dim), "hnorm": r(dim), "eh_proj": r(dim, 2 * dim),
             "final_layernorm": r(dim), "attention_norm": r(dim), "ffn_norm": r(dim), "wqkv": r(6 * 32, dim),
             "wo": r(dim, 64), "q_norm": r(32), "k_norm": r(32), "w1": r(256, dim), "w2": r(dim, 256), "w3": r(256, dim)}  # fmt: skip
        return model.TowerLayer(w, shape, capacity=256)

    m.shared, m.residual = [tower_layer() for _ in range(2)], [tower_layer() for _ in range(9)]
    m.shared = m.shared[:1] + m.shared[1:]
    m.heads = [torch.randn(4096, dim) * 0.3 for _ in range(9)]
    m.decoder = Decoder(m, graphs=False)
    for layer in m.backbone.layers:
        layer.mlp.experts = model.Experts(expert_stacks(experts=6, hidden=dim, inner=128), slots, "cpu")
    return m


def tiny_prompt(m, n=12):
    rows = torch.zeros(n, 11, dtype=torch.long)
    rows[:5, 0] = torch.randint(0, 250, (5,))
    codes = torch.randint(0, 4096, (n - 5, 10))
    rows[5:, 0], rows[5:, 1:] = codes[:, 0] + m.semantic_begin, codes
    return rows


@torch.inference_mode()
def test_the_decoding_step_matches_the_backbone():
    """After the prompt, our step for a new frame gives transformers' hidden state, for several frames."""
    m = tiny_drama3()
    rows = tiny_prompt(m)
    _, cache = m.prefill(rows)
    step = m.decoder.backbone
    step.load(cache, rows.shape[0])
    frame = rows[-1:].clone()
    for pos in range(rows.shape[0], rows.shape[0] + 3):
        frame[0, 1:] = torch.randint(0, 4096, (10,))
        frame[0, 0] = frame[0, 1] + m.semantic_begin
        x = m.backbone_input(frame)
        want = m.backbone(inputs_embeds=x[None], past_key_values=cache, use_cache=True).last_hidden_state[0]
        got = step.step(x, torch.tensor([pos]), 64)
        assert torch.allclose(got, want, atol=2e-5, rtol=1e-4), (got - want).abs().max()


@torch.inference_mode()
def test_the_decoder_speaks_as_plain_decoding(monkeypatch):
    """Greedy (one candidate) the decoder's frames equal the reference loop's."""
    m = tiny_drama3()
    monkeypatch.setattr(model, "REDRAW", model.Sampling(1.0, 0.95, 1))
    m.decoder.redraw[2].fill_(1)
    rows = tiny_prompt(m)
    greedy = model.Sampling(1.0, 0.9, 1)
    want = m.generate_reference(rows, greedy, 6)
    got = m.generate(rows, greedy, 6)
    assert got.shape == want.shape == (10, 6) and torch.equal(got, want)


def test_planning_holds_the_most_used_experts():
    layer = model.Experts(expert_stacks(), 2, "cpu")
    layer.counts.copy_(torch.tensor([5.0, 0, 9, 1, 0, 7]))
    assert layer.plan() == pytest.approx(16 / 22)
    assert set(layer.held) == {2, 5} and layer.table.tolist().count(-1) == 4
    hidden, weights = torch.randn(1, 128), torch.tensor([0.6, 0.4])
    picked = torch.tensor([2, 3])
    assert torch.allclose(
        layer.routed(hidden, picked, weights), dense_moe(layer.cpu, hidden, picked[None], weights[None]), atol=1e-4
    )


@torch.inference_mode()
def test_a_voices_prompt_part_is_kept_and_reused():
    """The prompt in two passes (references kept, then the text) gives the full pass's state and
    tower caches; a second text with the same references starts from the kept states."""
    m = tiny_drama3()
    rows = tiny_prompt(m, 14)
    head = 9
    towers = (*m.shared, *m.residual)

    def full(r):
        m.prefixes.clear()
        h, cache = m.prefill(r)
        return h, [t.k[: len(r) - 1].clone() for t in towers], cache

    for text_rows in (rows[head:], torch.cat((rows[head : head + 2], rows[head + 3 :]))):
        r = torch.cat((rows[:head], text_rows))
        want_h, want_k, want_cache = full(r)
        for _ in range(2):  # kept, then reused
            h, cache = m.prefill(r, head)
            assert torch.allclose(h, want_h, atol=1e-5)
            assert all(torch.allclose(t.k[: len(r) - 1], k, atol=1e-5) for t, k in zip(towers, want_k, strict=True))
            assert torch.allclose(
                cache.layers[0].recurrent_states[0], want_cache.layers[0].recurrent_states[0], atol=1e-5
            )
            assert cache.layers[3].keys.shape[2] == len(r)
        assert len(m.prefixes) == 1


def test_dense_linears_can_have_smaller_groups(tmp_path):
    fake_fp8_checkpoint(tmp_path / "fp8")
    convert.convert(tmp_path / "fp8", tmp_path / "int4", device="cpu", dense_group=32)
    out = read_all(tmp_path / "int4")
    dense = DENSE.removesuffix(".weight")
    assert out[dense + ".scales"].shape == (256 // 32, 256) and out[dense + ".qzeros"].shape[0] == 256 // 32
    assert out[EXPERT.removesuffix(".weight") + ".scales"].shape == (384 // 128, 256)  # experts keep 128
    config = json.loads((tmp_path / "int4" / "config.json").read_text())["quantization_config"]
    assert config["dense"]["group_size"] == 32 and config["experts"]["group_size"] == 128
