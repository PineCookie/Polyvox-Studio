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
    # The expert is stored rotated, on the grid: rotated back it's the weight with less error than
    # plain 4 bits (~0.10 on Gaussian weights).
    signs = out[convert.SIGNS.format(384)]
    grid = torch.tensor(convert.GRID)
    restored = convert.unrotate(
        convert.dequant(out[base + ".qweight"], out[base + ".scales"], torch.float32, grid).T, signs
    )
    assert ((restored - expert).norm() / expert.norm()) < 0.095
    experts = json.loads((tmp_path / "int4" / "config.json").read_text())["quantization_config"]["experts"]
    assert experts["quant_method"] == "higgs" and experts["grid"] == list(convert.GRID)
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


def test_the_standalone_script_writes_the_same_build(tmp_path):
    """tools/convert_drama3_higgs.py (published with the 4-bit build, needs no polyvox) = convert.py."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "tools" / "convert_drama3_higgs.py"
    spec = importlib.util.spec_from_file_location("convert_drama3_higgs", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    fake_fp8_checkpoint(tmp_path / "fp8")
    convert.convert(tmp_path / "fp8", tmp_path / "ours", device="cpu")
    script.convert(tmp_path / "fp8", tmp_path / "theirs", device="cpu", report=lambda line: None)
    ours = sorted(p.name for p in (tmp_path / "ours").iterdir())
    assert ours == sorted(p.name for p in (tmp_path / "theirs").iterdir())
    for name in ours:
        assert (tmp_path / "ours" / name).read_bytes() == (tmp_path / "theirs" / name).read_bytes(), name
    out = read_all(tmp_path / "theirs")
    base = EXPERT.removesuffix(".weight")
    signs = out[convert.SIGNS.format(384)]
    x = torch.randn(3, 384)
    w = script.dequant_expert(out[base + ".qweight"], out[base + ".scales"])  # rotated W^T [K, N]
    want = convert.unrotate(convert.dequant(out[base + ".qweight"], out[base + ".scales"], torch.float32,
                                            torch.tensor(convert.GRID)).T, signs)  # fmt: skip
    assert torch.allclose(script.rotate(x, signs) @ w, x @ want.T, atol=1e-4)  # the reader as documented


def test_plain_4_bit_experts_keep_the_community_layout(tmp_path):
    expert, _, _ = fake_fp8_checkpoint(tmp_path / "fp8")
    convert.convert(tmp_path / "fp8", tmp_path / "int4", device="cpu", experts="int4")
    out = read_all(tmp_path / "int4")
    base = EXPERT.removesuffix(".weight")
    restored = convert.dequant(out[base + ".qweight"], out[base + ".scales"], torch.float32).T
    assert ((restored - expert).norm() / expert.norm()) < 0.12
    assert not any(k.startswith("quant.") for k in out)
    experts = json.loads((tmp_path / "int4" / "config.json").read_text())["quantization_config"]["experts"]
    assert experts["quant_method"] == "gptq"


def test_the_rotation_keeps_products_and_spreads_outliers():
    torch.manual_seed(4)
    signs = convert.rotation_signs(256)
    w, x = torch.randn(32, 256), torch.randn(5, 256)
    w[3, 17] = 40.0
    assert torch.allclose(convert.rotate(x, signs) @ convert.rotate(w, signs).T, x @ w.T, atol=1e-3)
    assert torch.allclose(convert.unrotate(convert.rotate(w, signs), signs), w, atol=1e-5)
    assert convert.rotate(w, signs)[3].abs().max() < 10  # 40 spread over its block of 128: ~40 / sqrt(128) each
    q, s = convert.quantize_grid(w)
    assert q.max() <= 15 and s.shape == (32, 2)
    assert torch.equal(convert.rotation_signs(256), signs)  # fixed per size


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


def small_drama3_spec():
    """Drama 3's spec with a fallback that fetches one file of a made-up FP8 checkpoint."""
    spec = models.BY_SETTING["drama3_model"]
    fallback = models.Source("fp8", ("config.json",), spec.fallback.build, repo=spec.fallback.repo,
                             revision=spec.fallback.revision)  # fmt: skip
    files = ("config.json", "model.safetensors.index.json", convert.NOTICE, convert.STATS)
    return models.ModelSpec(spec.setting, spec.name, spec.purpose, spec.repo, spec.folder, files, 1,
                            revision=spec.revision, fallback=fallback)  # fmt: skip


def test_drama3_downloads_the_prebuilt_build(tmp_path, monkeypatch):
    small = small_drama3_spec()
    fetched = []

    def fake_fetch(repo, revision, folder, files, endpoint, report):
        fetched.append((repo, revision))
        for f in files:
            (folder / f).write_text("{}")
        report(1, 1)

    monkeypatch.setattr(models, "_fetch_hf", fake_fetch)
    models.download(small, tmp_path)
    assert fetched == [(models.DRAMA3_BUILD_REPO, models.DRAMA3_BUILD_REVISION)] and small.downloaded(tmp_path)


def test_drama3_is_built_from_the_official_weights_when_the_build_is_gone(tmp_path, monkeypatch):
    small = small_drama3_spec()
    fetched = []

    def fake_fetch(repo, revision, folder, files, endpoint, report):
        fetched.append((repo, revision))
        if repo == models.DRAMA3_BUILD_REPO:
            raise FileNotFoundError(f"{repo} has no config.json.")
        folder.rmdir()
        fake_fp8_checkpoint(folder)
        report(1 << 20, 1 << 20)

    monkeypatch.setattr(models, "_fetch_hf", fake_fetch)
    progress = []
    models.download(small, tmp_path, progress=lambda done, total: progress.append((done, total)))
    assert fetched == [(models.DRAMA3_BUILD_REPO, models.DRAMA3_BUILD_REVISION),
                       ("fishaudio/drama-3", models.BY_SETTING["drama3_model"].fallback.revision)]  # fmt: skip
    assert small.downloaded(tmp_path) and not (tmp_path / "fp8").exists()
    assert "Built with Fish Audio" in (tmp_path / small.folder / convert.NOTICE).read_text(encoding="utf-8")
    assert progress[-1][0] == progress[-1][1] > 0  # download + build reach the end of one bar


def test_a_network_error_is_not_a_reason_to_build(tmp_path, monkeypatch):
    def offline(repo, revision, folder, files, endpoint, report):
        raise ConnectionError("no network")

    monkeypatch.setattr(models, "_fetch_hf", offline)
    with pytest.raises(ConnectionError):
        models.download(small_drama3_spec(), tmp_path)
    assert not (tmp_path / "fp8").exists()


def expert_stacks(experts=6, hidden=128, inner=128, higgs=False):
    """4-bit stacks for ``experts`` made-up experts, as model.load builds them (HIGGS: rotated, on
    the grid)."""
    torch.manual_seed(1)
    stacks = {}
    for short, (k, n) in (("gate", (hidden, inner)), ("up", (hidden, inner)), ("down", (inner, hidden))):
        ws = [torch.randn(n, k) * 0.05 for _ in range(experts)]
        if higgs:
            qs = [convert.quantize_grid(convert.rotate(w, convert.rotation_signs(k))) for w in ws]
        else:
            qs = [convert.quantize(w) for w in ws]
        stacks[short + "_q"] = torch.stack([convert.pack(q) for q, _ in qs])
        stacks[short + "_s"] = torch.stack([s.t().contiguous().half() for _, s in qs])
    return stacks


def higgs_format(hidden=128, inner=128):
    """(grid, rotation) for HIGGS stacks, as model.load makes them."""
    signs = {k: convert.rotation_signs(k) for k in {hidden, inner}}
    return torch.tensor(convert.GRID), model.Rotation(signs, "cpu")


def dense_moe(stacks, hidden, index, weights, grid=None, rotation=None):
    """The experts' weighted sum from their weights unpacked (rotated back to the inputs' space)."""
    w = {k: convert.dequant(stacks[k + "_q"], stacks[k + "_s"], torch.float32, grid) for k in ("gate", "up", "down")}
    if rotation is not None:  # x @ W = rotate(x) @ W_stored for every x: W = rotate(I) @ W_stored
        w = {k: rotation(torch.eye(v.shape[1])) @ v for k, v in w.items()}
    out = torch.zeros_like(hidden)
    for t in range(hidden.shape[0]):
        for e, g in zip(index[t].tolist(), weights[t], strict=True):
            x = hidden[t]
            out[t] += g * ((torch.nn.functional.silu(x @ w["gate"][e]) * (x @ w["up"][e])) @ w["down"][e])
    return out


@pytest.mark.parametrize("higgs", [False, True])
@pytest.mark.parametrize("slots", [0, 2, 6])
def test_cached_and_streamed_experts_compute_the_same(slots, higgs):
    """Whatever the GPU slots hold (nothing, some, all), the layer's output is the experts'
    weighted sum, for many tokens and one; planning between calls moves the used ones into the
    slots. HIGGS experts: their inputs rotated, their values on the grid."""
    stacks = expert_stacks(higgs=higgs)
    fmt = higgs_format() if higgs else (None, None)
    layer = model.Experts(stacks, slots, "cpu", *fmt)
    for tokens in (5, 1, 5):
        hidden = torch.randn(tokens, 128)
        index = torch.stack([torch.randperm(6)[:2] for _ in range(tokens)])
        weights = torch.rand(tokens, 2)
        want = dense_moe(stacks, hidden, index, weights, *fmt)
        assert torch.allclose(layer(hidden, index, weights), want, atol=1e-4)
        layer.plan()
    assert len(layer.held) == slots


def test_higgs_experts_are_closer_to_their_weights_than_plain_4_bits():
    torch.manual_seed(5)
    w = torch.randn(64, 256) * 0.05
    w[:, 7] *= 8  # an outlier input channel, as real experts have
    signs = convert.rotation_signs(256)
    plain = convert.restore(*convert.quantize(w))
    higgs = convert.unrotate(convert.restore_grid(*convert.quantize_grid(convert.rotate(w, signs))), signs)
    assert (higgs - w).norm() < 0.8 * (plain - w).norm()


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
        self.guided = []  # per call: (the guidance weight, rows of the prompt without references)

    def generate(self, rows, sampling, max_frames, on_frame, head=0, plain=None):
        self.calls.append(max_frames)
        self.guided.append((sampling.guidance, None if plain is None else (len(plain), len(rows))))
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
    with pytest.raises(KeyError):  # no such voice (saved voices: test_drama3_voices)
        tts.synthesize(SynthesisRequest(text="你好。", lora_id="voxcpm/someone"))


def test_takes_lean_on_a_reference_only_when_there_is_one(settings, monkeypatch, tmp_path):
    m = FakeModel()
    tts = drama3(settings, monkeypatch, m)
    clip = tmp_path / "ref.wav"
    leaning = {"guidance": 0.5}
    tts.synthesize(SynthesisRequest(text="你好。", reference_wav=clip, prompt_text="参考。", options=leaning))
    weight, (plain, rows) = m.guided[-1]
    assert weight == 0.5 and plain < rows  # the same turn, without the reference
    tts.synthesize(SynthesisRequest(text="你好。", reference_wav=clip, prompt_text="参考。"))  # off unless asked
    tts.synthesize(SynthesisRequest(text="你好。", options=leaning))  # no reference: nothing to lean on
    assert m.guided[-2:] == [(0.0, None), (0.0, None)]
    assert engine.frame_cap(1000) == engine.MAX_FRAMES and engine.frame_cap(50) == 150


def tiny_drama3(slots=0, higgs=False):
    """A made-up Drama 3 small enough for the CPU: 3 Gated DeltaNet layers and one attention layer,
    6 experts in 4 bits (plain or HIGGS), a 2-layer tower at the real codebook size."""
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
    fmt = higgs_format(dim, 128) if higgs else (None, None)
    for layer in m.backbone.layers:
        layer.mlp.experts = model.Experts(
            expert_stacks(experts=6, hidden=dim, inner=128, higgs=higgs), slots, "cpu", *fmt
        )
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


@pytest.mark.parametrize("higgs", [False, True])
@torch.inference_mode()
def test_the_decoder_speaks_as_plain_decoding(monkeypatch, higgs):
    """Greedy (one candidate) the decoder's frames equal the reference loop's."""
    m = tiny_drama3(higgs=higgs)
    monkeypatch.setattr(model, "REDRAW", model.Sampling(1.0, 0.95, 1))
    m.decoder.redraw[2].fill_(1)
    rows = tiny_prompt(m)
    greedy = model.Sampling(1.0, 0.9, 1)
    want = m.generate_reference(rows, greedy, 6)
    got = m.generate(rows, greedy, 6)
    assert got.shape == want.shape == (10, 6) and torch.equal(got, want)
    # Reference guidance: a second state that read the prompt without its references.
    plain, leaning = rows[3:], model.Sampling(1.0, 0.9, 1, guidance=2.0)
    want_guided = m.generate_reference(rows, leaning, 6, plain=plain)
    assert torch.equal(m.generate(rows, leaning, 6, plain=plain), want_guided)
    assert not torch.equal(want_guided, want)  # (it changes what is said)
    assert torch.equal(m.generate(rows, leaning, 6), want)  # nothing to compare with: unguided


def test_guidance_strengthens_what_the_references_add():
    with_refs, without = torch.tensor([2.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 2.0])
    assert torch.allclose(model.guided(with_refs, without, 0.0), with_refs.log_softmax(-1))
    pushed = model.guided(with_refs, without, 1.0).softmax(-1)
    assert pushed[0] > with_refs.softmax(-1)[0] and pushed[2] < with_refs.softmax(-1)[2]


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
            assert all(
                torch.allclose(t.k[: len(r) - 1], k, atol=1e-4) for t, k in zip(towers, want_k, strict=True)
            )  # f32 rounding
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


def test_a_higgs_build_tells_the_model_its_grid_and_rotation(tmp_path):
    fake_fp8_checkpoint(tmp_path / "fp8")
    convert.convert(tmp_path / "fp8", tmp_path / "int4", device="cpu")
    out = read_all(tmp_path / "int4")
    config = json.loads((tmp_path / "int4" / "config.json").read_text())["quantization_config"]
    grid, rotation = model.expert_format(config["experts"], out, out.__getitem__, "cpu")
    assert torch.equal(grid, torch.tensor(convert.GRID))
    assert list(rotation.signs) == [384] and torch.equal(rotation.signs[384], convert.rotation_signs(384))
    assert model.expert_format({"quant_method": "gptq"}, out, out.__getitem__, "cpu") == (None, None)
    with pytest.raises(ValueError, match="format"):
        model.expert_format({"quant_method": "nf3"}, out, out.__getitem__, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernels need CUDA")
def test_the_kernels_read_the_grid():
    from polyvox.engines.fish_drama3 import kernels

    stacks = expert_stacks(higgs=True)
    grid = torch.tensor(convert.GRID, device="cuda")
    q, s = stacks["gate_q"].cuda(), stacks["gate_s"].cuda()
    want = convert.dequant(q, s, torch.float32, grid)  # [E, K, N]
    assert torch.equal(kernels.int4_dequant(q, s, grid), want.to(torch.bfloat16))
    # Expert 4 in a GPU slot, expert 1 read from pinned CPU memory.
    table = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    table[4] = 0
    host_q, host_s = (
        kernels.HostPointer(stacks["gate_q"].pin_memory()),
        kernels.HostPointer(stacks["gate_s"].pin_memory()),
    )
    x = torch.randn(1, 128, device="cuda").bfloat16()
    ids = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    rows = torch.zeros(2, dtype=torch.int32, device="cuda")
    got = kernels.int4_gemv_routed(x, rows, ids, table, q[4:5], s[4:5], host_q, host_s, grid)
    assert torch.allclose(got, (x.float() @ want[[1, 4]])[:, 0], rtol=1e-3, atol=1e-3)


def test_8_bit_linears_stay_close_to_their_weights():
    kernels = pytest.importorskip("polyvox.engines.fish_drama3.kernels")  # needs Triton
    torch.manual_seed(2)
    w = torch.randn(96, 256) * 0.02
    w[3, 17] = 0.5  # an outlier spoils only its own group of 128
    lin = kernels.Int8Linear(w)
    assert lin.q.dtype == torch.int8 and lin.scales.shape == (2, 96)
    assert ((lin.weight(torch.float32) - w).norm() / w.norm()) < 0.01
    x = torch.randn(3, 256)
    assert torch.allclose(lin(x), torch.nn.functional.linear(x, lin.weight(torch.float32)), atol=1e-6)
    assert torch.equal(model.linear(x, w), torch.nn.functional.linear(x, w))  # plain weights pass through


def test_long_texts_are_cut_into_even_parts_of_whole_sentences():
    assert engine.spoken_parts("  一句话。  ") == ["一句话。"]
    sentence = "今天天气很好，我们去公园散步。"  # 45 bytes
    assert engine.spoken_parts(sentence * 4) == [sentence * 4]  # a little over the aim: still one take
    text = sentence * 20
    parts = engine.spoken_parts(text)
    assert "".join(parts) == text and all(p.endswith("。") for p in parts)
    assert {len(p) // len(sentence) for p in parts} <= {3, 4}  # about TARGET_BYTES each, none left short
    long = "这是一个比较长的句子，" * 7 + "到这里才结束。"  # 252 bytes: over the aim, under the limit
    assert engine.spoken_parts(long * 3) == [long] * 3  # a sentence is cut only when over the limit
    assert engine.spoken_parts(sentence + long + sentence * 3) == [sentence, long, sentence * 3]
    # A blank line is the writer's cut: always a new part, however short; a line break alone isn't.
    assert engine.spoken_parts("第一段。\n还是第一段。\r\n \r\n\n第二段。") == ["第一段。\n还是第一段。", "第二段。"]
    assert engine.spoken_parts(text + "\n\n" + sentence)[-1] == sentence
    assert engine.spoken_parts("<whisper>a.\n\nb.</whisper>") == ["<whisper>a.\n\nb.</whisper>"]  # never in a tag pair


def test_symbols_the_model_cant_read_are_left_out():
    assert engine.speakable("The ⓘ button ① ★ works 🎉.") == "The i button 1 works ."
    assert (
        engine.speakable("你好，世界！[laughs] <whisper>嘘</whisper>") == "你好，世界！[laughs] <whisper>嘘</whisper>"
    )
    assert engine.speakable("first line\nsecond ½") == "first line\nsecond ½"


def test_takes_are_reported_as_offsets_into_the_text_as_typed(settings):
    tts = engine.FishDrama3Engine(settings)
    typed = "  第一段。 \r\n\r\n\r\n第二段，  two   words. \n"
    ends = tts.takes(typed, cloning=True)
    assert [typed[a:b].strip() for a, b in zip([0, *ends], ends, strict=False)] == [
        "第一段。",
        "第二段，  two   words.",
    ]
    sentence = "今天天气很好，我们去公园散步。"
    assert tts.takes(sentence * 6, cloning=True) == [len(sentence) * 3, len(sentence) * 6]
    assert tts.takes(sentence * 6, cloning=False) == [len(sentence) * 6]  # no voice to keep: one take
    assert tts.takes("  ", cloning=True) == []
    english = " ".join(f"Sentence number {i} is here." for i in range(30))
    parts = engine.spoken_parts(english, 100)
    assert all(p.endswith(".") for p in parts) and " ".join(parts) == english  # spaces kept between sentences
    whisper = "<whisper>" + "secret words. " * 10 + "</whisper>"  # longer than a part: stays whole
    parts = engine.spoken_parts("Pi is 3.14159 roughly. " * 6 + whisper + " [laughing] done.", 60)
    assert whisper.strip() in [p.strip() for p in parts] and not any(p.endswith("3.") for p in parts)
    assert any("[laughing]" in p for p in parts)
    clauses = "，".join(["这是一个很长的分句"] * 20)
    assert all(p.endswith("，") for p in engine.spoken_parts(clauses, 60)[:-1])
    assert all(engine.utf8_bytes(p) <= 50 for p in engine.spoken_parts("字" * 100, 50))


def test_style_and_hints_reach_every_part(settings, monkeypatch):
    m = FakeModel()
    seen = []

    def generate(rows, sampling, cap, on_frame, head=0):
        seen.append(rows)
        return torch.zeros(10, 20, dtype=torch.long)

    m.generate = generate
    tts = drama3(settings, monkeypatch, m)
    tts.synthesize(
        SynthesisRequest(text="第一句。" * 40 + "第二句。", style="softly", options={"language": "zh", "duration": 3})
    )
    texts = ["".join(chr(r) for r in rows[:, 0].tolist()) for rows in seen]
    assert len(texts) > 1 and all("[softly] " in t and "[language: zh]<|im_end|>" in t for t in texts)
    assert not any("audio tokens" in t for t in texts)  # a length only for a text in one take
    assert all(t.count("<|im_start|>user\n") == 1 for t in texts)  # each part on its own
    seen.clear()
    tts.synthesize(SynthesisRequest(text="今天天气很好，我们去公园散步。" * 6))  # no voice to keep: one take
    assert len(seen) == 1
    seen.clear()
    tts.synthesize(SynthesisRequest(text="一句。", options={"duration": 3}))
    assert "[audio tokens: 65]<|im_end|>" in "".join(chr(r) for r in seen[0][:, 0].tolist())
    seen.clear()
    tts.synthesize(SynthesisRequest(text="一句。\n\n🎉✓\n\n二句。"))  # a part of symbols only isn't spoken
    assert len(seen) == 2
    with pytest.raises(ValueError, match="only symbols"):
        tts.synthesize(SynthesisRequest(text="🎉 ★"))
    with pytest.raises(ValueError, match="can't contain"):
        tts.synthesize(SynthesisRequest(text="一行<|im_end|>"))


def test_expert_stacks_read_in_the_background_equal_the_stored_tensors(tmp_path):
    from polyvox.engines.fish_drama3.reader import Stacks

    torch.manual_seed(0)
    shards = {"a.safetensors": {}, "b.safetensors": {}}
    groups, weight_map = [], {}
    for layer in range(2):  # (a stack's tensors lie in both shards, among others, in no order)
        group = {}
        for name, shape, dtype in (("q", (3, 5), torch.int32), ("s", (4,), torch.bfloat16)):
            keys = [f"layers.{layer}.experts.{e}.{name}" for e in range(7)]
            for e, key in enumerate(keys):
                shard = "a.safetensors" if (e + layer) % 3 else "b.safetensors"
                shards[shard][key] = (torch.randn(shape) * 100).to(dtype)
                weight_map[key] = shard
            group[name] = keys
        groups.append(group)
    shards["a.safetensors"]["other"] = torch.ones(11)
    for shard, tensors in shards.items():
        save_file(tensors, str(tmp_path / shard))

    stacks = Stacks(tmp_path, weight_map, groups).result()

    assert [set(s) for s in stacks] == [{"q", "s"}, {"q", "s"}]
    for group, got in zip(groups, stacks, strict=True):
        for name, keys in group.items():
            assert torch.equal(got[name], torch.stack([shards[weight_map[k]][k] for k in keys]))

    (tmp_path / "b.safetensors").write_bytes((tmp_path / "b.safetensors").read_bytes()[:-8])
    with pytest.raises(OSError, match="ends inside a tensor"):
        Stacks(tmp_path, weight_map, groups).result()
