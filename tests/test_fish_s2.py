"""Fish Audio S2-Pro: our model / prompt / decoding on a tiny random model (no weights needed)."""

import dataclasses

import pytest
import torch

from polyvox.engines.common import fish_codec as codec
from polyvox.engines.fish_s2 import decode, generate, model, prompt, train
from polyvox.engines.fish_s2 import lora as s2lora
from polyvox.engines.fish_s2.engine import FishS2Engine, text_parts

SLOW = model.Shape(dim=32, layers=2, heads=4, kv_heads=2, head_dim=8, hidden=64, qk_norm=True, eps=1e-6, rope_base=1e4)
FAST = dataclasses.replace(SLOW, layers=1, qk_norm=False)
CONFIG = model.Config(
    slow=SLOW, fast=FAST, vocab_size=200, codebooks=3, codebook_size=32, semantic_begin=100, semantic_end=131
)
END = 99


class Letters:
    """A tokenizer: one id per character (special tokens as whole pieces)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 90 for ch in text]


def tiny_model() -> model.DualAR:
    torch.manual_seed(0)
    m = model.DualAR(CONFIG)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return m.eval().requires_grad_(False)


def test_long_texts_are_spoken_in_parts_of_whole_sentences():
    assert text_parts("一。二！三？", limit=4) == ["一。二！", "三？"]
    assert text_parts("这是一个很长的句子。", limit=3) == ["这是一个很长的句子。"]  # never cut inside a sentence
    assert text_parts("Hello there. How are you?") == ["Hello there. How are you?"]
    assert text_parts("你好。\nHello there.\n再见。") == ["你好。 Hello there. 再见。"]
    assert text_parts("  ") == []


def test_the_prompt_puts_reference_codes_under_their_semantic_tokens():
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])  # [codebooks, frames]
    rows = prompt.build(Letters(), 3, 100, "hi", (codes, "示例"))
    audio = (rows[0] >= 100) & (rows[0] <= 131)
    assert audio.sum() == 2 and rows.shape[0] == 4
    assert rows[0, audio].tolist() == [101, 102] and rows[1:, audio].tolist() == codes.tolist()
    assert rows[1:, ~audio].abs().sum() == 0  # text positions carry no codes
    plain = prompt.build(Letters(), 3, 100, "hi")
    assert not ((plain[0] >= 100) & (plain[0] <= 131)).any()


def test_checkpoint_keys_map_to_ours():
    assert model.checkpoint_name("text_model.model.layers.0.attention.wqkv.weight") == "layers.0.attention.wqkv.weight"
    assert model.checkpoint_name("audio_decoder.codebook_embeddings.weight") == "codebook_embeddings.weight"
    assert model.checkpoint_name("audio_decoder.layers.1.ffn_norm.weight") == "fast_layers.1.ffn_norm.weight"
    assert model.checkpoint_name("audio_decoder.output.weight") == "fast_output.weight"
    assert model.checkpoint_name("something.else") is None


def test_weight_norms_fold_into_plain_weights():
    conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 4, 5))
    state = {f"c.{k}": v for k, v in conv.state_dict().items()}
    state["c.freqs_cis"] = torch.zeros(2)  # a fixed buffer: left out
    folded = codec.folded(state)
    assert sorted(folded) == ["c.bias", "c.weight"]
    assert torch.allclose(folded["c.weight"], conv.weight, atol=1e-6)


def test_decoding_in_pieces_matches_one_pass():
    torch.manual_seed(0)
    up, frame = 2, 8  # 2 latent steps per frame, 4 samples per latent step
    stack = torch.nn.Sequential(
        codec.CausalConv(3, 4, 7, dilation=3),
        codec.Snake(4),
        codec.CausalTransConv(4, 4, 4, 2),
        codec.CausalTransConv(4, 1, 4, 2),
    ).eval()
    z = torch.randn(1, 3, up * 37)
    with torch.no_grad():
        whole = stack(z)
        pieces = codec.in_pieces(stack, z, up, frame, piece=5, context=16)
        no_context = codec.in_pieces(stack, z, up, frame, piece=5, context=0)

    assert whole.shape == pieces.shape == (1, 1, 37 * frame)
    assert torch.allclose(pieces, whole, atol=1e-6)
    assert not torch.allclose(no_context, whole, atol=1e-3)  # context is what makes them match
    assert codec.in_pieces(stack, z[..., :0], up, frame).shape == (1, 1, 0)


def test_the_frame_decoder_makes_the_same_frames_as_plain_decoding():
    m = tiny_model()
    torch.manual_seed(1)
    rows = torch.zeros(4, 12, dtype=torch.long)
    rows[0] = torch.randint(0, 90, (12,))
    rows[0, 4:7] = torch.tensor([100, 120, 131])  # a reference's frames
    rows[1:, 4:7] = torch.randint(0, 32, (3, 3))
    greedy = generate.Sampling(0.8, 0.8, 1)

    plain = generate.generate(m, rows, END, greedy, max_frames=20)
    framed = decode.Decoder(m, END, graphs=False).generate(rows, greedy, max_frames=20)

    assert plain.shape[0] == 3 and plain.shape[1] > 0
    assert torch.equal(plain, framed)
    assert ((plain >= 0) & (plain < 32)).all()

    # The linear layers stay module calls: hooks (a training run's LoRA) apply to both.
    m.fast_layers[0].feed_forward.w2.register_forward_hook(lambda mod, args, out: out * 3)
    hooked = generate.generate(m, rows, END, greedy, max_frames=20)
    assert not torch.equal(hooked, plain)
    assert torch.equal(decode.Decoder(m, END, graphs=False).generate(rows, greedy, max_frames=20), hooked)


def test_the_engine_needs_its_model_and_a_transcript(settings):
    engine = FishS2Engine(settings)
    assert "download" in engine.unavailable()
    caps = engine.capabilities
    assert caps.reference_inputs[0].transcript == "required" and caps.training and caps.voice_reference
    assert caps.tags and caps.free_tags and all(t.startswith("[") for g in caps.tag_groups for t in g.tags)


def sequence(m, text, frames, seed, reference=None):
    torch.manual_seed(seed)
    codes = torch.randint(0, 32, (3, frames))
    return train.Sequences(Letters(), m, END).rows(text, codes, reference)


def decoded_losses(m, rows, start):
    """The training losses of one sequence, the way decoding chooses each frame's
    semantic token (among the 32 semantic tokens and the end) and its codes."""
    m.allocate(rows.shape[1])
    allowed = torch.tensor([*range(100, 132), END])
    codes, semantic = torch.tensor(0.0), torch.tensor(0.0)
    ce = torch.nn.functional.cross_entropy
    for f in range(start, rows.shape[1]):
        logits, hidden = m.slow(rows[:, :f], 0)
        wanted = (allowed == rows[0, f]).nonzero()[0]
        semantic += ce(logits[allowed][None], wanted, reduction="sum")
        if f == rows.shape[1] - 1:  # the end token: no codes
            break
        m.fast(hidden, 0)
        for i in range(1, 3):
            logits = m.fast(m.fast_embeddings.weight[rows[i, f]], i)
            codes += ce(logits[None], rows[i + 1, f][None], reduction="sum")
    return codes, semantic


def test_training_scores_frames_as_decoding_produces_them():
    m = tiny_model()
    ref = (torch.randint(0, 32, (3, 4)), "示例")
    items = [sequence(m, "hello", 5, 1), sequence(m, "hi", 3, 2, ref)]  # different lengths: right padding
    trainer = train.Trainer(m, END, slow=True, recompute=False)

    losses = trainer.losses(*train.collate(items, "cpu"))

    (codes, n_codes), (semantic, n_semantic) = losses["codes"], losses["semantic"]
    assert n_codes == (5 + 3) * 2 and n_semantic == (5 + 1) + (3 + 1)  # codes 1..2 of each frame; its token, the end
    expected = [sum(parts) for parts in zip(*(decoded_losses(m, rows, start) for rows, start in items), strict=True)]
    assert torch.allclose(codes, expected[0], rtol=1e-4) and torch.allclose(semantic, expected[1], rtol=1e-4)
    rows, start = items[1]
    assert rows[0, -1] == END and ((rows[0, start:-1] >= 100) & (rows[0, start:-1] <= 131)).all()
    assert set(train.Trainer(m, END).losses(*train.collate(items, "cpu"))) == {"codes"}  # fast only: no semantic loss


def test_the_lora_trains_only_itself_and_merges_into_the_weights():
    import copy

    m = tiny_model()
    plain = copy.deepcopy(m)
    loras = s2lora.attach(m, rank=2, alpha=4.0)
    assert sorted(loras) == [f"fast_layers.0.{n}" for n in
                             ("attention.wo", "attention.wqkv", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")]  # fmt: skip
    trainer = train.Trainer(m, END, recompute=True)
    batch = train.collate([sequence(m, "hello", 4, 3)], "cpu")

    loss, n = trainer.losses(*batch)["codes"]
    loss.backward()
    assert all(lo.B.grad.abs().sum() > 0 and lo.A.grad.abs().sum() == 0 for lo in loras.values())  # B starts at 0
    assert all(p.grad is None for p in m.parameters())

    with torch.no_grad():
        for lo in loras.values():
            lo.B.normal_(std=0.5)
        hooked, _ = trainer.losses(*batch)["codes"]
    originals = {k: v.clone() for k, v in plain.state_dict().items()}
    merger = s2lora.Merger(plain)
    merger.apply(s2lora.state(loras), 4.0 / 2)
    merged, _ = train.Trainer(plain, END).losses(*batch)["codes"]
    assert torch.allclose(hooked, merged, rtol=1e-4) and not torch.allclose(merged, loss, rtol=1e-3)
    merger.apply(None)
    assert all(torch.equal(v, originals[k]) for k, v in plain.state_dict().items())
    with pytest.raises(ValueError, match="does not fit"):
        merger.apply(
            {
                "fast_layers.0.attention.wo.lora_A": torch.zeros(2, 5),
                "fast_layers.0.attention.wo.lora_B": torch.zeros(32, 2),
            }
        )
    with pytest.raises(ValueError, match="does not fit"):  # A and B of different ranks
        merger.apply(
            {
                "fast_layers.0.attention.wo.lora_A": torch.zeros(2, 32),
                "fast_layers.0.attention.wo.lora_B": torch.zeros(32, 3),
            }
        )


def test_a_voice_that_fails_to_load_leaves_the_base_model(tmp_path):
    import copy
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from polyvox import lora_library
    from polyvox.engines.fish_s2.engine import Runtime

    m = tiny_model()
    originals = {k: v.clone() for k, v in m.state_dict().items()}
    loras = s2lora.attach(copy.deepcopy(m), rank=2, alpha=2.0)
    with torch.no_grad():
        for lo in loras.values():
            lo.B.normal_(std=0.5)
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir(), bad.mkdir()
    save_file({k: v.contiguous() for k, v in s2lora.state(loras).items()}, str(good / lora_library.WEIGHTS_FILE))
    (bad / lora_library.WEIGHTS_FILE).write_bytes(b"half a file")  # e.g. read while training saves it
    voice = lambda path: SimpleNamespace(id=path.name, weights_mtime=1.0, path=path, rank=2, alpha=2.0)  # noqa: E731
    runtime = object.__new__(Runtime)
    runtime.model, runtime.merger, runtime.lora = m, s2lora.Merger(m), None

    def is_base() -> bool:
        return all(torch.equal(v, originals[k]) for k, v in m.state_dict().items())

    runtime.use_lora(voice(good))
    assert not is_base()
    with pytest.raises(RuntimeError, match="Unable to load voice 'bad'"):
        runtime.use_lora(voice(bad))
    assert is_base() and runtime.lora is None  # not the good voice's weights under the base model's name
    runtime.use_lora(None)
    assert is_base()


def test_reference_picks_follow_the_clips_not_their_order():
    keys = [f"{i:02x}" * 20 for i in range(6)]
    picks = train.reference_picks(keys, references=2)
    assert all(len(p) == 2 and all(1 <= len(c) <= 3 and i not in c for c in p) for i, p in enumerate(picks))
    shuffled = keys[::-1]  # another run listing the same clips in another order
    again = train.reference_picks(shuffled, references=2)
    as_keys = lambda ks, ps: {ks[i]: [[ks[j] for j in c] for c in p] for i, p in enumerate(ps)}  # noqa: E731
    assert as_keys(keys, picks) == as_keys(shuffled, again)


def test_cached_slow_states_score_as_live_ones():
    m = tiny_model()
    seq, trainer = train.Sequences(Letters(), m, END), train.Trainer(m, END)
    torch.manual_seed(4)
    clips = [{"text": t, "codes": torch.randint(0, 32, (3, n))} for t, n in (("一二", 3), ("三四五", 5), ("六", 2))]
    picks = train.reference_picks([f"{i}" * 12 for i in range(3)], references=2)

    cache = train.slow_states(trainer, seq, clips, picks, todo=[0, 2], tokens=300)  # several runs of the slow model

    assert sorted(cache) == [0, 2]
    for i in (0, 2):
        c = clips[i]
        references = [None, *(train.joined_reference(clips, chosen) for chosen in picks[i])]
        assert len(cache[i]) == 3
        for variant, reference in enumerate(references):
            live, n = trainer.losses(*train.collate([seq.rows(c["text"], c["codes"], reference)], "cpu"))["codes"]
            cached, n2 = trainer.code_loss(cache[i][variant], c["codes"].long().T)
            assert cache[i][variant].shape == (c["codes"].shape[1], 32) and n == n2
            assert torch.allclose(live, cached, rtol=1e-5)


def test_training_from_cached_states_needs_no_slow_transformer():
    m = tiny_model()
    seq, trainer = train.Sequences(Letters(), m, END), train.Trainer(m, END)
    clips = [
        {"text": "一二三", "codes": torch.randint(0, 32, (3, 4))},
        {"text": "四五", "codes": torch.randint(0, 32, (3, 3))},
    ]
    states = train.slow_states(trainer, seq, clips, train.reference_picks(["a" * 12, "b" * 12]), todo=[0, 1])
    loras = s2lora.attach(m, rank=2, alpha=2.0)

    for part in m.slow_parts():  # gone: any use of them would fail
        part.to("meta")
    loss, n = trainer.code_loss(
        torch.cat([states[0][1], states[1][0]]), torch.cat([c["codes"].long().T for c in clips])
    )
    loss.backward()

    assert n == (4 + 3) * 2 and torch.isfinite(loss)
    assert all(lo.B.grad is not None for lo in loras.values())
    assert not any(p.is_meta for p in m.fast_layers.parameters())
    m.allocate(16)
    m.release()
    assert m.capacity == 0 and m.layers[0].attention.cache is None


def test_the_slow_transformer_trains_only_when_asked():
    m = tiny_model()
    loras = s2lora.attach(m, rank=2, alpha=2.0, targets=(*s2lora.DEFAULT_TARGETS, *s2lora.SLOW_TARGETS))
    slow = {k: lo for k, lo in loras.items() if k.startswith("layers.")}
    assert sorted(slow) == [f"layers.{i}.attention.{n}" for i in range(2) for n in ("wo", "wqkv")]
    batch = train.collate([sequence(m, "hello", 4, 3)], "cpu")

    sum(s / n for s, n in train.Trainer(m, END).losses(*batch).values()).backward()
    assert all(lo.B.grad is None for lo in slow.values())  # fast only: the slow layers run without gradients

    sum(s / n for s, n in train.Trainer(m, END, slow=True).losses(*batch).values()).backward()
    assert all(lo.B.grad.abs().sum() > 0 for lo in slow.values())


def test_the_noise_floor_is_faint_and_repeatable():
    silence = torch.zeros(44_100)
    floored = train.with_noise_floor(silence, seed=3)
    assert torch.equal(floored, train.with_noise_floor(silence, seed=3))
    level = 20 * torch.log10(floored.pow(2).mean().sqrt())
    assert -73 < level < -71


def write_clips(folder, clips):
    """clips: (name, seconds, text) of a tone each; the dataset's manifest."""
    import numpy as np
    import soundfile as sf

    from polyvox.data import manifests

    (folder / "wavs").mkdir(parents=True)
    for name, seconds, _ in clips:
        t = np.arange(int(16000 * seconds)) / 16000
        sf.write(folder / "wavs" / f"{name}.wav", 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32), 16000)
    manifests.write_manifest([{"audio": f"wavs/{n}.wav", "text": text} for n, _, text in clips], folder / "train.jsonl")
    return folder / "train.jsonl"


def test_a_run_trains_the_fast_transformer_with_the_datasets_best_clips(settings, data_dir, tmp_path, monkeypatch):
    from polyvox.training import TrainParams

    write_clips(
        data_dir / "speaker_a",
        [(f"c{i}", 5.0, f"这是第{i}个示例句子，内容足够长。") for i in range(5)] + [("short", 1.0, "短。")],
    )
    engine = FishS2Engine(settings)
    params = TrainParams(name="run_a", engine="fish_s2", dataset="speaker_a/train.jsonl", optimizer="prodigy_sf")
    with pytest.raises(ValueError, match="download"):
        engine.training_config(params, tmp_path / "run_a", tmp_path / "train.jsonl", None)
    monkeypatch.setattr(engine, "training_unavailable", lambda: None)
    monkeypatch.setattr(engine, "training_gb", lambda: 16.0)

    config = engine.training_config(params, tmp_path / "run_a", tmp_path / "train.jsonl", None)

    assert config["lora"]["targets"] == ["fast_attn", "fast_mlp"] and config["optimizer"] == "prodigy_sf"
    assert (config["cached_references"], config["offload"]) == (2, False)  # offload: New run sets it per GPU
    slow = params.model_copy(update={"s2_slow": True})
    with pytest.raises(ValueError, match="AdamW"):  # Prodigy can't give the slow layers a smaller rate
        engine.training_config(slow, tmp_path / "b", tmp_path / "t.jsonl", None)
    slow = slow.model_copy(update={"optimizer": "adamw", "learning_rate": 5e-4})
    config_slow = engine.training_config(slow, tmp_path / "b", tmp_path / "t.jsonl", None)
    assert config_slow["lora"]["targets"] == ["fast_attn", "fast_mlp", "slow_attn"]  # only when asked
    assert (config_slow["learning_rate"], config_slow["slow_learning_rate"]) == (5e-4, 2e-5)
    assert "slow_learning_rate" not in config
    clips = config["reference"]["clips"]
    assert len(clips) == 3 and all(c["audio"].endswith(".wav") and "short" not in c["audio"] for c in clips)

    # A 12 GB card trains with the slow transformer in CPU memory, whatever was asked, and
    # can't train the slow transformer itself.
    monkeypatch.setattr(engine, "training_gb", lambda: 11.6)
    assert engine.training_config(params, tmp_path / "c", tmp_path / "t.jsonl", None)["offload"] is True
    with pytest.raises(ValueError, match="16 GB"):
        engine.training_config(slow, tmp_path / "d", tmp_path / "t.jsonl", None)


def test_training_needs_less_gpu_memory_than_synthesis(settings, monkeypatch):
    from polyvox import vram

    engine = FishS2Engine(dataclasses.replace(settings, s2_model="X:/models/s2-pro"))
    for gb, synthesis, training in ((16.0, True, True), (11.6, False, True), (7.6, False, False)):
        monkeypatch.setattr(vram, "total_gb", lambda device, gb=gb: gb)
        assert (engine.unavailable() is None, engine.training_unavailable() is None) == (synthesis, training), gb
    assert "12 GB" in engine.training_unavailable()
    assert engine.public()["training_unavailable"] == engine.training_unavailable()


def test_a_voice_package_carries_its_reference(lora_dir, tmp_path):
    import json

    import soundfile as sf
    from safetensors.torch import save_file

    from polyvox import lora_library, sharing

    ckpt = lora_dir / "fish_s2" / "voice" / "checkpoints" / "step_0000040"
    ckpt.mkdir(parents=True)
    save_file({"fast_layers.0.attention.wo.lora_A": torch.randn(4, 8), "fast_layers.0.attention.wo.lora_B": torch.zeros(8, 4)},
              ckpt / "lora_weights.safetensors")  # fmt: skip
    s2lora.write_config(
        ckpt, {"r": 4, "alpha": 4.0, "dropout": 0.0, "targets": ["fast_attn"]}, r"X:\old_pc\models\fishaudio__s2-pro"
    )
    sf.write(ckpt / "reference.wav", torch.zeros(1600).numpy(), 16000)
    (ckpt / "reference.json").write_text(
        json.dumps({"text": "示例。", "source": "speaker_a: c0.wav"}), encoding="utf-8"
    )

    package = sharing.export_package(
        lora_library.resolve(lora_dir, "fish_s2/voice/checkpoints/step_0000040"), "v", "fishaudio/s2-pro"
    )
    import zipfile

    with zipfile.ZipFile(package) as zf:
        names = sorted(zf.namelist())
        assert json.loads(zf.read("v/reference.json")) == {"text": "示例。"}  # not where it came from
        assert b"old_pc" not in zf.read("v/lora_config.json") and "Research License" in zf.read("v/README.md").decode()
    assert names == [
        "v/README.md",
        "v/lora_config.json",
        "v/lora_weights.safetensors",
        "v/reference.json",
        "v/reference.wav",
    ]

    other = tmp_path / "other"
    assert sharing.import_package(other, package, "v.zip") == ("fish_s2/v", "fish_s2")
    (imported,) = lora_library.scan(other)
    assert (imported.engine, imported.rank, (imported.path / "reference.wav").is_file()) == ("fish_s2", 4, True)
