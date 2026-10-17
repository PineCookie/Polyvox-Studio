"""Fish Audio S2-Pro: our model / prompt / decoding on a tiny random model (no weights needed)."""

import dataclasses

import pytest
import torch

from polyvox.engines.fish_s2 import codec, decode, generate, model, prompt
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
    assert text_parts("Hello there. How are you?") == ["Hello there.How are you?"]
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


def test_the_engine_needs_its_model_and_a_transcript(settings):
    engine = FishS2Engine(settings)
    assert "download" in engine.unavailable()
    assert engine.capabilities.reference_inputs[0].transcript == "required" and not engine.capabilities.training
    with pytest.raises(ValueError):
        engine.training_config(None, None, None, None)
