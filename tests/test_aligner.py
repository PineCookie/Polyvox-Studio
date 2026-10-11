import numpy as np
import pytest
import torch

from polyvox.prep import aligner
from polyvox.prep.aligner import AlignService


def tone(spans, seconds, rate=16000):
    """Silence (a faint noise floor) with a loud tone over each (start, end) span."""
    wav = np.random.default_rng(0).normal(0, 1e-4, int(seconds * rate)).astype(np.float32)
    t = np.arange(len(wav)) / rate
    for start, end in spans:
        on = (t >= start) & (t < end)
        wav[on] += 0.3 * np.sin(2 * np.pi * 220 * t[on])
    return wav


def test_split_words_keeps_cjk_characters_apart_and_drops_punctuation():
    assert aligner.split_words("你好，Hello world！かな2024年 it's") == [
        "你",
        "好",
        "Hello",
        "world",
        "か",
        "な",
        "2024",
        "年",
        "it's",
    ]
    assert aligner.split_words("……！") == []


def test_worst_window_finds_a_bad_stretch_the_mean_hides():
    confidences = [0.9] * 20 + [0.05, 0.05, 0.05] + [0.9] * 20

    assert np.mean(confidences) > 0.8
    assert aligner.worst_window(confidences, 3) == pytest.approx(0.05)
    assert aligner.worst_window([0.5, 0.02], 3, geometric=True) == pytest.approx(0.1)  # sqrt(0.5 * 0.02)
    assert aligner.worst_window([], 3) == 0.0


def emission(frames, path):
    """Log probabilities where frame t is almost surely class path[t] (0 = blank)."""
    probs = np.full((frames, 4), 1e-4)
    for t, c in enumerate(path):
        probs[t, c] = 1.0
    return torch.from_numpy(probs / probs.sum(1, keepdims=True)).log().float()


def test_ctc_alignment_maps_token_spans_to_words():
    em = emission(20, [0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0])

    result = aligner.ctc_alignment(em, [[1, 2], [], [3]], ["ab", "?", "c"], blank=0, frame_s=0.02)

    (ab, unknown, c) = result.words
    assert (ab.start, ab.end) == pytest.approx((0.04, 0.18))
    assert not unknown.aligned and unknown.start == unknown.end == ab.end  # no token: placed after "ab"
    assert (c.start, c.end) == pytest.approx((0.28, 0.32))
    assert result.span == pytest.approx((0.04, 0.32)) and result.score > 0.9


def test_ctc_alignment_of_text_too_long_for_the_audio_fails_softly():
    result = aligner.ctc_alignment(emission(3, [0, 1, 0]), [[1, 2, 3, 1, 2]], ["abcab"], blank=0, frame_s=0.02)

    assert result.score == 0.0 and result.span is None


def test_refine_edges_snaps_to_where_the_sound_starts_and_stops():
    wav = tone([(1.0, 2.0)], 3.0)

    # Inside the sound (CTC's late start, early end): move out.
    assert aligner.refine_edges(wav, 1.12, 1.68) == pytest.approx((1.0, 2.0), abs=0.02)
    # In the silence around it (a first word "starting" at 0): move in.
    assert aligner.refine_edges(wav, 0.0, 2.5) == pytest.approx((1.0, 2.0), abs=0.02)
    # ... but never past the first word's end / the last word's start.
    assert aligner.refine_edges(wav, 0.0, 2.5, inner=(0.4, 2.8)) == pytest.approx((0.0, 2.5), abs=0.02)
    # A short dip does not stop the sound.
    dipped = tone([(1.0, 1.5), (1.54, 2.0)], 3.0)
    assert aligner.refine_edges(dipped, 1.0, 1.3)[1] == pytest.approx(2.0, abs=0.02)


def test_service_loads_one_backend_at_a_time_and_keeps_order(align_service, align_loader):
    short, long = tone([(0.2, 0.8)], 1.0), tone([(0.5, 2.5)], 3.0)

    results = align_service.align([long, short, short], ["一二三", "……", "四"], "qwen3")

    assert [len(r.words) for r in results] == [3, 0, 1]
    assert results[1].score == 0.0  # nothing alignable: not sent to the model
    assert align_loader.backends[0].calls == [2]
    align_service.align([short], ["四"], "qwen3")
    align_service.align([short], ["四"], "ctc")
    assert align_loader.loaded == ["qwen3", "ctc"] and align_service.loaded
    align_service.unload()
    assert not align_service.loaded
    with pytest.raises(ValueError, match="Unknown aligner"):
        align_service.align([short], ["四"], "mms")


def test_service_uses_the_configured_model_per_backend(settings, align_loader):
    seen = []
    service = AlignService(
        settings, loader=lambda kind, model, device: seen.append(model) or align_loader(kind, model, device)
    )

    service.align([tone([(0.2, 0.8)], 1.0)], ["四"], "ctc")

    assert seen == [settings.ctc_align_model]
