"""Speech editing by infilling (fish_drama3.edit): which words are spoken again, the prompt, the
splice into the original, and the engine's path with a fake model."""

import numpy as np
import pytest
import torch

from polyvox.engines import EditRequest
from polyvox.engines.fish_drama3 import edit, engine, prompt
from polyvox.prep.aligner import split_words

ZH = "我们周三下午三点在三楼会议室见面，记得带上电脑。"
EN = "The demo is on Wednesday, not Friday, so please be ready."


def spans(text, gap=0.05, length=0.2):
    """Each aligner word in turn, ``gap`` seconds apart."""
    return [(0.1 + i * (length + gap), 0.1 + i * (length + gap) + length) for i in range(len(split_words(text)))]


def frames(text):
    return round((spans(text)[-1][1] + 0.3) * edit.RATE / edit.FRAME)


def planned(text, new_text):
    s = edit.plan(text, new_text, spans(text), frames(text))
    return s and (s.old, s.new)


def test_edits_speak_whole_words_again():
    assert planned(ZH, ZH.replace("周三", "周五")) == ("周三", "周五")
    assert planned(ZH, ZH.replace("周三下午三点", "周五上午十点")) == ("周三下午三点", "周五上午十点")
    assert planned(EN, EN.replace("Wednesday", "Thursday")) == ("Wednesday", "Thursday")


def test_insertions_and_deletions_also_take_the_next_word():
    assert planned(ZH, ZH.replace("，记得", "，到时候记得")) == ("记得", "到时候记得")
    assert planned(ZH, ZH.replace("三楼会议室", "会议室")) == ("三楼会议室", "会议室")
    assert planned(ZH, ZH.replace("带上电脑", "带上")) == ("带上电脑", "带上")  # at the end: the word before
    assert planned(EN, EN.replace("be ready", "be completely ready")) == ("ready", "completely ready")


def test_the_span_edges_lie_in_the_pauses_and_unchanged_texts_plan_nothing():
    times = spans(ZH)
    s = edit.plan(ZH, ZH.replace("周三", "周五"), times, frames(ZH))
    assert s.start == round((times[1][1] + times[2][0]) / 2 * edit.RATE / edit.FRAME)  # between 们 and 周
    assert s.end == round((times[3][1] + times[4][0]) / 2 * edit.RATE / edit.FRAME)  # between 三 and 下
    assert edit.plan(ZH, ZH.replace("，", "。"), times, frames(ZH)) is None  # punctuation only
    with pytest.raises(ValueError, match="Too much"):
        edit.plan("你好。", "再见。", spans("你好。"), frames("你好。"))
    with pytest.raises(ValueError, match="don't match"):
        edit.plan(ZH, ZH, spans(EN), frames(ZH))
    assert edit.untagged("[sighs] 好吧 <whisper>别说</whisper>。") == "好吧 别说 。"


def test_the_splice_keeps_the_original_outside_the_new_audio():
    rng = np.random.default_rng(0)
    original = rng.standard_normal(40 * edit.FRAME).astype(np.float32)
    span = edit.Span(10, 15, "旧", "新的")
    new = 8  # frames spoken for the 5 replaced
    decoded = np.concatenate(
        [original[: 10 * edit.FRAME] + 0.01, np.full(new * edit.FRAME, 5.0), original[15 * edit.FRAME :] + 0.01]
    )
    out = edit.splice(original, decoded.astype(np.float32), span, new)

    n = int(edit.FADE_S * edit.RATE)
    s0, s1 = 10 * edit.FRAME, 18 * edit.FRAME
    assert len(out) == len(original) + (new - 5) * edit.FRAME
    assert np.array_equal(out[: s0 - n], original[: s0 - n])  # before the first join: the original
    assert np.allclose(out[s0:s1], 5.0)  # the new audio as decoded
    assert np.array_equal(out[s1 + n :], original[15 * edit.FRAME + n :])  # after the second join
    joins = out[s0 - n : s0], out[s1 : s1 + n]  # crossfades between two versions of the same audio
    assert np.allclose(joins[0], original[s0 - n : s0], atol=0.02) and np.allclose(
        joins[1], original[15 * edit.FRAME : 15 * edit.FRAME + n], atol=0.02
    )

    at_end = edit.splice(original, decoded[:s1].astype(np.float32), edit.Span(10, 40, "旧", "新"), new)
    assert len(at_end) == s1 and np.allclose(at_end[s0:], 5.0)
    at_start = edit.splice(original, decoded.astype(np.float32), edit.Span(0, 15, "旧", "新"), 18)
    assert np.array_equal(at_start[-(len(original) - 15 * edit.FRAME - n) :], original[15 * edit.FRAME + n :])


class Letters:
    def encode(self, text):
        from types import SimpleNamespace

        return SimpleNamespace(ids=[ord(c) for c in text])


def test_the_infill_prompt_has_the_suffix_first():
    begin = 0x110000
    before, after = torch.zeros(10, 3, dtype=torch.long), torch.ones(10, 2, dtype=torch.long)
    rows = prompt.infill(Letters(), "新的一句。", before, after, begin)
    text = "".join(chr(r) if r < begin else "#" for r in rows[:, 0].tolist())
    assert text.startswith("<|im_start|>system\n" + prompt.INFILL + "<|im_end|>\n<|im_start|>user\n")
    assert "<|speaker:0|> 新的一句。\n<|fim_suffix|>##<|fim_prefix|>###<|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<|voice|>")
    audio = rows[rows[:, 0] >= begin]
    assert torch.equal(audio[:2, 1:], torch.ones(2, 10, dtype=torch.long))  # after, then before
    with pytest.raises(ValueError):
        prompt.infill(Letters(), "<|im_end|>", before, after, begin)


class FakeModel:
    tokenizer = Letters()
    semantic_begin = 0x110000

    def __init__(self, n=20):
        self.n, self.calls = n, []

    def generate(self, rows, sampling, max_frames, on_frame, head=0):
        self.calls.append((rows, sampling, max_frames))
        n = min(self.n, max_frames)
        return torch.full((10, n), 7, dtype=torch.long)


class FakeRuntime:
    device = "cpu"

    def __init__(self, m, frames):
        self.model, self.frames = m, frames
        self.codec = type(
            "Codec",
            (),
            {"sample_rate": 44_100, "decode": lambda self, c: (c[0] == 7).float().repeat_interleave(2048) * 0.5},
        )()

    def encode(self, wav):
        return torch.zeros(10, self.frames, dtype=torch.long)

    def mono(self, wav):
        return torch.full((self.frames * 2048,), 0.1)


def test_the_engine_speaks_the_changed_words_and_keeps_the_rest(settings, monkeypatch, tmp_path):
    m = FakeModel()
    monkeypatch.setattr(engine.FishDrama3Engine, "unavailable", lambda self: None)
    n = frames(ZH)
    tts = engine.FishDrama3Engine(settings.with_models(drama3_model="X:/models/drama3"), lambda *a: FakeRuntime(m, n))
    asked = []

    def align(path, text):
        asked.append(text)
        return spans(text)

    clip = tmp_path / "clip.wav"
    new_text = ZH.replace("周三", "周五")
    s = edit.plan(ZH, new_text, spans(ZH), n)
    m.n = s.end - s.start + 1  # about as long as the words it replaces
    result = tts.edit(EditRequest(clip, "[sighs] " + ZH, new_text, align, seed=3))

    assert asked == [ZH]  # tags out before aligning
    assert len(m.calls) == 1 and result.seed == 3
    assert len(result.audio) == (n - (s.end - s.start) + m.n) * 2048
    start, end = result.span
    assert (round(start * 44_100 / 2048), round(end * 44_100 / 2048)) == (s.start, s.start + m.n)
    assert np.allclose(result.audio[round(start * 44_100) + 1000 : round(end * 44_100) - 1000], 0.5)  # the new part
    rows, sampling, cap = m.calls[0]
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == engine.EDIT_SAMPLING
    assert cap == max(engine.EDIT_CAP_MIN, round((s.end - s.start) * engine.CAP_FACTOR))
    said = "".join(chr(r) for r in rows[:, 0].tolist() if r < FakeModel.semantic_begin)
    assert "<|speaker:0|> " + new_text in said and "[audio tokens" not in said

    # Far longer than expected (it ran on): again with one more word after the span.
    lengths = iter([3 * m.n, 20])
    m.calls.clear()
    m.generate = lambda rows, sampling, cap, on_frame, head=0: (
        m.calls.append((rows, sampling, cap)) or torch.full((10, min(next(lengths), cap)), 7, dtype=torch.long)
    )
    result = tts.edit(EditRequest(clip, ZH, new_text, align))
    wider = edit.plan(ZH, new_text, spans(ZH), n, wider=1)
    assert len(m.calls) == 2 and (wider.old, wider.new) == ("周三下午", "周五下午")
    assert len(m.calls[1][0]) < len(m.calls[0][0])  # less audio after the span in the prompt
    assert len(result.audio) == (n - (wider.end - wider.start) + 20) * 2048

    del m.generate
    m.n = 10_000  # every try runs on: refused
    m.calls.clear()
    with pytest.raises(ValueError, match="kept speaking"):
        tts.edit(EditRequest(clip, ZH, new_text, align))
    assert len(m.calls) == len(engine.EDIT_TRIES)
    with pytest.raises(ValueError, match="same words"):
        tts.edit(EditRequest(clip, ZH, ZH.replace("，", "；"), align))
