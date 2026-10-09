import json

import numpy as np
import pytest
import soundfile as sf

from voxcpm_studio import importer, slicer
from voxcpm_studio.slicer import SliceOptions

OPTS = SliceOptions(min_seconds=2.0, max_seconds=10.0, target_seconds=6.0, max_inner_gap=1.0, pad_seconds=0.1)


def durations(clips):
    return [round(e - s, 2) for s, e in clips]


def test_plan_cuts_at_the_long_pause_rather_than_the_short_ones():
    # Two "sentences" of three phrases each; short pauses inside, a 0.8 s pause between.
    segments = [(0.0, 1.5), (1.7, 3.0), (3.2, 4.5), (5.3, 6.8), (7.0, 8.3), (8.5, 9.8)]

    clips = slicer.plan_clips(segments, OPTS)

    assert clips == [(0.0, 4.5), (5.3, 9.8)]


def test_plan_respects_max_length_and_forces_cuts_at_long_silences():
    phrases = [(i * 2.0, i * 2.0 + 1.8) for i in range(10)]  # 20 s of speech, 0.2 s pauses
    clips = slicer.plan_clips(phrases, OPTS)
    assert all(e - s <= OPTS.max_seconds for s, e in clips)
    assert clips[0][0] == 0.0 and clips[-1][1] == phrases[-1][1]

    # A 3 s silence must end a clip even though both sides are short.
    clips = slicer.plan_clips([(0.0, 2.5), (5.5, 8.0)], OPTS)
    assert clips == [(0.0, 2.5), (5.5, 8.0)]


def test_split_long_cuts_at_the_quietest_point():
    rate = slicer.VAD_RATE
    wav = np.random.default_rng(0).normal(0, 0.3, 30 * rate).astype(np.float32)
    wav[int(13.0 * rate) : int(13.2 * rate)] *= 0.01  # a breath, no pause long enough for VAD

    parts = slicer.split_long([(0.0, 30.0)], wav, max_seconds=20.0)

    assert len(parts) == 2 and 13.0 <= parts[0][1] <= 13.2 and parts[0][1] == parts[1][0]


def test_pad_stops_halfway_to_the_neighbor_and_at_the_edges():
    assert slicer.pad_clips([(0.05, 2.0), (2.1, 4.0)], pad=0.2, duration=4.1) == [(0.0, 2.05), (2.05, 4.1)]


def test_find_clips_drops_what_stays_too_short():
    vad = lambda wav: [(0.5, 4.0), (9.0, 9.6), (15.0, 19.0)]  # 0.6 s "uh" alone in silence

    clips = slicer.find_clips(np.zeros(20 * slicer.VAD_RATE, dtype=np.float32), OPTS, vad)

    assert durations(clips) == [3.7, 4.2]


def test_slice_recording_writes_clips_at_the_source_rate(tmp_path):
    src = tmp_path / "long.wav"
    sf.write(src, np.zeros((44100 * 20, 2), dtype=np.float32), 44100)  # stereo, 44.1 kHz
    vad = lambda wav: [(1.0, 5.0), (7.0, 12.0)]

    files = [s.path for s in slicer.slice_recording(src, tmp_path / "out", OPTS, vad)]

    assert [f.name for f in files] == ["long_00000900_00005100.wav", "long_00006900_00012100.wav"]
    info = sf.info(str(files[0]))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 1, "PCM_16")
    assert info.duration == pytest.approx(4.2, abs=0.01)


def test_import_slices_long_recordings_then_transcribes(tmp_path, data_dir):
    folder = tmp_path / "recordings"
    folder.mkdir()
    sf.write(folder / "talk.wav", np.zeros(32000 * 40, dtype=np.float32), 32000)  # 40 s: sliced
    sf.write(folder / "short.wav", np.zeros(32000 * 3, dtype=np.float32), 32000)  # a clip already
    (folder / "short.lab").write_text("已有文本", encoding="utf-8")
    vad = lambda wav: [(1.0, 6.0), (8.0, 13.0), (20.0, 26.0)] if len(wav) > 20 * 16000 else [(0.2, 2.8)]
    phases = []

    preview = importer.preview_import(str(folder), None, data_dir)
    assert preview.long_recordings == 1

    result = importer.import_dataset(
        str(folder), None, "sliced", data_dir,
        transcriber=lambda paths: [(f"asr {p.stem[-8:]}", "Chinese") for p in paths],
        progress=lambda *p: phases.append(p), slicing=OPTS, vad=vad,
    )  # fmt: skip

    assert (result.imported, result.sliced, result.transcribed) == (4, 1, 3)
    entries = [
        json.loads(line) for line in (data_dir / "sliced" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert entries[0] == {"audio": "wavs/short.wav", "text": "已有文本"}
    assert [e["audio"] for e in entries[1:]] == [
        "wavs/talk_00000900_00006100.wav",
        "wavs/talk_00007900_00013100.wav",
        "wavs/talk_00019900_00026100.wav",
    ]
    assert all((data_dir / "sliced" / e["audio"]).is_file() for e in entries)
    assert not (data_dir / "sliced" / "_slices").exists()
    assert ("slicing", 40, 40) in phases  # seconds of recordings sliced


RATE = 16000


def bursts(spans, seconds, rate=RATE):
    """Silence with a loud tone per (start, end): one "word" each."""
    wav = np.zeros(int(seconds * rate), dtype=np.float32)
    t = np.arange(len(wav)) / rate
    for s, e in spans:
        on = (t >= s) & (t < e)
        wav[on] = 0.3 * np.sin(2 * np.pi * 220 * t[on])
    return wav


def loud_runs(wav, rate=RATE):
    """(start, end) of the loud stretches (10 ms frames): where the fake aligner puts words."""
    frame = rate // 100
    n = len(wav) // frame
    loud = np.abs(wav[: n * frame]).reshape(n, frame).max(axis=1) > 0.01
    runs, start = [], None
    for k, on in enumerate(list(loud) + [False]):
        if on and start is None:
            start = k
        elif not on and start is not None:
            runs.append((start / 100, k / 100))
            start = None
    return runs


def fake_guide(texts_by_words: dict[int, str]):
    """ASR that "hears" the text listed for a chunk's number of bursts; an
    aligner that puts word k on burst k."""
    from voxcpm_studio.aligner import Alignment, Word, split_words

    calls = {"transcribe": 0, "align": 0}

    def transcribe(wavs):
        calls["transcribe"] += 1
        return [(texts_by_words.get(len(loud_runs(w)), ""), "Chinese") for w in wavs]

    def align(wavs, texts):
        calls["align"] += 1
        out = []
        for w, t in zip(wavs, texts, strict=True):
            runs = loud_runs(w)
            out.append(Alignment([Word(x, s, e, 0.9) for x, (s, e) in zip(split_words(t), runs, strict=True)], 0.9))
        return out

    return slicer.Guide(transcribe=transcribe, align=align), calls


def test_word_spans_keep_offsets_into_the_text():
    from voxcpm_studio.aligner import word_spans

    text = "你好，world！再见。"
    assert [text[s:e] for s, e in word_spans(text)] == ["你", "好", "world", "再", "见"]


def test_word_plan_cuts_at_sentence_ends_even_without_a_longer_pause():
    # Ten 0.5 s words, 0.2 s apart; a sentence ends after the fifth.
    times = [(k * 0.7, k * 0.7 + 0.5) for k in range(10)]
    text_between = ["，" if k == 2 else "。" if k == 4 else "" for k in range(9)]
    breaks = [slicer._break_score(t, 0.2) for t in text_between]
    opts = SliceOptions(min_seconds=1.0, max_seconds=6.0, target_seconds=3.5, pad_seconds=0.1)

    assert slicer.plan_word_clips(times, breaks, opts) == [(0, 4), (5, 9)]


def test_guided_slicing_cuts_between_words_and_splits_the_text(tmp_path):
    # Two sentences of four words; the comma pause inside the second (0.4 s) is
    # longer than the one between them (0.2 s), yet the sentence end wins.
    words = [(1.0, 1.4), (1.6, 2.0), (2.2, 2.6), (2.8, 3.2), (3.4, 3.8), (4.0, 4.4), (4.8, 5.2), (5.4, 5.8)]
    src = tmp_path / "talk.wav"
    sf.write(src, bursts(words, 8.0, 44100), 44100)
    guide, calls = fake_guide({8: "一二三四。五六，七八。"})
    vad = lambda wav: [(1.0, 4.4), (4.8, 5.8)]
    opts = SliceOptions(min_seconds=2.0, max_seconds=4.0, target_seconds=2.5, max_inner_gap=1.0, pad_seconds=0.15)

    clips = slicer.slice_recording(src, tmp_path / "out", opts, vad, guide)

    assert [c.text for c in clips] == ["一二三四。", "五六，七八。"]
    assert all(c.language == "Chinese" for c in clips)
    first = sf.info(str(clips[0].path))
    assert first.samplerate == 44100
    # 0.15 s before the first word; cut in the middle of the 0.2 s pause, never into a word.
    assert first.duration == pytest.approx(3.3 - 0.85, abs=0.02)
    (start_ms, end_ms), (next_ms, _) = (map(int, c.path.stem.split("_")[1:]) for c in clips)
    assert start_ms == 850 and end_ms == next_ms == pytest.approx(3300, abs=10)
    assert calls == {"transcribe": 1, "align": 1}


def test_windows_end_in_pauses_and_cover_the_recording(monkeypatch):
    monkeypatch.setattr(slicer, "WINDOW_SLACK_S", 2.0)

    class Fake:
        duration = 25.0

        def read(self, start, end, rate=None):
            return np.zeros(int((min(end, 25.0) - start) * RATE), dtype=np.float32)

    # Speech everywhere except pauses at 9.0-9.6 and 19.5-20.5 (absolute seconds).
    pauses = [(9.0, 9.6), (19.5, 20.5)]

    def vad_from(t0):
        def vad(wav):
            end = t0 + len(wav) / RATE
            cuts = [p for p in pauses if t0 < p[0] < end]
            points = [t0] + [x for p in cuts for x in p] + [end]
            return [(points[i] - t0, points[i + 1] - t0) for i in range(0, len(points), 2)]

        return vad

    starts, t0 = [], 0.0
    for start, wav, _ in slicer.windows(Fake(), lambda w: vad_from(starts[-1] if starts else 0.0)(w), length=10.0):
        starts.append(start)
        t0 = start + len(wav) / RATE
        starts[-1] = t0  # the next window's vad offset
    assert t0 == pytest.approx(25.0)


def test_source_reads_ranges_through_either_backend(tmp_path):
    path = tmp_path / "x.wav"
    sf.write(path, bursts([(1.0, 2.0)], 3.0, 44100), 44100)

    src = slicer.Source(path)
    assert (src.duration, src.rate) == (pytest.approx(3.0), 44100)
    part = src.read(0.5, 1.5, 16000)
    assert len(part) == 16000 and np.abs(part[:7000]).max() < 0.01 < np.abs(part[9000:]).max()

    src._sf = False  # as for m4a / mp3: decoded by torchcodec
    decoded = src.read(0.5, 1.5, 16000)
    assert abs(len(decoded) - 16000) <= 16 and np.abs(decoded[9000:15000]).max() > 0.1


def test_slicing_needs_copy_and_transcription(tmp_path, data_dir):
    with pytest.raises(ValueError, match="needs copying and transcription"):
        importer.import_dataset(str(tmp_path), None, "x", data_dir, slicing=OPTS)
