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

    files = slicer.slice_recording(src, tmp_path / "out", OPTS, vad)

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
    assert ("slicing", 1, 1) in phases


def test_slicing_needs_copy_and_transcription(tmp_path, data_dir):
    with pytest.raises(ValueError, match="needs copying and transcription"):
        importer.import_dataset(str(tmp_path), None, "x", data_dir, slicing=OPTS)
