import json

import numpy as np
import soundfile as sf

from polyvox.data import record
from polyvox.prep import checks


def test_a_folder_without_a_record_is_all_clips(tmp_path):
    assert record.read_record(tmp_path) == record.Record()


def test_record_round_trip_and_step_settings(tmp_path):
    saved = record.Record(
        [record.Recording("X:/old_pc/talk.m4a", 600.0, 44100, sliced={"min_seconds": 2, "max_seconds": 15})],
        {"wavs/talk_1.wav": record.Origin("X:/old_pc/talk.m4a", 1.0, 5.0)},
        {"check": {"aligner": "qwen3"}},
    )
    record.write_record(tmp_path, saved)
    assert record.read_record(tmp_path) == saved

    # A step made of parts keeps the parts it did not run this time.
    record.remember(tmp_path, "tidy", {"fillers": True})
    record.remember(tmp_path, "tidy", {"normalize_lufs": -20.0})
    record.remember(tmp_path, "check", {"aligner": "ctc"})
    steps = record.read_record(tmp_path).steps
    assert steps == {"check": {"aligner": "ctc"}, "tidy": {"fillers": True, "normalize_lufs": -20.0}}


def test_trimming_a_clip_moves_its_origin(tmp_path):
    folder = tmp_path / "speaker_a"
    (folder / "wavs").mkdir(parents=True)
    sf.write(folder / "wavs" / "talk_1.wav", np.zeros(16000 * 4, dtype=np.float32), 16000)
    (folder / "train.jsonl").write_text(json.dumps({"audio": "wavs/talk_1.wav", "text": "示例"}) + "\n", "utf-8")
    record.write_record(
        folder, record.Record(origins={"wavs/talk_1.wav": record.Origin("X:/old_pc/talk.wav", 10.0, 14.0)})
    )

    result = checks.trim_clips(folder / "train.jsonl", None, lambda index, check: (0.5, 3.0))

    assert result == {"trimmed": 1, "skipped": 0}
    assert record.read_record(folder).origins["wavs/talk_1.wav"] == record.Origin("X:/old_pc/talk.wav", 10.5, 13.0)
