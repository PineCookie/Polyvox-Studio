"""Alignment checks on synthetic audio: tones stand in for speech."""

import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_aligner import tone
from test_importer import wait_job, write_list

from polyvox.data import manifests
from polyvox.prep import checks, importer, loudness, slicer
from polyvox.prep.aligner import Alignment, Word
from polyvox.prep.asr import load_audio
from polyvox.server import create_app


def aligned(start, end, score=0.9):
    return Alignment([Word("一", start, (start + end) / 2, score), Word("二", (start + end) / 2, end, score)], score)


def test_a_clean_clip_has_no_flags():
    wav = tone([(0.3, 2.0)], 2.4)

    check = checks.check_clip(wav, aligned(0.4, 1.8), "qwen3", [(0.3, 2.0)])

    assert check.flags == []
    assert (check.start, check.end) == pytest.approx((0.3, 2.0), abs=0.02)  # refined to the sound
    assert (check.lead, check.trail, check.duration) == pytest.approx((0.3, 0.4, 2.4), abs=0.02)


def test_flags_mismatch_extra_speech_cut_edges_and_silence():
    wav = tone([(0.0, 1.0), (1.8, 2.4)], 4.0)  # speech from the very start; a phrase the text lacks

    check = checks.check_clip(wav, aligned(0.0, 1.0, score=0.2), "qwen3", [(0.0, 1.0), (1.8, 2.4)])

    assert check.flags == ["mismatch", "edge_sound", "cut_edge", "long_silence"]
    assert check.unaligned == pytest.approx(0.6, abs=0.02)
    # Sound right after the words (no pause to cut in) is untranscribed speech instead.
    close = tone([(0.3, 1.0), (1.1, 1.6)], 1.8)
    assert checks.check_clip(close, aligned(0.3, 1.0), "qwen3", [(0.3, 1.0), (1.1, 1.6)]).flags == ["extra_speech"]
    # Thresholds are per aligner: the same score passes for CTC's scale.
    assert "mismatch" not in checks.check_clip(wav, aligned(0.0, 1.0, score=0.2), "ctc", [(0.0, 1.0)]).flags
    assert checks.check_clip(wav, Alignment([], 0.0), "ctc", []).flags[0] == "mismatch"


def test_a_noisy_background_is_not_mistaken_for_speech_at_the_edge():
    wav = tone([(0.4, 1.6)], 2.0)
    wav += np.random.default_rng(1).normal(0, 0.012, len(wav)).astype(np.float32)  # floor ~25 dB below the voice

    check = checks.check_clip(wav, aligned(0.5, 1.5), "qwen3", [(0.4, 1.6)])

    assert "cut_edge" not in check.flags
    assert (check.start, check.end) == pytest.approx((0.4, 1.6), abs=0.03)


def test_fillers_count_only_when_the_audio_lacks_them():
    words = [Word(c, 0.3 + k * 0.3, 0.6 + k * 0.3, 0.1 if c == "哈" else 0.9) for k, c in enumerate("一二哈哈三")]
    # 哈哈 (0.9-1.5 s) is there: unsure fillers are no mismatch and no flag.
    heard = tone([(0.3, 1.8)], 2.2)
    check = checks.check_clip(heard, Alignment(words, 0.1), "qwen3", [(0.3, 1.8)])
    assert check.flags == [] and check.score == pytest.approx(0.9)

    # 哈哈 written over silence: a soft flag, still no mismatch.
    missing = tone([(0.3, 0.9), (1.5, 1.8)], 2.2)
    assert checks.check_clip(missing, Alignment(words, 0.1), "qwen3", [(0.3, 0.9), (1.5, 1.8)]).flags == ["filler"]

    real_words_wrong = [Word(w.text, w.start, w.end, 0.1) for w in words]
    assert checks.check_clip(heard, Alignment(real_words_wrong, 0.1), "qwen3", [(0.3, 1.8)]).flags == ["mismatch"]


def test_trim_cuts_off_untranscribed_sound_a_pause_separates():
    wav = tone([(0.2, 0.6), (1.0, 2.0)], 2.6)  # a laugh, a pause, then the transcribed words
    check = checks.check_clip(wav, aligned(1.0, 2.0), "qwen3", [(0.2, 0.6), (1.0, 2.0)])
    assert "edge_sound" in check.flags and check.sound_before == pytest.approx((0.8, 0.4), abs=0.02)

    start, end = check.trim_range()
    after = check.trimmed(start, end)

    assert (start, end) == pytest.approx((0.85, 2.15), abs=0.02)
    assert after.flags == [] and after.unaligned == pytest.approx(0.0)
    # Sound running into the words without a pause is not cut off: it counts as part of the speech.
    joined = checks.check_clip(tone([(0.4, 2.0)], 2.4), aligned(1.0, 2.0), "qwen3", [(0.4, 2.0)])
    assert joined.sound_before is None and joined.start == pytest.approx(0.4, abs=0.02)
    assert joined.trim_range()[0] == pytest.approx(0.25, abs=0.02)


def test_a_text_starting_with_a_laugh_keeps_the_laugh():
    # The aligner squeezed 哈哈 against the words; the laugh itself is before them, a pause apart.
    wav = tone([(0.2, 0.6), (1.0, 2.0)], 2.4)
    words = [Word("哈", 1.0, 1.0, 0.2), Word("哈", 1.0, 1.0, 0.2), Word("一", 1.0, 1.5, 0.9), Word("二", 1.5, 2.0, 0.9)]

    check = checks.check_clip(wav, Alignment(words, 0.5), "qwen3", [(0.2, 0.6), (1.0, 2.0)])

    assert check.flags == []  # squeezed fillers are no evidence either way
    assert check.sound_before is None
    assert checks.squeezed_edge_filler(check)  # Trim all leaves it for review
    assert check.trim_range()[0] == pytest.approx(0.05, abs=0.02)  # 0.15 s before the laugh, not the words


def test_sound_vad_misses_is_not_silence_but_a_breath_is():
    # A laugh VAD passed over, a pause, then the words: edge sound, not long silence.
    laugh = tone([(0.3, 0.8), (1.5, 2.5)], 2.7)
    check = checks.check_clip(laugh, aligned(1.5, 2.5), "qwen3", [(1.5, 2.5)])
    assert check.flags == ["edge_sound"] and check.lead == pytest.approx(0.3, abs=0.02)

    # The same stretch 40 dB down (a breath): silence.
    breath = tone([(1.5, 2.5)], 2.7)
    t = np.arange(len(breath)) / 16000
    breath[(t >= 0.3) & (t < 0.8)] += 0.003 * np.sin(2 * np.pi * 220 * t[(t >= 0.3) & (t < 0.8)])
    check = checks.check_clip(breath, aligned(1.5, 2.5), "qwen3", [(1.5, 2.5)])
    assert check.flags == ["long_silence"] and check.lead == pytest.approx(1.5, abs=0.02)


def test_trim_keeps_padding_around_all_speech():
    wav = tone([(1.5, 3.0)], 5.0)
    check = checks.check_clip(wav, aligned(1.6, 2.9), "qwen3", [(1.5, 3.0)])
    assert "long_silence" in check.flags

    start, end = check.trim_range()
    after = check.trimmed(start, end)

    assert (start, end) == pytest.approx((1.35, 3.15), abs=0.02)
    assert (after.lead, after.trail, after.start) == pytest.approx((0.15, 0.15, 0.15), abs=0.02)
    assert [w[1] for w in after.words] == pytest.approx([w[1] - start for w in check.words], abs=0.002)
    assert "long_silence" not in after.flags
    assert after.trim_range() is None  # nothing left worth cutting


def test_trailing_silence_is_flagged_sooner_than_leading():
    def flags(lead, trail):
        wav = tone([(lead, lead + 1.0)], lead + 1.0 + trail)
        return checks.check_clip(wav, aligned(lead, lead + 1.0), "qwen3", [(lead, lead + 1.0)]).flags

    assert flags(0.8, 0.3) == []
    assert flags(0.3, 0.8) == ["long_silence"]
    assert flags(1.2, 0.3) == ["long_silence"]


def write_dataset(data_dir, texts):
    folder = data_dir / "speaker_a"
    (folder / "wavs").mkdir(parents=True)
    lines = []
    for i, text in enumerate(texts):
        sf.write(folder / "wavs" / f"{i}.wav", tone([(0.5, 1.5)], 2.0), 16000)
        lines.append(json.dumps({"audio": f"wavs/{i}.wav", "text": text}, ensure_ascii=False))
    (folder / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder / "train.jsonl"


def no_vad(wav):
    return []


def test_check_dataset_writes_results_the_review_shows(data_dir, align_service):
    manifest = write_dataset(data_dir, ["第一句", "错错错句", "第三句"])
    phases = []
    checker = checks.make_checker(lambda w, t: align_service.align(w, t, "qwen3"), "qwen3", no_vad)

    summary = checks.check_dataset(manifest, None, checker, "qwen3", load_audio, lambda *p: phases.append(p))

    assert summary == {
        "aligner": "qwen3",
        "checked": 3,
        "flagged": 1,
        "flags": {
            "singing": 0,
            "other_speaker": 0,
            "clipping": 0,
            "mismatch": 1,
            "asr_more": 0,
            "filler": 0,
            "edge_sound": 0,
            "extra_speech": 0,
            "cut_edge": 0,
            "long_silence": 0,
        },
    }
    assert phases[-1] == ("aligning", 3, 3)
    page = manifests.read_rows(manifest, flag="any")
    assert [(r.file, r.flags) for r in page.rows] == [("wavs/1.wav", ["mismatch"])]

    # A text ASR made from the clip itself: scored, but not flagged as a mismatch.
    (manifest.parent / manifests.ASR_FILE).write_text('{"wavs/1.wav": {"language": "Chinese"}}', encoding="utf-8")
    assert checks.check_dataset(manifest, None, checker, "qwen3", load_audio)["flagged"] == 0
    (row,) = [r for r in manifests.read_rows(manifest).rows if r.file == "wavs/1.wav"]
    assert row.flags == [] and row.score is not None and row.score < checks.MISMATCH_BELOW["qwen3"]
    check = checks.check_one(manifest, None, 1, checker, "qwen3", load_audio)
    assert "mismatch" not in check.flags


def test_clipping_is_flat_tops_at_the_peak():
    t = np.arange(16000) / 16000
    loud = np.sin(2 * np.pi * 220 * t)
    assert loudness.clipped_runs(0.9 * loud) == 0
    clipped = np.clip(1.5 * loud, -1, 1)
    assert loudness.clipped_runs(clipped) == 440  # every half cycle
    assert loudness.clipped_runs(0.5 * clipped) == 440  # a gain after clipping keeps the tops flat
    assert loudness.clipped_runs(0.1 * clipped) == 0  # too quiet to judge
    # A low voice's peaks span a few samples, but no flat run at 16-bit precision.
    low = np.round(0.3 * np.sin(2 * np.pi * 60 * np.arange(48000) / 48000 + 0.3) * 32767) / 32767
    assert loudness.clipped_runs(low) == 0


def test_the_voice_leaves_other_speakers_out():
    rng = np.random.default_rng(0)

    def near(direction, n):
        v = np.asarray(direction, float) + rng.normal(0, 0.1, (n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    vectors = np.concatenate([near([1, 0, 0], 17), near([0, 1, 0], 3)])
    voice = checks.voice_of(vectors)
    assert voice @ [1, 0, 0] > 0.99
    assert checks.voice_of(vectors[:5]) is None  # too few clips
    # Short clips need a lower score to count as another voice.
    assert checks.other_speaker(0.52, 3.0) and not checks.other_speaker(0.52, 2.0)
    assert checks.other_speaker(0.25, 1.0) and not checks.other_speaker(0.4, 1.0)


def test_check_dataset_flags_other_speakers_and_clipping(data_dir, align_service):
    folder = data_dir / "speaker_a"
    (folder / "wavs").mkdir(parents=True)
    lines = []
    for i in range(12):
        wav = tone([(0.5, 2.7 if i == 3 else 1.5)], 3.0 if i == 3 else 2.0)  # clip 3: "another voice"
        if i == 7:
            wav = np.clip(5 * wav, -1, 1)
        sf.write(folder / "wavs" / f"{i}.wav", wav, 16000)
        lines.append(json.dumps({"audio": f"wavs/{i}.wav", "text": "第一句"}, ensure_ascii=False))
    manifest = folder / "train.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def embed(wavs):  # stands in for CAM++: the 3 s clip's voice points elsewhere
        return np.array([[0.0, 1.0] if len(w) > 2.5 * 16000 else [1.0, 0.0] for w in wavs])

    checker = checks.make_checker(lambda w, t: align_service.align(w, t, "qwen3"), "qwen3", no_vad)
    summary = checks.check_dataset(manifest, None, checker, "qwen3", load_audio, embed=embed)

    assert summary["flags"]["other_speaker"] == 1 and summary["flags"]["clipping"] == 1
    rows = {r.file: r.flags for r in manifests.read_rows(manifest, flag="any").rows}
    assert rows == {"wavs/3.wav": ["other_speaker"], "wavs/7.wav": ["clipping"]}
    # Worst first by the voice's similarity, for that flag.
    page = manifests.read_rows(manifest, flag="other_speaker", worst_first=True)
    assert [r.file for r in page.rows] == ["wavs/3.wav"]
    # One clip checked again is scored against the saved voice.
    check = checks.check_one(manifest, None, 3, checker, "qwen3", load_audio, embed=embed)
    assert check.speaker == pytest.approx(0.0) and "other_speaker" in check.flags
    assert checks.check_one(manifest, None, 0, checker, "qwen3", load_audio, embed=embed).speaker == pytest.approx(1)


def test_import_checks_and_trims_the_copies(tmp_path, data_dir, align_service):
    src = tmp_path / "recordings"
    src.mkdir()
    sf.write(src / "long_pause.flac", tone([(2.0, 3.0)], 4.5), 16000)  # trimmed, rewritten as WAV
    sf.write(src / "tight.wav", tone([(0.1, 0.9)], 1.0), 16000)
    (src / "long_pause.lab").write_text("第一句", encoding="utf-8")
    (src / "tight.lab").write_text("第二句", encoding="utf-8")
    checker = checks.make_checker(lambda w, t: align_service.align(w, t, "ctc"), "ctc", no_vad)

    result = importer.import_dataset(str(src), None, "trimmed", data_dir, checker=checker, aligner="ctc", trim=True)

    assert (result.imported, result.trimmed) == (2, 1)
    assert result.checks["flagged"] == 0  # the long silence was cut away
    target = data_dir / "trimmed"
    assert sorted(p.name for p in (target / "wavs").iterdir()) == ["long_pause.wav", "tight.wav"]
    assert sf.info(str(target / "wavs" / "long_pause.wav")).duration == pytest.approx(1.3, abs=0.03)
    assert sf.info(str(target / "wavs" / "tight.wav")).duration == pytest.approx(1.0)
    saved = json.loads((target / manifests.ALIGN_FILE).read_text(encoding="utf-8"))
    assert saved["aligner"] == "ctc" and saved["clips"]["wavs/long_pause.wav"]["lead"] == pytest.approx(0.15, abs=0.02)
    assert (src / "long_pause.flac").is_file()  # the source is untouched

    with pytest.raises(ValueError, match="needs copying and checking"):
        importer.import_dataset(str(src), None, "x", data_dir, trim=True)


def laugh_then_words(data_dir, flags_of_second=("edge_sound",)):
    """A dataset of two clips with a laugh before the words, checked as such
    (the second also with ``flags_of_second``)."""
    folder = data_dir / "speaker_a"
    (folder / "wavs").mkdir(parents=True)
    wav = tone([(0.2, 0.6), (1.0, 2.0)], 2.4)
    check = checks.check_clip(wav, aligned(1.0, 2.0), "qwen3", [(0.2, 0.6), (1.0, 2.0)])
    stereo = np.stack([wav, wav], axis=1)
    sf.write(folder / "wavs" / "0.wav", stereo, 16000, subtype="PCM_24")
    sf.write(folder / "wavs" / "1.flac", wav, 16000)
    texts = ["第一句", "第二句"]
    (folder / "train.jsonl").write_text(
        "".join(
            json.dumps({"audio": f"wavs/{f}", "text": t}) + "\n"
            for f, t in zip(["0.wav", "1.flac"], texts, strict=True)
        ),
        encoding="utf-8",
    )
    second = replace(check, flags=sorted(set(check.flags) | set(flags_of_second)))
    checks.write_checks(folder, "qwen3", {"wavs/0.wav": (texts[0], check), "wavs/1.flac": (texts[1], second)})
    return folder / "train.jsonl"


def test_trim_in_place_keeps_the_original_and_updates_the_check(data_dir):
    manifest = laugh_then_words(data_dir)
    folder = manifest.parent

    result = checks.trim_clips(manifest, None, lambda i, c: c.trim_range() if i == 0 else None)

    assert result == {"trimmed": 1, "skipped": 0}
    info = sf.info(str(folder / "wavs" / "0.wav"))
    assert (info.channels, info.subtype) == (2, "PCM_24")  # written back as it was
    assert info.duration == pytest.approx(1.3, abs=0.03)  # the laugh and the silence are gone
    assert sf.info(str(folder / checks.ORIGINALS_DIR / "wavs" / "0.wav")).duration == pytest.approx(2.4)
    saved = json.loads((folder / manifests.ALIGN_FILE).read_text(encoding="utf-8"))["clips"]
    assert saved["wavs/0.wav"]["flags"] == [] and saved["wavs/0.wav"]["duration"] == pytest.approx(1.3, abs=0.03)
    page = manifests.read_rows(manifest)
    assert page.rows[0].seconds == pytest.approx(1.3, abs=0.03)  # the cached duration follows the file
    assert sf.info(str(folder / "wavs" / "1.flac")).duration == pytest.approx(2.4)  # not picked

    # Trimming again keeps the first original.
    checks.trim_clips(manifest, None, lambda i, c: c.trim_range())
    assert sf.info(str(folder / checks.ORIGINALS_DIR / "wavs" / "0.wav")).duration == pytest.approx(2.4)
    assert sf.info(str(folder / "wavs" / "1.flac")).duration == pytest.approx(1.3, abs=0.03)


def test_trim_flagged_leaves_doubtful_clips_for_review(data_dir):
    manifest = laugh_then_words(data_dir, flags_of_second=("edge_sound", "asr_more"))

    assert checks.trim_flagged(manifest, None, "edge_sound") == {"trimmed": 1, "skipped": 0, "review": 1}
    assert sf.info(str(manifest.parent / "wavs" / "1.flac")).duration == pytest.approx(2.4)


def test_trim_and_hear_endpoints(service, training, asr_service, align_service, data_dir):
    laugh_then_words(data_dir)
    client = TestClient(create_app(service.settings, service, training, asr_service, align=align_service))
    ds = "speaker_a/train.jsonl"

    clip = client.get("/api/dataset-clip", params={"id": ds, "index": 0}).json()
    assert clip["flags"] == ["edge_sound"] and clip["sound_before"] == pytest.approx(0.8, abs=0.02)
    assert clip["sound_after"] is None and clip["trim"] == pytest.approx([0.85, 2.15], abs=0.02)
    rows = client.get("/api/dataset-rows", params={"id": ds}).json()["rows"]
    assert rows[0]["trim"] == pytest.approx([0.85, 2.15], abs=0.02)
    heard = client.post("/api/dataset-clip/hear", json={"id": ds, "index": 0, "start": 0, "end": 0.8}).json()
    assert heard == {"text": f"clip of {int(0.8 * 16000)} samples"}

    trimmed = client.post("/api/dataset-clip/trim", json={"id": ds, "index": 0}).json()
    assert trimmed["flags"] == [] and trimmed["trim"] is None and trimmed["seconds"] == pytest.approx(1.3, abs=0.03)
    assert client.post("/api/dataset-clip/trim", json={"id": ds, "index": 0}).status_code == 409
    # The fixed clip left the flag's list, but stays in it while kept in view.
    flagged = {"id": ds, "flag": "edge_sound"}
    assert [r["index"] for r in client.get("/api/dataset-rows", params=flagged).json()["rows"]] == [1]
    kept = client.get("/api/dataset-rows", params={**flagged, "keep": [0]}).json()["rows"]
    assert [(r["index"], r["flags"]) for r in kept] == [(0, []), (1, ["edge_sound"])]

    # A range of one's own, on a clip trimmed before: cut further.
    ranged = client.post("/api/dataset-clip/trim", json={"id": ds, "index": 0, "start": 0.1, "end": 1.2}).json()
    assert ranged["seconds"] == pytest.approx(1.1, abs=0.01)
    assert sf.info(str(data_dir / "speaker_a" / checks.ORIGINALS_DIR / "wavs" / "0.wav")).duration == pytest.approx(2.4)
    too_short = {"id": ds, "index": 0, "start": 0.1, "end": 0.2}
    assert client.post("/api/dataset-clip/trim", json=too_short).status_code == 422

    # Check one clip again on request, also asking ASR (the dataset's checks didn't).
    rechecked = client.post("/api/dataset-clip/check", json={"id": ds, "index": 0, "compare_asr": True}).json()
    assert rechecked["heard"]["text"].startswith("clip of") and rechecked["words"]
    assert client.post("/api/dataset-clip/check", json={"id": ds, "index": 9}).status_code == 404

    result = client.post("/api/datasets/trim-flagged", json={"id": ds, "flag": "edge_sound"}).json()
    assert result == {"trimmed": 1, "skipped": 0, "review": 0}
    assert client.post("/api/datasets/trim-flagged", json={"id": ds, "flag": "mismatch"}).status_code == 422


def test_check_job_endpoint(service, training, asr_service, align_service, align_loader, data_dir, monkeypatch):
    monkeypatch.setattr(slicer, "silero_vad_many", lambda wavs: [[] for _ in wavs])
    write_dataset(data_dir, ["第一句", "错错错句"])
    client = TestClient(create_app(service.settings, service, training, asr_service, align=align_service))

    resp = client.post("/api/datasets/check", json={"id": "speaker_a/train.jsonl", "aligner": "qwen3"})
    assert resp.status_code == 202 and resp.json()["kind"] == "check"
    job = wait_job(client, resp.json()["id"])
    assert job["state"] == "done" and job["result"]["flagged"] == 1, job
    assert align_loader.loaded == ["qwen3"]

    page = client.get("/api/dataset-rows", params={"id": "speaker_a/train.jsonl", "order": "worst"}).json()
    assert page["checks"]["aligner"] == "qwen3" and page["rows"][0]["flags"] == ["mismatch"]
    flagged = client.get("/api/dataset-rows", params={"id": "speaker_a/train.jsonl", "flag": "any"}).json()
    assert flagged["total"] == 1

    clip = client.get("/api/dataset-clip", params={"id": "speaker_a/train.jsonl", "index": 1}).json()
    assert [(w["text"], w["weak"]) for w in clip["words"]] == [
        ("错", True),
        ("错", True),
        ("错", True),
        ("句", False),
    ]
    assert clip["aligner"] == "qwen3" and len(clip["peaks"]) == manifests.DETAIL_BINS
    assert 0.5 <= clip["words"][0]["start"] < clip["words"][-1]["end"] <= 1.5  # the tone's span
    assert client.get("/api/dataset-clip", params={"id": "speaker_a/train.jsonl", "index": 9}).status_code == 404
    assert client.get("/api/status").json()["aligner_loaded"] is True
    assert client.post("/api/model/unload").json()["aligner_loaded"] is False

    assert client.post("/api/datasets/check", json={"id": "nope/train.jsonl"}).status_code == 404
    assert client.post("/api/datasets/check", json={"id": "speaker_a/train.jsonl", "aligner": "x"}).status_code == 422


def test_import_endpoint_with_checks(service, training, asr_service, align_service, data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(slicer, "silero_vad_many", lambda wavs: [[] for _ in wavs])
    src = tmp_path / "speaker_b"
    src.mkdir()
    sf.write(src / "a.wav", tone([(0.3, 1.2)], 1.5), 16000)
    lst = write_list(tmp_path / "speaker_b.list", [f"{src / 'a.wav'}|speaker_b|ZH|示例"])
    client = TestClient(create_app(service.settings, service, training, asr_service, align=align_service))

    body = {"source": str(lst), "name": "speaker_b", "check": True, "trim": True}
    job = wait_job(client, client.post("/api/datasets/import", json=body).json()["id"])

    assert job["state"] == "done", job
    assert job["result"]["checks"]["checked"] == 1 and job["result"]["trimmed"] == 1  # 0.3 s silent edges
    assert (data_dir / "speaker_b" / manifests.ALIGN_FILE).is_file()
    no_check = {**body, "name": "other", "check": False}
    assert client.post("/api/datasets/import", json=no_check).status_code == 422
