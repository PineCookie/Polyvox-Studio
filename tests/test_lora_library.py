import pytest

from polyvox import lora_library

from conftest import make_lora


def test_scan_finds_nested_checkpoints_newest_first(lora_dir):
    make_lora(lora_dir, "voice/checkpoints/step_0001000", r=64)
    make_lora(lora_dir, "voice/checkpoints/step_0002000", r=64)
    (lora_dir / "not_a_checkpoint").mkdir()

    ids = [c.id for c in lora_library.scan(lora_dir)]

    assert ids == ["voice/checkpoints/step_0002000", "voice/checkpoints/step_0001000"]


def test_scan_reads_config_and_base_model(lora_dir):
    make_lora(lora_dir, "a", r=64, base_model="models/VoxCPM2")

    (ckpt,) = lora_library.scan(lora_dir)

    assert ckpt.config.r == 64
    assert ckpt.base_model == "models/VoxCPM2"


def test_missing_or_broken_config_yields_none(lora_dir):
    make_lora(lora_dir, "no_config", r=None)
    (make_lora(lora_dir, "broken", r=None) / "lora_config.json").write_text("{not json")

    assert all(c.config is None for c in lora_library.scan(lora_dir))


def test_scan_missing_dir_is_empty(tmp_path):
    assert lora_library.scan(tmp_path / "absent") == []


@pytest.mark.parametrize("bad_id", ["../outside", "missing", "a/../../outside"])
def test_resolve_rejects_unknown_or_escaping_ids(lora_dir, bad_id):
    make_lora(lora_dir.parent, "outside")
    make_lora(lora_dir, "a")

    with pytest.raises(KeyError):
        lora_library.resolve(lora_dir, bad_id)


def test_resolve_returns_checkpoint(lora_dir):
    make_lora(lora_dir, "voice/step_1", r=16)

    assert lora_library.resolve(lora_dir, "voice/step_1").config.r == 16


def test_checkpoints_know_their_run_or_voice(lora_dir):
    make_lora(lora_dir, "voxcpm/speaker_a/checkpoints/step_0000100")
    make_lora(lora_dir, "legacy_b/checkpoints/step_0000100")  # from before the engine folders
    by_id = {c.id: c for c in lora_library.scan(lora_dir)}
    new, old = by_id["voxcpm/speaker_a/checkpoints/step_0000100"], by_id["legacy_b/checkpoints/step_0000100"]
    assert (new.voice_path, new.voice) == ("voxcpm/speaker_a", "speaker_a")
    assert (old.voice_path, old.voice) == ("legacy_b", "legacy_b")


def test_migrate_moves_runs_and_voices_into_engine_folders(lora_dir):
    import json

    import yaml

    # A run made here (absolute paths into its folder) and a webui run (relative ones).
    made = lora_dir / "made_here"
    (made / "checkpoints" / "step_0000010").mkdir(parents=True)
    config = {"engine": "dots_tts", "save_path": str(made / "checkpoints"), "train_manifest": str(made / "m.jsonl"),
              "pretrained_path": "X:/old_pc/models/dots", "elsewhere": str(lora_dir / "made_here_2" / "x")}  # fmt: skip
    (made / "train_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    webui = lora_dir / "webui_run"
    webui.mkdir()
    (webui / "train_config.yaml").write_text(
        yaml.safe_dump({"save_path": "lora\\webui_run\\checkpoints", "tensorboard": "lora/webui_run/logs"}),
        encoding="utf-8",
    )
    voice = lora_dir / "voice_c"  # an imported GPT-SoVITS voice
    voice.mkdir()
    (voice / "sovits_weights.pth").write_bytes(b"x")
    (voice / "lora_config.json").write_text(json.dumps({"engine": "gpt_sovits"}), encoding="utf-8")
    (lora_dir / "notes").mkdir()  # nothing to tell its engine by: stays
    (lora_dir / "gpt_sovits" / "voice_d").mkdir(parents=True)  # already in place
    (lora_dir / "voice_d").mkdir()
    (lora_dir / "voice_d" / "sovits_weights.pth").write_bytes(b"x")
    (lora_dir / "voice_d" / "lora_config.json").write_text(json.dumps({"engine": "gpt_sovits"}), encoding="utf-8")

    moved = lora_library.migrate(lora_dir)

    assert sorted(moved) == [
        ("made_here", "dots_tts/made_here"),
        ("voice_c", "gpt_sovits/voice_c"),
        ("webui_run", "voxcpm/webui_run"),
    ]
    new = lora_dir.resolve() / "dots_tts" / "made_here"
    config = yaml.safe_load((new / "train_config.yaml").read_text(encoding="utf-8"))
    assert config["save_path"] == str(new / "checkpoints") and config["train_manifest"] == str(new / "m.jsonl")
    assert config["pretrained_path"] == "X:/old_pc/models/dots"  # not inside the run: kept
    assert config["elsewhere"] == str(lora_dir / "made_here_2" / "x")  # only the run's own folder moves
    webui_config = yaml.safe_load((lora_dir / "voxcpm" / "webui_run" / "train_config.yaml").read_text(encoding="utf-8"))
    assert webui_config == {
        "save_path": "lora\\voxcpm\\webui_run\\checkpoints",
        "tensorboard": "lora/voxcpm/webui_run/logs",
    }
    assert (lora_dir / "notes").is_dir() and (lora_dir / "voice_d").is_dir()  # left (name taken)
    assert lora_library.migrate(lora_dir) == []  # once
