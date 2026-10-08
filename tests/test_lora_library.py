import pytest

from voxcpm_studio import lora_library

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
