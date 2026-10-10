import io
import json
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load, save_file

from voxcpm_studio import lora_library, sharing
from voxcpm_studio.api import create_app

from conftest import make_lora

LOCAL_BASE = r"C:\Users\someone\models\openbmb__VoxCPM2"


def make_checkpoint(lora_dir, rel="voice/checkpoints/step_0005000", r=32):
    ckpt = make_lora(lora_dir, rel, r=r, base_model=LOCAL_BASE)
    save_file(
        {
            "base_lm.layers.0.self_attn.q_proj.lora_A": torch.randn(r, 64),
            "base_lm.layers.0.self_attn.q_proj.lora_B": torch.zeros(64, r),
        },
        ckpt / "lora_weights.safetensors",
    )
    (ckpt / "optimizer.pth").write_bytes(b"optimizer state")
    return lora_library.resolve(lora_dir, rel)


def read_zip(path_or_bytes):
    with zipfile.ZipFile(path_or_bytes) as zf:
        return {i.filename: zf.read(i) for i in zf.infolist()}


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.mark.parametrize(
    "base, expected",
    [
        (LOCAL_BASE, "openbmb/VoxCPM2"),
        ("/home/x/models/openbmb__VoxCPM1.5/", "openbmb/VoxCPM1.5"),
        ("openbmb/VoxCPM2", "openbmb/VoxCPM2"),
        (r"C:\models\VoxCPM2", None),
        (None, None),
    ],
)
def test_hub_id(base, expected):
    assert sharing.hub_id(base) == expected


def test_export_contains_only_the_essentials_without_local_info(lora_dir):
    ckpt = make_checkpoint(lora_dir)

    files = read_zip(sharing.export_package(ckpt, "voice", "openbmb/VoxCPM2"))

    assert sorted(files) == ["voice/README.md", "voice/lora_config.json", "voice/lora_weights.safetensors"]
    info = json.loads(files["voice/lora_config.json"])
    assert info["base_model"] == "openbmb/VoxCPM2" and info["lora_config"]["r"] == 32
    weights = load(files["voice/lora_weights.safetensors"])
    assert all(t.dtype == torch.bfloat16 for t in weights.values())
    readme = files["voice/README.md"].decode()
    assert "| Training steps | 5000 |" in readme and "--lora-r 32 --lora-alpha 32" in readme
    for data in files.values():
        assert b"someone" not in data and str(lora_dir).encode() not in data and b"__metadata__" not in data


def test_export_carries_no_time_and_packs_the_same_bytes_each_time(lora_dir):
    import zipfile

    ckpt = make_checkpoint(lora_dir)
    first = sharing.export_package(ckpt, "voice", "openbmb/VoxCPM2")
    with zipfile.ZipFile(first) as zf:
        assert {i.date_time for i in zf.infolist()} == {sharing.ZIP_EPOCH}
        assert {i.external_attr >> 16 for i in zf.infolist()} == {0o644}
    assert sharing.export_package(ckpt, "voice", "openbmb/VoxCPM2").read_bytes() == first.read_bytes()


def test_export_keeps_float32_on_request_and_validates_input(lora_dir):
    ckpt = make_checkpoint(lora_dir)

    files = read_zip(sharing.export_package(ckpt, "v", "openbmb/VoxCPM2", half=False))
    assert all(t.dtype == torch.float32 for t in load(files["v/lora_weights.safetensors"]).values())
    with pytest.raises(ValueError):
        sharing.export_package(ckpt, "../x", "openbmb/VoxCPM2")
    with pytest.raises(ValueError):
        sharing.export_package(ckpt, "v", r"C:\models\VoxCPM2")


def test_import_round_trip(lora_dir, tmp_path):
    package = sharing.export_package(make_checkpoint(lora_dir, r=16), "shared_voice", "openbmb/VoxCPM2")
    other = tmp_path / "other_lora"

    assert sharing.import_package(other, package, "whatever.zip") == "shared_voice"

    (ckpt,) = lora_library.scan(other)
    assert ckpt.id == "shared_voice" and ckpt.config.r == 16 and ckpt.base_model == "openbmb/VoxCPM2"
    with pytest.raises(FileExistsError):
        sharing.import_package(other, package, "whatever.zip")


def test_import_accepts_files_without_a_folder(lora_dir, tmp_path):
    files = read_zip(sharing.export_package(make_checkpoint(lora_dir), "v", "openbmb/VoxCPM2"))
    package = tmp_path / "upload.zip"
    package.write_bytes(make_zip({k.split("/")[1]: v for k, v in files.items()}))

    assert sharing.import_package(tmp_path / "lora2", package, "My_Voice.zip") == "My_Voice"


@pytest.mark.parametrize(
    "entries, message",
    [
        ({"../evil/lora_weights.safetensors": b"x"}, "Unexpected entry"),
        ({"a/b/lora_weights.safetensors": b"x"}, "Unexpected entry"),
        ({"v/lora_weights.safetensors": b"x"}, "needs"),
        ({"v/lora_weights.safetensors": b"x", "v/lora_config.json": b"{}"}, "safetensors"),
    ],
)
def test_import_rejects_bad_packages(tmp_path, entries, message):
    package = tmp_path / "p.zip"
    package.write_bytes(make_zip(entries))

    with pytest.raises(ValueError, match=message):
        sharing.import_package(tmp_path / "lora", package, "p.zip")
    assert not (tmp_path / "lora").exists()


def test_export_and_import_endpoints(service, training, lora_dir):
    make_checkpoint(lora_dir)
    client = TestClient(create_app(service.settings, service, training))

    lora = next(x for x in client.get("/api/loras").json() if x["id"].endswith("step_0005000"))
    assert (lora["base_model_id"], lora["step"]) == ("openbmb/VoxCPM2", 5000)

    resp = client.get(
        "/api/loras/export",
        params={"id": lora["id"], "name": "voice_r32", "base_model": lora["base_model_id"]},
    )
    assert resp.status_code == 200
    assert 'filename="voice_r32.zip"' in resp.headers["content-disposition"]

    upload = {"package": ("voice_r32.zip", resp.content, "application/zip")}
    assert client.post("/api/loras/import", files=upload).json() == {"id": "voice_r32"}
    assert client.post("/api/loras/import", files=upload).status_code == 409
    bad = {"package": ("x.zip", b"not a zip", "application/zip")}
    assert client.post("/api/loras/import", files=bad).status_code == 422
    assert client.get("/api/loras/export", params={"id": "nope", "name": "a", "base_model": "a/b"}).status_code == 404
