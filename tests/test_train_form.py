"""Training forms per engine (Engine.train_form), sizing a run (Engine.plan), TrainParams.options."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from polyvox import engines, vram
from polyvox.server import create_app
from polyvox.training import TrainParams

TRAINABLE = [cls for cls in engines.classes().values() if cls.capabilities.training]


@pytest.mark.parametrize("cls", TRAINABLE, ids=lambda c: c.id)
def test_each_form_is_consistent(cls):
    for profile in (None, vram.profile_for(16), vram.profile_for(8)):
        form = cls.train_form(profile)
        names = [p.name for p in form]
        assert len(names) == len(set(names)), names
        run_fields = {p.name for p in form if p.scope == "run"}
        assert run_fields <= set(TrainParams.model_fields), run_fields - set(TrainParams.model_fields)
        for p in form:
            p.coerce(p.default)  # every default is a value it accepts
            for condition in (p.shown_if or {}, p.disabled_if or {}, *(c for c, _ in p.labels)):
                assert set(condition) <= set(names), (p.name, condition)
            for implied in (p.implies or {}).values():
                assert set(implied) <= set(names), (p.name, implied)


def test_a_lora_run_is_sized_in_epochs_as_the_form_sized_it():
    form = {p.name: p.default for p in engines.classes()["voxcpm"].train_form(vram.profile_for(16))}
    assert (form["batch_size"], form["grad_accum_steps"], form["optimizer"]) == (4, 4, "prodigy_sf")
    plan = engines.classes()["voxcpm"].plan({**form, "epochs": 4}, 656, 0)
    # 41 steps of 16 per epoch: 164 steps, checkpoints every half epoch (20.5 -> 21), warmup 5%.
    assert plan["values"] == {
        "num_iters": 164,
        "warmup_steps": 8,
        "save_interval": 21,
        "valid_interval": 21,
        "log_interval": 4,
    }
    assert plan["name_tag"] == "voxcpm2-r32" and plan["summary"]["epochs"] == pytest.approx(4.0, abs=0.01)
    # Steps set by hand stay as they are; the epochs follow from them.
    plan = engines.classes()["dots_tts"].plan({**form, "epochs": 4, "num_iters": 82}, 656, 0, ["num_iters"])
    assert "num_iters" not in plan["values"] and plan["summary"]["epochs"] == pytest.approx(2.0, abs=0.01)
    assert plan["name_tag"] == "dots-r32"
    assert engines.classes()["voxcpm"].plan(form, 0, 0)["values"] == {}  # no dataset yet


def test_full_fine_tuning_runs_are_sized_by_their_own_epochs():
    gsv = engines.classes()["gpt_sovits"]
    values = {p.name: p.default for p in gsv.train_form(vram.profile_for(16))}
    plan = gsv.plan(values, 20, 0)
    # SoVITS 8 epochs of ceil(20 / 8) batches; GPT 15 epochs of 4 batches of 5 clips, 4 per update.
    assert plan["values"] == {"num_iters": 8 * 3 + 15} and plan["summary"]["gpt_batch"] == 5
    assert plan["name_tag"] == "v2pp" and plan["samples"]["unit"] == "epochs"
    assert gsv.plan({**values, "gsv_version": "v4", "lora_rank": 16}, 20, 0)["name_tag"] == "v4-r16"
    version = next(p for p in gsv.train_form(vram.profile_for(16)) if p.name == "gsv_version")
    assert version.implies["v4"] == {"batch_size": 4} and version.implies["v2Pro"] == {"batch_size": 8}

    rvc = engines.classes()["rvc"]
    values = {p.name: p.default for p in rvc.train_form(vram.profile_for(16))}
    plan = rvc.plan(values, 10, 100.0)  # ~33 pieces: 100 s / 3.4 + 0.37 a clip
    assert plan["values"] == {"num_iters": 3 * 100} and plan["summary"]["pieces"] == 33
    assert (
        plan["name_tag"] == "rvc" and rvc.plan({**values, "rvc_sample_rate": 48000}, 10, 100.0)["name_tag"] == "rvc48k"
    )


def test_engine_settings_are_checked_options():
    params = TrainParams(name="a", engine="gpt_sovits", dataset="d/train.jsonl", options={"gsv_version": "v4"})
    assert params.options == {"gsv_version": "v4"}
    with pytest.raises(ValidationError, match="must be one of"):
        TrainParams(name="a", engine="gpt_sovits", dataset="d/train.jsonl", options={"gsv_version": "v9"})
    with pytest.raises(ValidationError, match="no training setting"):  # another engine's
        TrainParams(name="a", engine="voxcpm", dataset="d/train.jsonl", options={"gsv_version": "v4"})
    rate = TrainParams(name="a", engine="rvc", dataset="d/train.jsonl", options={"rvc_sample_rate": "48000"})
    assert rate.options == {"rvc_sample_rate": 48000}  # a choice keeps its type
    assert engines.classes()["rvc"].train_options(rate.options)["rvc_epochs"] == 100  # defaults filled in


def test_a_runs_values_come_back_from_its_config():
    old = engines.classes()["voxcpm"].values_from_config({"num_iters": 5000, "lora": {"r": 64, "alpha": 32}})
    # A run from before the optimizer option: AdamW at 1e-4, batch 1, no token budget.
    assert (old["optimizer"], old["learning_rate"], old["batch_size"], old["max_batch_tokens"]) == ("adamw", 1e-4, 1, 0)
    assert (old["lora_rank"], old["num_iters"], old["enable_lm"]) == (64, 5000, True)
    s2 = engines.classes()["fish_s2"].values_from_config(
        {"lora": {"targets": ["fast_attn", "slow_attn"]}, "slow_learning_rate": 3e-5, "optimizer": "adamw"}
    )
    assert s2["s2_slow"] and s2["s2_slow_lr"] == 3e-5 and s2["s2_offload"] is True  # older runs offloaded
    gsv = engines.classes()["gpt_sovits"].values_from_config({"version": "v4", "lora_rank": 16, "sovits_epochs": 4})
    assert (gsv["gsv_version"], gsv["lora_rank"], gsv["sovits_epochs"], gsv["gpt_epochs"]) == ("v4", 16, 4, 0)


def test_forms_follow_the_gpu_and_plans_come_from_the_server(monkeypatch, service, training):
    monkeypatch.setenv("POLYVOX_VRAM_GB", "8")
    client = TestClient(create_app(service.settings, service, training))
    listed = {e["id"]: e for e in client.get("/api/engines").json()}
    form = {p["name"]: p for p in listed["fish_s2"]["train_form"]}
    assert form["s2_offload"]["default"] is True and form["s2_offload"]["disabled"]  # 8 GB: always offloaded
    assert form["s2_slow"]["disabled"] and form["batch_size"]["default"] == 4
    vox = {p["name"]: p["default"] for p in listed["voxcpm"]["train_form"]}
    assert (vox["batch_size"], vox["grad_accum_steps"]) == (1, 16)  # the same effective batch, accumulated
    assert listed["fish_drama3"]["train_form"] == []  # doesn't train

    body = {"engine": "voxcpm", "values": {**vox, "epochs": 2}, "clips": 160, "seconds": 600}
    plan = client.post("/api/training/plan", json=body).json()
    assert plan["values"]["num_iters"] == 20 and plan["summary"]["kind"] == "epochs"
    assert client.post("/api/training/plan", json={**body, "engine": "fish_drama3"}).status_code == 404
