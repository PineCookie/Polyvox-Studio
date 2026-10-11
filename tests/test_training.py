import json
from pathlib import Path

import pytest
import yaml

from polyvox import training
from polyvox.training import TrainingConflict, TrainParams

from conftest import make_dataset, wait_for, wait_idle


def test_parse_metrics_drops_steps_redone_after_resume(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "[train] step 0: loss/diff: 0.9, lr: 0.000001\n"
        "[train] step 10: loss/diff: 0.8, lr: 0.00001, log interval: 2.00s\n"
        "[train] step 20: loss/diff: 0.7, lr: 0.00002, log interval: 3.00s\n"
        "[val] step 0: loss/total: 1.0\n"
        "Resuming from step 10\n"
        "[train] step 10: loss/diff: 0.6, lr: 0.00001\n"
        "[train] step 15: garbage\n"
    )

    m = training.parse_metrics(log)

    assert [(p["step"], p["loss/diff"]) for p in m["train"]] == [(0, 0.9), (10, 0.6)]
    assert m["val"] == [{"step": 0, "loss/total": 1.0}]


def test_seconds_per_step_and_downsample():
    points = [{"step": s, "interval": 2.0} for s in range(0, 100, 10)]
    assert training.seconds_per_step(points) == pytest.approx(0.2)

    many = [{"step": s} for s in range(2500)]
    kept = training.downsample(many, 1000)
    assert len(kept) <= 1001 and kept[0]["step"] == 0 and kept[-1]["step"] == 2499


def test_tail_lines_keeps_final_progress_bar_state(tmp_path):
    log = tmp_path / "console.log"
    log.write_bytes(b"one\r\nbar 10%\rbar 100%\r\nlast\r\n")
    assert training.tail_lines(log) == ["one", "bar 100%", "last"]


def test_start_prepares_run_and_completes(training, data_dir, lora_dir):
    make_dataset(data_dir, "voice")

    training.start(TrainParams(name="voice_r32", dataset="voice/train.jsonl", num_iters=20))
    wait_idle(training)

    run = lora_dir / "voxcpm" / "voice_r32"
    config = yaml.safe_load((run / "train_config.yaml").read_text(encoding="utf-8"))
    assert config["sample_rate"] == 16000 and config["out_sample_rate"] == 48000
    assert config["lora"]["r"] == 32 and config["save_path"] == str(run / "checkpoints")
    train = [json.loads(line) for line in Path(config["train_manifest"]).read_text(encoding="utf-8").splitlines()]
    val = [json.loads(line) for line in Path(config["val_manifest"]).read_text(encoding="utf-8").splitlines()]
    assert len(train) == 2 and len(val) == 2  # default holdout
    assert all(Path(e["audio"]).is_absolute() and Path(e["audio"]).is_file() for e in train + val)

    detail = training.detail("voice_r32")
    assert detail.status == "complete" and detail.saved_step == 20
    assert [p["step"] for p in detail.train] == [0, 10, 20]
    assert detail.seconds_per_step == pytest.approx(0.2)
    assert [s.file for s in detail.samples] == ["step_0000000_0.wav"]
    assert detail.sample_texts == [v["text"] for v in val]
    assert detail.val_clips == 2
    assert "progress 100%" in detail.log


def test_stop_saves_checkpoint_and_resume_continues(training, data_dir, lora_dir):
    make_dataset(data_dir, "voice")
    training.start(TrainParams(name="long", dataset="voice/train.jsonl", num_iters=1000))
    wait_for(lora_dir / "voxcpm" / "long" / "console.log", "waiting for stop")

    assert training.detail("long").status == "running"
    training.stop()
    wait_idle(training)

    summary = training.detail("long")
    assert (summary.status, summary.saved_step) == ("incomplete", 20)

    training.resume("long")
    samples = lora_dir / "voxcpm" / "long" / "samples"
    wait_for(lora_dir / "voxcpm" / "long" / "checkpoints" / "train.log", "[val] step 20:")
    training.stop()
    wait_idle(training)

    detail = training.detail("long")
    assert detail.saved_step == 40
    assert [p["step"] for p in detail.train] == [0, 10, 20, 30, 40]
    assert (samples / "step_0000020_0.wav").is_file()


def test_conflicts(training, data_dir):
    make_dataset(data_dir, "voice")
    with pytest.raises(TrainingConflict):
        training.stop()

    training.start(TrainParams(name="a", dataset="voice/train.jsonl", num_iters=1000))
    with pytest.raises(TrainingConflict):
        training.start(TrainParams(name="b", dataset="voice/train.jsonl"))
    training.stop()
    wait_idle(training)

    with pytest.raises(TrainingConflict):
        training.start(TrainParams(name="a", dataset="voice/train.jsonl"))
    with pytest.raises(KeyError):
        training.start(TrainParams(name="c", dataset="nope.jsonl"))


def test_resume_complete_run_needs_more_steps(training, data_dir):
    make_dataset(data_dir, "voice")
    training.start(TrainParams(name="a", dataset="voice/train.jsonl", num_iters=20))
    wait_idle(training)

    with pytest.raises(ValueError):
        training.resume("a")
    with pytest.raises(ValueError):
        training.resume("a", num_iters=20)

    training.resume("a", num_iters=1000)
    config = training.detail("a").config
    assert config["num_iters"] == config["max_steps"] == 1000


def test_resume_adopts_webui_run(training, data_dir, lora_dir, model_dir):
    make_dataset(data_dir, "voice")
    run = lora_dir / "legacy"
    (run / "checkpoints" / "latest").mkdir(parents=True)
    (run / "checkpoints" / "latest" / "training_state.json").write_text('{"step": 10}')
    webui_config = {
        "pretrained_path": str(model_dir),
        "train_manifest": "data/voice/train.jsonl",  # relative to the webui's cwd
        "val_manifest": "",
        "num_iters": 1000,
        "max_steps": 1000,
        "save_path": "lora\\legacy\\checkpoints",
        "tensorboard": "lora\\legacy\\logs",
    }
    (run / "train_config.yaml").write_text(yaml.safe_dump(webui_config), encoding="utf-8")

    assert [r.name for r in training.list_runs()] == ["legacy"]
    assert training.list_runs()[0].status == "incomplete"

    training.resume("legacy")
    wait_for(run / "console.log", "waiting for stop")

    config = yaml.safe_load((run / "train_config.yaml").read_text(encoding="utf-8"))
    assert config["save_path"] == str(run / "checkpoints")
    assert config["tensorboard"] == str(lora_dir / "legacy" / "logs")
    entries = Path(config["train_manifest"]).read_text(encoding="utf-8").splitlines()
    assert len(entries) == 4 and all(Path(json.loads(e)["audio"]).is_file() for e in entries)
    assert yaml.safe_load((run / "train_config.orig.yaml").read_text(encoding="utf-8")) == webui_config


def test_latest_only_checkpoint_takes_its_step_from_the_log(training, lora_dir, model_dir):
    run = lora_dir / "old"
    (run / "checkpoints" / "latest").mkdir(parents=True)
    (run / "checkpoints" / "train.log").write_text(
        "[train] step 3990: loss/diff: 0.6\n[train] step 3999: loss/diff: 0.6\n"
    )
    config = {"pretrained_path": str(model_dir), "train_manifest": "", "num_iters": 4000, "save_path": "x"}
    (run / "train_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    (summary,) = training.list_runs()
    assert (summary.status, summary.saved_step) == ("complete", 3999)

    training.resume("old", num_iters=5000)
    assert json.loads((run / "checkpoints" / "latest" / "training_state.json").read_text()) == {"step": 3999}


def test_failed_job_is_reported(training, data_dir, lora_dir):
    make_dataset(data_dir, "voice")
    training.start(TrainParams(name="a", dataset="voice/train.jsonl", num_iters=1000))
    wait_for(lora_dir / "voxcpm" / "a" / "console.log", "waiting for stop")

    (lora_dir / "voxcpm" / "a" / "crash").touch()
    wait_idle(training)

    assert training.detail("a").status == "failed"
    assert "exited with code 1" in training.detail("a").log[-1]
