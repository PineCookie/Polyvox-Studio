import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from voxcpm_studio import lora_library, training
from voxcpm_studio.engines import SynthesisRequest
from voxcpm_studio.engines.dots import lora as dots_lora
from voxcpm_studio.engines.dots.engine import DotsEngine
from voxcpm_studio.engines.dots.worker import cache_key, cached, format_metrics, make_optimizer, rates

from conftest import make_lora

DIM = 8


def attention(names=("q_proj", "k_proj", "v_proj", "o_proj")) -> nn.Module:
    attn = nn.Module()
    for name in names:
        setattr(attn, name, nn.Linear(DIM, DIM))
    return attn


class TinyDots(nn.Module):
    """dots.tts's module layout in miniature: LLM layers and DiT blocks with
    attention projections, and the fields its inference fusion touches."""

    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.llm = nn.Module()
        self.core.llm.model = nn.Module()
        self.core.llm.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.core.llm.model.layers:
            layer.self_attn = attention()
        dit = nn.Module()
        dit.blocks = nn.ModuleList([nn.Module()])
        for block in dit.blocks:
            block.attn = attention()
            block.adaLN_modulation = nn.Linear(DIM, DIM)
        dit.output_layer = nn.Module()
        dit.output_layer.adaLN_modulation = nn.Linear(DIM, DIM)
        self.core.velocity_field_predictor = dit
        self._fm_dit_solvers = {}

    def fuse(self):
        """What dit_inference.fuse_dit_for_inference does to the modules."""
        dit = self.core.velocity_field_predictor
        for block in dit.blocks:
            old, fused = block.attn, nn.Module()
            fused.qkv_proj = nn.Linear(DIM, 3 * DIM, bias=False)
            with torch.no_grad():
                fused.qkv_proj.weight.copy_(torch.cat([old.q_proj.weight, old.k_proj.weight, old.v_proj.weight]))
            fused.o_proj = old.o_proj  # kept, as FusedQKVMultiHeadAttention does
            block.attn = fused
            block.adaLN_modulation = nn.Identity()
        dit.output_layer.adaLN_modulation = nn.Identity()
        self.core.velocity_field_predictor = nn.Module()
        self._fm_dit_solvers["fm"] = object()


def lora_tensors(model, rank=4, seed=0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (_, layer) in dots_lora.target_layers(model).items():
        tensors[f"{name}.lora_A"] = torch.randn(rank, layer.in_features, generator=gen)
        tensors[f"{name}.lora_B"] = torch.randn(layer.out_features, rank, generator=gen)
    return tensors


def test_targets_are_the_attention_projections():
    names = dots_lora.target_layers(TinyDots())
    assert len(names) == 2 * 4 + 4
    assert {net for net, _ in names.values()} == {"llm", "dit"}
    assert "core.velocity_field_predictor.blocks.0.attn.adaLN_modulation" not in names


def test_merged_lora_equals_the_trained_hooks():
    torch.manual_seed(0)
    trained, merged = TinyDots(), TinyDots()
    merged.load_state_dict(trained.state_dict())
    loras = dots_lora.attach(trained, rank=4, alpha=8, dropout=0.0, networks=("llm", "dit"))
    for _, lora in loras.values():
        nn.init.normal_(lora.B)
    dots_lora.Merger(merged).apply(dots_lora.state(loras), scale=8 / 4)

    x = torch.randn(3, DIM)
    for name in dots_lora.target_layers(merged):
        assert torch.allclose(trained.get_submodule(name)(x), merged.get_submodule(name)(x), atol=1e-5)

    for _, lora in loras.values():
        lora.enabled = False  # samples: merged weights, hooks off
    layer = "core.llm.model.layers.0.self_attn.q_proj"
    assert not torch.allclose(trained.get_submodule(layer)(x), merged.get_submodule(layer)(x))


def test_merger_switches_loras_and_undoes_inference_fusion():
    model = TinyDots()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    merger = dots_lora.Merger(model)
    dit = model.core.velocity_field_predictor

    first, second = lora_tensors(model, seed=1), lora_tensors(model, seed=2)
    merger.apply(first)
    model.fuse()  # a generation
    merger.apply(second, scale=0.5)

    assert model.core.velocity_field_predictor is dit and not model._fm_dit_solvers
    name = "core.velocity_field_predictor.blocks.0.attn.q_proj"
    expected = original[f"{name}.weight"] + 0.5 * second[f"{name}.lora_B"] @ second[f"{name}.lora_A"]
    assert torch.allclose(model.get_submodule(name).weight, expected, atol=1e-6)

    model.fuse()
    merger.apply(None)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_merger_keeping_fusion_writes_the_fused_copies_in_place():
    model = TinyDots()
    merger = dots_lora.Merger(model, keep_fusion=True)
    dit = model.core.velocity_field_predictor
    old_attn = dit.blocks[0].attn
    model.fuse()  # warm-up
    fused = dit.blocks[0].attn.qkv_proj.weight
    pointer, before = fused.data_ptr(), fused.clone()

    merger.apply(lora_for(merger))

    assert dit.blocks[0].attn is not old_attn and model._fm_dit_solvers  # still fused
    assert fused.data_ptr() == pointer  # the same tensor: recorded CUDA graphs stay valid
    expected = torch.cat([old_attn.q_proj.weight, old_attn.k_proj.weight, old_attn.v_proj.weight])
    assert torch.equal(fused, expected) and not torch.equal(fused, before)
    merger.apply(None)
    assert torch.equal(fused, before)


def lora_for(merger: dots_lora.Merger, seed: int = 1) -> dict[str, torch.Tensor]:
    """LoRA tensors for the merger's (unfused) layer names."""
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in merger._modules.items():
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and "qkv" not in name:
            tensors[f"{name}.lora_A"] = torch.randn(4, module.in_features, generator=gen)
            tensors[f"{name}.lora_B"] = torch.randn(module.out_features, 4, generator=gen)
    return tensors


def test_merger_rejects_a_lora_that_does_not_fit():
    model = TinyDots()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    merger = dots_lora.Merger(model)
    merger.apply(lora_tensors(model))
    bad = {"core.llm.model.layers.0.self_attn.q_proj.lora_A": torch.zeros(4, DIM + 1),
           "core.llm.model.layers.0.self_attn.q_proj.lora_B": torch.zeros(DIM, 4)}  # fmt: skip

    with pytest.raises(ValueError, match="does not fit"):
        merger.apply(bad)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="unpaired"):
        merger.apply({"x.lora_A": torch.zeros(1)})


class FakeTokenizer:
    def __call__(self, text):
        return {"input_ids": list(range(len(text)))}


class FakeRuntime:
    """Stands in for DotsTtsRuntime: generate() runs the model's patch stream."""

    sample_rate = 48_000

    def __init__(self):
        self.model = TinyDots()
        self.model.tokenizer = FakeTokenizer()
        self.calls: list[dict] = []
        type(self.model)._generate_latents_stream = lambda model, *a, **k: iter([torch.zeros(1)] * 5)

    def run_warmup(self):
        self.model.fuse()  # as the optimized runtime's warm-up generations do

    def generate(self, **kwargs):
        self.calls.append({**kwargs, "q": self.model.get_submodule(DOTS_LAYER).weight.clone()})
        patches = list(self.model._generate_latents_stream())
        return {"audio": torch.zeros(len(patches) * 7680)}


DOTS_LAYER = "core.llm.model.layers.0.self_attn.q_proj"


def make_dots_lora(lora_dir, rel, model, seed=0, rank=4, alpha=8.0):
    ckpt = lora_dir / rel
    ckpt.mkdir(parents=True)
    save_file(lora_tensors(model, rank, seed), ckpt / lora_library.WEIGHTS_FILE)
    dots_lora.write_config(ckpt, {"r": rank, "alpha": alpha}, "org/dots")
    return ckpt


@pytest.fixture
def runtime():
    return FakeRuntime()


@pytest.fixture
def engine(settings, runtime, monkeypatch):
    monkeypatch.setattr(DotsEngine, "unavailable", lambda self: None)
    return DotsEngine(settings, loader=lambda model, optimize: runtime)


def test_dots_synthesis_options_seed_and_progress(engine, runtime):
    result = engine.synthesize(SynthesisRequest(text="你好世界", seed=3, options={"guidance_scale": 2.0}))

    call = runtime.calls[0]
    assert (call["guidance_scale"], call["num_steps"], call["speaker_scale"]) == (2.0, 10, 1.5)
    assert call["language"] == "zh" and call["normalize_text"] is False
    assert "prompt_audio_path" not in call
    assert (result.seed, result.sample_rate, len(result.audio)) == (3, 48_000, 5 * 7680)
    assert engine.progress()["phase"] == "idle" and engine.progress()["step"] == 5
    assert "_generate_latents_stream" not in vars(runtime.model)  # the counting wrapper is gone
    with pytest.raises(ValueError):
        engine.synthesize(SynthesisRequest(text="hi", options={"cfg_value": 2.0}))  # VoxCPM's setting


def test_dots_reference_and_transcript(engine, runtime, tmp_path):
    ref = tmp_path / "ref.wav"
    engine.synthesize(SynthesisRequest(text="hi", reference_wav=ref, prompt_text="  "))
    engine.synthesize(SynthesisRequest(text="hi", reference_wav=ref, prompt_text="spoken"))

    assert runtime.calls[0]["prompt_audio_path"] == str(ref) and "prompt_text" not in runtime.calls[0]
    assert runtime.calls[1]["prompt_text"] == "spoken"


def test_dots_loras_merge_switch_and_reject_voxcpm(engine, runtime, lora_dir):
    base = runtime.model.get_submodule(DOTS_LAYER).weight.clone()
    make_dots_lora(lora_dir, "a/step_1", runtime.model, seed=1)
    make_dots_lora(lora_dir, "b/step_1", runtime.model, seed=2)
    make_lora(lora_dir, "vox/step_1", r=8)

    for lora_id in ("a/step_1", "b/step_1", None, "a/step_1"):
        engine.synthesize(SynthesisRequest(text="hi", lora_id=lora_id))
    weights = [c["q"] for c in runtime.calls]

    assert not torch.equal(weights[0], base) and not torch.equal(weights[0], weights[1])
    assert torch.equal(weights[2], base) and torch.allclose(weights[3], weights[0])
    assert engine.status()["active_lora"] == "a/step_1"
    with pytest.raises(ValueError, match="another model"):
        engine.synthesize(SynthesisRequest(text="hi", lora_id="vox/step_1"))
    engine.unload()
    assert not engine.loaded


def test_dots_training_config(engine, tmp_path):
    params = training.TrainParams(name="r", dataset="d", engine="dots_tts", lora_rank=16, enable_dit=False)
    config = engine.training_config(params, tmp_path, tmp_path / "train.jsonl", None)

    assert config["lora"] == {"r": 16, "alpha": 32, "dropout": 0.0, "enable_lm": True, "enable_dit": False}
    assert (config["save_path"], config["val_manifest"]) == (str(tmp_path / "checkpoints"), "")
    with pytest.raises(ValueError):
        engine.training_config(params.model_copy(update={"enable_lm": False}), tmp_path, tmp_path, None)


def test_worker_log_lines_parse_as_the_studio_reads_them(tmp_path):
    groups = [{"name": "llm", "params": [nn.Parameter(torch.zeros(2))]}]
    opt, scheduler = make_optimizer("prodigy_sf", groups, {"learning_rate": 1.0, "weight_decay": 0.0}, 10)
    values = {"loss/diff": 0.5, "loss/stop": 0.1, **rates(opt), "epoch": 0.2, "grad_norm": 0.3, "log interval": 2.5}
    (tmp_path / "train.log").write_text(f"[train] step 10: {format_metrics(values)}\n", encoding="utf-8")

    (point,) = training.parse_metrics(tmp_path / "train.log")["train"]
    assert scheduler is None and point["lr_llm"] == pytest.approx(1e-6) and "elr_llm" in point
    assert (point["step"], point["loss/diff"], point["interval"]) == (10, 0.5, 2.5)

    groups = [{"name": "llm", "params": [nn.Parameter(torch.zeros(2))]}]
    opt, scheduler = make_optimizer(
        "adamw", groups, {"learning_rate": 1e-4, "weight_decay": 0.0, "warmup_steps": 2}, 10
    )
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    assert lrs[0] < lrs[2] == pytest.approx(1e-4) and lrs[-1] < lrs[5]


def test_encoded_clips_are_cached_per_manifest_and_model(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"audio": "a.wav", "text": "x"}\n', encoding="utf-8")
    cache = tmp_path / "cache" / "encoded_train.pt"
    calls = []

    def make(n=1):
        calls.append(n)
        return [{"latents": torch.ones(4, 3, dtype=torch.float16), "xvectors": [torch.zeros(2)]}] * n

    first = cached(cache, cache_key(manifest, "model"), make, 1)
    again = cached(cache, cache_key(manifest, "model"), make, 1)
    assert calls == [1] and torch.equal(again[0]["latents"], first[0]["latents"])

    cached(cache, cache_key(manifest, "other_model"), make, 1)  # another base model
    manifest.write_text('{"audio": "b.wav", "text": "y"}\n', encoding="utf-8")
    cached(cache, cache_key(manifest, "other_model"), make, 1)  # edited clips
    assert len(calls) == 3

    stopped = tmp_path / "cache" / "stopped.pt"
    cached(stopped, cache_key(manifest, "model"), lambda: make(0), 1)  # a stop cut the pass short
    assert not stopped.exists()
    cache.write_bytes(b"not a cache")
    assert len(cached(cache, cache_key(manifest, "model"), make, 1)) == 1 and len(calls) == 5


def test_dots_checkpoints_are_listed_with_their_engine(lora_dir):
    make_dots_lora(lora_dir, "voice/checkpoints/step_0000010", TinyDots())
    (ckpt,) = lora_library.scan(lora_dir)
    info = json.loads((ckpt.path / "lora_config.json").read_text())
    assert (ckpt.engine, ckpt.rank, ckpt.alpha, ckpt.config) == ("dots_tts", 4, 8.0, None)
    assert info["base_model"] == "org/dots"
