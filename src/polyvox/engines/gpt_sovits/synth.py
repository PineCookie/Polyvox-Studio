"""GPT-SoVITS v2Pro / v2ProPlus / v4 (and the webui's v2 voices) synthesis on the
vendored modules (vendor/gpt_sovits).

What upstream's inference_webui.get_tts_wav does for these versions, without the
webui: text -> phonemes + RoBERTa features; the reference clip -> semantic tokens
(HuBERT through the voice's quantizer), a spectrogram and a speaker embedding (v2Pro);
the GPT continues the reference's tokens with the text's; SoVITS decodes them
to 32 kHz audio in the reference's timbre. v4 instead turns them into a mel
spectrogram with a flow-matching DiT (continuing the reference's mel, in
windows) and a vocoder makes 48 kHz audio; it needs the reference's
transcript. Long texts are cut into chunks of up to four sentences, joined with
short pauses.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

REF_MIN_S, REF_MAX_S = 3.0, 10.0  # upstream refuses references outside this range
PAUSE_S = 0.3  # between chunks, and after the reference's semantic prompt
SPLITS = set("，。？！,.?!~:：—…")
SEMANTIC_HZ = 50  # upstream's early stop: max_sec x 50 tokens
BERT_DIM = 1024
BASE_VERSIONS = ("v2ProPlus", "v2Pro", "v4")  # base models offered (models\)
VERSIONS = (*BASE_VERSIONS, "v2")  # voices that load (v2: the webui's older voices, no base model here)
V4 = "v4"
V4_RATE = 48_000  # the vocoder's output
# v4's mel: 32 kHz audio, 100 bands, 100 frames/s; values scaled from [-12, 2] to [-1, 1].
V4_MEL = {"n_fft": 1280, "num_mels": 100, "sampling_rate": 32000, "hop_size": 320, "win_size": 1280}
V4_SPEC_MIN, V4_SPEC_MAX = -12.0, 2.0
V4_REF_FRAMES, V4_WINDOW = 500, 1000  # mel frames of reference prompt / per DiT window (prompt included)


def sovits_version(path: Path) -> tuple[str, bool]:
    """(model version, LoRA weights?) of a SoVITS file. Untagged webui files
    are told apart by size, and v1 / v2 sizes overlap (v2 files of 81 MB pass
    as v1): there, as the webui does, the text embedding decides (322 symbols
    = v1)."""
    from ...vendor.gpt_sovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

    _, version, lora = get_sovits_version_from_path_fast(str(path))
    if version == "v1":
        embedding = load_sovits_new(path)["weight"].get("enc_p.text_embedding.weight")
        if embedding is not None and embedding.shape[0] != 322:
            version = "v2"
    return version, lora


@dataclass(frozen=True)
class Options:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed: float = 1.0
    sample_steps: int = 8  # v4: flow-matching steps per mel window


@dataclass
class Voice:
    """A voice's GPT and SoVITS weights, loaded."""

    key: object
    model_version: str  # v2Pro | v2ProPlus | v4
    sovits: torch.nn.Module
    gpt: torch.nn.Module
    hps: object  # the SoVITS config (filter length, hop, sampling rate ...)
    max_sec: float
    decoder: object = None  # decode.Decoder for the GPT, made on first use (CUDA)

    @property
    def data_rate(self) -> int:
        """The rate SoVITS reads references at (its spectrogram)."""
        return int(self.hps.data.sampling_rate)

    @property
    def sample_rate(self) -> int:
        """The rate of the audio it makes."""
        return V4_RATE if self.model_version == V4 else self.data_rate


@dataclass
class Reference:
    key: object
    prompt: torch.Tensor | None  # semantic tokens [1, T]; None without a transcript
    phones: list[int]
    bert: torch.Tensor | None
    spec: torch.Tensor
    sv_emb: torch.Tensor | None  # v2Pro only
    mel: torch.Tensor | None = None  # v4: the reference's last mel frames, normalized [1, 100, T]
    fea: torch.Tensor | None = None  # v4: DiT conditions for those frames [1, 512, T]
    ge: torch.Tensor | None = None  # v4: the reference's style vector


class Config(dict):
    """A saved config (dict or the webui's HParams) read with attribute access."""

    def __init__(self, value):
        super().__init__(value.items() if hasattr(value, "items") else value)

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return Config(value) if hasattr(value, "items") else value


class Pipeline:
    """The shared front ends (RoBERTa, HuBERT, speaker encoder) and the G2PW
    pronunciation model, loaded once; voices and references on top."""

    def __init__(self, base_dir: Path, g2pw_dir: Path, device: str):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        from ...vendor.gpt_sovits.feature_extractor.cnhubert import CNHubert
        from ...vendor.gpt_sovits.sv import SV
        from ...vendor.gpt_sovits.text import chinese2

        self.base_dir, self.device = Path(base_dir), device
        self.half = device.startswith("cuda")
        # GPT decoding as a CUDA graph (decode.py), its layers fused by torch.compile;
        # off: upstream's loop.
        self.fast_decode = self.compile_decode = device.startswith("cuda")
        self.dtype = torch.float16 if self.half else torch.float32
        bert_dir = str(self.base_dir / "chinese-roberta-wwm-ext-large")
        chinese2.configure(str(g2pw_dir), bert_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_dir).to(device, self.dtype).eval()
        self.hubert = CNHubert(str(self.base_dir / "chinese-hubert-base")).to(device, self.dtype).eval()
        self.sv = SV(device, self.half, str(self.base_dir / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"))
        self.v4_dir = self.base_dir / "gsv-v4-pretrained"  # v4's SoVITS base and vocoder
        self.vocoder: torch.nn.Module | None = None  # v4's, loaded with the first v4 voice

    # ------------------------------------------------------------ weights
    def base_weights(self, version: str = "v2ProPlus") -> tuple[Path, Path]:
        """(SoVITS, GPT) of a base model."""
        if version == V4:
            return self.v4_dir / "s2Gv4.pth", self.base_dir / "s1v3.ckpt"
        return self.base_dir / "v2Pro" / f"s2G{version}.pth", self.base_dir / "s1v3.ckpt"

    def load_voice(self, sovits_path: Path, gpt_path: Path, key: object = None) -> Voice:
        from ...vendor.gpt_sovits.AR.models.t2s_model import Text2SemanticDecoder
        from ...vendor.gpt_sovits.module.models import SynthesizerTrn, SynthesizerTrnV3
        from ...vendor.gpt_sovits.process_ckpt import load_sovits_new, load_weights

        model_version, lora = sovits_version(sovits_path)
        if model_version not in VERSIONS or (lora and model_version != V4):
            raise ValueError(
                f"{Path(sovits_path).name} is GPT-SoVITS {model_version}; only {' / '.join(VERSIONS)} run here."
            )
        saved = load_sovits_new(sovits_path)
        hps = Config(saved["config"])
        model = {**hps["model"], "semantic_frame_rate": "25hz", "version": model_version}
        sizes = (hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length)
        model_class = SynthesizerTrnV3 if model_version == V4 else SynthesizerTrn
        sovits = model_class(*sizes, n_speakers=hps.data.n_speakers, **model)
        weights = self.plain_weights(saved, lora)
        if hasattr(sovits, "enc_q"):
            del sovits.enc_q  # training only
        missing, _ = sovits.load_state_dict(weights, strict=False)
        if missing:
            raise ValueError(f"{Path(sovits_path).name} lacks {len(missing)} SoVITS weights, e.g. {missing[0]}.")
        if model_version == V4:
            from .decode import fast_dit

            fast_dit(sovits, compiled=self.fast_decode and self.compile_decode)
            self._ensure_vocoder()
        gpt_saved = load_weights(gpt_path)
        gpt = Text2SemanticDecoder(gpt_saved["config"], top_k=3)
        gpt.load_state_dict({k.removeprefix("model."): v for k, v in gpt_saved["weight"].items()})
        return Voice(
            key=key,
            model_version=model_version,
            sovits=sovits.to(self.device, self.dtype).eval(),
            gpt=gpt.to(self.device, self.dtype).eval(),
            hps=hps,
            max_sec=float(gpt_saved["config"]["data"]["max_sec"]),
        )

    def plain_weights(self, saved: dict, lora: bool) -> dict[str, torch.Tensor]:
        """A loaded SoVITS file's weights as a full state dict: v4 LoRA voices
        (the webui's and ours) hold only their trained parts and the DiT's LoRA,
        laid over the v4 base and merged."""
        from ...vendor.gpt_sovits.process_ckpt import load_sovits_new

        if not lora:
            return saved["weight"]
        return merge_lora({**load_sovits_new(self.base_weights(V4)[0])["weight"], **saved["weight"]})

    def _ensure_vocoder(self) -> torch.nn.Module:
        """v4's vocoder (mel -> 48 kHz), shared by all v4 voices."""
        if self.vocoder is None:
            from ...vendor.gpt_sovits.module.models import Generator

            vocoder = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            vocoder.remove_weight_norm()  # saved without weight norm
            path = self.v4_dir / "vocoder.pth"
            vocoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.vocoder = vocoder.to(self.device, self.dtype).eval()
        return self.vocoder

    # ------------------------------------------------------------ text
    def phones_and_bert(self, text: str, language: str = "zh", lead_in: bool = True) -> tuple[list[int], torch.Tensor]:
        """Phoneme ids and per-phoneme RoBERTa features (zeros for English) of
        Chinese / English text, as the webui's mixed Chinese-English mode.
        ``lead_in``: very short texts get a "." in front, as upstream synthesizes
        them (not for training, where the audio has no such pause)."""
        from ...vendor.gpt_sovits.text import cleaned_text_to_sequence
        from ...vendor.gpt_sovits.text.cleaner import clean_text
        from ...vendor.gpt_sovits.text.LangSegmenter import LangSegmenter

        phones_all: list[int] = []
        features = []
        for part in LangSegmenter.getTexts(re.sub(r" {2,}", " ", text)):
            lang = "en" if part["lang"] == "en" else language
            phones, word2ph, norm = clean_text(part["text"], lang, "v2")
            phones = cleaned_text_to_sequence(phones, "v2")
            if lang == "zh":
                features.append(self._bert(norm, word2ph))
            else:
                features.append(torch.zeros((BERT_DIM, len(phones)), device=self.device))
            phones_all += phones
        if lead_in and len(phones_all) < 6 and not text.startswith("."):
            return self.phones_and_bert("." + text, language)  # as upstream: very short texts get a lead-in
        return phones_all, torch.cat(features, 1).to(self.dtype)

    def _bert(self, text: str, word2ph: list[int]) -> torch.Tensor:
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in self.tokenizer(text, return_tensors="pt").items()}
            hidden = self.bert(**inputs, output_hidden_states=True)["hidden_states"][-3][0][1:-1]
        assert len(word2ph) == len(text)
        return torch.cat([hidden[i].repeat(n, 1) for i, n in enumerate(word2ph)]).T.float()

    # ------------------------------------------------------------ reference
    def reference(self, voice: Voice, wav: Path, text: str | None, key: object = None) -> Reference:
        """A reference clip (3-10 s) as the voice hears it; ``text`` its transcript
        (None: the voice follows only its timbre, upstream's "reference-free" mode)."""
        import librosa

        sr = voice.data_rate
        audio16, _ = librosa.load(str(wav), sr=16000)
        seconds = len(audio16) / 16000
        if not REF_MIN_S <= seconds <= REF_MAX_S:
            raise ValueError(f"The reference clip is {seconds:.1f} s; GPT-SoVITS needs {REF_MIN_S:g}-{REF_MAX_S:g} s.")
        v4 = voice.model_version == V4
        if v4 and not (text and text.strip()):
            raise ValueError("GPT-SoVITS v4 needs the reference clip's transcript.")
        from ...vendor.gpt_sovits.module.mel_processing import spectrogram_torch

        with torch.no_grad():
            prompt = phones = bert = None
            if text and text.strip():
                silence = torch.zeros(int(sr * PAUSE_S), device=self.device, dtype=self.dtype)
                padded = torch.cat([torch.from_numpy(audio16).to(self.device, self.dtype), silence])
                feats = self.hubert.model(padded[None])["last_hidden_state"].transpose(1, 2)
                prompt = voice.sovits.extract_latent(feats)[0, 0][None]
                text = text.strip()
                phones, bert = self.phones_and_bert(text if text[-1] in SPLITS else text + "。")
            audio, _ = librosa.load(str(wav), sr=sr)
            peak = float(np.abs(audio).max()) if len(audio) else 0.0
            if peak > 1:
                audio = audio / min(2.0, peak)
            hps = voice.hps.data
            spec = spectrogram_torch(
                torch.from_numpy(audio).to(self.device)[None],
                hps.filter_length,
                sr,
                hps.hop_length,
                hps.win_length,
                center=False,
            ).to(self.dtype)
            if v4:
                mel, fea, ge = self._v4_prompt(voice, wav, spec, prompt, phones)
                return Reference(key, prompt, phones, bert, spec, sv_emb=None, mel=mel, fea=fea, ge=ge)
            sv_emb = None  # v2: no speaker embedding
            if voice.sovits.is_v2pro:
                audio16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sv_emb = self.sv.compute_embedding3(torch.from_numpy(audio16k).to(self.device, self.dtype)[None])
        return Reference(key=key, prompt=prompt, phones=phones or [], bert=bert, spec=spec, sv_emb=sv_emb)

    def _v4_prompt(self, voice: Voice, wav: Path, spec: torch.Tensor, prompt: torch.Tensor, phones: list[int]):
        """The reference's normalized mel and its DiT conditions (last V4_REF_FRAMES
        frames of both), and its style vector."""
        import librosa

        from ...vendor.gpt_sovits.module.mel_processing import mel_spectrogram_torch

        audio, _ = librosa.load(str(wav), sr=V4_MEL["sampling_rate"])
        mel = mel_spectrogram_torch(torch.from_numpy(audio).to(self.device)[None], fmin=0, fmax=None, **V4_MEL)
        mel = ((mel - V4_SPEC_MIN) / (V4_SPEC_MAX - V4_SPEC_MIN) * 2 - 1).to(self.dtype)
        fea, ge = voice.sovits.decode_encp(prompt[None], torch.LongTensor(phones).to(self.device)[None], spec)
        n = min(mel.shape[2], fea.shape[2])  # as upstream: both cut to the shorter, then to their last frames
        start = max(0, n - V4_REF_FRAMES)
        return mel[:, :, start:n], fea[:, :, start:n], ge

    # ------------------------------------------------------------ synthesis
    def synthesize(
        self,
        voice: Voice,
        ref: Reference,
        text: str,
        options: Options,
        seed: int,
        on_token: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """32 kHz mono float audio of ``text`` in the reference's voice;
        ``on_token`` is called for each semantic token the GPT makes."""
        chunks = split_text(text)
        if not chunks:
            raise ValueError("Text to synthesize is empty.")
        with counting_tokens(on_token):
            return self._synthesize(voice, ref, chunks, options, seed)

    def _synthesize(self, voice: Voice, ref: Reference, chunks: list[str], options: Options, seed: int) -> np.ndarray:
        pause = np.zeros(int(voice.sample_rate * PAUSE_S), dtype=np.float32)
        torch.manual_seed(seed)
        out: list[np.ndarray] = []
        for chunk in chunks:
            phones, bert = self.phones_and_bert(chunk)
            if ref.prompt is not None:
                ids, features = ref.phones + phones, torch.cat([ref.bert, bert], 1)
            else:
                ids, features = phones, bert
            ids_t = torch.LongTensor(ids).to(self.device)[None]
            sampling = {
                "top_k": options.top_k,
                "top_p": options.top_p,
                "temperature": options.temperature,
                "early_stop_num": int(SEMANTIC_HZ * voice.max_sec),
                "repetition_penalty": options.repetition_penalty,
            }
            decoder = self._decoder(voice)
            with torch.no_grad():
                if decoder is not None and decoder.fits(ids_t, ref.prompt):
                    tokens, n = self._generate(voice, decoder, ids_t, ref.prompt, features[None], sampling)
                else:
                    tokens, n = voice.gpt.infer_panel(
                        ids_t,
                        torch.tensor([ids_t.shape[-1]], device=self.device),
                        ref.prompt,
                        features[None],
                        **sampling,
                    )
                semantic = tokens[:, -n:][None]
                if voice.model_version == V4:
                    audio = self._v4_audio(voice, ref, semantic, phones, options).float()
                else:
                    audio = voice.sovits.decode(
                        semantic,
                        torch.LongTensor(phones).to(self.device)[None],
                        [ref.spec],
                        speed=options.speed,
                        sv_emb=[ref.sv_emb],
                    )[0][0].float()
            peak = float(audio.abs().max())
            if peak > 1:  # as upstream: avoid clipping
                audio = audio / peak
            out += [audio.cpu().numpy(), pause]
        return np.concatenate(out[:-1])

    def _v4_audio(self, voice: Voice, ref: Reference, semantic: torch.Tensor, phones: list[int], options: Options):
        """Semantic tokens -> mel (the DiT, window by window, each continuing the
        last frames before it, the first the reference's) -> 48 kHz audio."""
        fea, _ = voice.sovits.decode_encp(
            semantic, torch.LongTensor(phones).to(self.device)[None], ref.spec, ref.ge, options.speed
        )
        mel, cond = ref.mel, ref.fea
        keep = mel.shape[2]
        step = V4_WINDOW - keep
        out = []
        for i in range(0, fea.shape[2], step):
            chunk = fea[:, :, i : i + step]
            x = torch.cat([cond, chunk], 2).transpose(2, 1)
            lengths = torch.LongTensor([x.shape[1]]).to(self.device)
            made = self._dit(voice, x, lengths, mel, options.sample_steps)
            made = made[:, :, mel.shape[2] :]
            mel, cond = made[:, :, -keep:], chunk[:, :, -keep:]
            out.append(made)
        mel = (torch.cat(out, 2) + 1) / 2 * (V4_SPEC_MAX - V4_SPEC_MIN) + V4_SPEC_MIN
        with torch.inference_mode():
            return self._ensure_vocoder()(mel)[0][0]

    def _dit(self, voice: Voice, *args):
        """cfm.inference; if its fused blocks fail (torch.compile: no working
        Triton, say), this voice and later ones run them unfused."""
        try:
            return voice.sovits.cfm.inference(*args, inference_cfg_rate=0)
        except Exception as e:
            if not self.compile_decode or isinstance(e, torch.OutOfMemoryError):
                raise
            logger.warning("GPT-SoVITS: fusing the v4 DiT failed (%s); running it unfused", e)
            from .decode import fast_dit

            self.compile_decode = False
            fast_dit(voice.sovits, compiled=False)
            return voice.sovits.cfm.inference(*args, inference_cfg_rate=0)

    def _decoder(self, voice: Voice):
        if not self.fast_decode:
            return None
        if voice.decoder is None:
            from .decode import Decoder

            voice.decoder = Decoder(voice.gpt, compiled=self.compile_decode)
        return voice.decoder

    def _generate(self, voice: Voice, decoder, *args):
        """decoder.generate; if fusing its step fails (torch.compile: no working
        Triton, say), the rest of the process decodes unfused."""
        *inputs, sampling = args
        try:
            return decoder.generate(*inputs, **sampling)
        except Exception as e:
            if not self.compile_decode or isinstance(e, torch.OutOfMemoryError):
                raise
            logger.warning("GPT-SoVITS: fusing the decoding step failed (%s); decoding unfused", e)
            from .decode import Decoder

            self.compile_decode = False
            voice.decoder = Decoder(voice.gpt)
            return voice.decoder.generate(*inputs, **sampling)


def merge_lora(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The webui's v4 voice weights as plain SoVITS weights: its DiT was wrapped
    by peft (keys ``cfm.base_model.model.<module>.lora_{A,B}.default.weight``),
    so each LoRA folds into its layer, W + B @ A (the webui trains with
    alpha = rank: scale 1)."""
    plain: dict[str, torch.Tensor] = {}
    lora: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in weights.items():
        key = key.replace("cfm.base_model.model.", "cfm.").replace(".base_layer.", ".")
        if m := re.match(r"(.+)\.lora_([AB])\.default\.weight$", key):
            lora.setdefault(m[1], {})[m[2]] = value
        else:
            plain[key] = value
    for module, ab in lora.items():
        w = plain[f"{module}.weight"]
        plain[f"{module}.weight"] = (w.float() + ab["B"].float() @ ab["A"].float()).to(w.dtype)
    return plain


@contextlib.contextmanager
def counting_tokens(on_token: Callable[[], None] | None):
    """The GPT's decoding loop is ``for _ in tqdm(range(1500))`` (one step per
    token, stopping at end of speech): count its steps instead of printing a bar."""
    from ...vendor.gpt_sovits.AR.models import t2s_model

    original = t2s_model.tqdm

    def counted(iterable, *args, **kwargs):
        for item in iterable:
            if on_token:
                on_token()
            yield item

    t2s_model.tqdm = counted
    try:
        yield
    finally:
        t2s_model.tqdm = original


def split_text(text: str) -> list[str]:
    """Chunks of up to four sentences (upstream's default cut), very short ones
    merged into the one before, each ending in punctuation."""
    from ...vendor.gpt_sovits.TTS_infer_pack.text_segmentation_method import cut1

    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    for line in cut1(text).split("\n"):
        line = line.strip()
        if not line or set(line) <= SPLITS:
            continue
        if chunks and len(line) < 5:
            chunks[-1] += line
        else:
            chunks.append(line)
    return [c if c[-1] in SPLITS else c + "。" for c in chunks]
