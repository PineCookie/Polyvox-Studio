"""GPT-SoVITS v2Pro / v2ProPlus synthesis on the vendored modules (vendor/gpt_sovits).

What upstream's inference_webui.get_tts_wav does for these versions, without the
webui: text -> phonemes + RoBERTa features; the reference clip -> semantic tokens
(HuBERT through the voice's quantizer), a spectrogram and a speaker embedding;
the GPT continues the reference's tokens with the text's; SoVITS decodes them
to 32 kHz audio in the reference's timbre. Long texts are cut into chunks of up
to four sentences, joined with short pauses.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REF_MIN_S, REF_MAX_S = 3.0, 10.0  # upstream refuses references outside this range
PAUSE_S = 0.3  # between chunks, and after the reference's semantic prompt
SPLITS = set("，。？！,.?!~:：—…")
SEMANTIC_HZ = 50  # upstream's early stop: max_sec x 50 tokens
BERT_DIM = 1024


@dataclass(frozen=True)
class Options:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed: float = 1.0


@dataclass
class Voice:
    """A voice's GPT and SoVITS weights, loaded."""

    key: object
    model_version: str  # v2Pro | v2ProPlus
    sovits: torch.nn.Module
    gpt: torch.nn.Module
    hps: object  # the SoVITS config (filter length, hop, sampling rate ...)
    max_sec: float
    decoder: object = None  # decode.Decoder for the GPT, made on first use (CUDA)

    @property
    def sample_rate(self) -> int:
        return int(self.hps.data.sampling_rate)


@dataclass
class Reference:
    key: object
    prompt: torch.Tensor | None  # semantic tokens [1, T]; None without a transcript
    phones: list[int]
    bert: torch.Tensor | None
    spec: torch.Tensor
    sv_emb: torch.Tensor


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
        # GPT decoding as a CUDA graph (decode.py); off: upstream's loop.
        self.fast_decode = device.startswith("cuda")
        self.dtype = torch.float16 if self.half else torch.float32
        bert_dir = str(self.base_dir / "chinese-roberta-wwm-ext-large")
        chinese2.configure(str(g2pw_dir), bert_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_dir).to(device, self.dtype).eval()
        self.hubert = CNHubert(str(self.base_dir / "chinese-hubert-base")).to(device, self.dtype).eval()
        self.sv = SV(device, self.half, str(self.base_dir / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"))

    # ------------------------------------------------------------ weights
    def base_weights(self, version: str = "v2ProPlus") -> tuple[Path, Path]:
        """(SoVITS, GPT) of a base model."""
        return self.base_dir / "v2Pro" / f"s2G{version}.pth", self.base_dir / "s1v3.ckpt"

    def load_voice(self, sovits_path: Path, gpt_path: Path, key: object = None) -> Voice:
        from ...vendor.gpt_sovits.AR.models.t2s_model import Text2SemanticDecoder
        from ...vendor.gpt_sovits.module.models import SynthesizerTrn
        from ...vendor.gpt_sovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new, load_weights

        _, model_version, lora = get_sovits_version_from_path_fast(str(sovits_path))
        if lora or model_version not in ("v2Pro", "v2ProPlus"):
            raise ValueError(
                f"{Path(sovits_path).name} is GPT-SoVITS {model_version}; only v2Pro / v2ProPlus run here."
            )
        saved = load_sovits_new(sovits_path)
        hps = Config(saved["config"])
        model = {**hps["model"], "semantic_frame_rate": "25hz", "version": model_version}
        sovits = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model,
        )
        if hasattr(sovits, "enc_q"):
            del sovits.enc_q  # training only
        missing, _ = sovits.load_state_dict(saved["weight"], strict=False)
        if missing:
            raise ValueError(f"{Path(sovits_path).name} lacks {len(missing)} SoVITS weights, e.g. {missing[0]}.")
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

        sr = voice.sample_rate
        audio16, _ = librosa.load(str(wav), sr=16000)
        seconds = len(audio16) / 16000
        if not REF_MIN_S <= seconds <= REF_MAX_S:
            raise ValueError(f"The reference clip is {seconds:.1f} s; GPT-SoVITS needs {REF_MIN_S:g}-{REF_MAX_S:g} s.")
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
            audio16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sv_emb = self.sv.compute_embedding3(torch.from_numpy(audio16k).to(self.device, self.dtype)[None])
        return Reference(key=key, prompt=prompt, phones=phones or [], bert=bert, spec=spec, sv_emb=sv_emb)

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
                    tokens, n = decoder.generate(ids_t, ref.prompt, features[None], **sampling)
                else:
                    tokens, n = voice.gpt.infer_panel(
                        ids_t,
                        torch.tensor([ids_t.shape[-1]], device=self.device),
                        ref.prompt,
                        features[None],
                        **sampling,
                    )
                semantic = tokens[:, -n:][None]
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

    def _decoder(self, voice: Voice):
        if not self.fast_decode:
            return None
        if voice.decoder is None:
            from .decode import Decoder

            voice.decoder = Decoder(voice.gpt)
        return voice.decoder


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
