# Polyvox Studio guide

The details behind the [README](../README.md): how each part works, what the
settings do, and how to configure and develop the studio.

- [Synthesizing](#synthesizing)
- [Converting voices (RVC)](#converting-voices-rvc): covers of songs, voices from the RVC WebUI
- [Preparing datasets](#preparing-datasets): sources, steps, clean-up, checks, review, export
- [Training](#training)
- [Sharing voices](#sharing-voices)
- [Speed](#speed)
- [Hardware](#hardware)
- [Models](#models)
- [Configuration](#configuration)
- [Development](#development)

## Synthesizing

At the top of the editor, choose the **model**, then its **voice**: the base model,
or a voice trained or saved for it. Choosing a model again brings back the voice
you last used with it, and a model that can't run on this computer says why in the
list. The settings column shows the chosen voice; a voice with several
checkpoints lets you pick one there. Then:

- **Text**, with the model's inline tags a click away: VoxCPM2's sounds and
  reactions (laugh, sigh, ...); S2-Pro's emotions, sounds, delivery and pauses,
  plus any description in brackets (**Custom…**), which acts on the words after
  it; Drama 3's point tags ("[sighs]", at that spot) and tags around words
  ("<whisper>…</whisper>": select the words, then pick the tag). Tags a model
  doesn't know are marked in the text.
- **Style**: VoxCPM2 takes a description of the voice, e.g. "a calm, warm
  voice, speaking slowly"; Drama 3 a direction for how to say the text, e.g.
  "with bright eager excitement".
- **Reference audio**: each model shows its own inputs. VoxCPM2 has a
  Reference (who it sounds like) and a Prompt (speech it continues from, with
  its transcript); dots.tts one clip; GPT-SoVITS a prompt clip and timbre clips;
  S2-Pro and Drama 3 a reference with its transcript. Each can be an upload or **Dataset
  clips**: three of a dataset's clips, joined into one for models that take a
  single clip. They are chosen as the clips **most like the voice** (measured
  with a speaker model in seconds; the default for Drama 3) or those with the
  **cleanest transcripts** (the default for the others); each can be heard or
  removed, and **Other clips** brings the next ones. With Drama 3, **Test by
  speaking** goes further: a few sets each speak the same sentences and the one
  that sounds most like the speaker is put in (about a minute; see saved voices
  below). **Transcribe** fills in a
  transcript. On every model, about 20 s of reference clones clearly better
  than one short clip.
- **Advanced**: guidance, steps, seed and the model's own settings; for Drama 3
  a language hint, a length in seconds (it keeps to it closely; for a text
  short enough for one take) and **Reference strength**: each frame is predicted
  with and without the reference, and what the reference adds is strengthened,
  which can bring a take a little closer to the voice. It is off (0) by default: it
  costs speed (the backbone runs twice per frame), and in a blind listening test
  it was not preferred.
- **History** keeps this session's clips to compare and download. **Edit
  words** changes what a clip says without generating it again: change, add or
  remove words in its text and **Apply**. With Fish Audio Drama 3 downloaded,
  this works on clips of any model: Drama 3 speaks only the changed words again
  (the whole words around a change, plus the word after an insertion or a
  deletion), in the voice of the audio around them, and the rest of the clip is
  kept as it was. The edited clip is added to History, showing where its new
  audio is.

GPT-SoVITS always speaks from a reference clip, and S2-Pro voices trained here
keep the clips they were trained with: each such voice has a default reference,
picked from its dataset, that you can change in the settings column.

Drama 3 voices are saved references, with no training: with Drama 3's base
model chosen, **Save as a voice…** asks for a name and its clips: dataset clips,
chosen and shown just as in Synthesize (the voice is saved with exactly those,
about 20 s joined), or a clip with its transcript. The voice is listed with the others and speaks with its reference whenever no other clip
is given; change the clips in the settings column. (Training Drama 3 was measured
first: with a reference its codebook decoder already renders a voice as well as
the real recordings, so a LoRA there would add little.)

Which clips make the reference matters a lot for Drama 3: it takes the voice's
pitch and pace from them, and the clips with the cleanest transcripts aren't
always typical of the speaker. **Most like the voice** usually gets better
ones; **Test by speaking** checks by listening, in effect: three sets (the
cleanest transcripts, the clips most like the voice, and clips like the voice
with its usual pace and pauses) each speak the same dozen sentences from the
dataset, and the set whose takes sound most like the speaker is put in: the most
similar voice, with the pitch closest to the real recordings of those sentences.
It takes about a minute; the ⓘ beside its note lists each set's scores. On one
mixed dataset this raised the likeness from 0.65 to 0.75; where the cleanest
clips are already typical, it keeps them.

The first generation after a model loads takes longer while the model is
compiled for your GPU: from a few seconds to about half a minute, depending on
the model and the computer, and several minutes the very first time (torch then
keeps the compiled code on disk for later starts). Later generations are fast.
The progress shows how long loading and compiling took the last time on your
computer.

**Fish Audio Drama 3** clones zero-shot only (no training yet), at about 0.6 s
per second of audio on a 16 GB GPU. The model's state after a reference is
kept, so the lines after the first one with the same reference start sooner. It is a 35B mixture of experts, of which 3B
work for each frame: its routed experts (17 GB in 4 bits) stay in the computer's
memory, and the most used ones are also kept on the GPU (the rest are read over
PCIe as needed). The conversion needs no calibration data: the experts are
stored as HIGGS (each block of 128 inputs rotated by a Hadamard matrix, then
rounded to 16 levels fitted to a Gaussian), which on real speech came closer to
the official weights than plain 4-bit rounding or GPTQ. A long text is spoken in parts of whole sentences (a line
break ends one too), about 50 Chinese characters or 25 English words each: within
one take the voice slowly drifts from the reference, and each part starts from
the reference again. The editor marks where each part ends and counts
them; a blank line (the **New take** button puts one at the cursor) always starts
a new part, so you can choose the cuts yourself. Without a reference or a saved
voice there is no voice to keep, and only texts too long for one take are cut. Each part gets the same reference, style and hints. A take
that runs far longer than its text is drawn again once.

Editing (History's **Edit words**) follows the "speech editing by infilling"
format of Drama 3's technical report: the audio before and after the change
and the clip's whole new text go into the prompt, and the model speaks the gap.
The words are found on the clip with the forced aligner (on the CPU when
Drama 3 holds the aligner's GPU). A new part that comes out much longer than
the words it replaces has run on into the words after it: it is made again
with one more word, and once more; when all three run on, the edit is refused
with a hint to change a few more words. An edit takes about 2 s.

## Converting voices (RVC)

The **Convert** tab sings or speaks a recording again in another voice, with
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
(Retrieval-based Voice Conversion, v2 with pitch). No text is involved: the melody,
timing and words come from the recording, the timbre from the voice. The typical
use is a cover of a song in a trained voice.

Choose a recording (a **song**, or **vocals or speech only**) and a voice, then
**Convert**. The side panel lists the steps, each with its option:

1. **Take the vocals out of the music** (songs): the vocal separation model of the
   Clean up step; the instrumental is kept for the mix.
2. **Remove reverb** (on by default): reverb tails turn into noise in the converted
   voice, and most studio vocals have some.
3. **Split lead and backing vocals** (songs; on by default): UVR's MDX-Net
   Karaoke 2 model. RVC follows one pitch at a time, so harmonies sung over the lead
   would blur into it; split, the lead is converted on its own, and the backing
   vocals are kept as sung or converted separately.
4. **Convert** with the voice's settings: **Key change** in semitones (e.g. +12 for
   a voice an octave above the singer), **Index strength** (how much of the voice's
   own training features replace the singer's: more like the voice, slightly less
   clear words), **Protect consonants**, **Pitch clean-up** (folds back the pitch
   tracker's octave jumps on reverb, harmonies and breathy notes; ours, not part of
   RVC) and **Loudness of the voice**.
5. **Mix back** over the instrumental, which moves with a key change so the song
   stays in tune.

The results list the cover and the parts it was made from (converted vocals,
instrumental, the vocals used, lead and backing), to play and download. A voice
trained on speech alone often can't follow a singer's high or low notes: singing
in its training clips helps, and so does a key change towards its range.

Voices come from three places: trained in the Train tab (see [Training](#training)),
imported from the RVC WebUI or Applio (**From the RVC WebUI…**: the voice's `.pth`,
v2 with pitch at 32, 40 or 48 kHz), or added as a voice package. An imported voice's
index is rebuilt here, from the dataset it was trained on or from the WebUI's logs
folder (its `total_fea.npy`); the WebUI's `.index` files aren't read.

## Preparing datasets

### Sources

The Prepare tab turns raw recordings and transcript lists into datasets in the
data folder. **New dataset** makes an empty one; on its page, **Add source** takes
a path on this computer:

- a GPT-SoVITS `.list` (`path|speaker|lang|text`),
- a `file|text` CSV (LJSpeech style),
- a `.jsonl` manifest, or
- a folder of audio files, with same-name `.txt` / `.lab` transcripts or without.

Lists often hold stale paths (another machine, a moved folder). Each clip is
looked up as written, by file name next to the list and in its `wavs/` folder,
and by file name in an optional audio folder. **Scan** shows what was found
before anything is written; rows without audio are skipped. Clips go into
`data/<name>/train.jsonl` and, by default, are copied into `data/<name>/wavs/` so
the folder is self-contained. Long recordings (over 20 s) are listed in
`data/<name>/dataset.json` and stay where they are until they are sliced.
Sources can be added at any time; audio already in the dataset is skipped.

### Steps

The page lists the dataset's steps in order: **Sources**, **Clean up**
(optional), **Slice recordings**, **Transcribe**, **Check clips** and **Tidy**.
Each step shows what still needs it (a dataset may mix recordings and ready
clips, or have stopped halfway), keeps its settings for next time, works only on
what is left, and saves as it goes, so a stopped step loses little. **Run the
remaining steps** runs the needed ones in order; Add source can start that right
after adding. Steps run in the background with progress and can be cancelled.

**Slicing.** Long recordings (WAV, FLAC, mp3, m4a, video files, ...) are read 10
minutes at a time, so hours of audio are fine. Voice activity detection (Silero
VAD) finds the speech, Qwen3-ASR transcribes it and a forced aligner places every
word. Cuts fall only between words, at real pauses, preferring sentence ends and
keeping clips within a length range (2-15 s by default); each clip keeps its
exact part of the transcript, punctuation included.

**Transcribing.** Clips without text are kept, and the Transcribe step fills
them in with Qwen3-ASR, optionally with a fixed language and names or terms to
spell right; until then they are left out of training. Transcribed clips are
listed in `data/<name>/asr.json` so they can be reviewed. By default the step
also shortens runs of one filler in every text (哈哈哈哈 → 哈哈, 嗯嗯嗯 → 嗯嗯): for
training, their count matters less than consistency. Different fillers stay
different (哈哈 is a laugh, 呵呵 a "he he"), and transcripts keep plain characters
rather than tags like `[laughing]`, which can add stray sounds.

**Tidying.** The Tidy step trims edges (long silence, and a sigh or 嗯 a pause
away from the words, keeping 0.15 s) and can bring every clip to one loudness.
Clips flagged Mismatch or Missing words are left for review. Clips are rewritten
in place; originals are kept in `originals/` next to the manifest.

**Deleting.** **Delete…** on a dataset's page removes its folder after showing
what goes with it (clips, size, training runs that used it) and asking to
confirm.

### Clean-up: separation, de-reverb, denoise

The optional Clean up step cleans the audio with the RoFormer models that UVR5
and MSST-WebUI use (architectures vendored from ZFTurbo's Music-Source-Separation-
Training, MIT): **separate** the voice from music and background, **remove
reverb**, **remove noise**. Each is off by default and available once its model
is downloaded (Models tab). Clips are cleaned in place; recordings are cleaned
while they are sliced. The de-reverb weights are GPL-3.0.

The default vocal model is a "bleedless" one: for training data, music left in
the voice is worse than a slightly thinner voice. Separation adds time and
leaves small artifacts, so use it for recordings with music or background under
the voice, not for clean ones. It does not remove singing.

Any MSST-style checkpoint works: set `POLYVOX_VOCALS_MODEL`,
`POLYVOX_DEREVERB_MODEL` or `POLYVOX_DENOISE_MODEL` to `model.ckpt` (its config:
the same-name or only `.yaml` beside it) or `model.ckpt|config.yaml`.

### Checking clips

Forced alignment places each word of a transcript in its audio. The **Check
clips** step uses it, and a few other measures, to flag clips for review (new
clips and edited texts only; **Check all again** redoes every clip):

| Flag | Meaning |
|---|---|
| Singing | a sung clip (karaoke, a cover): not wanted in training data |
| Other voice? | the voice is unlike the rest of the dataset (CAM++ speaker embeddings): maybe a guest or played-back audio |
| Clipped | the waveform is cut flat at its peak (recorded or saved too loud) |
| Mismatch | part of the text does not match the audio |
| Missing words? | speech recognition hears words the transcript lacks (with **Compare with ASR**) |
| Filler? | a filler or laugh in the text (嗯, 哈哈, 呵呵, ...) placed on quiet audio: maybe not said |
| Edge sound | sound before or after the words, a pause apart (a sigh, 嗯, a laugh, or words the text misses) |
| Untranscribed | speech right next to the aligned words: the text may miss something |
| Cut edge | the sound is still loud at the start or end of the file: a word may be cut |
| Long silence | over 0.5 s of silence after the speech (the model learns when to stop from it), or over 1 s before it |

Two aligners are available: **Qwen3-ForcedAligner** (11 languages, mixed
scripts) and a **wav2vec2 CTC** model (sharper on single wrong words; Chinese by
default, `POLYVOX_CTC_ALIGN_MODEL` for another language). A transcript that
leaves out a few words is mostly not caught by alignment, so **Compare with
ASR** transcribes each clip again and flags Missing words. Singing detection
(an AudioSet tagger) is off by default: it slows checks down and most datasets
have no songs. Results are saved in `data/<name>/align.json`.

### Reviewing clips

The clips table can show only flagged clips, worst first. Each row shows the
clip's waveform with the span the text covers shaded; click it to play from that
point. Open a clip to see every word where the aligner placed it (words it is
unsure of are marked) and:

- **Edit** its transcript (or **Use ASR text**); saving checks that clip again.
- **Trim…**: drag the handles on the waveform; letting go plays the new edge.
- **Check again**, or **Ask ASR** to compare with what speech recognition hears.
- **Drop** it: it stays in the table, dimmed, but training leaves it out
  (listed in `data/<name>/excluded.json`); **Restore** brings it back.

**Trim all** trims every clip flagged Edge sound or Long silence (the table greys
what it would cut), and with one flag shown, **Drop all** drops every clip that
has it, e.g. all sung clips.

### Exporting for other TTS tools

**Export for other TTS** on a dataset's page writes its training clips (dropped
ones left out) in another tool's layout, by default to `exports/<dataset>_<format>`
next to `data/`:

| Format | Files |
|---|---|
| GPT-SoVITS | `<speaker>.list`: `path\|speaker\|language\|text`, language `zh` / `ja` / `en` / `ko` from each text's script (or one forced for all) |
| LJSpeech | `metadata.csv`: `id\|text\|text` |
| CSV (F5-TTS) | `metadata.csv` with an `audio_file\|text` header |
| Kaldi | `wav.scp`, `text`, `utt2spk`, `spk2utt` |
| Folder | `wavs/<id>.wav` with `wavs/<id>.lab` |

Audio goes to `wavs/` as WAV with plain numbered names, copied or resampled; an
optional clip length range leaves out clips outside it (GPT-SoVITS suggests 1-15 s).

## Training

The Train tab starts, stops and resumes runs, one at a time. **New run** starts
from a dataset (its stats, warnings and clips to browse), then the model and the
training settings; the name is suggested from the dataset and model.

| Model | What is trained | Defaults |
|---|---|---|
| VoxCPM2 | a LoRA (rank 32) on the language models and the diffusion decoder | 4 epochs, Prodigy Schedule-Free |
| dots.tts | a LoRA (rank 32) on the language model and the diffusion head | 4 epochs, Prodigy Schedule-Free |
| GPT-SoVITS v2Pro / v2ProPlus | both models in full, SoVITS then GPT, as the webui does | 8 + 15 epochs |
| GPT-SoVITS v4 | a LoRA on the SoVITS diffusion model, then the GPT in full | 8 + 15 epochs |
| Fish Audio S2-Pro | a LoRA (rank 32) on the fast transformer, which adds each audio frame's detail | 4 epochs, Prodigy Schedule-Free |
| RVC v2 | the whole voice model, from RVC's base models, as the WebUI does | 100 epochs, a voice saved every 25 |

- **Stop** saves a checkpoint first; **Resume** continues from it, and
  **Continue to** adds more steps to a finished run.
- **Progress** shows the loss charts (each folds away), the learning rate,
  and, from clips held out of the training set, validation loss and audio samples
  at each checkpoint. **Try** opens a checkpoint in Synthesize. Logged losses are
  means over the log interval.
- **Speaker similarity** scores each sample against the training set's voice
  (CAM++ speaker embeddings) next to the score the set's own clips reach, so you
  can pick the checkpoint that sounds most like the speaker.
- **Optimizer:** Prodigy Schedule-Free
  ([prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free))
  finds its own step size for each part of the model, so the learning rate is a
  multiplier (1). AdamW with a cosine schedule and Prodigy are the alternatives.

**S2-Pro** learns the voice from its dataset and keeps three of its clips as
the voice's reference, which it speaks with unless you give another. Half the
training sequences clone another clip of the dataset, as synthesis with a
reference does. Training clips get a faint noise floor (-72 dBFS): in separated
vocals, stretches of digital silence taught the model long pauses mid-sentence.
The large slow transformer stays frozen; its output for each clip is computed
once and kept in the dataset's `.cache/` (about 1 GB per hour of audio), so later
runs on the same dataset start training right away. Under Advanced, **Cached
references per clip** sets how many reference variants are kept, and **Slow
transformer in CPU memory** cuts training's GPU memory from about 12 GB to 3.5 GB
(it comes back for the samples, about 1 s each way). **Also
train the slow transformer** is experimental and off by default: it needs AdamW
with its own small learning rate, and in our tests it sounded no more like the
voice. Synthesis needs a 16 GB GPU and training a 12 GB one (below 16 GB the
slow transformer always waits in CPU memory and can't train itself); the recipe
was tuned on one voice.

A run lives in `lora/<model>/<name>/` (`voxcpm`, `dots_tts`, `gpt_sovits` or `fish_s2`). Runs
from VoxCPM's `lora_ft_webui.py` use the same layout, so they appear too and can be
resumed or continued. Runs and voices lying directly in `lora/` (from older
versions, or copied in) are moved into their model's folder when the server
starts.

**RVC** cuts the training clips into pieces of up to 3.7 s, as the WebUI does, at 32,
40 or 48 kHz (**Sample rate**, 40 kHz by default), and caches their content
features and pitch per clip in the dataset. It reads only the audio, but like the
other models it trains on the dataset's transcribed clips. Each saved voice comes
with samples: held-out clips converted into it. **Try** opens it in the Convert
tab. Singing clips in the dataset help the voice sing.

## Sharing voices

Select a voice and choose **Export** in the settings column (an RVC voice: in the
Convert tab). A LoRA zip holds only
the weights (bfloat16 by default: half the size, same output), its configuration
with the base model as a Hugging Face id, and a README with usage. A GPT-SoVITS
voice exports both models in the GPT-SoVITS webui's format, with its reference
clip and transcript; an S2-Pro voice includes its reference too, and its README
states the weights' non-commercial license. An RVC voice exports its voice file in
the RVC WebUI's format (usable there and in Applio) and its index as content
features (no audio). Local paths, optimizer state, logs and other dataset audio are
left out.

**Voice package…** at the foot of the voice list adds such a zip (RVC packages also
in the Convert tab). **GPT-SoVITS…** imports a pair
of weights trained in the GPT-SoVITS webui (v2, v2Pro, v2ProPlus or v4), with a
reference clip from a dataset or a file.

## Speed

Synthesis and training run on optimized paths written for this studio. Each was
checked against the original code: the same loss curves in training, and in
synthesis the same output up to floating-point rounding (with the same speaker
similarity and transcripts), so the speed costs nothing in quality. Measured on
a 16 GB NVIDIA GPU; the baseline is the model's own code where it has one, else a
plain PyTorch loop:

| | What is done | Gain |
|---|---|---|
| Switching voices (VoxCPM2, dots.tts) | LoRAs of any rank are written into the running, compiled model | about 1 s instead of a 30-40 s reload |
| dots.tts synthesis | compiled; every decoding step replayed as a CUDA graph | about 4x faster |
| GPT-SoVITS synthesis | the GPT's decoding step as one CUDA graph with a static cache and fused layer math; v4's diffusion model compiled | about 2.5x faster (v2Pro / v2ProPlus), 1.3x (v4) |
| S2-Pro synthesis | our own implementation: each audio frame (the slow step and the fast transformer's ten) as one CUDA graph, the steps between its linear layers fused | about 0.6 s per second of audio |
| Drama 3 synthesis | our own decoding step, each audio frame one CUDA graph, the steps between its linear layers fused; the 4-bit weights read in Triton kernels, the experts from GPU slots or straight from CPU memory, so routing needs no CPU; the codebook tower in 8 bits (its codes' cross-entropy on real clips +0.02%); a voice's prompt states kept | about 10x faster than a plain loop over transformers' modules |
| VoxCPM2 training | frozen weights in bfloat16 (half the memory, so batches of 4 fit 16 GB); each clip's audio encoded once; transformer layers fused | about 1.6x faster per step; 2.3x on 8 GB cards |
| dots.tts training | each clip encoded once and cached on disk; no vocabulary logits; fused layers | about 3x faster per step |
| GPT-SoVITS training | no per-parameter GPU syncs; features cached per clip in the dataset; v4's diffusion blocks compiled | about 1.2x faster per step; starts in seconds |
| S2-Pro training | the frozen slow transformer's output cached per clip in the dataset; the fast transformer's layers compiled | about 2.8x faster per step; later runs skip the first pass |
| Preparing datasets | compiled vocal separation; voice detection and speaker embeddings batched on the GPU | separation 2x, voice detection 3.5x, the other-voice check about 10x |

Compiling takes a little time once per model load (and longer on the very first
run, while caches fill); `POLYVOX_OPTIMIZE=0` turns it off.

## Hardware

[uv](https://docs.astral.sh/uv/) installs Python 3.13, PyTorch and everything
else on the first run, and `run.bat` / `run.sh` pick the PyTorch build for the
machine (`POLYVOX_TORCH` overrides):

| Build | For |
|---|---|
| `cu130` | NVIDIA RTX 20 series or newer (incl. RTX 50) with a driver for CUDA 13 |
| `cu126` | older NVIDIA GPUs (e.g. GTX 10 series) or drivers (CUDA 12.6) |
| `rocm` | AMD GPUs on Linux (ROCm 7.2: recent Radeon and Instinct; untested) |
| `cpu` | no GPU: datasets can be prepared, but synthesis is slow and training impractical |

The first start downloads several GB (PyTorch alone is about 3 GB on Windows). Before
it, `run.bat` / `run.sh` ask whether to use mirrors in mainland China, suggesting yes
when the system is set to China's time zone or a zh-CN locale: Tsinghua University's
PyPI mirror, Nanjing University's for PyTorch and for Python itself, and hf-mirror.com
as the Models tab's default source (a source chosen there still wins). The answer is
kept in `.mirror` in the studio folder (`cn` or `off`; updates keep it; change or
delete it to choose again, or set `POLYVOX_MIRROR`). With the mirrors, the scripts
point the download addresses in `uv.lock` at the same files on the mirrors and install
with `--frozen`: the versions stay exactly those locked, and uv checks each file
against its locked hash. Turning the mirrors off points the addresses back.

The studio picks batch sizes and which models share the GPU from its memory:

| GPU memory | Synthesis | Training | Preparing datasets |
|---|---|---|---|
| 16 GB or more | yes | yes | all models stay loaded |
| 12 GB | yes; unloaded while preparing | yes; some models with gradient checkpointing (slower) | smaller batches |
| 8 GB | yes; reference denoising on the CPU | yes, with gradient checkpointing (slower) | one model on the GPU at a time |
| under 8 GB | no (needs about 8 GB) | no | as for 8 GB |

Fish Audio S2-Pro is the exception: synthesis needs 16 GB (its weights alone are
9 GB in bfloat16) and training 12 GB. Fish Audio Drama 3 needs a 12 GB GPU and
32 GB of RAM; a larger GPU keeps more of its experts and runs faster.

Smaller cards keep the effective batch, so they train slower, not differently. A
batch that still runs out of memory is retried in halves. Starting a training run
unloads the synthesis model if both use the same GPU; to keep synthesizing while
training, put them on different GPUs (e.g. `POLYVOX_DEVICE=cuda:1`).

No FFmpeg install is needed: audio in other formats (m4a, video files) is
decoded with PyAV, which brings its own. On Linux, Triton (used to compile the
models) needs a C compiler (`apt install build-essential`).

To reach the studio from another machine, start it with
`./run.sh --host 0.0.0.0`. It has no login: use it on a trusted network or
through an SSH tunnel. It answers to `localhost` and IP addresses; to open it by
a machine name, add the name to `POLYVOX_ALLOWED_HOSTS`. Changes sent from
other websites open in the browser are refused.

## Models

The **Models** tab lists every model the studio uses, grouped into text to speech,
voice conversion, dataset tools and audio clean-up, with its size, license and whether it is
downloaded, and downloads the missing ones with progress into `models/`. None is
required: download the speech models you want to use, and the others for what
they do (until one speech model is there, the other tabs point to the Models tab):

| Model | For | Size |
|---|---|---|
| VoxCPM2 | synthesis and LoRA training | 5 GB |
| dots.tts SOAR | synthesis and LoRA training | 4.9 GB |
| GPT-SoVITS v2Pro / v2ProPlus, G2PW | synthesis and voice training (G2PW: Chinese pronunciation) | 1.7 + 0.6 GB |
| GPT-SoVITS v4 | synthesis and voice training at 48 kHz | 0.8 GB |
| Fish Audio S2-Pro | synthesis and LoRA training at 44.1 kHz; research and non-commercial use only | 11 GB |
| Fish Audio Drama 3 | zero-shot synthesis at 44.1 kHz; a 4-bit build of the official weights (`PineCookie/drama-3-int4-higgs`, made by `convert.py`), or (chosen in its row) the official weights (39.6 GB) downloaded and converted on your computer to the same files, then removed or kept; the conversion is also the fallback when the build can't be downloaded; Fish Audio Community License; Built with Fish Audio | 22 GB |
| Qwen3-ASR 1.7B | transcribing, slicing, Compare with ASR | 4.1 GB |
| Qwen3-ForcedAligner 0.6B | slicing and checking clips | 1.9 GB |
| wav2vec2 Chinese | second aligner for checks | 1.3 GB |
| CAM++, ZipEnhancer, AST | speaker similarity and Other voice; denoising reference clips; singing detection | 0.4 GB |
| RVC pitch tracker (RMVPE), ContentVec | voice conversion (Convert tab) | 0.2 + 0.4 GB |
| RVC v2 base models | training RVC voices (32, 40 and 48 kHz) | 0.65 GB |
| RoFormer vocals / de-reverb / denoise | the optional Clean up step; the Convert tab's vocals and de-reverb steps | 0.9 GB each |
| UVR MDX-Net Karaoke 2 | splitting lead and backing vocals (Convert tab) | 53 MB |

Downloads come from Hugging Face, a mirror of it (hf-mirror.com, or any URL) or
ModelScope, fetch only the files that are used, and resume when interrupted. The
source chosen in the Models tab is kept in the models folder (`.download-source`;
until one is chosen, `POLYVOX_HF_ENDPOINT` sets it). A model downloaded while the
server runs is used from its next load, no restart needed. A model that is not
downloaded and doesn't say "option unavailable" downloads itself into the models
folder, from the same source, the first time a feature needs it: Synthesize,
Edit words and the dataset steps show its progress (and the steps' Cancel stops
it). If that fails (no network, the site blocked), the message names the model
and points to the Models tab; without any network that can take a few minutes,
as each file is retried.

## Configuration

Environment variables, all optional:

| Variable | Default | |
|---|---|---|
| `POLYVOX_MODELS_DIR` | `models` | downloaded models |
| `POLYVOX_HF_ENDPOINT` | | Hugging Face mirror for downloads, e.g. `https://hf-mirror.com` (until one is chosen in the Models tab) |
| `POLYVOX_MIRROR` | (`.mirror`) | `run.bat` / `run.sh`: `cn` installs from mirrors in mainland China, `off` doesn't (see [Hardware](#hardware)) |
| `POLYVOX_UPDATE_CHECK` | `1` | ask GitHub at startup and once a day whether a newer release is out (`0`: never) |
| `POLYVOX_ALLOWED_HOSTS` | | host names the server answers to besides `localhost` and IP addresses, comma-separated (e.g. a machine name when it listens on the network) |
| `POLYVOX_LORA_DIR` | `lora` | training runs, LoRAs and voices |
| `POLYVOX_DATA_DIR` | `data` | datasets |
| `POLYVOX_DEVICE` | `auto` | synthesis device: `cuda`, `cuda:1`, `cpu`, ... |
| `POLYVOX_TRAIN_GPU` | | CUDA index for training (default: the first GPU) |
| `POLYVOX_ASR_DEVICE` | `auto` | speech recognition and aligner device; `auto` = the synthesis device |
| `POLYVOX_VRAM_GB` | | use at most this much GPU memory (a shared GPU), and pick the settings for it |
| `POLYVOX_OPTIMIZE` | `auto` | compile models with Triton (`torch.compile`); `auto` = on for CUDA |
| `POLYVOX_DENOISER` | `1` | load the ZipEnhancer denoiser for reference clips |
| `POLYVOX_MODEL` | `openbmb/VoxCPM2` | VoxCPM2: a local folder or Hugging Face id |
| `POLYVOX_DOTS_MODEL` | `rednote-hilab/dots.tts-soar` | dots.tts: a local folder or Hugging Face id |
| `POLYVOX_GSV_MODEL` | (downloaded copy) | GPT-SoVITS: a folder laid out like `lj1995/GPT-SoVITS` (v2Pro, s1v3.ckpt, encoders) |
| `POLYVOX_GSV_G2PW_MODEL` | (downloaded copy) | GPT-SoVITS: the G2PW folder |
| `POLYVOX_GSV_V4_MODEL` | (downloaded copy) | GPT-SoVITS v4: a folder with `s2Gv4.pth` and `vocoder.pth` |
| `POLYVOX_S2_MODEL` | (downloaded copy) | Fish Audio S2-Pro: a folder laid out like `fishaudio/s2-pro` |
| `POLYVOX_DRAMA3_MODEL` | (converted copy) | Fish Audio Drama 3: a 4-bit build (`python -m polyvox.engines.fish_drama3.convert` makes one from `fishaudio/drama-3`) |
| `POLYVOX_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B-hf` | speech recognition model |
| `POLYVOX_ALIGN_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | forced aligner |
| `POLYVOX_CTC_ALIGN_MODEL` | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | wav2vec2 CTC aligner (one language) |
| `POLYVOX_DENOISER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | its folder or ModelScope id |

`run.bat` / `run.sh` use the project's own `models/`, `lora/` and `data/`; set
the variables to use other folders.

## Updating

When a newer release is out, the studio shows a notice (it asks GitHub's release
list at startup and once a day; nothing else is sent; `POLYVOX_UPDATE_CHECK=0`
turns it off). To update, close the studio and run `update.bat` (`./update.sh`
on Linux) in its folder. It downloads the latest release and replaces only the
program files: `models/`, `data/`, `lora/`, `output/` and the Python environment
(`.venv/`) stay, and files of your own at the top of the folder too. The old
program files are set aside first and put back if anything fails. The next
`run.bat` installs what the new version needs. A git checkout updates with
`git pull` instead.

From 0.5.0 (which had no updater), once: unpack the new release next to the old
folder and move `models/`, `data/` and `lora/` into it.

## Development

```powershell
uv run pytest                            # no GPU needed (fake model loader and training worker)
uv run ruff check; uv run ruff format    # Python lint / format
uv run polyvox                           # API on :8808
cd web; npm run dev                      # UI on :5173, proxies /api to :8808
cd web; npm run lint                     # ESLint + Prettier check; `npm run format` fixes
```

From a git checkout, the first `run.bat` / `run.sh` also builds the web UI, which
needs Node.js 20+. GitHub Actions run the tests, linters and web build on Ubuntu
and Windows for every push; pushing a tag like `v0.4.0` publishes a release whose
archives include the built web UI.

VoxCPM is used as a library, pinned to an upstream commit in `pyproject.toml`
(PyPI 2.0.3 predates `generate(seed=...)`). To develop against a local VoxCPM
checkout, temporarily set `voxcpm = { path = "../VoxCPM", editable = true }`
under `[tool.uv.sources]`. `src/polyvox/vendor/` is excluded from Ruff: it stays
as upstream has it, with any changes listed in each package's `__init__.py`.

```
models/            downloaded models (git-ignored)
data/              datasets: <name>/train.jsonl, audio, dataset.json (git-ignored)
lora/              training runs, LoRAs and voices: voxcpm/, dots_tts/, gpt_sovits/, fish_s2/,
                   fish_drama3/ (git-ignored)
src/polyvox/
  config.py        settings from POLYVOX_* env vars
  models.py        the models used, and downloading them
  vram.py          GPU size classes (profiles): batches of the dataset tools, which models share the
                   GPU; each engine keeps its own training batches per class
  engines/         TTS engines behind one interface (base.py: Engine, Capabilities;
                   __init__.py: the registry). No engine imports another; what several share
                   is in common/. An engine describes its New-run form (Engine.train_form:
                   fields, defaults per GPU, when they show) and sizes a run (Engine.plan);
                   the UI draws the form from that. It also owns its voices' files: which
                   mark a checkpoint (checkpoint_files), its packages (package_files /
                   import_package) and voices from other tools (import_voice)
    common/        train_form.py (shared training fields, runs sized in epochs),
                   lora_package.py (LoRA packages: weights, config, README),
                   lora.py (LoRA hooks), worker.py (training-worker helpers), cache.py
                   (per-clip dataset caches), reference.py (voices' default references,
                   saved-reference voices), layers.py, fish_codec.py (Fish Audio's codec)
    voxcpm/        engine.py; inference.py (model cache, LoRA hot-swap); worker.py (training);
                   package.py (its packages)
    dots/          engine.py; lora.py (LoRA layers, merging); worker.py (training); package.py
    gpt_sovits/    engine.py; synth.py (synthesis); voices.py (voice folders, webui import);
                   features.py + worker.py (training: SoVITS then GPT)
    fish_s2/       engine.py; model.py (our implementation); decode.py (CUDA graphs);
                   lora.py, train.py, worker.py (training); package.py
    fish_drama3/   engine.py; model.py + decode.py + kernels.py (4-bit Triton kernels);
                   prompt.py; edit.py (speech editing); convert.py (FP8 -> 4 bits, experts as HIGGS)
    rvc/           engine.py; model.py (the voice model); pitch.py (RMVPE); convert.py
                   (conversion, index); train.py + worker.py (training); voices.py (import, packages)
  data/            manifests.py (training manifests); record.py (dataset.json);
                   audio_io.py (audio without a system FFmpeg)
  prep/            dataset preparation: steps.py (the dataset page's steps), importer.py,
                   slicer.py, asr.py, aligner.py, checks.py, separation.py, singing.py,
                   fillers.py, loudness.py, exporter.py, karaoke.py (lead / backing split)
  lora_library.py  checkpoint discovery (by the engines' checkpoint files) and metadata
  jobs.py          background jobs with progress and cancel
  training.py      training runs: config, job control, metrics
  speaker.py       speaker similarity (CAM++)
  sharing.py       voice packages: zips in and out, each handed to its engine
  vendor/          train_voxcpm_finetune.py (unmodified); roformer/ (MSST, MIT);
                   dots_tts/ (Apache-2.0); gpt_sovits/ (MIT; rebuilt by
                   tools/vendor_gpt_sovits.py from a pinned upstream commit)
  server/          the FastAPI app: app.py, context.py (services, GPU rules),
                   schemas.py, one module of routes per area
web/               Vite + React + TypeScript frontend
tests/             backend tests
```

### Roadmap

- Streaming synthesis: playback starts before generation ends.
- VoxCPM 1.5 as another base model.
- Keeping a long recording's own transcript when it is sliced.
