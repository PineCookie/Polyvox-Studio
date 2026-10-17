// Typed client for the Polyvox Studio API (src/polyvox/server/).

export interface LoraConfig {
  r: number
  alpha: number
  dropout: number
  enable_lm: boolean
  enable_dit: boolean
  enable_proj: boolean
  target_modules_lm: string[]
  target_modules_dit: string[]
}

export interface Lora {
  id: string // its path in the LoRA folder, e.g. "gpt_sovits/speaker_a/checkpoints/step_0000100"
  engine: string // the engine it was trained for (Engine.id)
  voice_path: string // the run / voice folder it belongs to, e.g. "gpt_sovits/speaker_a"
  voice: string // that folder's name
  base_model: string | null
  // Hugging Face id guessed from base_model (the export default).
  base_model_id: string | null
  step: number | null
  rank: number | null
  alpha: number | null
  config: LoraConfig | null // VoxCPM's layout (VoxCPM LoRAs only)
  reference_text: string | null // its default reference clip's transcript (GPT-SoVITS voices); null: none
  version: string | null // model version, when recorded (e.g. v2ProPlus)
}

// A TTS model family (src/polyvox/engines/base.py).
export interface EngineParam {
  name: string
  kind: 'float' | 'int' | 'bool' | 'choice'
  default: number | boolean | string
  min: number | null
  max: number | null
  step: number | null
  choices: string[] | null // kind 'choice': the allowed values
  base_only: boolean // only used with the base model (no voice chosen)
  versions: string[] // only used by these model versions (the voice's, or base_version); empty: all
}

export type ReferenceSlot = 'reference' | 'prompt'

/** One audio input of an engine's cloning (engines/base.py ReferenceInput). */
export interface ReferenceInput {
  name: string // the engine's own name for it (labels: t.referenceInputs['<engine>.<name>'])
  slot: ReferenceSlot // the /api/tts fields it fills: <slot>_audio / <slot>_clips (+ prompt_text)
  transcript: 'none' | 'optional' | 'required'
  several: boolean // takes several clips as they are
  max_seconds: number | null // one short clip (never several joined)
  versions: string[] // model versions that have it; empty: all
  same_as: string | null // may reuse this other input's audio
}

export interface Engine {
  id: string
  name: string
  unavailable: string | null // why it can't run here
  capabilities: {
    sample_rate: number
    training_rate: number
    languages: string[]
    reference_audio: boolean
    reference_transcript: boolean
    style: boolean
    tags: boolean
    lora_hot_swap: boolean
    reference_required: boolean // always clones a clip: the request's or the voice's own
    training: boolean // New run offers it
    full_finetune: boolean // trains every weight: no LoRA rank or optimizer choice
    reference_inputs: ReferenceInput[] // the audio inputs its cloning takes
  }
  params: EngineParam[] // synthesis settings
  model_setting: string // its base model in the Models list
  loaded: boolean
}

export const DEFAULT_ENGINE = 'voxcpm' // runs and LoRAs from before engines

export async function listEngines(): Promise<Engine[]> {
  return (await request('/api/engines')).json()
}

export interface Status {
  device: string
  optimize: boolean
  model: string
  // Hugging Face id of the base model, when it can be inferred.
  model_id: string | null
  torch: string // PyTorch build, e.g. 2.14.1+cu130
  loaded: boolean
  asr_loaded: boolean
  aligner_loaded: boolean
  separation_loaded: boolean
  singing_loaded: boolean
  // Import clean-up options whose model is configured.
  cleanup: CleanupOption[]
  // Required models not in the models folder yet (names).
  missing_required: string[]
  // The GPU's memory and the settings picked for it (src/polyvox/vram.py).
  gpu: GpuProfile
  lora_rank: number | null
  active_lora: string | null
  loaded_engines: string[] // engines with a model in memory
}

export type EngineOptions = Record<string, number | boolean | string>

export interface SynthesisParams {
  engine: string
  text: string
  control: string // style (engines with capabilities.style)
  loraId: string | null
  // Audio for the engine's reference inputs, by slot.
  references: Partial<Record<ReferenceSlot, ReferenceAudio>>
  options: EngineOptions // the engine's params by name
  seed: number | null
}

export interface ReferenceClips {
  dataset: string
  indices: number[]
}

// An uploaded clip (with its transcript, where the input takes one), or dataset clips.
export type ReferenceAudio = { file: File; text: string } | { clips: ReferenceClips }

export interface ReferenceClip {
  index: number
  text: string
  seconds: number
}

/** A dataset's best clips to clone its voice from, leaving out `exclude` (indices). */
export async function pickReferenceClips(dataset: string, count: number, exclude: number[]): Promise<ReferenceClip[]> {
  const q = new URLSearchParams({ id: dataset, count: String(count) })
  for (const index of exclude) q.append('exclude', String(index))
  return (await request(`/api/dataset-references?${q}`)).json()
}

export interface SynthesisOutput {
  audio: Blob
  seed: number | null
}

async function errorMessage(resp: Response): Promise<string> {
  try {
    const body = await resp.json()
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    // FastAPI validation errors: [{ loc, msg }, ...]
    if (Array.isArray(detail)) return detail.map((d) => `${d.loc?.at(-1) ?? 'input'}: ${d.msg}`).join('; ')
  } catch {
    // Non-JSON error body.
  }
  return `${resp.status} ${resp.statusText}`
}

async function request(input: string, init?: RequestInit): Promise<Response> {
  const resp = await fetch(input, init)
  if (!resp.ok) throw new Error(await errorMessage(resp))
  return resp
}

export interface GpuProfile {
  total_gb: number | null // null: no CUDA GPU
  name: string // "16 GB or more", "12 GB", "8 GB", "under 8 GB", "CPU"
  exclusive: boolean // TTS and the preparation models take turns on the GPU
  denoise_on_cpu: boolean
  train_batch: number // VoxCPM's training micro-batch that fits; accumulation makes up the rest
  train_batches: Record<string, number> // the same per engine id
  note: string // what doesn't fit, if anything
}

export async function getStatus(): Promise<Status> {
  return (await request('/api/status')).json()
}

// Models the studio uses (src/polyvox/models.py).
export interface ModelInfo {
  setting: string
  name: string
  purpose: string
  repo: string
  hub: 'huggingface' | 'modelscope'
  size_mb: number
  license: string
  required: boolean
  // downloaded: in the models folder; custom: configured explicitly; missing: not downloaded.
  state: 'downloaded' | 'custom' | 'missing'
  path: string
  fallback: boolean // when missing, still loads (from the hub's cache); else the feature is unavailable
}

export interface ModelList {
  models_dir: string
  endpoint: string // the server's default download source ("" = huggingface.co)
  models: ModelInfo[]
}

export type DownloadJob = Job<{ downloaded: string[] }>

export async function listModels(): Promise<ModelList> {
  return (await request('/api/models')).json()
}

/** Download models into the models folder: a job (phase = the setting downloading, done / total in MB). */
export async function downloadModels(settings: string[], endpoint: string): Promise<DownloadJob> {
  return (await postJson('/api/models/download', { settings, endpoint })).json()
}

export async function listLoras(): Promise<Lora[]> {
  return (await request('/api/loras')).json()
}

export async function exportLora(id: string, name: string, baseModel: string, half: boolean): Promise<Blob> {
  const q = new URLSearchParams({ id, name, base_model: baseModel, half: String(half) })
  return (await request(`/api/loras/export?${q}`)).blob()
}

/** Unpack a shared package into the LoRA folder: its id and engine. */
export async function importLora(file: File): Promise<{ id: string; engine: string }> {
  const form = new FormData()
  form.set('package', file)
  return (await request('/api/loras/import', { method: 'POST', body: form })).json()
}

export interface GptSovitsImport {
  name: string
  sovits: string // server-local path of the webui's SoVITS weights (.pth)
  gpt: string // and of its GPT weights (.ckpt)
  // The default reference: the best clip of a dataset, or a clip (server path) and its transcript.
  dataset?: string
  reference?: string
  reference_text?: string
}

/** A voice from weights the GPT-SoVITS webui trained (v2Pro / v2ProPlus). */
export async function importGptSovits(body: GptSovitsImport): Promise<{ id: string; engine: string }> {
  return (
    await request('/api/loras/import-gpt-sovits', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  ).json()
}

/** A GPT-SoVITS voice's default reference clip. */
export function voiceReferenceUrl(id: string, version: string | null = null): string {
  const q = new URLSearchParams({ id })
  if (version) q.set('v', version) // changes when the reference does (no stale cached audio)
  return `/api/loras/reference?${q}`
}

/** A new default reference: the best clip of a dataset, or a clip and its transcript. */
export async function setVoiceReference(
  id: string,
  source: { dataset: string } | { audio: File; text: string },
): Promise<Lora> {
  const form = new FormData()
  form.set('id', id)
  if ('dataset' in source) {
    form.set('dataset', source.dataset)
  } else {
    form.set('audio', source.audio)
    form.set('text', source.text)
  }
  return (await request('/api/loras/reference', { method: 'POST', body: form })).json()
}

export async function unloadModel(): Promise<Status> {
  return (await request('/api/model/unload', { method: 'POST' })).json()
}

export async function transcribe(audio: File, language?: string): Promise<{ text: string; language: string | null }> {
  const form = new FormData()
  form.set('audio', audio)
  if (language) form.set('language', language)
  return (await request('/api/asr', { method: 'POST', body: form })).json()
}

// The running synthesis (src/polyvox/inference.py Progress).
export interface TtsProgress {
  phase: 'idle' | 'loading' | 'preparing' | 'compiling' | 'generating'
  step: number
  expected: number | null // an estimate: the text decides when generation stops
  attempt: number // VoxCPM retries a generation that runs far too long
  elapsed: number // seconds
}

export async function getTtsProgress(): Promise<TtsProgress> {
  return (await request('/api/tts/progress')).json()
}

export async function synthesize(p: SynthesisParams): Promise<SynthesisOutput> {
  const form = new FormData()
  form.set('engine', p.engine)
  form.set('text', p.text)
  form.set('control', p.control)
  if (p.loraId) form.set('lora_id', p.loraId)
  for (const [slot, audio] of Object.entries(p.references)) {
    if ('clips' in audio) {
      form.set(`${slot}_clips`, JSON.stringify(audio.clips))
    } else {
      form.set(`${slot}_audio`, audio.file)
      if (slot === 'prompt' && audio.text.trim()) form.set('prompt_text', audio.text)
    }
  }
  form.set('options', JSON.stringify(p.options))
  if (p.seed !== null) form.set('seed', String(p.seed))

  const resp = await request('/api/tts', { method: 'POST', body: form })
  const seed = resp.headers.get('X-Seed')
  return { audio: await resp.blob(), seed: seed === null ? null : Number(seed) }
}

// Training

export interface DatasetReport {
  id: string
  samples: number
  dropped: number // dropped in review: not in training, nor in the numbers here
  untranscribed: number // clips without text yet: not in training
  short_clips: number // shorter than 0.3 s
  long_clips: number // longer than 30 s
  valid: number
  total_seconds: number
  min_seconds: number | null
  median_seconds: number | null
  max_seconds: number | null
  sample_rates: Record<string, number>
  errors: string[]
  warnings: string[]
}

export type RunStatus = 'running' | 'stopping' | 'complete' | 'incomplete' | 'failed'

export interface RunSummary {
  name: string
  engine: string
  path: string // its folder in the LoRA folder ("<engine>/<name>"): its checkpoints' ids start with it
  status: RunStatus
  step: number | null
  saved_step: number | null
  num_iters: number
  lora_rank: number | null
  lora_alpha: number | null
  train_manifest: string
  val_manifest: string
  updated: number
}

// { step, "loss/diff", "loss/stop", lr, ... } as logged by the training script.
export type MetricPoint = { step: number } & Record<string, number>

export interface RunDetail extends RunSummary {
  config: Record<string, unknown>
  train: MetricPoint[]
  val: MetricPoint[]
  seconds_per_step: number | null
  samples: { step: number; index: number; file: string }[]
  references: string[]
  model_inputs: string[] // reference_<i>_16k.wav: the references as the model hears them
  sample_texts: string[]
  val_clips: number
  log: string[] // console lines: the tail, or with logFrom the lines after it
  log_next: number // logFrom for the next poll
  log_partial: boolean // the last line may still change: the next poll's first line replaces it
  log_appended: boolean // log continues the poller's lines
  since: number | null // set: train / val hold only the points after this step
}

const CONSOLE_LINES = 200 // as many as the server's tail

export const OPTIMIZERS = ['adamw', 'prodigy', 'prodigy_sf'] as const
export type Optimizer = (typeof OPTIMIZERS)[number]

export interface TrainParams {
  name: string
  engine: string
  dataset: string
  val_dataset: string | null
  val_holdout: number
  num_iters: number
  batch_size: number
  grad_accum_steps: number
  max_batch_tokens: number // clips too long for this per-batch budget are left out; 0 = no limit
  learning_rate: number
  optimizer: Optimizer
  prodigy_steps: number // prodigy_sf: step size fixed from this step on; 0 = adapts throughout
  weight_decay: number
  warmup_steps: number
  max_grad_norm: number
  log_interval: number
  valid_interval: number
  save_interval: number
  num_workers: number
  lora_rank: number
  lora_alpha: number
  lora_dropout: number
  enable_lm: boolean
  enable_dit: boolean
  enable_proj: boolean
  // GPT-SoVITS (full fine-tuning in two stages): num_iters follows from these
  sovits_epochs: number
  gpt_epochs: number
  gpt_batch_size: number
  gsv_version: string // the GPT-SoVITS base model trained from: v2ProPlus | v2Pro
}

function postJson(url: string, body?: unknown): Promise<Response> {
  return request(url, {
    method: 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

const runUrl = (name: string) => `/api/training/runs/${encodeURIComponent(name)}`

export async function listDatasets(): Promise<string[]> {
  return (await request('/api/datasets')).json()
}

export interface ImportPreview {
  kind: 'gsv' | 'csv' | 'jsonl' | 'folder'
  source: string
  rows: number
  found: number // audio and transcript
  untranscribed: number // audio, no transcript: ASR can fill it in
  missing: number
  duplicates: number
  unreadable: number
  long_recordings: number // too long for a training clip: slice them
  total_seconds: number
  median_seconds: number | null
  sample_rates: Record<string, number>
  examples: { text: string; audio: string }[]
  missing_examples: string[]
  suggested_name: string
}

export async function previewImport(source: string, audioDir: string): Promise<ImportPreview> {
  return (await postJson('/api/datasets/import/preview', { source, audio_dir: audioDir || null })).json()
}

export interface Job<R = unknown> {
  id: string
  kind: string
  title: string
  state: 'running' | 'done' | 'failed' | 'cancelled'
  phase: string
  done: number
  total: number
  /** Estimate for the current phase, from its rate so far; null until it has one. */
  seconds_left: number | null
  rate: number | null // done per second over the last few seconds (downloads: MB/s)
  error: string | null
  result: R
  subject: string // e.g. the dataset it works on
  label: string // e.g. which dataset step ('add', 'remaining', 'slice', ...)
}

// Optional clean-up before import (src/polyvox/separation.py).
export type CleanupOption = 'separate' | 'dereverb' | 'denoise'

// Forced aligners (src/polyvox/aligner.py).
export type Aligner = 'qwen3' | 'ctc'

export type CheckFlag =
  | 'singing'
  | 'other_speaker'
  | 'clipping'
  | 'mismatch'
  | 'asr_more'
  | 'filler'
  | 'edge_sound'
  | 'extra_speech'
  | 'cut_edge'
  | 'long_silence'

// Flags trimming fixes (trim-flagged).
export const TRIMMABLE_FLAGS: CheckFlag[] = ['edge_sound', 'long_silence']

// Alignment checks over a whole dataset (checks.summarize).
export interface CheckSummary {
  aligner: Aligner | null
  checked: number
  flagged: number
  flags: Partial<Record<CheckFlag, number>>
}

// Seconds per stage of a job ("separation", "asr", ...), longest first.
export type Timings = Record<string, number>

// Processing steps of a dataset (src/polyvox/prep/steps.py), in order after its sources.
export type StepName = 'clean_up' | 'slice' | 'transcribe' | 'check' | 'tidy'
export const STEP_NAMES: StepName[] = ['clean_up', 'slice', 'transcribe', 'check', 'tidy']

export interface StepSettings {
  clean_up: { models: CleanupOption[] }
  slice: { min_seconds: number; max_seconds: number }
  // language: 'auto' = estimate from the dataset, '' = detect per clip, else a code
  transcribe: { language: string; prompt: string; tidy_fillers: boolean }
  check: { aligner: Aligner; compare_asr: boolean; detect_singing: boolean }
  tidy: { trim: boolean; normalize_lufs: number | null }
}

// Until a step has run on a dataset (steps.DEFAULTS).
export const STEP_DEFAULTS: StepSettings = {
  clean_up: { models: [] },
  slice: { min_seconds: 2, max_seconds: 15 },
  transcribe: { language: 'auto', prompt: '', tidy_fillers: true },
  check: { aligner: 'qwen3', compare_asr: false, detect_singing: false },
  tidy: { trim: true, normalize_lufs: null },
}

// What each step still has to do (dropped clips don't count), and the settings it last ran with.
export interface StepStatus {
  sources: { clips: number; recordings: number; dropped: number }
  clean_up: { clips: number } // not cleaned yet
  slice: { recordings: number; seconds: number } // not sliced yet
  transcribe: { clips: number } // without text
  check: { clips: number } // with text, not checked for it
  tidy: { trimmable: number }
  settings: { [K in StepName]?: Partial<StepSettings[K]> }
  job: Job<Record<string, unknown>> | null // running on this dataset (started from a page closed since)
}

export type StepJob = Job<Record<string, unknown> & { timings?: Timings }>

export async function getStepStatus(id: string): Promise<StepStatus> {
  return (await request(`/api/datasets/steps?id=${encodeURIComponent(id)}`)).json()
}

/** Run one step on what still needs it (`everything`: check every clip again). */
export async function runStep<S extends StepName>(
  id: string,
  step: S,
  settings: StepSettings[S],
  everything = false,
): Promise<StepJob> {
  return (await postJson('/api/datasets/step', { id, step, ...settings, everything })).json()
}

/** Run the steps with work left, in order, with these settings. */
export async function runRemaining(id: string, settings: StepSettings): Promise<StepJob> {
  return (await postJson('/api/datasets/run-remaining', { id, settings })).json()
}

// What deleting a dataset removes and affects (for the confirmation).
export interface DatasetDeletion {
  folder: string
  clips: number
  bytes: number
  runs: string[] // trained on its clips: can't be resumed or continued after
  outside_recordings: number // recordings it only points at: not deleted
  blocked: 'shared' | 'job' | 'training' | null // why it can't be deleted now
}

export async function getDeletion(id: string): Promise<DatasetDeletion> {
  return (await request(`/api/datasets/deletion?id=${encodeURIComponent(id)}`)).json()
}

/** Delete a dataset's folder for good. */
export async function deleteDataset(id: string): Promise<void> {
  await postJson('/api/datasets/delete', { id })
}

/** A new, empty dataset (sources are added on its page): its id. */
export async function newDataset(name: string): Promise<string> {
  return (await (await postJson('/api/datasets/new', { name })).json()).id
}

export type AddSourceJob = Job<{
  clips: number
  recordings: number
  steps: Record<string, Record<string, unknown>> // the remaining steps' results, with `runSteps`
  timings?: Timings
}>

/** Add a source's clips and recordings; with `runSteps`, the remaining steps run after with `settings`. */
export async function addSource(
  id: string,
  source: string,
  audioDir: string,
  copyAudio: boolean,
  runSteps: boolean,
  settings: StepSettings,
): Promise<AddSourceJob> {
  const body = { id, source, audio_dir: audioDir || null, copy_audio: copyAudio, run_steps: runSteps, settings }
  return (await postJson('/api/datasets/add-source', body)).json()
}

// Loudness normalization (src/polyvox/loudness.py).
export const DEFAULT_LUFS = -20

export interface ExportFormat {
  id: string
  label: string
  description: string
  min_seconds: number | null // suggested clip length range
  max_seconds: number | null
}

export const EXPORT_LANGUAGES = ['zh', 'ja', 'en', 'ko', 'yue'] as const // GPT-SoVITS codes

export interface ExportOptions {
  format: string
  out_dir: string // '' = the server's default folder
  min_seconds: number | null
  max_seconds: number | null
  sample_rate: number | null // null = keep each clip's rate
  language: string | null // GPT-SoVITS; null = per clip from the text
}

export type ExportJob = Job<{
  dir: string
  format: string
  clips: number
  seconds: number
  skipped: { short: number; long: number; unreadable: number }
}>

export async function listExportFormats(): Promise<ExportFormat[]> {
  return (await request('/api/export-formats')).json()
}

export async function exportDir(id: string, format: string): Promise<string> {
  const query = new URLSearchParams({ id, format })
  return (await (await request(`/api/datasets/export-dir?${query}`)).json()).dir
}

export async function exportDataset(id: string, options: ExportOptions): Promise<ExportJob> {
  return (await postJson('/api/datasets/export', { id, ...options })).json()
}

/** Drop every clip whose check has the flag (e.g. all sung clips); returns how many were newly dropped. */
export async function dropFlagged(id: string, flag: CheckFlag): Promise<number> {
  return (await (await postJson('/api/datasets/drop-flagged', { id, flag })).json()).dropped
}

export interface TrimResult {
  trimmed: number
  skipped: number // can't be rewritten in place (format, or outside the dataset folder)
  review: number // also flagged Mismatch / Missing words: left for review
}

/** Trim every clip with one flag at its checked edges (originals kept in originals/). */
export async function trimFlagged(id: string, flag: CheckFlag): Promise<TrimResult> {
  return (await postJson('/api/datasets/trim-flagged', { id, flag })).json()
}

export async function getJob<J extends Job>(id: string): Promise<J> {
  return (await request(`/api/jobs/${encodeURIComponent(id)}`)).json()
}

export async function cancelJob(id: string): Promise<void> {
  await postJson(`/api/jobs/${encodeURIComponent(id)}/cancel`)
}

export interface ClipRow {
  index: number // stable id within the dataset (for audio)
  file: string
  text: string
  seconds: number | null // null: unreadable or missing
  asr: boolean // transcript made by ASR
  score: number | null // alignment check; null: not checked (for this text)
  flags: CheckFlag[]
  speech: [number, number] | null // aligned words' span in seconds, when checked
  peaks: number[] | null // waveform outline: peak level 0..1 per equal slice
  dropped: boolean // dropped in review: left out of training runs
  trim: [number, number] | null // what trimming keeps, seconds; null: not checked, or little to trim
}

export interface AlignedWord {
  text: string
  start: number
  end: number
  confidence: number
  weak: boolean // unsure enough that it is likely not what was said
}

// What ASR heard, compared with the transcript (checks.compare_asr).
export interface Heard {
  text: string
  extra: number // words ASR heard that the transcript lacks
  missing: number
  changed: number
  diff: ['equal' | 'missing' | 'extra' | 'changed', string, string][] // (op, transcript part, ASR part)
}

export interface ClipDetail extends ClipRow {
  aligner: Aligner | null
  words: AlignedWord[] // empty when not checked (or checked before word timings were kept)
  heard: Heard | null // when the check compared with ASR
  // Untranscribed sound a pause separates from the words lies before / after this second.
  sound_before: number | null
  sound_after: number | null
  speaker: number | null // similarity of the voice to the dataset's, when scored
  clipped: number | null // flat tops at the peak, when measured
}

/** Trim one clip in place to `keep` seconds, or at its checked edges (the original is kept). */
export async function trimClip(id: string, index: number, keep?: [number, number]): Promise<ClipDetail> {
  const range = keep ? { start: keep[0], end: keep[1] } : {}
  return (await postJson('/api/dataset-clip/trim', { id, index, ...range })).json()
}

/** Check one clip again (alignment; `compareAsr`: also what ASR hears). */
export async function checkClip(id: string, index: number, compareAsr = false): Promise<ClipDetail> {
  return (await postJson('/api/dataset-clip/check', { id, index, compare_asr: compareAsr })).json()
}

/** What ASR hears in part of a clip. */
export async function hearClipPart(id: string, index: number, start: number, end: number): Promise<string> {
  return (await (await postJson('/api/dataset-clip/hear', { id, index, start, end })).json()).text
}

/** Change a clip's transcript; it is checked again when the dataset has checks. */
export async function setClipText(id: string, index: number, text: string): Promise<ClipDetail> {
  return (await postJson('/api/dataset-clip/text', { id, index, text })).json()
}

/** Leave a clip out of training, or bring it back. */
export async function setClipDropped(id: string, index: number, dropped: boolean): Promise<void> {
  await postJson('/api/dataset-clip/drop', { id, index, dropped })
}

export async function getDatasetClip(source: DatasetSource, index: number): Promise<ClipDetail> {
  return (await request(`/api/dataset-clip?${new URLSearchParams({ ...source, index: String(index) })}`)).json()
}

export interface ClipPage {
  total: number // clips matching the search / filter
  rows: ClipRow[]
  checks: CheckSummary | null // over the whole dataset; null: never checked
  dropped: number // clips dropped in review, whole dataset
}

// A dataset in the data dir, or the copy a training run trains on.
export type DatasetSource = { id: string } | { run: string }

export async function getDatasetReport(source: DatasetSource): Promise<DatasetReport> {
  return (await request(`/api/dataset-report?${new URLSearchParams(source)}`)).json()
}

// '' = all clips, 'any' = any check flag, 'dropped', or one flag.
export type FlagFilter = '' | 'any' | 'dropped' | CheckFlag

export interface ClipFilter {
  query: string
  flag: FlagFilter
  worstFirst: boolean // lowest alignment scores first
  keep?: number[] // clip indices listed whatever the filter (just edited)
}

export async function getDatasetRows(
  source: DatasetSource,
  offset: number,
  limit: number,
  filter: ClipFilter,
): Promise<ClipPage> {
  const q = new URLSearchParams({
    ...source,
    offset: String(offset),
    limit: String(limit),
    q: filter.query,
    flag: filter.flag,
    order: filter.worstFirst ? 'worst' : 'file',
  })
  for (const index of filter.keep ?? []) q.append('keep', String(index))
  return (await request(`/api/dataset-rows?${q}`)).json()
}

// `version` changes the URL after a clip was trimmed, so the browser fetches it again.
export function datasetAudioUrl(source: DatasetSource, index: number, version = 0): string {
  return `/api/dataset-audio?${new URLSearchParams({ ...source, index: String(index), v: String(version) })}`
}

export async function listRuns(): Promise<RunSummary[]> {
  return (await request('/api/training/runs')).json()
}

/** A run; polls pass what they have (`known`) and get only what is new: metrics after
 * its last step, console lines after its log_next. Merge the answer with `appendNew`. */
export async function getRun(name: string, known?: RunDetail): Promise<RunDetail> {
  if (!known) return (await request(runUrl(name))).json()
  const since = Math.max(known.train.at(-1)?.step ?? -1, known.val.at(-1)?.step ?? -1)
  return (await request(`${runUrl(name)}?since=${since}&log_from=${known.log_next}`)).json()
}

/** A poll's answer (`run`) with its new points and console lines added to `known`'s. */
export function appendNew(known: RunDetail, run: RunDetail): RunDetail {
  // A known point at or after the first new one was logged again (a run resumed from an
  // earlier checkpoint): the new one replaces it.
  const append = (old: MetricPoint[], added: MetricPoint[]) =>
    added.length ? [...old.filter((p) => p.step < added[0].step), ...added] : old
  const train = run.since === null ? run.train : append(known.train, run.train)
  const val = run.since === null ? run.val : append(known.val, run.val)
  const kept = known.log_partial ? known.log.slice(0, -1) : known.log
  const log = run.log_appended ? [...kept, ...run.log].slice(-CONSOLE_LINES) : run.log
  return { ...run, since: null, train, val, log, log_appended: false }
}

export async function startRun(params: TrainParams): Promise<RunDetail> {
  return (await postJson('/api/training/runs', params)).json()
}

export async function resumeRun(name: string, numIters?: number): Promise<RunDetail> {
  return (await postJson(`${runUrl(name)}/resume`, { num_iters: numIters ?? null })).json()
}

export async function stopTraining(): Promise<void> {
  await postJson('/api/training/stop')
}

export interface Similarity {
  target: number | null // how real clips of the training set score
  scores: Record<string, number> // sample / reference file -> similarity to the voice
}

/** Speaker similarity (CAM++) of a run's samples to the voice it trains on. */
export async function getSimilarity(name: string): Promise<Similarity> {
  return (await request(`${runUrl(name)}/similarity`)).json()
}

export function sampleUrl(name: string, file: string): string {
  return `${runUrl(name)}/samples/${encodeURIComponent(file)}`
}
