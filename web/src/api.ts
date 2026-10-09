// Typed client for the VoxCPM Studio API (src/voxcpm_studio/api.py).

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
  id: string
  base_model: string | null
  // Hugging Face id guessed from base_model (the export default).
  base_model_id: string | null
  step: number | null
  config: LoraConfig | null
}

export interface Status {
  device: string
  optimize: boolean
  model: string
  // Hugging Face id of the base model, when it can be inferred.
  model_id: string | null
  loaded: boolean
  asr_loaded: boolean
  aligner_loaded: boolean
  separation_loaded: boolean
  singing_loaded: boolean
  // Import clean-up options whose model is configured.
  cleanup: CleanupOption[]
  lora_rank: number | null
  active_lora: string | null
}

export interface SynthesisParams {
  text: string
  control: string
  loraId: string | null
  referenceAudio: File | null
  promptText: string
  cfgValue: number
  steps: number
  normalize: boolean
  denoise: boolean
  seed: number | null
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

export async function getStatus(): Promise<Status> {
  return (await request('/api/status')).json()
}

export async function listLoras(): Promise<Lora[]> {
  return (await request('/api/loras')).json()
}

export async function exportLora(id: string, name: string, baseModel: string, half: boolean): Promise<Blob> {
  const q = new URLSearchParams({ id, name, base_model: baseModel, half: String(half) })
  return (await request(`/api/loras/export?${q}`)).blob()
}

export async function importLora(file: File): Promise<string> {
  const form = new FormData()
  form.set('package', file)
  return (await (await request('/api/loras/import', { method: 'POST', body: form })).json()).id
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

export async function synthesize(p: SynthesisParams): Promise<SynthesisOutput> {
  const form = new FormData()
  form.set('text', p.text)
  form.set('control', p.control)
  if (p.loraId) form.set('lora_id', p.loraId)
  if (p.referenceAudio) {
    form.set('reference_audio', p.referenceAudio)
    if (p.promptText.trim()) form.set('prompt_text', p.promptText)
  }
  form.set('cfg_value', String(p.cfgValue))
  form.set('inference_timesteps', String(p.steps))
  form.set('normalize', String(p.normalize))
  form.set('denoise', String(p.denoise))
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
  sample_texts: string[]
  val_clips: number
  log: string[]
}

export interface TrainParams {
  name: string
  dataset: string
  val_dataset: string | null
  val_holdout: number
  num_iters: number
  batch_size: number
  grad_accum_steps: number
  max_batch_tokens: number // clips too long for this per-batch budget are left out; 0 = no limit
  learning_rate: number
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
  error: string | null
  result: R
}

// Optional clean-up before import (src/voxcpm_studio/separation.py).
export type CleanupOption = 'separate' | 'dereverb' | 'denoise'

// Forced aligners (src/voxcpm_studio/aligner.py).
export type Aligner = 'qwen3' | 'ctc'

export type CheckFlag =
  'singing' | 'mismatch' | 'asr_more' | 'filler' | 'edge_sound' | 'extra_speech' | 'cut_edge' | 'long_silence'

// Flags trimming fixes (trim-flagged).
export const TRIMMABLE_FLAGS: CheckFlag[] = ['edge_sound', 'long_silence']

// Alignment checks over a whole dataset (checks.summarize).
export interface CheckSummary {
  aligner: Aligner | null
  checked: number
  flagged: number
  flags: Partial<Record<CheckFlag, number>>
}

export interface ImportOptions {
  source: string
  audioDir: string
  name: string
  copyAudio: boolean
  transcribe: boolean
  language: string // '' = detect per clip
  prompt: string // names / terms for ASR
  slice: boolean // cut long recordings into clips at pauses
  sliceMin: number
  sliceMax: number
  check: boolean // check each clip against its text by forced alignment
  aligner: Aligner
  trim: boolean // cut long silent edges from the copies
  cleanup: CleanupOption[] // applied first, in this order: separate, dereverb, denoise
  tidyFillers: boolean // 哈哈哈哈 -> 哈哈 in every transcript
  detectSinging: boolean // with check: flag sung clips
}

export type ImportJob = Job<{
  id: string
  imported: number
  transcribed: number
  sliced: number
  trimmed: number
  checks: CheckSummary | null
}>

export async function importDataset(o: ImportOptions): Promise<ImportJob> {
  const body = {
    source: o.source,
    audio_dir: o.audioDir || null,
    name: o.name,
    copy_audio: o.copyAudio,
    transcribe: o.transcribe,
    language: o.language || null,
    prompt: o.prompt.trim() || null,
    slice: o.slice,
    slice_min_seconds: o.sliceMin,
    slice_max_seconds: o.sliceMax,
    check: o.check,
    aligner: o.aligner,
    trim: o.trim,
    separate: o.cleanup.includes('separate'),
    dereverb: o.cleanup.includes('dereverb'),
    denoise: o.cleanup.includes('denoise'),
    tidy_fillers: o.tidyFillers,
    detect_singing: o.detectSinging,
  }
  return (await postJson('/api/datasets/import', body)).json()
}

export type CheckJob = Job<CheckSummary>

/** Shorten runs of fillers in a dataset's transcripts, in place (the original manifest is kept). */
export async function tidyFillers(id: string): Promise<{ changed: number }> {
  return (await postJson('/api/datasets/tidy-fillers', { id })).json()
}

export async function checkDataset(
  id: string,
  aligner: Aligner,
  compareAsr: boolean,
  detectSinging: boolean,
): Promise<CheckJob> {
  const body = { id, aligner, compare_asr: compareAsr, detect_singing: detectSinging }
  return (await postJson('/api/datasets/check', body)).json()
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
  trim: [number, number] | null // what trimming keeps; null: little to trim
}

/** Trim one clip at its checked edges, in place (the original is kept). */
export async function trimClip(id: string, index: number): Promise<ClipDetail> {
  return (await postJson('/api/dataset-clip/trim', { id, index })).json()
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
  return (await request(`/api/dataset-rows?${q}`)).json()
}

// `version` changes the URL after a clip was trimmed, so the browser fetches it again.
export function datasetAudioUrl(source: DatasetSource, index: number, version = 0): string {
  return `/api/dataset-audio?${new URLSearchParams({ ...source, index: String(index), v: String(version) })}`
}

export async function listRuns(): Promise<RunSummary[]> {
  return (await request('/api/training/runs')).json()
}

export async function getRun(name: string): Promise<RunDetail> {
  return (await request(runUrl(name))).json()
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

export function sampleUrl(name: string, file: string): string {
  return `${runUrl(name)}/samples/${encodeURIComponent(file)}`
}
