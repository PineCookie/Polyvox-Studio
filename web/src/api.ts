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
  config: LoraConfig | null
}

export interface Status {
  device: string
  optimize: boolean
  model: string
  loaded: boolean
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

export async function unloadModel(): Promise<Status> {
  return (await request('/api/model/unload', { method: 'POST' })).json()
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

export async function inspectDataset(id: string): Promise<DatasetReport> {
  return (await request(`/api/datasets/${id.split('/').map(encodeURIComponent).join('/')}`)).json()
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
