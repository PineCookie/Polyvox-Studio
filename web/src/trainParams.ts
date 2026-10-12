// Training parameters of a run: defaults, and reading them back from a run's config.
import { DEFAULT_ENGINE, OPTIMIZERS, type GpuProfile, type Optimizer, type TrainParams } from './api'

export type Params = Omit<TrainParams, 'name' | 'dataset'>

// An effective batch of 16 (4 clips x 4 accumulated on a 16 GB card; see forGpu)
// and max_batch_tokens 8192, as in VoxCPM's fine-tuning guide. Its AdamW at 1e-4
// for 1-3 epochs left voices close to the base model; Prodigy Schedule-Free finds a
// larger step size itself and sounds like the speaker within a few epochs. Steps,
// warmup and checkpoint intervals follow from the epochs and the dataset's size
// (scheduleFor).
export const DEFAULT_EPOCHS = 4
export const WARMUP_SHARE = 0.05 // of the steps
export const CHECKPOINTS_PER_EPOCH = 2

export const DEFAULT_PARAMS: Params = {
  engine: DEFAULT_ENGINE,
  val_dataset: null,
  val_holdout: 2,
  num_iters: 200,
  batch_size: 4,
  grad_accum_steps: 4,
  max_batch_tokens: 8192,
  learning_rate: 1,
  optimizer: 'prodigy_sf',
  prodigy_steps: 0,
  weight_decay: 0.01,
  warmup_steps: 10,
  max_grad_norm: 1.0,
  log_interval: 5,
  valid_interval: 50,
  save_interval: 50,
  num_workers: 2,
  lora_rank: 32,
  lora_alpha: 32,
  lora_dropout: 0,
  enable_lm: true,
  enable_dit: true,
  enable_proj: false,
  sovits_epochs: 8,
  gpt_epochs: 15,
  gpt_batch_size: 8,
  gsv_version: 'v2ProPlus',
}

// Prodigy finds its own step size: learning_rate is a multiplier, 1 by its authors' advice.
export const OPTIMIZER_LR: Record<Optimizer, number> = { adamw: 1e-4, prodigy: 1, prodigy_sf: 1 }

export const EFFECTIVE_BATCH = 16

// Full fine-tuning (GPT-SoVITS, as its webui trains voices): SoVITS with AdamW at
// the upstream rate, the batch as is (a GAN: no accumulation); then the GPT with
// upstream's own optimizer, an update per 4 batches. Epochs: the webui's defaults.
export const FULL_FINETUNE = { learning_rate: 1e-4, gpt_accum: 4 }

/** Steps of a full fine-tuning run (as the GPT-SoVITS worker plans them): whole
 * batches per SoVITS epoch, GPT updates of 4 batches of at most a quarter of the clips. */
export function fullSteps(clips: number, params: Params): { sovits: number; gpt: number } {
  const gptBatch = Math.max(1, Math.min(params.gpt_batch_size, Math.floor(clips / 4)))
  return {
    sovits: Math.ceil(params.sovits_epochs * Math.ceil(clips / Math.max(1, params.batch_size))),
    gpt: Math.ceil((params.gpt_epochs * Math.ceil(clips / gptBatch)) / FULL_FINETUNE.gpt_accum),
  }
}

/** The batch that fits the GPU (its profile's micro-batch for the params' engine),
 * accumulated to the same effective batch: smaller cards train slower, not differently.
 * `full`: a full fine-tuning engine, whose batch is not accumulated. */
// GPT-SoVITS v4's SoVITS stage (a LoRA on its mel DiT): the batch it was tuned at
// (on smaller GPUs the worker recomputes activations instead of shrinking it).
export const V4_BATCH = 4

export function forGpu(params: Params, gpu: GpuProfile, full = false): Params {
  const batch = Math.max(1, Math.min(gpu.train_batches?.[params.engine] ?? gpu.train_batch, EFFECTIVE_BATCH))
  if (full && params.gsv_version === 'v4') return { ...params, batch_size: V4_BATCH, gpt_batch_size: batch }
  if (full) return { ...params, batch_size: batch, grad_accum_steps: 1, gpt_batch_size: batch } // both stages, as the webui
  return { ...params, batch_size: batch, grad_accum_steps: Math.max(1, Math.round(EFFECTIVE_BATCH / batch)) }
}

/** The optimizer settings that go with an engine (`full`: full fine-tuning). */
export function forEngine(params: Params, full: boolean): Params {
  return full
    ? { ...params, optimizer: 'adamw', learning_rate: FULL_FINETUNE.learning_rate, grad_accum_steps: 1 }
    : { ...params, optimizer: DEFAULT_PARAMS.optimizer, learning_rate: DEFAULT_PARAMS.learning_rate }
}

export type Schedule = Pick<Params, 'num_iters' | 'warmup_steps' | 'save_interval' | 'valid_interval' | 'log_interval'>

/** Steps for `epochs` passes over `clips` at the params' effective batch, and the
 * warmup, checkpoint / validation and log intervals that go with them. */
export function scheduleFor(epochs: number, clips: number, params: Params): Schedule {
  const perStep = Math.max(1, params.batch_size * params.grad_accum_steps)
  const stepsPerEpoch = Math.max(1, clips / perStep)
  const steps = Math.max(1, Math.ceil(epochs * stepsPerEpoch))
  const interval = Math.max(1, Math.round(stepsPerEpoch / CHECKPOINTS_PER_EPOCH))
  return {
    num_iters: steps,
    warmup_steps: Math.max(1, Math.round(steps * WARMUP_SHARE)),
    save_interval: interval,
    valid_interval: interval,
    log_interval: Math.max(1, Math.min(10, Math.round(stepsPerEpoch / 10))),
  }
}

/** Passes over the training clips a run makes (num_iters x effective batch / clips). */
export function epochsOf(params: Params, clips: number): number | null {
  return clips > 0 ? (params.num_iters * params.batch_size * params.grad_accum_steps) / clips : null
}

/** The parameters a run was started with, from its train_config.yaml. Runs from
 * before these defaults keep their own values; missing keys mean the old ones. */
export function paramsFromConfig(config: Record<string, unknown>): Params {
  const lora = (config.lora ?? {}) as Record<string, unknown>
  const num = (v: unknown, fallback: number) => (typeof v === 'number' ? v : fallback)
  const flag = (v: unknown, fallback: boolean) => (typeof v === 'boolean' ? v : fallback)
  const d = DEFAULT_PARAMS
  return {
    ...d,
    engine: typeof config.engine === 'string' ? config.engine : DEFAULT_ENGINE,
    num_iters: num(config.num_iters, d.num_iters),
    batch_size: num(config.batch_size, 1),
    grad_accum_steps: num(config.grad_accum_steps, 1),
    max_batch_tokens: num(config.max_batch_tokens, 0),
    // Runs from before the optimizer option used AdamW.
    learning_rate: num(config.learning_rate, OPTIMIZER_LR.adamw),
    optimizer: OPTIMIZERS.find((o) => o === config.optimizer) ?? 'adamw',
    prodigy_steps: num(config.prodigy_steps, 0),
    weight_decay: num(config.weight_decay, d.weight_decay),
    warmup_steps: num(config.warmup_steps, d.warmup_steps),
    max_grad_norm: num(config.max_grad_norm, d.max_grad_norm),
    log_interval: num(config.log_interval, d.log_interval),
    valid_interval: num(config.valid_interval, d.valid_interval),
    save_interval: num(config.save_interval, d.save_interval),
    num_workers: num(config.num_workers, d.num_workers),
    lora_rank: num(lora.r ?? config.lora_rank, d.lora_rank), // GPT-SoVITS v4: its DiT LoRA's rank
    lora_alpha: num(lora.alpha, d.lora_alpha),
    lora_dropout: num(lora.dropout, d.lora_dropout),
    enable_lm: flag(lora.enable_lm, d.enable_lm),
    enable_dit: flag(lora.enable_dit, d.enable_dit),
    enable_proj: flag(lora.enable_proj, d.enable_proj),
    sovits_epochs: num(config.sovits_epochs, d.sovits_epochs),
    gpt_epochs: num(config.gpt_epochs, 0),
    gpt_batch_size: num(config.gpt_batch_size, d.gpt_batch_size),
    gsv_version: typeof config.version === 'string' ? config.version : d.gsv_version,
  }
}
