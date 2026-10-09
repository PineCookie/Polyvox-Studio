// Training parameters of a run: defaults, and reading them back from a run's config.
import type { TrainParams } from './api'

export type Params = Omit<TrainParams, 'name' | 'dataset'>

export const DEFAULT_PARAMS: Params = {
  val_dataset: null,
  val_holdout: 2,
  num_iters: 5000,
  batch_size: 1,
  grad_accum_steps: 1,
  learning_rate: 1e-4,
  weight_decay: 0.01,
  warmup_steps: 100,
  max_grad_norm: 1.0,
  log_interval: 10,
  valid_interval: 500,
  save_interval: 1000,
  num_workers: 2,
  lora_rank: 32,
  lora_alpha: 32,
  lora_dropout: 0,
  enable_lm: true,
  enable_dit: true,
  enable_proj: false,
}

/** The parameters a run was started with, from its train_config.yaml. */
export function paramsFromConfig(config: Record<string, unknown>): Params {
  const lora = (config.lora ?? {}) as Record<string, unknown>
  const num = (v: unknown, fallback: number) => (typeof v === 'number' ? v : fallback)
  const flag = (v: unknown, fallback: boolean) => (typeof v === 'boolean' ? v : fallback)
  const d = DEFAULT_PARAMS
  return {
    ...d,
    num_iters: num(config.num_iters, d.num_iters),
    batch_size: num(config.batch_size, d.batch_size),
    grad_accum_steps: num(config.grad_accum_steps, d.grad_accum_steps),
    learning_rate: num(config.learning_rate, d.learning_rate),
    weight_decay: num(config.weight_decay, d.weight_decay),
    warmup_steps: num(config.warmup_steps, d.warmup_steps),
    max_grad_norm: num(config.max_grad_norm, d.max_grad_norm),
    log_interval: num(config.log_interval, d.log_interval),
    valid_interval: num(config.valid_interval, d.valid_interval),
    save_interval: num(config.save_interval, d.save_interval),
    num_workers: num(config.num_workers, d.num_workers),
    lora_rank: num(lora.r, d.lora_rank),
    lora_alpha: num(lora.alpha, d.lora_alpha),
    lora_dropout: num(lora.dropout, d.lora_dropout),
    enable_lm: flag(lora.enable_lm, d.enable_lm),
    enable_dit: flag(lora.enable_dit, d.enable_dit),
    enable_proj: flag(lora.enable_proj, d.enable_proj),
  }
}
