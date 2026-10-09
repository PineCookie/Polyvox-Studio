import type { ReactNode } from 'react'
import type { Params } from '../trainParams'

type NumericKey = { [K in keyof Params]: Params[K] extends number ? K : never }[keyof Params]
type FlagKey = 'enable_lm' | 'enable_dit' | 'enable_proj'

interface Props {
  params: Params
  // Omitted: read-only (an existing run shows what it was started with).
  onChange?: <K extends keyof Params>(key: K, value: Params[K]) => void
  // Above the main fields (e.g. the run name).
  header?: ReactNode
  // Under the main fields: the epochs estimate.
  hint?: ReactNode
  // The validation section differs between a new and an existing run.
  validation: ReactNode
}

/** Training parameters, editable for a new run and read-only for an existing one. */
export function ParamsFields({ params, onChange, header, hint, validation }: Props) {
  const readOnly = !onChange

  function numberField(
    key: NumericKey,
    label: string,
    props: { min?: number; max?: number; step?: number | 'any' } = {},
  ) {
    return (
      <label className="field">
        <span>{label}</span>
        <input
          type="number"
          value={Number.isFinite(params[key]) ? params[key] : ''}
          onChange={(e) => onChange?.(key, e.target.valueAsNumber)}
          {...props}
        />
      </label>
    )
  }

  function flag(key: FlagKey, label: string) {
    return (
      <label>
        <input type="checkbox" checked={params[key]} onChange={(e) => onChange?.(key, e.target.checked)} /> {label}
      </label>
    )
  }

  return (
    <fieldset className="params" disabled={readOnly}>
      {header}
      <div className="form-grid">
        {numberField('num_iters', 'Steps', { min: 1, step: 100 })}
        {numberField('lora_rank', 'LoRA rank', { min: 1, max: 512 })}
        {numberField('lora_alpha', 'LoRA alpha', { min: 0.1, step: 'any' })}
        {numberField('learning_rate', 'Learning rate', { min: 0, step: 'any' })}
      </div>
      {hint}
      {validation}
      {/* Read-only shows everything; a new run keeps the rarely changed ones folded. */}
      <details className="advanced" open={readOnly || undefined}>
        <summary>Advanced</summary>
        <div className="form-grid">
          {numberField('batch_size', 'Batch size', { min: 1 })}
          {numberField('grad_accum_steps', 'Gradient accumulation', { min: 1 })}
          {numberField('warmup_steps', 'Warmup steps', { min: 0 })}
          {numberField('weight_decay', 'Weight decay', { min: 0, step: 'any' })}
          {numberField('max_grad_norm', 'Max grad norm', { min: 0, step: 'any' })}
          {numberField('lora_dropout', 'LoRA dropout', { min: 0, max: 0.99, step: 'any' })}
          {numberField('save_interval', 'Save every (steps)', { min: 1 })}
          {numberField('valid_interval', 'Validate every (steps)', { min: 1 })}
          {numberField('log_interval', 'Log every (steps)', { min: 1 })}
          {numberField('num_workers', 'Data loader workers', { min: 0, max: 16 })}
          <div className="toggles">
            {flag('enable_lm', 'LoRA on LM')}
            {flag('enable_dit', 'LoRA on DiT')}
            {flag('enable_proj', 'LoRA on projections')}
          </div>
        </div>
      </details>
    </fieldset>
  )
}
