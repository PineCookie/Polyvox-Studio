import type { ReactNode } from 'react'
import { useT } from '../i18n'
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
  // A new run is sized in epochs (steps follow from the dataset); an existing one shows its steps.
  epochs?: { value: number; onChange: (value: number) => void }
}

/** Training parameters, editable for a new run and read-only for an existing one. */
export function ParamsFields({ params, onChange, header, hint, validation, epochs }: Props) {
  const t = useT()
  const readOnly = !onChange

  function numberField(key: NumericKey, props: { min?: number; max?: number; step?: number | 'any' } = {}) {
    return (
      <label className="field">
        <span>{t.params[key as keyof typeof t.params]}</span>
        <input
          type="number"
          value={Number.isFinite(params[key]) ? params[key] : ''}
          onChange={(e) => onChange?.(key, e.target.valueAsNumber)}
          {...props}
        />
      </label>
    )
  }

  function flag(key: FlagKey) {
    return (
      <label>
        <input type="checkbox" checked={params[key]} onChange={(e) => onChange?.(key, e.target.checked)} />{' '}
        {t.params[key]}
      </label>
    )
  }

  return (
    <fieldset className="params" disabled={readOnly}>
      {header}
      <div className="form-grid">
        {epochs ? (
          <label className="field">
            <span>{t.params.epochs}</span>
            <input
              type="number"
              min={0.5}
              step={0.5}
              value={Number.isFinite(epochs.value) ? epochs.value : ''}
              onChange={(e) => epochs.onChange(e.target.valueAsNumber)}
            />
          </label>
        ) : (
          numberField('num_iters', { min: 1, step: 100 })
        )}
        {numberField('lora_rank', { min: 1, max: 512 })}
        {numberField('lora_alpha', { min: 0.1, step: 'any' })}
        {numberField('learning_rate', { min: 0, step: 'any' })}
      </div>
      {hint}
      {validation}
      {/* Read-only shows everything; a new run keeps the rarely changed ones folded. */}
      <details className="advanced" open={readOnly || undefined}>
        <summary>{t.advanced}</summary>
        <div className="form-grid">
          {epochs && numberField('num_iters', { min: 1 })}
          {numberField('batch_size', { min: 1 })}
          {numberField('grad_accum_steps', { min: 1 })}
          {numberField('warmup_steps', { min: 0 })}
          {numberField('max_batch_tokens', { min: 0, step: 1024 })}
          {numberField('weight_decay', { min: 0, step: 'any' })}
          {numberField('max_grad_norm', { min: 0, step: 'any' })}
          {numberField('lora_dropout', { min: 0, max: 0.99, step: 'any' })}
          {numberField('save_interval', { min: 1 })}
          {numberField('valid_interval', { min: 1 })}
          {numberField('log_interval', { min: 1 })}
          {numberField('num_workers', { min: 0, max: 16 })}
          <div className="toggles">
            {flag('enable_lm')}
            {flag('enable_dit')}
            {flag('enable_proj')}
          </div>
        </div>
      </details>
    </fieldset>
  )
}
