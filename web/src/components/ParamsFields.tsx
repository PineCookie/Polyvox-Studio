import type { ReactNode } from 'react'
import { useT } from '../i18n'
import { DEFAULT_ENGINE, OPTIMIZERS, type Engine, type Optimizer } from '../api'
import { OPTIMIZER_LR, type Params } from '../trainParams'

type NumericKey = { [K in keyof Params]: Params[K] extends number ? K : never }[keyof Params]
type FlagKey = 'enable_lm' | 'enable_dit' | 'enable_proj'
// GPT-SoVITS base models a run can start from (TrainParams.gsv_version), the default first.
const GSV_VERSIONS = ['v2ProPlus', 'v2Pro', 'v4']

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
  // The models a run can train (the choice shows when there is more than one).
  engines?: Engine[]
  // The engine trains every weight: no LoRA, optimizer or accumulation settings.
  fullFinetune?: boolean
}

/** Training parameters, editable for a new run and read-only for an existing one. */
export function ParamsFields({
  params,
  onChange,
  header,
  hint,
  validation,
  epochs,
  engines = [],
  fullFinetune = false,
}: Props) {
  const t = useT()
  const readOnly = !onChange
  const voxcpm = params.engine === DEFAULT_ENGINE // token budget and projections are VoxCPM's
  const lora = !fullFinetune
  const showEngine = engines.length > 1 || !voxcpm
  const engineOptions = engines.some((e) => e.id === params.engine)
    ? engines
    : [...engines, { id: params.engine, name: params.engine, unavailable: null } as Engine]

  function numberField(
    key: NumericKey,
    props: { min?: number; max?: number; step?: number | 'any' } = {},
    label: string = t.params[key as keyof typeof t.params],
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
        {showEngine && (
          <label className="field">
            <span>{t.params.engine}</span>
            <select value={params.engine} onChange={(e) => onChange?.('engine', e.target.value)}>
              {engineOptions.map((e) => (
                <option key={e.id} value={e.id} disabled={!!e.unavailable} title={e.unavailable ?? undefined}>
                  {e.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {!lora ? (
          <>
            {numberField('sovits_epochs', { min: 0, max: 100 })}
            {numberField('gpt_epochs', { min: 0, max: 100 })}
          </>
        ) : epochs ? (
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
        {(lora || params.gsv_version === 'v4') && numberField('lora_rank', { min: 1, max: 512 })}
        {lora && numberField('lora_alpha', { min: 0.1, step: 'any' })}
        {lora && (
          <label className="field">
            <span>{t.params.optimizer}</span>
            <select
              value={params.optimizer}
              onChange={(e) => {
                const optimizer = e.target.value as Optimizer
                onChange?.('optimizer', optimizer)
                onChange?.('learning_rate', OPTIMIZER_LR[optimizer])
              }}
            >
              {OPTIMIZERS.map((o) => (
                <option key={o} value={o}>
                  {t.optimizers[o]}
                </option>
              ))}
            </select>
          </label>
        )}
        {numberField(
          'learning_rate',
          { min: 0, step: 'any' },
          lora && params.optimizer !== 'adamw' ? t.params.lr_multiplier : undefined,
        )}
        {!lora && numberField('batch_size', { min: 1 })}
        {!lora && numberField('gpt_batch_size', { min: 1, max: 64 })}
        {!lora && (
          <label className="field">
            <span>{t.params.gsv_version}</span>
            <select value={params.gsv_version} onChange={(e) => onChange?.('gsv_version', e.target.value)}>
              {GSV_VERSIONS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {hint}
      {validation}
      {/* Read-only shows everything; a new run keeps the rarely changed ones folded. */}
      <details className="advanced" open={readOnly || undefined}>
        <summary>{t.advanced}</summary>
        <div className="form-grid">
          {(epochs || (!lora && readOnly)) && numberField('num_iters', { min: 1 })}
          {lora && numberField('batch_size', { min: 1 })}
          {lora && numberField('grad_accum_steps', { min: 1 })}
          {lora &&
            (params.optimizer === 'adamw' || params.optimizer === 'prodigy'
              ? numberField('warmup_steps', { min: 0 })
              : numberField('prodigy_steps', { min: 0 }))}
          {voxcpm && numberField('max_batch_tokens', { min: 0, step: 1024 })}
          {lora && numberField('weight_decay', { min: 0, step: 'any' })}
          {lora && numberField('max_grad_norm', { min: 0, step: 'any' })}
          {lora && numberField('lora_dropout', { min: 0, max: 0.99, step: 'any' })}
          {lora && numberField('save_interval', { min: 1 })}
          {lora && numberField('valid_interval', { min: 1 })}
          {numberField('log_interval', { min: 1 })}
          {lora && numberField('num_workers', { min: 0, max: 16 })}
          {lora && (
            <div className="toggles">
              {flag('enable_lm')}
              {flag('enable_dit')}
              {voxcpm && flag('enable_proj')}
            </div>
          )}
        </div>
      </details>
    </fieldset>
  )
}
