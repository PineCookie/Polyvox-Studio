import { useEffect, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { inspectDataset } from '../api'
import type { DatasetReport, TrainParams } from '../api'
import { DatasetClips } from './DatasetClips'
import { DatasetSummary } from './DatasetSummary'

interface Props {
  datasets: string[]
  runNames: string[]
  // Name of a run that is currently training, if any; only one job runs at a time.
  activeRun: string | null
  onStart: (params: TrainParams) => Promise<void>
  // Preselected from the Data tab's "Train on this".
  initialDataset?: string
}

const DEFAULTS: Omit<TrainParams, 'name' | 'dataset'> = {
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

// Matches training.RUN_NAME: letters of any language, digits, "." "_" "-".
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

type NumericKey = { [K in keyof TrainParams]: TrainParams[K] extends number ? K : never }[keyof TrainParams]
type FlagKey = 'enable_lm' | 'enable_dit' | 'enable_proj'

function suggestName(dataset: string, rank: number, taken: string[]): string {
  const base = `${dataset.split('/').at(-2) ?? dataset.replace(/\.jsonl$/, '')}_r${rank}`.replace(/[^\p{L}\p{N}_.-]/gu, '_')
  let name = base
  for (let i = 2; taken.includes(name); i++) name = `${base}_${i}`
  return name
}

export function NewRunForm({ datasets, runNames, activeRun, onStart, initialDataset }: Props) {
  const [params, setParams] = useState<Omit<TrainParams, 'name' | 'dataset'>>(DEFAULTS)
  const [dataset, setDataset] = useState(initialDataset ?? '')
  const [customName, setCustomName] = useState<string | null>(null)
  const [useValDataset, setUseValDataset] = useState(false)
  const [report, setReport] = useState<DatasetReport | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const name = customName ?? (dataset ? suggestName(dataset, params.lora_rank, runNames) : '')

  useEffect(() => {
    setReport(null)
    setReportError(null)
    if (!dataset) return
    let cancelled = false
    inspectDataset(dataset).then(
      (r) => !cancelled && setReport(r),
      (e: Error) => !cancelled && setReportError(e.message),
    )
    return () => {
      cancelled = true
    }
  }, [dataset])

  function set<K extends keyof typeof params>(key: K, value: (typeof params)[K]) {
    setParams((p) => ({ ...p, [key]: value }))
  }

  const nameError = !name
    ? null
    : !NAME_PATTERN.test(name)
      ? 'Letters, digits, ".", "_" and "-" only.'
      : runNames.includes(name)
        ? 'A run with this name exists.'
        : null
  const numbersValid = Object.values(params).every((v) => typeof v !== 'number' || Number.isFinite(v))
  const valValid = !useValDataset || !!params.val_dataset
  // Unreadable clips would crash the job mid-epoch, so they block the start.
  const datasetValid = !!report && report.valid > 0 && report.errors.length === 0
  const canSubmit = !submitting && !activeRun && datasetValid && !!name && !nameError && numbersValid && valValid

  const trainClips = report ? report.valid - (useValDataset ? 0 : Math.min(params.val_holdout, report.valid - 1)) : 0
  const epochs = trainClips > 0 ? (params.num_iters * params.batch_size * params.grad_accum_steps) / trainClips : null

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)
    try {
      await onStart({ ...params, val_dataset: useValDataset ? params.val_dataset : null, name, dataset })
    } finally {
      setSubmitting(false)
    }
  }

  function numberField(key: NumericKey, label: ReactNode, props: { min?: number; max?: number; step?: number | 'any' } = {}) {
    return (
      <label className="field">
        <span>{label}</span>
        <input type="number" value={Number.isFinite(params[key]) ? params[key] : ''} onChange={(e) => set(key, e.target.valueAsNumber)} {...props} />
      </label>
    )
  }

  function flag(key: FlagKey, label: string) {
    return (
      <label>
        <input type="checkbox" checked={params[key]} onChange={(e) => set(key, e.target.checked)} /> {label}
      </label>
    )
  }

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h2>1 · Dataset</h2>
        </header>
        <label className="field">
          <span>Manifest</span>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="">Choose a dataset…</option>
            {datasets.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        {datasets.length === 0 && (
          <p className="muted empty">No datasets yet. Prepare one from raw recordings or lists in the Prepare tab.</p>
        )}
        {dataset && (
          <>
            <DatasetSummary report={report} error={reportError} />
            <DatasetClips key={dataset} datasetId={dataset} />
          </>
        )}
      </section>

      <form className="panel" onSubmit={submit}>
        <header className="panel-head">
          <h2>2 · Training</h2>
        </header>

        <label className="field">
          <span>
            Run name {nameError && <span className="field-error">{nameError}</span>}
          </span>
          <input value={name} onChange={(e) => setCustomName(e.target.value)} placeholder="Choose a dataset first" />
        </label>

        <div className="form-grid">
          {numberField('num_iters', 'Steps', { min: 1, step: 100 })}
          {numberField('lora_rank', 'LoRA rank', { min: 1, max: 512 })}
          {numberField('lora_alpha', 'LoRA alpha', { min: 0.1, step: 'any' })}
          {numberField('learning_rate', 'Learning rate', { min: 0, step: 'any' })}
        </div>
        {epochs !== null && (
          <p className="muted hint">
            ≈ {epochs.toFixed(1)} epochs over {trainClips} training clips
          </p>
        )}

        <fieldset className="field">
          <legend>Validation</legend>
          <div className="segmented" role="radiogroup">
            <label>
              <input type="radio" checked={!useValDataset} onChange={() => setUseValDataset(false)} /> Hold out clips
            </label>
            <label>
              <input type="radio" checked={useValDataset} onChange={() => setUseValDataset(true)} /> Separate dataset
            </label>
          </div>
          {useValDataset ? (
            <select value={params.val_dataset ?? ''} onChange={(e) => set('val_dataset', e.target.value || null)}>
              <option value="">Choose a manifest…</option>
              {datasets
                .filter((d) => d !== dataset)
                .map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
            </select>
          ) : (
            <div className="inline-field">
              <input
                type="number"
                min={0}
                max={50}
                value={Number.isFinite(params.val_holdout) ? params.val_holdout : ''}
                onChange={(e) => set('val_holdout', e.target.valueAsNumber)}
              />
              <span className="muted hint">
                clips set aside for validation loss and audio samples every {params.valid_interval} steps (0 = none)
              </span>
            </div>
          )}
        </fieldset>

        <details className="advanced">
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

        <div className="actions">
          {activeRun && <span className="muted hint">“{activeRun}” is training; one run at a time.</span>}
          <button type="submit" className="primary" disabled={!canSubmit}>
            {submitting ? 'Starting…' : 'Start training'}
          </button>
        </div>
      </form>
    </>
  )
}
