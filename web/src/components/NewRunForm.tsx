import { useState } from 'react'
import type { DatasetReport, TrainParams } from '../api'
import { datasetName, formatStep } from '../format'
import { DatasetReview } from './DatasetReview'
import { DEFAULT_PARAMS } from '../trainParams'
import type { Params } from '../trainParams'
import { ParamsFields } from './ParamsFields'
import { SubTabs } from './SubTabs'

interface Props {
  datasets: string[]
  runNames: string[]
  // Name of a run that is currently training, if any; only one job runs at a time.
  activeRun: string | null
  onStart: (params: TrainParams) => Promise<void>
  // Preselected from the Prepare tab ("Review and train").
  initialDataset?: string
}

type Tab = 'progress' | 'dataset' | 'parameters'

// Matches training.RUN_NAME: letters of any language, digits, "." "_" "-".
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

function suggestName(dataset: string, rank: number, taken: string[]): string {
  const base = `${datasetName(dataset).split('/').at(-1)}_r${rank}`.replace(/[^\p{L}\p{N}_.-]/gu, '_')
  let name = base
  for (let i = 2; taken.includes(name); i++) name = `${base}_${i}`
  return name
}

/** A run before it starts: the same page as an existing run, with editable sections. */
export function NewRunForm({ datasets, runNames, activeRun, onStart, initialDataset }: Props) {
  const [tab, setTab] = useState<Tab>('dataset')
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS)
  const [dataset, setDataset] = useState(initialDataset ?? '')
  const [customName, setCustomName] = useState<string | null>(null)
  const [useValDataset, setUseValDataset] = useState(false)
  const [report, setReport] = useState<DatasetReport | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const name = customName ?? (dataset ? suggestName(dataset, params.lora_rank, runNames) : '')
  const set = <K extends keyof Params>(key: K, value: Params[K]) => setParams((p) => ({ ...p, [key]: value }))

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

  const trainClips = report ? report.valid - (useValDataset ? 0 : Math.min(params.val_holdout, report.valid - 1)) : 0
  const epochs = trainClips > 0 ? (params.num_iters * params.batch_size * params.grad_accum_steps) / trainClips : null

  // What still blocks the start, in the order the page is filled in.
  const blocker = !dataset
    ? 'Choose a dataset'
    : !report
      ? 'Checking the dataset…'
      : !datasetValid
        ? 'The dataset has unreadable clips (see Dataset)'
        : !name || nameError
          ? `Run name: ${nameError ?? 'required'} (see Parameters)`
          : !numbersValid
            ? 'Fill in every parameter (see Parameters)'
            : !valValid
              ? 'Choose a validation dataset (see Parameters)'
              : activeRun
                ? `“${activeRun}” is training; one run at a time`
                : null

  async function start() {
    if (blocker || submitting) return
    setSubmitting(true)
    try {
      await onStart({ ...params, val_dataset: useValDataset ? params.val_dataset : null, name, dataset })
    } finally {
      setSubmitting(false)
    }
  }

  const validation = (
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
          <option value="">Choose a dataset…</option>
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
  )

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h2 className="run-title">
            {name || 'New run'} <span className="pill">New</span>
          </h2>
          <div className="run-actions">
            <button type="button" className="primary" onClick={start} disabled={!!blocker || submitting}>
              {submitting ? 'Starting…' : 'Start training'}
            </button>
          </div>
        </header>
        <p className="run-stats">
          {blocker ? (
            <span className="muted">{blocker}</span>
          ) : (
            <>
              <span>
                <strong>{trainClips.toLocaleString()}</strong> training clips from {datasetName(dataset)}
              </span>
              <span>{formatStep(params.num_iters)} steps</span>
              {epochs !== null && <span className="muted">≈ {epochs.toFixed(1)} epochs</span>}
              <span className="muted">
                rank {params.lora_rank} · alpha {params.lora_alpha}
              </span>
            </>
          )}
        </p>
        <SubTabs
          label="Run sections"
          value={tab}
          onChange={setTab}
          tabs={[
            {
              id: 'progress',
              label: 'Progress',
              disabled: true,
              title: 'Loss and samples appear once training starts',
            },
            { id: 'dataset', label: 'Dataset' },
            { id: 'parameters', label: 'Parameters' },
          ]}
        />
      </section>

      {tab === 'dataset' && (
        <section className="panel">
          <label className="field">
            <span>Dataset</span>
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
          {dataset && <DatasetReview source={{ id: dataset }} onReport={setReport} />}
        </section>
      )}

      {tab === 'parameters' && (
        <section className="panel">
          <ParamsFields
            params={params}
            onChange={set}
            validation={validation}
            header={
              <label className="field">
                <span>Run name {nameError && <span className="field-error">{nameError}</span>}</span>
                <input
                  value={name}
                  onChange={(e) => setCustomName(e.target.value)}
                  placeholder="Choose a dataset first"
                />
              </label>
            }
            hint={
              epochs !== null && (
                <p className="muted hint">
                  ≈ {epochs.toFixed(1)} epochs over {trainClips.toLocaleString()} training clips
                </p>
              )
            }
          />
        </section>
      )}
    </>
  )
}
