import { useEffect, useState } from 'react'
import { getStatus } from '../api'
import type { DatasetReport, GpuProfile, TrainParams } from '../api'
import { datasetName, formatStep } from '../format'
import { DatasetReview } from './DatasetReview'
import { DEFAULT_EPOCHS, DEFAULT_PARAMS, epochsOf, forGpu, scheduleFor } from '../trainParams'
import type { Params, Schedule } from '../trainParams'
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

const SCHEDULE_KEYS: (keyof Schedule)[] = [
  'num_iters',
  'warmup_steps',
  'save_interval',
  'valid_interval',
  'log_interval',
]

function suggestName(dataset: string, rank: number, taken: string[]): string {
  const base = `${datasetName(dataset).split('/').at(-1)}_r${rank}`.replace(/[^\p{L}\p{N}_.-]/gu, '_')
  let name = base
  for (let i = 2; taken.includes(name); i++) name = `${base}_${i}`
  return name
}

/** A run before it starts: the same page as an existing run, with editable sections. */
export function NewRunForm({ datasets, runNames, activeRun, onStart, initialDataset }: Props) {
  const [tab, setTab] = useState<Tab>('dataset')
  const [base, setBase] = useState<Params>(DEFAULT_PARAMS)
  const [epochTarget, setEpochTarget] = useState(DEFAULT_EPOCHS)
  // Schedule values set by hand in Advanced; the rest follow from the epochs.
  const [overrides, setOverrides] = useState<Partial<Schedule>>({})
  const [dataset, setDataset] = useState(initialDataset ?? '')
  const [customName, setCustomName] = useState<string | null>(null)
  const [useValDataset, setUseValDataset] = useState(false)
  const [report, setReport] = useState<DatasetReport | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [gpu, setGpu] = useState<GpuProfile | null>(null)

  // The batch that fits this GPU (the defaults are for 16 GB).
  useEffect(() => {
    getStatus().then(
      (s) => {
        if (!s.gpu) return
        setGpu(s.gpu)
        setBase((p) => forGpu(p, s.gpu.train_batch))
      },
      () => {},
    )
  }, [])

  const trainClips = report ? report.valid - (useValDataset ? 0 : Math.min(base.val_holdout, report.valid - 1)) : 0
  const params: Params = {
    ...base,
    ...(trainClips > 0 && Number.isFinite(epochTarget) ? scheduleFor(epochTarget, trainClips, base) : {}),
    ...overrides,
  }
  const name = customName ?? (dataset ? suggestName(dataset, params.lora_rank, runNames) : '')
  const set = <K extends keyof Params>(key: K, value: Params[K]) =>
    SCHEDULE_KEYS.includes(key as keyof Schedule)
      ? setOverrides((o) => ({ ...o, [key]: value }))
      : setBase((p) => ({ ...p, [key]: value }))

  const nameError = !name
    ? null
    : !NAME_PATTERN.test(name)
      ? 'Letters, digits, ".", "_" and "-" only.'
      : runNames.includes(name)
        ? 'A run with this name exists.'
        : null
  const numbersValid =
    Number.isFinite(epochTarget) && Object.values(params).every((v) => typeof v !== 'number' || Number.isFinite(v))
  const valValid = !useValDataset || !!params.val_dataset
  // Unreadable clips would crash the job mid-epoch, so they block the start.
  const datasetValid = !!report && report.valid > 0 && report.errors.length === 0

  const epochs = epochsOf(params, trainClips)
  const perStep = params.batch_size * params.grad_accum_steps

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
              {epochs !== null && <span>{epochs.toFixed(1)} epochs</span>}
              <span className="muted">
                {formatStep(params.num_iters)} steps × {perStep} clips
              </span>
              <span className="muted">
                rank {params.lora_rank} · alpha {params.lora_alpha}
              </span>
            </>
          )}
        </p>
        {gpu?.note && <p className="notice">{gpu.note}</p>}
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
            epochs={{
              value: epochTarget,
              onChange: (v) => {
                setEpochTarget(v)
                // Steps follow the epochs again.
                setOverrides((o) => {
                  const rest = { ...o }
                  delete rest.num_iters
                  return rest
                })
              },
            }}
            hint={
              trainClips > 0 ? (
                <p className="muted hint">
                  = {formatStep(params.num_iters)} steps of {perStep} clips ({params.batch_size} ×{' '}
                  {params.grad_accum_steps} accumulated
                  {gpu?.total_gb ? `, the batch that fits this ${gpu.total_gb} GB GPU` : ''}) over{' '}
                  {trainClips.toLocaleString()} training clips
                  {epochs !== null && Math.abs(epochs - epochTarget) > 0.05 && ` (≈ ${epochs.toFixed(1)} epochs)`};
                  checkpoint and validation every {formatStep(params.save_interval)} steps, warmup{' '}
                  {formatStep(params.warmup_steps)}. VoxCPM recommends 1–3 epochs for one speaker; compare the
                  checkpoints by listening.
                </p>
              ) : (
                <p className="muted hint">Steps follow from the epochs once a dataset is chosen.</p>
              )
            }
          />
        </section>
      )}
    </>
  )
}
