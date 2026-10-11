import { useEffect, useState } from 'react'
import { DEFAULT_ENGINE, getStatus, listEngines } from '../api'
import type { DatasetReport, Engine, GpuProfile, TrainParams } from '../api'
import { datasetName, formatStep } from '../format'
import { useT } from '../i18n'
import { DatasetReview } from './DatasetReview'
import { DEFAULT_EPOCHS, DEFAULT_PARAMS, epochsOf, forEngine, forGpu, fullSteps, scheduleFor } from '../trainParams'
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

function suggestName(dataset: string, engine: string, rank: number | null, taken: string[]): string {
  // e.g. voice_r32 (VoxCPM, as before engines), voice_dots_r32, voice_gsv (full fine-tuning: no rank)
  const tag = engine === 'gpt_sovits' ? 'gsv' : engine.split('_')[0]
  const model = engine === DEFAULT_ENGINE ? '' : `_${tag}`
  const suffix = rank === null ? '' : `_r${rank}`
  const base = `${datasetName(dataset).split('/').at(-1)}${model}${suffix}`.replace(/[^\p{L}\p{N}_.-]/gu, '_')
  let name = base
  for (let i = 2; taken.includes(name); i++) name = `${base}_${i}`
  return name
}

/** A run before it starts: the same page as an existing run, with editable sections. */
export function NewRunForm({ datasets, runNames, activeRun, onStart, initialDataset }: Props) {
  const t = useT()
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
  const [engines, setEngines] = useState<Engine[]>([])

  // The batch that fits this GPU (the defaults are for 16 GB).
  useEffect(() => {
    getStatus().then(
      (s) => {
        if (!s.gpu) return
        setGpu(s.gpu)
        setBase((p) => forGpu(p, s.gpu))
      },
      () => {},
    )
    // Only engines that can train yet.
    listEngines().then(
      (list) => setEngines(list.filter((e) => e.capabilities.training)),
      () => {},
    )
  }, [])

  const trainClips = report ? report.valid - (useValDataset ? 0 : Math.min(base.val_holdout, report.valid - 1)) : 0
  const isFull = (id: string) => !!engines.find((e) => e.id === id)?.capabilities.full_finetune
  const full = isFull(base.engine)
  // Full fine-tuning (GPT-SoVITS): steps follow from each stage's epochs.
  const stages = full && trainClips > 0 ? fullSteps(trainClips, base) : null
  const params: Params = full
    ? { ...base, ...(stages ? { num_iters: stages.sovits + stages.gpt } : {}), ...overrides }
    : {
        ...base,
        ...(trainClips > 0 && Number.isFinite(epochTarget) ? scheduleFor(epochTarget, trainClips, base) : {}),
        ...overrides,
      }
  const name =
    customName ?? (dataset ? suggestName(dataset, params.engine, full ? null : params.lora_rank, runNames) : '')
  const engineName = engines.find((e) => e.id === params.engine)?.name ?? params.engine
  const set = <K extends keyof Params>(key: K, value: Params[K]) => {
    if (SCHEDULE_KEYS.includes(key as keyof Schedule)) {
      setOverrides((o) => ({ ...o, [key]: value }))
      return
    }
    if (key === 'engine') {
      // Each model has its own batch that fits the GPU, and optimizer settings.
      const nextFull = isFull(value as string)
      setOverrides({})
      setBase((p) => {
        const next = forEngine({ ...p, engine: value as string }, nextFull)
        return gpu ? forGpu(next, gpu, nextFull) : next
      })
      return
    }
    setBase((p) => ({ ...p, [key]: value }))
  }

  const nameError = !name ? null : !NAME_PATTERN.test(name) ? t.nameChars : runNames.includes(name) ? t.runExists : null
  const numbersValid =
    Number.isFinite(epochTarget) && Object.values(params).every((v) => typeof v !== 'number' || Number.isFinite(v))
  const valValid = !useValDataset || !!params.val_dataset
  // Unreadable clips would crash the job mid-epoch, so they block the start.
  const datasetValid = !!report && report.valid > 0 && report.errors.length === 0

  const epochs = full ? null : epochsOf(params, trainClips)
  const perStep = params.batch_size * params.grad_accum_steps

  // What still blocks the start, in the order the page is filled in.
  const blocker = !dataset
    ? t.chooseDataset
    : !report
      ? t.checkingDataset
      : !datasetValid
        ? t.unreadableClips
        : !name || nameError
          ? t.runNameBlocker(nameError ?? t.requiredLower)
          : !numbersValid
            ? t.fillEveryParam
            : !valValid
              ? t.chooseValDataset
              : activeRun
                ? t.oneRunAtATime(activeRun)
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
      <legend>{t.validation}</legend>
      <div className="segmented" role="radiogroup">
        <label>
          <input type="radio" checked={!useValDataset} onChange={() => setUseValDataset(false)} /> {t.holdOutClips}
        </label>
        <label>
          <input type="radio" checked={useValDataset} onChange={() => setUseValDataset(true)} /> {t.separateDataset}
        </label>
      </div>
      {useValDataset ? (
        <select value={params.val_dataset ?? ''} onChange={(e) => set('val_dataset', e.target.value || null)}>
          <option value="">{t.chooseDatasetOption}</option>
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
          <span className="muted hint">{t.holdoutHint(params.valid_interval)}</span>
        </div>
      )}
    </fieldset>
  )

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h2 className="run-title">
            {name || t.newRun} <span className="pill">{t.newPill}</span>
          </h2>
          <div className="run-actions">
            <button type="button" className="primary" onClick={start} disabled={!!blocker || submitting}>
              {submitting ? t.starting2 : t.startTraining}
            </button>
          </div>
        </header>
        <p className="run-stats">
          {blocker ? (
            <span className="muted">{blocker}</span>
          ) : (
            <>
              <span>{t.trainingClipsFrom(trainClips.toLocaleString(), datasetName(dataset))}</span>
              {epochs !== null && <span>{t.epochsValue(epochs.toFixed(1))}</span>}
              <span className="muted">
                {stages
                  ? t.stageSteps(formatStep(stages.sovits), formatStep(stages.gpt))
                  : t.stepsTimesClips(formatStep(params.num_iters), perStep)}
              </span>
              <span className="muted">
                {engines.length > 1 && `${engineName} · `}
                {full ? t.fullFinetune : t.rankAlpha(params.lora_rank, params.lora_alpha)}
              </span>
            </>
          )}
        </p>
        {gpu?.note && <p className="notice">{gpu.note}</p>}
        <SubTabs
          label={t.runSections}
          value={tab}
          onChange={setTab}
          tabs={[
            { id: 'progress', label: t.subTabs.progress, disabled: true, title: t.progressDisabled },
            { id: 'dataset', label: t.subTabs.dataset },
            { id: 'parameters', label: t.subTabs.parameters },
          ]}
        />
      </section>

      {tab === 'dataset' && (
        <section className="panel">
          <label className="field">
            <span>{t.datasetLabel}</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="">{t.chooseDatasetOption}</option>
              {datasets.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          {datasets.length === 0 && <p className="muted empty">{t.noDatasetsYet}</p>}
          {dataset && <DatasetReview source={{ id: dataset }} onReport={setReport} />}
        </section>
      )}

      {tab === 'parameters' && (
        <section className="panel">
          <ParamsFields
            params={params}
            engines={engines}
            fullFinetune={full}
            onChange={set}
            validation={validation}
            header={
              <label className="field">
                <span>
                  {t.runName} {nameError && <span className="field-error">{nameError}</span>}
                </span>
                <input
                  value={name}
                  onChange={(e) => setCustomName(e.target.value)}
                  placeholder={t.chooseDatasetFirst}
                />
              </label>
            }
            epochs={
              full
                ? undefined
                : {
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
                  }
            }
            hint={
              stages ? (
                <p className="muted hint">
                  {t.fullScheduleHint(
                    formatStep(stages.sovits),
                    params.batch_size,
                    formatStep(stages.gpt),
                    Math.max(1, Math.min(params.gpt_batch_size, Math.floor(trainClips / 4))),
                    gpu?.total_gb ? String(gpu.total_gb) : '',
                    trainClips.toLocaleString(),
                  )}
                </p>
              ) : trainClips > 0 ? (
                <p className="muted hint">
                  {t.scheduleHint(
                    formatStep(params.num_iters),
                    perStep,
                    params.batch_size,
                    params.grad_accum_steps,
                    gpu?.total_gb ? String(gpu.total_gb) : '',
                    trainClips.toLocaleString(),
                    epochs !== null && Math.abs(epochs - epochTarget) > 0.05 ? epochs.toFixed(1) : '',
                    formatStep(params.save_interval),
                    formatStep(params.warmup_steps),
                  )}
                </p>
              ) : (
                <p className="muted hint">{t.stepsFollow}</p>
              )
            }
          />
        </section>
      )}
    </>
  )
}
