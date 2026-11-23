import { useEffect, useState } from 'react'
import { getStatus, planRun } from '../api'
import type {
  DatasetReport,
  Engine,
  EngineParam,
  FormValue,
  FormValues,
  GpuProfile,
  RunPlan,
  TrainRequest,
} from '../api'
import { datasetName, formatStep } from '../format'
import { useT } from '../i18n'
import { DatasetReview } from './DatasetReview'
import { SCHEDULE_KEYS, allNumbers, formDefaults, withValue } from '../trainParams'
import { ParamsFields } from './ParamsFields'
import { SubTabs } from './SubTabs'

interface Props {
  datasets: string[]
  runNames: string[]
  // Name of a run that is currently training, if any; only one job runs at a time.
  activeRun: string | null
  onStart: (params: TrainRequest) => Promise<void>
  // Preselected from the Prepare tab ("Review and train").
  initialDataset?: string
  // Every engine (those that train are offered).
  engines: Engine[]
}

type Tab = 'progress' | 'dataset' | 'parameters'

// Matches training.RUN_NAME: letters of any language, digits, "." "_" "-".
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u
const PLAN_DELAY_MS = 150 // after the last change

/** dataset-model, the model part from the engine (RunPlan.name_tag: voice-voxcpm2-r32,
 * voice-v2pp, voice-rvc48k ...); a repeat adds the month and day (-1012), then a counter (-1012-2). */
function suggestName(dataset: string, tag: string, taken: string[], now = new Date()): string {
  const base = [datasetName(dataset).split('/').at(-1), tag].join('-').replace(/[^\p{L}\p{N}_.-]/gu, '_')
  if (!taken.includes(base)) return base
  const day = `${base}-${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`
  let name = day
  for (let i = 2; taken.includes(name); i++) name = `${day}-${i}`
  return name
}

/** A run before it starts: the same page as an existing run, with editable sections. */
export function NewRunForm({ datasets, runNames, activeRun, onStart, initialDataset, engines }: Props) {
  const t = useT()
  const [tab, setTab] = useState<Tab>('dataset')
  const trainable = engines.filter((e) => e.capabilities.training)
  const [engineId, setEngineId] = useState<string | null>(null) // null: the first
  const engine = trainable.find((e) => e.id === engineId) ?? trainable[0]
  const form = engine?.train_form ?? []
  // Values changed so far (null: the engine's defaults), and the sizing fields set by hand.
  const [edited, setEdited] = useState<FormValues | null>(null)
  const [overrides, setOverrides] = useState<string[]>([])
  const values = edited ?? formDefaults(form)
  const [dataset, setDataset] = useState(initialDataset ?? '')
  const [customName, setCustomName] = useState<string | null>(null)
  const [useValDataset, setUseValDataset] = useState(false)
  const [valDataset, setValDataset] = useState<string | null>(null)
  const [valHoldout, setValHoldout] = useState(2)
  const [report, setReport] = useState<DatasetReport | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [gpu, setGpu] = useState<GpuProfile | null>(null)
  const [plan, setPlan] = useState<{ engine: string; plan: RunPlan } | null>(null)

  useEffect(() => {
    getStatus().then(
      (s) => setGpu(s.gpu ?? null),
      () => {},
    )
  }, [])

  const trainClips = report ? report.valid - (useValDataset ? 0 : Math.min(valHoldout, report.valid - 1)) : 0
  const seconds = report && report.valid > 0 ? (report.total_seconds * trainClips) / report.valid : 0

  // What the run will do (its steps etc. follow from the values and the clips): asked of the server.
  const planKey = JSON.stringify([engine?.id, values, trainClips, seconds, overrides])
  useEffect(() => {
    if (!engine) return
    let alive = true
    const id = engine.id
    const timer = setTimeout(() => {
      planRun({ engine: id, values, clips: Math.max(0, trainClips), seconds, overrides }).then(
        (p) => alive && setPlan({ engine: id, plan: p }),
        () => {}, // an unfinished number: the last plan stays until the values are complete
      )
    }, PLAN_DELAY_MS)
    return () => {
      alive = false
      clearTimeout(timer)
    }
    // The key holds everything the plan depends on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [planKey])
  const current = plan && plan.engine === engine?.id ? plan.plan : null
  const params: FormValues = { ...values, ...(current?.values ?? {}) }
  const summary = current?.summary

  const name = customName ?? (dataset && current ? suggestName(dataset, current.name_tag, runNames) : '')
  const change = (p: EngineParam, value: FormValue) => {
    if (SCHEDULE_KEYS.includes(p.name)) setOverrides((o) => (o.includes(p.name) ? o : [...o, p.name]))
    if (p.name === 'epochs') setOverrides((o) => o.filter((k) => k !== 'num_iters')) // the steps follow again
    setEdited(withValue(params, p, value))
  }
  const chooseEngine = (id: string) => {
    // Each model has its own fields and defaults (batch for this GPU, optimizer).
    setEngineId(id)
    setEdited(null)
    setOverrides([])
  }

  const nameError = !name ? null : !NAME_PATTERN.test(name) ? t.nameChars : runNames.includes(name) ? t.runExists : null
  const numbersValid = allNumbers(params) && Number.isFinite(valHoldout)
  const valValid = !useValDataset || !!valDataset
  // Unreadable clips would crash the job mid-epoch, so they block the start.
  const datasetValid = !!report && report.valid > 0 && report.errors.length === 0

  // What still blocks the start, in the order the page is filled in.
  const blocker = !engine
    ? t.loading
    : !dataset
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
    if (blocker || submitting || !engine) return
    setSubmitting(true)
    try {
      const scoped = (scope: EngineParam['scope']) =>
        Object.fromEntries(form.filter((p) => p.scope === scope).map((p) => [p.name, params[p.name]]))
      await onStart({
        ...scoped('run'),
        options: scoped('option'),
        engine: engine.id,
        name,
        dataset,
        val_dataset: useValDataset ? valDataset : null,
        val_holdout: valHoldout,
      })
    } finally {
      setSubmitting(false)
    }
  }

  const gpuGb = gpu?.total_gb ? String(gpu.total_gb) : ''
  const step = (n: number | null | undefined) => formatStep(n ?? 0)

  // One line on what the run does, and the longer note under the fields.
  const stats = !summary
    ? null
    : summary.kind === 'stages'
      ? t.stageSteps(step(summary.sovits), step(summary.gpt))
      : summary.kind === 'pieces'
        ? t.rvcSteps(step(summary.steps), (summary.pieces ?? 0).toLocaleString())
        : t.stepsTimesClips(step(summary.steps), summary.per_step ?? 1)
  const model = !summary
    ? ''
    : summary.model === 'v4_lora'
      ? t.v4Lora(summary.rank ?? 0)
      : summary.model === 'full'
        ? t.fullFinetune
        : t.rankAlpha(summary.rank ?? 0, summary.alpha ?? 0)
  const target = typeof params.epochs === 'number' ? params.epochs : null
  const note =
    !summary || trainClips <= 0 ? (
      <p className="muted hint">{t.stepsFollow}</p>
    ) : (
      <p className="muted hint">
        {summary.kind === 'stages'
          ? t.fullScheduleHint(
              step(summary.sovits),
              summary.batch ?? 1,
              step(summary.gpt),
              summary.gpt_batch ?? 1,
              gpuGb,
              trainClips.toLocaleString(),
            )
          : summary.kind === 'pieces'
            ? t.rvcScheduleHint(
                summary.epochs ?? 0,
                (summary.pieces ?? 0).toLocaleString(),
                summary.batch ?? 1,
                gpuGb,
                step(summary.steps),
                summary.save_every ?? 0,
              )
            : t.scheduleHint(
                step(summary.steps),
                summary.per_step ?? 1,
                summary.batch ?? 1,
                summary.accum ?? 1,
                gpuGb,
                trainClips.toLocaleString(),
                summary.epochs != null && target !== null && Math.abs(summary.epochs - target) > 0.05
                  ? summary.epochs.toFixed(1)
                  : '',
                step(summary.save_interval),
                step(summary.warmup_steps),
              )}
      </p>
    )

  const samples = current?.samples ?? { every: Number(params.valid_interval) || 0, unit: 'steps' }
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
        <select value={valDataset ?? ''} onChange={(e) => setValDataset(e.target.value || null)}>
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
            value={Number.isFinite(valHoldout) ? valHoldout : ''}
            onChange={(e) => setValHoldout(e.target.valueAsNumber)}
          />
          <span className="muted hint">
            {samples.unit === 'epochs' ? t.holdoutEpochsHint(samples.every) : t.holdoutHint(samples.every)}
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
              {summary?.kind === 'epochs' && summary.epochs != null && (
                <span>{t.epochsValue(summary.epochs.toFixed(1))}</span>
              )}
              {stats && <span className="muted">{stats}</span>}
              <span className="muted">
                {trainable.length > 1 && `${engine?.name} · `}
                {model}
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
          {engine ? (
            <ParamsFields
              form={form}
              values={params}
              engine={engine.id}
              engines={trainable}
              onEngine={chooseEngine}
              onChange={change}
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
              hint={note}
            />
          ) : (
            <p className="muted empty">{t.loading}</p>
          )}
        </section>
      )}
    </>
  )
}
