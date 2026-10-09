import { useEffect, useMemo, useRef, useState } from 'react'
import { sampleUrl } from '../api'
import type { Lora, RunDetail as Run } from '../api'
import { formatDuration, formatStep, formatValue } from '../format'
import { DatasetReview } from './DatasetReview'
import { LossChart } from './LossChart'
import { paramsFromConfig } from '../trainParams'
import { ParamsFields } from './ParamsFields'
import { ProgressBar, StatusPill } from './RunList'
import { SubTabs } from './SubTabs'

interface Props {
  run: Run
  loras: Lora[]
  activeRun: string | null
  onStop: () => void
  onResume: (numIters?: number) => void
  onTryCheckpoint: (loraId: string) => void
}

const checkpointId = (run: string, step: number) => `${run}/checkpoints/step_${String(step).padStart(7, '0')}`

type Tab = 'progress' | 'dataset' | 'parameters'

export function RunDetail({ run, loras, activeRun, onStop, onResume, onTryCheckpoint }: Props) {
  const [tab, setTab] = useState<Tab>('progress')
  const [extendTo, setExtendTo] = useState(run.num_iters * 2)
  const live = run.status === 'running' || run.status === 'stopping'
  const step = run.step ?? 0
  const eta = live && run.seconds_per_step ? (run.num_iters - step) * run.seconds_per_step : null
  const otherJob = activeRun !== null && activeRun !== run.name

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h2 className="run-title">
            {run.name} <StatusPill status={run.status} />
          </h2>
          <div className="run-actions">
            {run.status === 'running' && (
              <button type="button" className="ghost" onClick={onStop} title="Saves a checkpoint, then stops">
                Stop
              </button>
            )}
            {run.status === 'stopping' && (
              <button type="button" className="ghost" disabled>
                Saving checkpoint…
              </button>
            )}
            {(run.status === 'incomplete' || run.status === 'failed') && (
              <button type="button" className="primary" onClick={() => onResume()} disabled={otherJob}>
                Resume from step {formatStep(run.saved_step ?? 0)}
              </button>
            )}
            {run.status === 'complete' && (
              <span className="inline-field">
                <label className="muted hint" htmlFor="extend">
                  Continue to
                </label>
                <input
                  id="extend"
                  type="number"
                  min={run.num_iters + 1}
                  step={100}
                  value={Number.isFinite(extendTo) ? extendTo : ''}
                  onChange={(e) => setExtendTo(e.target.valueAsNumber)}
                />
                <button
                  type="button"
                  className="ghost"
                  onClick={() => onResume(extendTo)}
                  disabled={otherJob || !(extendTo > run.num_iters)}
                >
                  Continue
                </button>
              </span>
            )}
          </div>
        </header>

        <ProgressBar value={step} max={run.num_iters} />
        <p className="run-stats">
          <span>
            Step <strong>{formatStep(step)}</strong> / {formatStep(run.num_iters)}
          </span>
          {eta !== null && <span>{formatDuration(eta)} left</span>}
          {run.seconds_per_step !== null && <span className="muted">{run.seconds_per_step.toFixed(2)} s/step</span>}
          {run.saved_step !== null && <span className="muted">checkpoint at step {formatStep(run.saved_step)}</span>}
          {otherJob && <span className="muted">“{activeRun}” is training</span>}
        </p>
        <SubTabs
          label="Run sections"
          value={tab}
          onChange={setTab}
          tabs={[
            { id: 'progress', label: 'Progress' },
            { id: 'dataset', label: 'Dataset' },
            { id: 'parameters', label: 'Parameters' },
          ]}
        />
      </section>

      {tab === 'progress' && (
        <>
          <section className="panel">
            <header className="panel-head">
              <h2>Loss</h2>
            </header>
            <LossChart title="Diffusion loss" metric="loss/diff" train={run.train} val={run.val} />
            <LossChart title="Stop loss" metric="loss/stop" train={run.train} val={run.val} />
            <MetricsTable run={run} />
          </section>
          <Samples run={run} loras={loras} onTryCheckpoint={onTryCheckpoint} />
          <Console lines={run.log} open={run.status === 'failed'} follow={live} />
        </>
      )}

      {tab === 'dataset' && (
        <section className="panel">
          <p className="muted hint">The clips this run trains on (validation clips not included).</p>
          <DatasetReview source={{ run: run.name }} />
        </section>
      )}

      {tab === 'parameters' && (
        <section className="panel">
          <ParamsFields
            params={paramsFromConfig(run.config)}
            validation={
              <div className="field">
                <span>Validation</span>
                <p className="muted hint">
                  {run.val_clips > 0
                    ? `${run.val_clips} clips, for validation loss and audio samples`
                    : 'None: no validation loss or samples'}
                </p>
              </div>
            }
          />
          <ConfigFiles config={run.config} />
        </section>
      )}
    </>
  )
}

function MetricsTable({ run }: { run: Run }) {
  const rows = useMemo(() => {
    const val = new Map(run.val.map((p) => [p.step, p]))
    return run.train
      .slice(-30)
      .reverse()
      .map((p) => ({ train: p, val: val.get(p.step) }))
  }, [run.train, run.val])
  if (rows.length === 0) return null
  return (
    <details className="advanced">
      <summary>Table (latest {rows.length} logged steps)</summary>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              <th>Step</th>
              <th>Diffusion</th>
              <th>Stop</th>
              <th>Val diffusion</th>
              <th>Val stop</th>
              <th>LR</th>
              <th>Epoch</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ train, val }) => (
              <tr key={train.step}>
                <td>{formatStep(train.step)}</td>
                <td>{formatValue(train['loss/diff'])}</td>
                <td>{formatValue(train['loss/stop'])}</td>
                <td>{val ? formatValue(val['loss/diff']) : ''}</td>
                <td>{val ? formatValue(val['loss/stop']) : ''}</td>
                <td>{formatValue(train.lr)}</td>
                <td>{train.epoch?.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}

function Samples({ run, loras, onTryCheckpoint }: { run: Run; loras: Lora[]; onTryCheckpoint: (id: string) => void }) {
  const byStep = useMemo(() => {
    const steps = new Map<number, Map<number, string>>()
    for (const s of run.samples) {
      if (!steps.has(s.step)) steps.set(s.step, new Map())
      steps.get(s.step)!.set(s.index, s.file)
    }
    return [...steps.entries()].sort((a, b) => b[0] - a[0])
  }, [run.samples])
  const loraIds = useMemo(() => new Set(loras.map((l) => l.id)), [loras])
  const columns = Math.max(run.sample_texts.length, run.references.length, ...byStep.map(([, m]) => m.size), 1)

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Samples</h2>
        {byStep.length > 0 && <span className="muted">{byStep.length} validations</span>}
      </header>

      {byStep.length === 0 ? (
        <p className="muted empty">
          {run.val_manifest
            ? `Samples of the validation clips appear every ${String(run.config.valid_interval ?? '')} steps.`
            : 'This run has no validation set, so no samples are generated.'}
        </p>
      ) : (
        <div className="samples" style={{ gridTemplateColumns: `auto repeat(${columns}, minmax(0, 1fr))` }}>
          <span />
          {Array.from({ length: columns }, (_, i) => (
            <p key={i} className="sample-text" title={run.sample_texts[i]}>
              {run.sample_texts[i] ?? `Sample ${i + 1}`}
            </p>
          ))}

          {run.references.length > 0 && (
            <>
              <span className="sample-step muted">Reference</span>
              {Array.from({ length: columns }, (_, i) => {
                const file = `reference_${i}.wav`
                return run.references.includes(file) ? (
                  <audio key={i} controls preload="none" src={sampleUrl(run.name, file)} />
                ) : (
                  <span key={i} />
                )
              })}
            </>
          )}

          {byStep.map(([step, files]) => {
            const id = checkpointId(run.name, step)
            return (
              <div key={step} className="sample-row">
                <span className="sample-step">
                  Step {formatStep(step)}
                  {loraIds.has(id) && (
                    <button type="button" className="ghost small" onClick={() => onTryCheckpoint(id)} title={id}>
                      Try
                    </button>
                  )}
                </span>
                {Array.from({ length: columns }, (_, i) => {
                  const file = files.get(i)
                  return file ? (
                    <audio key={i} controls preload="none" src={sampleUrl(run.name, file)} />
                  ) : (
                    <span key={i} />
                  )
                })}
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function Console({ lines, open, follow }: { lines: string[]; open: boolean; follow: boolean }) {
  const ref = useRef<HTMLPreElement>(null)
  useEffect(() => {
    const el = ref.current
    if (el && follow) el.scrollTop = el.scrollHeight
  }, [lines, follow])

  return (
    <details className="panel console" open={open || undefined}>
      <summary>Console</summary>
      <pre ref={ref}>{lines.length ? lines.join('\n') : 'No output yet.'}</pre>
    </details>
  )
}

const FILE_KEYS: [string, string][] = [
  ['pretrained_path', 'Base model'],
  ['train_manifest', 'Train manifest'],
  ['val_manifest', 'Validation manifest'],
  ['save_path', 'Checkpoints'],
]

/** Where a run's inputs and outputs live (the parameters are shown above). */
function ConfigFiles({ config }: { config: Record<string, unknown> }) {
  return (
    <dl className="lora-details">
      {FILE_KEYS.filter(([key]) => config[key] !== undefined && config[key] !== '').map(([key, label]) => (
        <div key={key} className="dl-row">
          <dt>{label}</dt>
          <dd className="path">{String(config[key])}</dd>
        </div>
      ))}
    </dl>
  )
}
