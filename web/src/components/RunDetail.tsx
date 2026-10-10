import { useEffect, useMemo, useRef, useState } from 'react'
import { sampleUrl } from '../api'
import type { Lora, RunDetail as Run } from '../api'
import { formatDuration, formatStep, formatValue } from '../format'
import { useT } from '../i18n'
import { usePeaks } from '../peaks'
import { DatasetReview } from './DatasetReview'
import { LossChart } from './LossChart'
import { paramsFromConfig } from '../trainParams'
import { ParamsFields } from './ParamsFields'
import { ProgressBar, StatusPill } from './RunList'
import { SubTabs } from './SubTabs'
import { Waveform } from './Waveform'

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
  const t = useT()
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
              <button type="button" className="ghost" onClick={onStop} title={t.stopTitle}>
                {t.stop}
              </button>
            )}
            {run.status === 'stopping' && (
              <button type="button" className="ghost" disabled>
                {t.savingCheckpoint}
              </button>
            )}
            {(run.status === 'incomplete' || run.status === 'failed') && (
              <button type="button" className="primary" onClick={() => onResume()} disabled={otherJob}>
                {t.resumeFrom(formatStep(run.saved_step ?? 0))}
              </button>
            )}
            {run.status === 'complete' && (
              <span className="inline-field">
                <label className="muted hint" htmlFor="extend">
                  {t.continueTo}
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
                  {t.continue}
                </button>
              </span>
            )}
          </div>
        </header>

        <ProgressBar value={step} max={run.num_iters} />
        <p className="run-stats">
          <span>
            {t.stepOf} <strong>{formatStep(step)}</strong> / {formatStep(run.num_iters)}
          </span>
          {eta !== null && <span>{t.timeLeftShort(formatDuration(eta))}</span>}
          {run.seconds_per_step !== null && (
            <span className="muted">{t.secondsPerStep(run.seconds_per_step.toFixed(2))}</span>
          )}
          {run.saved_step !== null && <span className="muted">{t.checkpointAt(formatStep(run.saved_step))}</span>}
          {otherJob && <span className="muted">{t.isTraining(activeRun!)}</span>}
        </p>
        <SubTabs
          label={t.runSections}
          value={tab}
          onChange={setTab}
          tabs={[
            { id: 'progress', label: t.subTabs.progress },
            { id: 'dataset', label: t.subTabs.dataset },
            { id: 'parameters', label: t.subTabs.parameters },
          ]}
        />
      </section>

      {tab === 'progress' && (
        <>
          <section className="panel">
            <header className="panel-head">
              <h2>{t.loss}</h2>
            </header>
            <LossChart title={t.diffusionLoss} metric="loss/diff" train={run.train} val={run.val} />
            <LossChart title={t.stopLoss} metric="loss/stop" train={run.train} val={run.val} />
            <MetricsTable run={run} />
          </section>
          <Samples run={run} loras={loras} onTryCheckpoint={onTryCheckpoint} />
          <Console lines={run.log} open={run.status === 'failed'} follow={live} />
        </>
      )}

      {tab === 'dataset' && (
        <section className="panel">
          <p className="muted hint">{t.runDatasetHint}</p>
          <DatasetReview source={{ run: run.name }} />
        </section>
      )}

      {tab === 'parameters' && (
        <section className="panel">
          <ParamsFields
            params={paramsFromConfig(run.config)}
            validation={
              <div className="field">
                <span>{t.validation}</span>
                <p className="muted hint">{run.val_clips > 0 ? t.valClips(run.val_clips) : t.noValidation}</p>
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
  const t = useT()
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
      <summary>{t.tableLatest(rows.length)}</summary>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              <th>{t.columns.step}</th>
              <th>{t.columns.diffusion}</th>
              <th>{t.columns.stop}</th>
              <th>{t.columns.valDiffusion}</th>
              <th>{t.columns.valStop}</th>
              <th>{t.columns.lr}</th>
              <th>{t.columns.epoch}</th>
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
  const t = useT()
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
  // One player for the whole grid: starting a sample stops the one playing.
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState<string | null>(null)

  function play(file: string, at?: number) {
    const el = audio.current
    if (!el) return
    if (at === undefined && playing === file && !el.paused) {
      el.pause()
      return
    }
    const url = sampleUrl(run.name, file)
    if (!el.src.endsWith(url)) el.src = url
    if (at !== undefined) el.currentTime = at
    el.play().then(
      () => setPlaying(file),
      () => setPlaying(null),
    )
  }

  const cell = (file: string | undefined, key: number) =>
    file ? (
      <SampleCell
        key={key}
        url={sampleUrl(run.name, file)}
        playing={playing === file}
        audio={playing === file ? audio.current : null}
        onPlay={(at) => play(file, at)}
      />
    ) : (
      <span key={key} />
    )

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{t.samples}</h2>
        {byStep.length > 0 && <span className="muted">{t.validations(byStep.length)}</span>}
      </header>

      {byStep.length === 0 ? (
        <p className="muted empty">
          {run.val_manifest ? t.samplesEvery(String(run.config.valid_interval ?? '')) : t.noSamples}
        </p>
      ) : (
        <div className="samples" style={{ gridTemplateColumns: `auto repeat(${columns}, minmax(0, 1fr))` }}>
          <span />
          {Array.from({ length: columns }, (_, i) => (
            <p key={i} className="sample-text" title={run.sample_texts[i]}>
              {run.sample_texts[i] ?? t.sampleN(i + 1)}
            </p>
          ))}

          {run.references.length > 0 && (
            <>
              <span className="sample-step muted">{t.reference}</span>
              {Array.from({ length: columns }, (_, i) => {
                const file = `reference_${i}.wav`
                return cell(run.references.includes(file) ? file : undefined, i)
              })}
            </>
          )}

          {byStep.map(([step, files]) => {
            const id = checkpointId(run.name, step)
            return (
              <div key={step} className="sample-row">
                <span className="sample-step">
                  <span className="sample-step-label">{t.stepLabel(formatStep(step))}</span>
                  {loraIds.has(id) && (
                    <button type="button" className="ghost small" onClick={() => onTryCheckpoint(id)} title={id}>
                      {t.tryCheckpoint}
                    </button>
                  )}
                </span>
                {Array.from({ length: columns }, (_, i) => cell(files.get(i), i))}
              </div>
            )
          })}
        </div>
      )}
      <audio ref={audio} onPause={() => setPlaying(null)} onEnded={() => setPlaying(null)} hidden />
    </section>
  )
}

function Console({ lines, open, follow }: { lines: string[]; open: boolean; follow: boolean }) {
  const t = useT()
  const ref = useRef<HTMLPreElement>(null)
  useEffect(() => {
    const el = ref.current
    if (el && follow) el.scrollTop = el.scrollHeight
  }, [lines, follow])

  return (
    <details className="panel console" open={open || undefined}>
      <summary>{t.console}</summary>
      <pre ref={ref}>{lines.length ? lines.join('\n') : t.noOutput}</pre>
    </details>
  )
}

const FILE_KEYS = ['pretrained_path', 'train_manifest', 'val_manifest', 'save_path'] // labels: t.fileKeys

/** Where a run's inputs and outputs live (the parameters are shown above). */
function ConfigFiles({ config }: { config: Record<string, unknown> }) {
  const t = useT()
  return (
    <dl className="lora-details">
      {FILE_KEYS.filter((key) => config[key] !== undefined && config[key] !== '').map((key) => (
        <div key={key} className="dl-row">
          <dt>{t.fileKeys[key]}</dt>
          <dd className="path">{String(config[key])}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Play button and waveform of one sample (click the waveform to play from there). */
function SampleCell({
  url,
  playing,
  audio,
  onPlay,
}: {
  url: string
  playing: boolean
  audio: HTMLAudioElement | null
  onPlay: (at?: number) => void
}) {
  const t = useT()
  const shape = usePeaks(url)
  return (
    <div className="sample-cell">
      <button
        type="button"
        className="ghost small"
        onClick={() => onPlay()}
        aria-label={playing ? t.pause : t.playLabel}
      >
        {playing ? '❚❚' : '▶'}
      </button>
      {shape ? (
        <Waveform peaks={shape.peaks} seconds={shape.seconds} speech={null} audio={audio} onSeek={onPlay} />
      ) : (
        <span className="sample-wave-placeholder" />
      )}
      {shape && <span className="muted hint">{t.seconds(shape.seconds.toFixed(1))}</span>}
    </div>
  )
}
