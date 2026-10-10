import type { RunStatus, RunSummary } from '../api'
import { formatStep } from '../format'
import { useT } from '../i18n'

interface Props {
  runs: RunSummary[]
  selected: string | null // run name, or NEW_RUN
  onSelect: (name: string) => void
  onNew: () => void
  loading: boolean
}

export const NEW_RUN = '\0new'

export function StatusPill({ status }: { status: RunStatus }) {
  const t = useT()
  return <span className={`pill ${status}`}>{t.runStatus[status]}</span>
}

export function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={max} aria-valuenow={value}>
      <div style={{ width: `${pct}%` }} />
    </div>
  )
}

export function RunList({ runs, selected, onSelect, onNew, loading }: Props) {
  const t = useT()
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{t.runs}</h2>
        <button type="button" className={`small ${selected === NEW_RUN ? 'primary' : 'ghost'}`} onClick={onNew}>
          {t.newRun}
        </button>
      </header>

      <div className="voice-list">
        {runs.map((run) => (
          <button
            key={run.name}
            type="button"
            className={`voice-card run-card ${run.name === selected ? 'active' : ''}`}
            onClick={() => onSelect(run.name)}
            aria-current={run.name === selected}
          >
            <span className="run-card-head">
              <span className="voice-name">{run.name}</span>
              <StatusPill status={run.status} />
            </span>
            <ProgressBar value={run.step ?? 0} max={run.num_iters} />
            <span className="muted">
              {t.runSteps(run.step !== null ? formatStep(run.step) : '–', formatStep(run.num_iters))}
              {run.lora_rank !== null && t.rankSuffix(run.lora_rank)}
            </span>
          </button>
        ))}
        {!loading && runs.length === 0 && <p className="muted empty">{t.noRuns}</p>}
      </div>
    </section>
  )
}
