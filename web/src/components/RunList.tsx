import type { RunStatus, RunSummary } from '../api'
import { formatStep } from '../format'

interface Props {
  runs: RunSummary[]
  selected: string | null // run name, or NEW_RUN
  onSelect: (name: string) => void
  onNew: () => void
  loading: boolean
}

export const NEW_RUN = '\0new'

const STATUS_LABEL: Record<RunStatus, string> = {
  running: 'Training',
  stopping: 'Stopping',
  complete: 'Complete',
  incomplete: 'Paused',
  failed: 'Failed',
}

export function StatusPill({ status }: { status: RunStatus }) {
  return <span className={`pill ${status}`}>{STATUS_LABEL[status]}</span>
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
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Runs</h2>
        <button type="button" className={`small ${selected === NEW_RUN ? 'primary' : 'ghost'}`} onClick={onNew}>
          New run
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
              {run.step !== null ? formatStep(run.step) : '–'} / {formatStep(run.num_iters)} steps
              {run.lora_rank !== null && ` · rank ${run.lora_rank}`}
            </span>
          </button>
        ))}
        {!loading && runs.length === 0 && <p className="muted empty">No training runs yet.</p>}
      </div>
    </section>
  )
}
