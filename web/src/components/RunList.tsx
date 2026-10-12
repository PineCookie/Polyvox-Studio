import { useState } from 'react'
import type { RunStatus, RunSummary } from '../api'
import { engineName, engineRank } from '../engines'
import { useFolded } from '../folded'
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

const isLive = (run: RunSummary) => run.status === 'running' || run.status === 'stopping'

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
  const [filter, setFilter] = useState('')
  const [folded, toggleGroup] = useFolded('polyvox.runGroups') // engine ids whose group is folded

  const shown = runs.filter((r) => r.name.toLowerCase().includes(filter.trim().toLowerCase()))
  // Runs grouped by model (in the list's order within a group).
  const engines = [...new Set(runs.map((r) => r.engine))].sort((a, b) => engineRank(a) - engineRank(b))
  const groups = engines
    .map((id) => ({ id, runs: shown.filter((r) => r.engine === id) }))
    .filter((g) => g.runs.length > 0)

  // One line as in Synthesize's voice list; the status only when the run isn't complete.
  function row(run: RunSummary) {
    const complete = run.status === 'complete'
    const steps = complete
      ? t.stepsCount(formatStep(run.num_iters))
      : `${run.step !== null ? formatStep(run.step) : '–'} / ${formatStep(run.num_iters)}`
    return (
      <button
        key={run.name}
        type="button"
        className={`voice-row run-row ${run.name === selected ? 'active' : ''}`}
        onClick={() => onSelect(run.name)}
        aria-current={run.name === selected}
        title={t.runSteps(run.step !== null ? formatStep(run.step) : '–', formatStep(run.num_iters))}
      >
        <span className="voice-row-name">{run.name}</span>
        <span className="muted">
          {!complete && <StatusPill status={run.status} />}
          {[run.lora_rank !== null && t.rank(run.lora_rank), steps].filter(Boolean).join(' · ')}
        </span>
        {isLive(run) && <ProgressBar value={run.step ?? 0} max={run.num_iters} />}
      </button>
    )
  }

  return (
    <section className="panel voice-panel">
      <header className="panel-head">
        <h2>{t.runs}</h2>
        <button type="button" className={`small ${selected === NEW_RUN ? 'primary' : 'ghost'}`} onClick={onNew}>
          {t.newRun}
        </button>
      </header>

      {runs.length > 6 && (
        <input
          className="voice-filter"
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={t.filterRuns(runs.length)}
          aria-label={t.filterRunsLabel}
        />
      )}

      <div className="voice-list">
        {groups.map((group) => {
          // While filtering every group is open; a folded one still shows the open run and live ones.
          const open = !!filter || !folded.has(group.id)
          const visible = (r: RunSummary) => open || r.name === selected || isLive(r)
          return (
            <div key={group.id} className="voice-group run-group">
              <button
                type="button"
                className="voice-group-head"
                aria-expanded={open}
                onClick={() => !filter && toggleGroup(group.id)}
              >
                <span className="chevron" aria-hidden>
                  {open ? '▾' : '▸'}
                </span>
                <span>{engineName(group.id)}</span>
                <span className="muted">{group.runs.length}</span>
              </button>
              {group.runs.filter(visible).map(row)}
            </div>
          )
        })}
        {!loading && runs.length === 0 && <p className="muted empty">{t.noRuns}</p>}
        {runs.length > 0 && shown.length === 0 && <p className="muted empty">{t.noRunMatches(filter)}</p>}
      </div>
    </section>
  )
}
