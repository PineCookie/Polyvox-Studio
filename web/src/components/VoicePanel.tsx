import { useMemo } from 'react'
import type { Lora } from '../api'

interface Props {
  loras: Lora[]
  selected: string | null
  onSelect: (id: string | null) => void
  onRefresh: () => void
  loading: boolean
}

interface Run {
  name: string
  checkpoints: { lora: Lora; label: string }[]
}

// "speaker_r64/checkpoints/step_0005000" -> run "speaker_r64", label "Step 5,000"
function groupByRun(loras: Lora[]): Run[] {
  const runs = new Map<string, Run>()
  for (const lora of loras) {
    const [name, ...rest] = lora.id.split('/')
    const step = /step_0*(\d+)$/.exec(lora.id)
    const label = step ? `Step ${Number(step[1]).toLocaleString()}` : rest.join('/') || 'Latest'
    if (!runs.has(name)) runs.set(name, { name, checkpoints: [] })
    runs.get(name)!.checkpoints.push({ lora, label })
  }
  return [...runs.values()]
}

export function VoicePanel({ loras, selected, onSelect, onRefresh, loading }: Props) {
  const runs = useMemo(() => groupByRun(loras), [loras])
  const current = loras.find((l) => l.id === selected) ?? null

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Voice</h2>
        <button type="button" className="ghost small" onClick={onRefresh} disabled={loading}>
          {loading ? 'Scanning…' : 'Refresh'}
        </button>
      </header>

      <div className="voice-list" role="radiogroup" aria-label="Voice">
        <button
          type="button"
          role="radio"
          aria-checked={selected === null}
          className={`voice-card ${selected === null ? 'active' : ''}`}
          onClick={() => onSelect(null)}
        >
          <span className="voice-name">Base model</span>
          <span className="muted">No LoRA</span>
        </button>

        {runs.map((run) => {
          const runActive = run.checkpoints.some((c) => c.lora.id === selected)
          return (
            <div key={run.name} className={`voice-card run ${runActive ? 'active' : ''}`}>
              <span className="voice-name">{run.name}</span>
              <div className="chips">
                {run.checkpoints.map(({ lora, label }) => (
                  <button
                    key={lora.id}
                    type="button"
                    role="radio"
                    aria-checked={lora.id === selected}
                    className={`chip ${lora.id === selected ? 'active' : ''}`}
                    onClick={() => onSelect(lora.id)}
                    title={lora.id}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )
        })}

        {!loading && loras.length === 0 && (
          <p className="muted empty">No LoRA checkpoints found. Train one, or set VOXCPM_STUDIO_LORA_DIR.</p>
        )}
      </div>

      {current && (
        <dl className="lora-details">
          <dt>Rank / alpha</dt>
          <dd>{current.config ? `${current.config.r} / ${current.config.alpha}` : 'default (no lora_config.json)'}</dd>
          {current.config && (
            <>
              <dt>Layers</dt>
              <dd>
                {[
                  current.config.enable_lm && 'LM',
                  current.config.enable_dit && 'DiT',
                  current.config.enable_proj && 'Projections',
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </dd>
              <dt>Targets</dt>
              <dd>{[...new Set([...current.config.target_modules_lm, ...current.config.target_modules_dit])].join(', ')}</dd>
            </>
          )}
          <dt>Base model</dt>
          <dd className="path">{current.base_model ?? 'not recorded'}</dd>
        </dl>
      )}
    </section>
  )
}
