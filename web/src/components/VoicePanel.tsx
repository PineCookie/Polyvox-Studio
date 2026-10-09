import { useMemo, useRef, useState } from 'react'
import type { Lora } from '../api'
import { formatStep } from '../format'
import { ExportDialog } from './ExportDialog'

interface Props {
  loras: Lora[]
  selected: string | null
  // Hub id of the model the server runs; LoRAs for another base model can't load.
  modelId: string | null
  onSelect: (id: string | null) => void
  onRefresh: () => void
  onImport: (file: File) => Promise<void>
  onError: (message: string) => void
  loading: boolean
}

interface Checkpoint {
  lora: Lora
  step: number | null
  label: string
}

interface Voice {
  name: string
  checkpoints: Checkpoint[] // best (highest step) first
  baseModelId: string | null
}

// Groups checkpoints by run ("speaker_r64/checkpoints/step_0005000" -> "speaker_r64") and drops
// copies of the final weights: the script saves the last step twice (N-1 and N), and
// "latest" duplicates whichever step it was copied from.
function groupVoices(loras: Lora[]): Voice[] {
  const byName = new Map<string, Lora[]>()
  for (const lora of loras) {
    const name = lora.id.split('/')[0]
    byName.set(name, [...(byName.get(name) ?? []), lora])
  }

  return [...byName.entries()]
    .map(([name, list]) => {
      const stepDirs = list.filter((l) => /step_\d+$/.test(l.id))
      const steps = new Set(stepDirs.map((l) => l.step))
      const max = Math.max(-1, ...stepDirs.map((l) => l.step ?? -1))
      const checkpoints = list
        .filter((l) => {
          if (/step_\d+$/.test(l.id)) return !(l.step === max - 1 && steps.has(max))
          return l.step === null || !steps.has(l.step) // "latest" or an imported package
        })
        .map((lora) => ({
          lora,
          step: lora.step,
          label: lora.step !== null ? `Step ${formatStep(lora.step)}` : 'Latest',
        }))
        .sort((a, b) => (b.step ?? -1) - (a.step ?? -1))
      return { name, checkpoints, baseModelId: checkpoints[0]?.lora.base_model_id ?? null }
    })
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
}

function shortModel(id: string): string {
  return id.split('/').at(-1) ?? id
}

export function VoicePanel({ loras, selected, modelId, onSelect, onRefresh, onImport, onError, loading }: Props) {
  const voices = useMemo(() => groupVoices(loras), [loras])
  const [filter, setFilter] = useState('')
  const [exporting, setExporting] = useState<Lora | null>(null)
  const [importing, setImporting] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const current = loras.find((l) => l.id === selected) ?? null
  const currentVoice = current ? voices.find((v) => v.name === current.id.split('/')[0]) : null
  // A hidden duplicate (e.g. picked from the Train tab) still needs an option.
  const options =
    currentVoice && current && !currentVoice.checkpoints.some((c) => c.lora.id === current.id)
      ? [...currentVoice.checkpoints, { lora: current, step: current.step, label: current.id.split('/').at(-1)! }]
      : (currentVoice?.checkpoints ?? [])
  const incompatible = (v: Voice) => !!modelId && !!v.baseModelId && v.baseModelId !== modelId
  const shown = voices.filter((v) => v.name.toLowerCase().includes(filter.trim().toLowerCase()))

  async function importFile(file: File | undefined) {
    if (!file) return
    setImporting(true)
    try {
      await onImport(file)
    } finally {
      setImporting(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  return (
    <section className="panel voice-panel">
      <header className="panel-head">
        <h2>Voice</h2>
        <span className="run-actions">
          <input
            ref={fileInput}
            type="file"
            accept=".zip,application/zip"
            hidden
            onChange={(e) => importFile(e.target.files?.[0])}
          />
          <button
            type="button"
            className="ghost small"
            onClick={() => fileInput.current?.click()}
            disabled={importing}
            title="Add a LoRA package (.zip) exported from VoxCPM Studio"
          >
            {importing ? 'Importing…' : 'Import'}
          </button>
          <button type="button" className="ghost small" onClick={onRefresh} disabled={loading}>
            {loading ? 'Scanning…' : 'Refresh'}
          </button>
        </span>
      </header>

      <div className="voice-current">
        {current && currentVoice ? (
          <>
            <div className="voice-current-head">
              <span className="voice-name">{currentVoice.name}</span>
              <button type="button" className="ghost small" onClick={() => setExporting(current)}>
                Export
              </button>
            </div>
            {options.length > 1 && (
              <select value={current.id} onChange={(e) => onSelect(e.target.value)} aria-label="Checkpoint">
                {options.map((c, i) => (
                  <option key={c.lora.id} value={c.lora.id}>
                    {c.label}
                    {i === 0 ? ' (final)' : ''}
                  </option>
                ))}
              </select>
            )}
            <p className="voice-meta">
              {current.config
                ? `Rank ${current.config.r} · alpha ${current.config.alpha} · ${[
                    current.config.enable_lm && 'LM',
                    current.config.enable_dit && 'DiT',
                    current.config.enable_proj && 'projections',
                  ]
                    .filter(Boolean)
                    .join(' + ')}`
                : 'Default layout (no lora_config.json)'}
              <br />
              <span title={current.base_model ?? undefined}>
                Base {current.base_model_id ?? current.base_model ?? 'not recorded'}
              </span>
            </p>
          </>
        ) : (
          <>
            <span className="voice-name">Base model</span>
            <p className="voice-meta">No LoRA{modelId ? ` · ${modelId}` : ''}</p>
          </>
        )}
      </div>

      {voices.length > 6 && (
        <input
          className="voice-filter"
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={`Filter ${voices.length} voices`}
          aria-label="Filter voices"
        />
      )}

      <div className="voice-list" role="radiogroup" aria-label="Voice">
        {!filter && (
          <button
            type="button"
            role="radio"
            aria-checked={selected === null}
            className={`voice-row ${selected === null ? 'active' : ''}`}
            onClick={() => onSelect(null)}
          >
            <span className="voice-row-name">Base model</span>
            <span className="muted">no LoRA</span>
          </button>
        )}

        {shown.map((voice) => {
          const active = voice === currentVoice
          const best = voice.checkpoints[0]
          const blocked = incompatible(voice)
          return (
            <button
              key={voice.name}
              type="button"
              role="radio"
              aria-checked={active}
              className={`voice-row ${active ? 'active' : ''}`}
              onClick={() => onSelect(best.lora.id)}
              disabled={blocked}
              title={blocked ? `Trained on ${voice.baseModelId}; the server runs ${modelId}` : best.lora.id}
            >
              <span className="voice-row-name">{voice.name}</span>
              {blocked ? (
                <span className="tag subtle">{shortModel(voice.baseModelId!)}</span>
              ) : (
                <span className="muted">
                  {[
                    best.lora.config && `r${best.lora.config.r}`,
                    best.step !== null && `${formatStep(best.step)} steps`,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </span>
              )}
            </button>
          )
        })}

        {!loading && voices.length === 0 && (
          <p className="muted empty">No LoRAs yet. Train one, or import a shared package.</p>
        )}
        {voices.length > 0 && shown.length === 0 && <p className="muted empty">No voice matches “{filter}”.</p>}
      </div>

      {exporting && (
        <ExportDialog
          lora={exporting}
          final={currentVoice?.checkpoints[0]?.lora.id === exporting.id}
          onError={onError}
          onClose={() => setExporting(null)}
        />
      )}
    </section>
  )
}
