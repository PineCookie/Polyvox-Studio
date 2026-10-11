import { useMemo, useRef, useState } from 'react'
import { DEFAULT_ENGINE } from '../api'
import type { Engine, Lora } from '../api'
import type { VoiceChoice } from '../engines'
import { formatStep } from '../format'
import { useT } from '../i18n'
import type { Messages } from '../i18n'
import { ExportDialog } from './ExportDialog'

interface Props {
  engines: Engine[]
  loras: Lora[]
  selected: VoiceChoice
  // Hub id of the VoxCPM model the server runs; VoxCPM LoRAs for another base model can't load.
  modelId: string | null
  onSelect: (choice: VoiceChoice) => void
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
  engine: string
  checkpoints: Checkpoint[] // best (highest step) first
  baseModelId: string | null
}

// Groups checkpoints by run ("speaker_r64/checkpoints/step_0005000" -> "speaker_r64") and drops
// copies of the final weights: the script saves the last step twice (N-1 and N), and
// "latest" duplicates whichever step it was copied from.
function groupVoices(loras: Lora[], t: Messages): Voice[] {
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
          label: lora.step !== null ? t.stepLabel(formatStep(lora.step)) : t.latest,
        }))
        .sort((a, b) => (b.step ?? -1) - (a.step ?? -1))
      const best = checkpoints[0]?.lora
      return { name, engine: best?.engine ?? DEFAULT_ENGINE, checkpoints, baseModelId: best?.base_model_id ?? null }
    })
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
}

function shortModel(id: string): string {
  return id.split('/').at(-1) ?? id
}

export function VoicePanel({
  engines,
  loras,
  selected,
  modelId,
  onSelect,
  onRefresh,
  onImport,
  onError,
  loading,
}: Props) {
  const t = useT()
  const voices = useMemo(() => groupVoices(loras, t), [loras, t])
  const [filter, setFilter] = useState('')
  const [exporting, setExporting] = useState<Lora | null>(null)
  const [importing, setImporting] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const engineName = (id: string) => engines.find((e) => e.id === id)?.name ?? id
  const current = loras.find((l) => l.id === selected.lora) ?? null
  const currentVoice = current ? voices.find((v) => v.name === current.id.split('/')[0]) : null
  // A hidden duplicate (e.g. picked from the Train tab) still needs an option.
  const options =
    currentVoice && current && !currentVoice.checkpoints.some((c) => c.lora.id === current.id)
      ? [...currentVoice.checkpoints, { lora: current, step: current.step, label: current.id.split('/').at(-1)! }]
      : (currentVoice?.checkpoints ?? [])
  // Only VoxCPM records a base model the server could differ from; other engines have one model each.
  const incompatible = (v: Voice) =>
    v.engine === DEFAULT_ENGINE && !!modelId && !!v.baseModelId && v.baseModelId !== modelId
  const unknownEngine = (v: Voice) => engines.length > 0 && !engines.some((e) => e.id === v.engine)
  const shown = voices.filter((v) => v.name.toLowerCase().includes(filter.trim().toLowerCase()))
  const select = (lora: Lora) => onSelect({ engine: lora.engine, lora: lora.id })
  const layout = (lora: Lora) =>
    lora.config
      ? t.loraLayout(
          lora.config.r,
          lora.config.alpha,
          [lora.config.enable_lm && 'LM', lora.config.enable_dit && 'DiT', lora.config.enable_proj && t.projections]
            .filter(Boolean)
            .join(' + '),
        )
      : lora.rank !== null && lora.alpha !== null
        ? t.rankAlpha(lora.rank, lora.alpha)
        : t.defaultLayout

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
        <h2>{t.voice}</h2>
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
            title={t.importLoraTitle}
          >
            {importing ? t.importing : t.import}
          </button>
          <button type="button" className="ghost small" onClick={onRefresh} disabled={loading}>
            {loading ? t.scanning : t.refresh}
          </button>
        </span>
      </header>

      <div className="voice-current">
        {current && currentVoice ? (
          <>
            <div className="voice-current-head">
              <span className="voice-name">{currentVoice.name}</span>
              {current.engine === DEFAULT_ENGINE && (
                <button type="button" className="ghost small" onClick={() => setExporting(current)}>
                  {t.export}
                </button>
              )}
            </div>
            {options.length > 1 && (
              <select
                value={current.id}
                onChange={(e) => onSelect({ engine: current.engine, lora: e.target.value })}
                aria-label={t.checkpoint}
              >
                {options.map((c, i) => (
                  <option key={c.lora.id} value={c.lora.id}>
                    {c.label}
                    {i === 0 ? t.final : ''}
                  </option>
                ))}
              </select>
            )}
            <p className="voice-meta">
              {engineName(current.engine)} · {layout(current)}
              <br />
              <span title={current.base_model ?? undefined}>
                {t.baseOf(current.base_model_id ?? current.base_model ?? t.notRecorded)}
              </span>
            </p>
          </>
        ) : (
          <>
            <span className="voice-name">{t.baseModel}</span>
            <p className="voice-meta">
              {engineName(selected.engine)} · {t.noLora}
              {selected.engine === DEFAULT_ENGINE && modelId ? ` · ${modelId}` : ''}
            </p>
          </>
        )}
      </div>

      {voices.length > 6 && (
        <input
          className="voice-filter"
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={t.filterVoices(voices.length)}
          aria-label={t.filterVoicesLabel}
        />
      )}

      <div className="voice-list" role="radiogroup" aria-label={t.voice}>
        {/* A base model per engine, in the server's order (VoxCPM first). */}
        {!filter &&
          engines.map((engine) => {
            const active = selected.lora === null && selected.engine === engine.id
            return (
              <button
                key={`base:${engine.id}`}
                type="button"
                role="radio"
                aria-checked={active}
                className={`voice-row ${active ? 'active' : ''}`}
                onClick={() => onSelect({ engine: engine.id, lora: null })}
                disabled={!!engine.unavailable}
                title={engine.unavailable ? t.engineUnavailable(engine.name, engine.unavailable) : undefined}
              >
                <span className="voice-row-name">{t.baseModel}</span>
                <span className="muted">{engine.name}</span>
              </button>
            )
          })}

        {shown.map((voice) => {
          const active = voice === currentVoice
          const best = voice.checkpoints[0]
          const blocked = incompatible(voice) || unknownEngine(voice)
          const otherEngine = voice.engine !== DEFAULT_ENGINE
          return (
            <button
              key={voice.name}
              type="button"
              role="radio"
              aria-checked={active}
              className={`voice-row ${active ? 'active' : ''}`}
              onClick={() => select(best.lora)}
              disabled={blocked}
              title={incompatible(voice) ? t.trainedOn(voice.baseModelId!, modelId!) : best.lora.id}
            >
              <span className="voice-row-name">{voice.name}</span>
              {incompatible(voice) ? (
                <span className="tag subtle">{shortModel(voice.baseModelId!)}</span>
              ) : (
                <span className="muted">
                  {otherEngine && <span className="tag subtle">{engineName(voice.engine)}</span>}{' '}
                  {[
                    best.lora.rank !== null && t.rank(best.lora.rank),
                    best.step !== null && t.stepsCount(formatStep(best.step)),
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </span>
              )}
            </button>
          )
        })}

        {!loading && voices.length === 0 && <p className="muted empty">{t.noLoras}</p>}
        {voices.length > 0 && shown.length === 0 && <p className="muted empty">{t.noVoiceMatches(filter)}</p>}
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
