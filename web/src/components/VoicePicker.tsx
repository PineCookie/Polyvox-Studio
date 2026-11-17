import { useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import type { Engine, Lora } from '../api'
import type { VoiceChoice } from '../engines'
import { useFolded } from '../folded'
import { formatStep } from '../format'
import { useT } from '../i18n'
import { groupVoices, incompatible, shortModel, unknownEngine, voiceHue } from '../voices'
import type { Voice } from '../voices'
import { Chevron } from './Fold'
import { GptSovitsImportDialog } from './GptSovitsImportDialog'
import { ChevronDownIcon, PlusIcon, RefreshIcon, SearchIcon } from './icons'
import { Popover } from './Popover'

interface Props {
  engines: Engine[]
  loras: Lora[]
  selected: VoiceChoice
  // Hub id of the VoxCPM model the server runs; VoxCPM LoRAs for another base model can't load.
  modelId: string | null
  onSelect: (choice: VoiceChoice) => void
  onRefresh: () => void
  onImport: (file: File) => Promise<void>
  onImported: (id: string, engine: string) => void // a voice made here (GPT-SoVITS weights)
  onError: (message: string) => void
  loading: boolean
}

/** A voice's round initial, coloured from its name. */
export function VoiceAvatar({ name, base = false }: { name: string; base?: boolean }) {
  return (
    <span className={`voice-avatar${base ? ' base' : ''}`} style={{ '--hue': voiceHue(name) } as CSSProperties}>
      {base ? '' : [...name][0]?.toUpperCase()}
    </span>
  )
}

/** The voice Synthesize speaks with: a button showing it, opening the searchable list of base
 * models and voices (by model), with the import actions at its foot. */
export function VoicePicker({
  engines,
  loras,
  selected,
  modelId,
  onSelect,
  onRefresh,
  onImport,
  onImported,
  onError,
  loading,
}: Props) {
  const t = useT()
  const voices = useMemo(() => groupVoices(loras, t), [loras, t])
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [importing, setImporting] = useState(false)
  const [importingGsv, setImportingGsv] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const [collapsed, toggleGroup] = useFolded('polyvox.voiceGroups') // engine ids whose group is folded

  const engineOf = (id: string) => engines.find((e) => e.id === id) ?? null
  const engineName = (id: string) => engineOf(id)?.name ?? id
  const keepsReference = (id: string) => !!engineOf(id)?.capabilities.voice_reference
  const current = loras.find((l) => l.id === selected.lora) ?? null
  const currentVoice = current ? voices.find((v) => v.key === current.voice_path) : null
  const shown = voices.filter((v) => v.name.toLowerCase().includes(filter.trim().toLowerCase()))

  // Voices grouped by model, in the server's engine order (VoxCPM first); checkpoints of an
  // engine this server doesn't know go last.
  const groups: { id: string; name: string; engine: Engine | null; voices: Voice[] }[] =
    engines.length === 0
      ? [{ id: 'all', name: '', engine: null, voices: shown }]
      : [
          ...engines.map((e) => ({
            id: e.id,
            name: e.name,
            engine: e,
            voices: shown.filter((v) => v.engine === e.id),
          })),
          { id: '?', name: t.otherModels, engine: null, voices: shown.filter((v) => unknownEngine(v, engines)) },
        ].filter((g) => (filter ? g.voices.length > 0 : g.engine || g.voices.length > 0))

  function choose(choice: VoiceChoice) {
    onSelect(choice)
    setOpen(false)
    setFilter('')
  }

  function baseRow(engine: Engine) {
    const active = selected.lora === null && selected.engine === engine.id
    return (
      <button
        key={`base:${engine.id}`}
        type="button"
        role="option"
        aria-selected={active}
        className={`voice-row ${active ? 'active' : ''}`}
        onClick={() => choose({ engine: engine.id, lora: null })}
        disabled={!!engine.unavailable}
        title={engine.unavailable ? t.engineUnavailable(engine.name, engine.unavailable) : undefined}
      >
        <VoiceAvatar name={engine.name} base />
        <span className="voice-row-name">{t.baseModel}</span>
        {engine.unavailable && <span className="muted">{t.unavailableShort}</span>}
      </button>
    )
  }

  function voiceRow(voice: Voice) {
    const active = voice === currentVoice
    const best = voice.checkpoints[0]
    const blocked = incompatible(voice, modelId) || unknownEngine(voice, engines)
    return (
      <button
        key={voice.key}
        type="button"
        role="option"
        aria-selected={active}
        className={`voice-row ${active ? 'active' : ''}`}
        onClick={() => choose({ engine: best.lora.engine, lora: best.lora.id })}
        disabled={blocked}
        title={incompatible(voice, modelId) ? t.trainedOn(voice.baseModelId!, modelId!) : best.lora.id}
      >
        <VoiceAvatar name={voice.name} />
        <span className="voice-row-name">{voice.name}</span>
        {incompatible(voice, modelId) ? (
          <span className="tag subtle">{shortModel(voice.baseModelId!)}</span>
        ) : (
          <span className="muted">
            {unknownEngine(voice, engines) && <span className="tag subtle">{voice.engine}</span>}{' '}
            {[
              best.lora.version,
              best.step !== null && t.stepsCount(formatStep(best.step)),
              keepsReference(best.lora.engine) && best.lora.reference_text === null && t.noReference,
            ]
              .filter(Boolean)
              .join(' · ')}
          </span>
        )}
      </button>
    )
  }

  async function importFile(file: File | undefined) {
    if (!file) return
    setImporting(true)
    setOpen(false)
    try {
      await onImport(file)
    } finally {
      setImporting(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const name = currentVoice?.name ?? t.baseModel
  const sub = current
    ? [engineName(current.engine), current.step !== null && t.stepLabel(formatStep(current.step))]
        .filter(Boolean)
        .join(' · ')
    : engineName(selected.engine)

  return (
    <>
      <Popover
        open={open}
        onClose={() => setOpen(false)}
        align="start"
        className="voice-popover"
        anchor={
          <button
            type="button"
            className="voice-picker"
            aria-haspopup="listbox"
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
            title={t.chooseVoice}
          >
            <VoiceAvatar name={currentVoice ? name : engineName(selected.engine)} base={!currentVoice} />
            <span className="voice-picker-text">
              <span className="voice-picker-name">{name}</span>
              <span className="voice-picker-sub">{sub}</span>
            </span>
            <ChevronDownIcon />
          </button>
        }
      >
        <label className="voice-search">
          <SearchIcon />
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={t.filterVoices(voices.length)}
            aria-label={t.filterVoicesLabel}
            autoFocus
          />
        </label>
        <div className="voice-list" role="listbox" aria-label={t.voice}>
          {groups.map((group) => {
            // While filtering every group is open and shows only matches (no base rows).
            const isOpen = !!filter || !collapsed.has(group.id)
            const baseActive = !!group.engine && selected.lora === null && selected.engine === group.engine.id
            return (
              <div key={group.id} className="voice-group">
                {group.name && (
                  <button
                    type="button"
                    className="voice-group-head"
                    aria-expanded={isOpen}
                    onClick={() => !filter && toggleGroup(group.id)}
                  >
                    <Chevron open={isOpen} />
                    <span>{group.name}</span>
                    <span className="muted">{group.voices.length}</span>
                  </button>
                )}
                {group.engine && !filter && (isOpen || baseActive) && baseRow(group.engine)}
                {group.voices.filter((v) => isOpen || v === currentVoice).map(voiceRow)}
              </div>
            )
          })}
          {!loading && voices.length === 0 && <p className="muted empty">{t.noLoras}</p>}
          {voices.length > 0 && shown.length === 0 && <p className="muted empty">{t.noVoiceMatches(filter)}</p>}
        </div>
        <div className="voice-popover-foot">
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
            <PlusIcon /> {importing ? t.importing : t.importPackage}
          </button>
          <button
            type="button"
            className="ghost small"
            onClick={() => {
              setOpen(false)
              setImportingGsv(true)
            }}
            title={t.gsvImportButtonTitle}
          >
            <PlusIcon /> {t.gsvImportButton}
          </button>
          <span className="spacer" />
          <button
            type="button"
            className="icon-button"
            onClick={onRefresh}
            disabled={loading}
            title={loading ? t.scanning : t.refresh}
            aria-label={t.refresh}
          >
            <RefreshIcon />
          </button>
        </div>
      </Popover>
      {importingGsv && (
        <GptSovitsImportDialog onImported={onImported} onError={onError} onClose={() => setImportingGsv(false)} />
      )}
    </>
  )
}
