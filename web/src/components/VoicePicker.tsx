import { useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import type { Engine, Lora } from '../api'
import type { VoiceChoice } from '../engines'
import { importDialog } from '../importDialogs'
import { formatStep } from '../format'
import { useT } from '../i18n'
import { engineModel, groupVoices, incompatible, shortModel, unknownEngine, voiceHue } from '../voices'
import type { Voice } from '../voices'
import { CheckIcon, ChevronDownIcon, ChevronRightIcon, PlusIcon, RefreshIcon, SearchIcon } from './icons'
import { Popover } from './Popover'

const LAST_VOICE_KEY = 'polyvox.lastVoice' // {engine id: the LoRA / voice id last chosen with it}
const SEARCH_FROM = 7 // voices of a model from which their list gets a filter box

interface Props {
  engines: Engine[]
  loras: Lora[]
  selected: VoiceChoice
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

function lastVoices(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(LAST_VOICE_KEY) ?? '{}') ?? {}
  } catch {
    return {}
  }
}

function rememberVoice(engine: string, lora: string | null) {
  try {
    const last = lastVoices()
    if (lora) last[engine] = lora
    else delete last[engine]
    localStorage.setItem(LAST_VOICE_KEY, JSON.stringify(last))
  } catch {
    // Not remembered (storage blocked): choosing the model again starts from its base model.
  }
}

/** What Synthesize speaks with, chosen in two steps: the model, then its base model or one of its
 * voices (trained LoRAs, imported or saved voices). Choosing a model brings back the voice last
 * used with it. Import actions sit at the foot of the voice list. */
export function VoicePicker({
  engines,
  loras,
  selected,
  onSelect,
  onRefresh,
  onImport,
  onImported,
  onError,
  loading,
}: Props) {
  const t = useT()
  const voices = useMemo(() => groupVoices(loras, t), [loras, t])
  const [opened, setOpened] = useState<'model' | 'voice' | null>(null)
  const [filter, setFilter] = useState('')
  const [importing, setImporting] = useState(false)
  const [importingVoice, setImportingVoice] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const engine = engines.find((e) => e.id === selected.engine) ?? null
  const importer = importDialog(engine)
  const engineName = engine?.name ?? selected.engine
  const keepsReference = !!engine?.capabilities.voice_reference
  const usable = (v: Voice) => !incompatible(v, engines) && !unknownEngine(v, engines)
  const voicesOf = (id: string) => voices.filter((v) => v.engine === id)
  const own = voicesOf(selected.engine)
  const shown = own.filter((v) => v.name.toLowerCase().includes(filter.trim().toLowerCase()))
  const others = voices.filter((v) => unknownEngine(v, engines)).length // of models this server lacks
  const current = loras.find((l) => l.id === selected.lora) ?? null
  const currentVoice = current ? voices.find((v) => v.key === current.voice_path) : null

  function close() {
    setOpened(null)
    setFilter('')
  }

  function choose(choice: VoiceChoice) {
    rememberVoice(choice.engine, choice.lora)
    onSelect(choice)
    close()
  }

  /** A model: the voice last used with it if it is still there, else its base model. */
  function chooseModel(id: string) {
    if (id === selected.engine) return close()
    const last = lastVoices()[id]
    const voice = voicesOf(id).find((v) => usable(v) && v.checkpoints.some((c) => c.lora.id === last))
    onSelect({ engine: id, lora: voice ? last : null })
    close()
  }

  async function importFile(file: File | undefined) {
    if (!file) return
    setImporting(true)
    close()
    try {
      await onImport(file)
    } finally {
      setImporting(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  function modelRow(e: Engine) {
    const active = e.id === selected.engine
    const count = voicesOf(e.id).length
    return (
      <button
        key={e.id}
        type="button"
        role="option"
        aria-selected={active}
        className={`model-row${active ? ' active' : ''}`}
        onClick={() => chooseModel(e.id)}
        disabled={!!e.unavailable}
      >
        <span className="model-row-text">
          <span className="model-row-name">{e.name}</span>
          <span className="model-row-sub">
            {e.unavailable ? t.engineUnavailableReason(e.unavailable) : t.voicesCount(count)}
          </span>
        </span>
        {e.loaded && !e.unavailable && <span className="model-row-loaded" title={t.statusLoaded} />}
        {active && <CheckIcon />}
      </button>
    )
  }

  function baseRow() {
    const active = selected.lora === null
    return (
      <button
        type="button"
        role="option"
        aria-selected={active}
        className={`voice-row${active ? ' active' : ''}`}
        onClick={() => choose({ engine: selected.engine, lora: null })}
      >
        <VoiceAvatar name={engineName} base />
        <span className="voice-row-name">{t.baseModel}</span>
        <span className="muted">{t.baseModelSub}</span>
        {active && <CheckIcon />}
      </button>
    )
  }

  function voiceRow(voice: Voice) {
    const active = voice === currentVoice
    const best = voice.checkpoints[0]
    return (
      <button
        key={voice.key}
        type="button"
        role="option"
        aria-selected={active}
        className={`voice-row${active ? ' active' : ''}`}
        onClick={() => choose({ engine: best.lora.engine, lora: best.lora.id })}
        disabled={!usable(voice)}
        title={
          incompatible(voice, engines) ? t.trainedOn(voice.baseModelId!, engineModel(voice, engines)!) : best.lora.id
        }
      >
        <VoiceAvatar name={voice.name} />
        <span className="voice-row-name">{voice.name}</span>
        {incompatible(voice, engines) ? (
          <span className="tag subtle">{shortModel(voice.baseModelId!)}</span>
        ) : (
          <span className="muted">
            {[
              best.lora.version,
              best.step !== null && t.stepsCount(formatStep(best.step)),
              keepsReference && best.lora.reference_text === null && t.noReference,
            ]
              .filter(Boolean)
              .join(' · ')}
          </span>
        )}
        {active && <CheckIcon />}
      </button>
    )
  }

  const voiceName = currentVoice?.name ?? t.baseModel
  const voiceSub = current
    ? current.step !== null
      ? t.stepLabel(formatStep(current.step))
      : (current.version ?? t.voiceKind(!!engine?.capabilities.reference_voices))
    : t.baseModelSub

  return (
    <div className="voice-pickers">
      <Popover
        open={opened === 'model'}
        onClose={close}
        align="start"
        className="voice-popover model-popover"
        anchor={
          <button
            type="button"
            className="voice-picker model-picker"
            aria-haspopup="listbox"
            aria-expanded={opened === 'model'}
            onClick={() => (opened === 'model' ? close() : setOpened('model'))}
            title={t.chooseModel}
          >
            <span className="voice-picker-text">
              <span className="voice-picker-label">{t.modelLabel}</span>
              <span className="voice-picker-name">{engineName}</span>
            </span>
            <ChevronDownIcon />
          </button>
        }
      >
        <div className="voice-list" role="listbox" aria-label={t.modelLabel}>
          {engines.map(modelRow)}
          {others > 0 && <p className="muted model-others">{t.otherModelVoices(others)}</p>}
        </div>
      </Popover>
      <ChevronRightIcon />
      <Popover
        open={opened === 'voice'}
        onClose={close}
        align="start"
        className="voice-popover"
        anchor={
          <button
            type="button"
            className="voice-picker"
            aria-haspopup="listbox"
            aria-expanded={opened === 'voice'}
            onClick={() => (opened === 'voice' ? close() : setOpened('voice'))}
            title={t.chooseVoice}
          >
            <VoiceAvatar name={currentVoice ? voiceName : engineName} base={!currentVoice} />
            <span className="voice-picker-text">
              <span className="voice-picker-name">{voiceName}</span>
              <span className="voice-picker-sub">{voiceSub}</span>
            </span>
            <ChevronDownIcon />
          </button>
        }
      >
        <div className="voice-popover-head">{t.voicesOf(engineName)}</div>
        {own.length >= SEARCH_FROM && (
          <label className="voice-search">
            <SearchIcon />
            <input
              type="search"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder={t.filterVoices(own.length)}
              aria-label={t.filterVoicesLabel}
              autoFocus
            />
          </label>
        )}
        <div className="voice-list" role="listbox" aria-label={t.voice}>
          {!filter && baseRow()}
          {shown.map(voiceRow)}
          {!loading && own.length === 0 && <p className="muted empty">{t.noVoicesYet}</p>}
          {own.length > 0 && filter && shown.length === 0 && <p className="muted empty">{t.noVoiceMatches(filter)}</p>}
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
          {importer && (
            <button
              type="button"
              className="ghost small"
              onClick={() => {
                close()
                setImportingVoice(true)
              }}
              title={importer.title?.(t)}
            >
              <PlusIcon /> {importer.label(t)}
            </button>
          )}
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
      {importingVoice && importer && (
        <importer.Dialog onImported={onImported} onError={onError} onClose={() => setImportingVoice(false)} />
      )}
    </div>
  )
}
