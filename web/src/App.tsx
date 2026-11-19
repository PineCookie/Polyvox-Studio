import { useCallback, useEffect, useRef, useState } from 'react'
import {
  DEFAULT_ENGINE,
  editSpeech,
  getStatus,
  importLora,
  listEngines,
  listLoras,
  synthesize,
  unloadModel,
} from './api'
import type { Engine, Lora, Status, SynthesisParams } from './api'
import { BASE_VOICE, engineName } from './engines'
import { formatStep } from './format'
import type { VoiceChoice } from './engines'
import { forgetPeaks } from './peaks'
import { History } from './components/History'
import { ConvertView } from './components/ConvertView'
import { ModelsView } from './components/ModelsView'
import type { Clip } from './components/History'
import { PrepareView } from './components/PrepareView'
import { StatusBar } from './components/StatusBar'
import { ThemeToggle } from './components/ThemeToggle'
import { LanguageToggle } from './components/LanguageToggle'
import { useT } from './i18n'
import { SynthesisPanel } from './components/SynthesisPanel'
import { TrainingView } from './components/TrainingView'
import { VoiceDetails } from './components/VoiceDetails'
import { VoicePicker } from './components/VoicePicker'

// Speech first (text, then recordings), then the workflow behind voices: raw audio -> dataset -> training.
const TABS = ['synthesize', 'convert', 'prepare', 'train', 'models'] as const
type Tab = (typeof TABS)[number]
const UPDATE_LOOK_MS = 8000 // when to look again for the server's update check
const DISMISSED_UPDATE_KEY = 'polyvox.dismissedUpdate' // the newer version whose notice was dismissed

function tabFromHash(): Tab {
  const t = location.hash.slice(1)
  if (t === 'data') return 'prepare' // the tab's earlier name
  return (TABS as readonly string[]).includes(t) ? (t as Tab) : 'synthesize'
}

export default function App() {
  const t = useT()
  const [tab, setTab] = useState<Tab>(tabFromHash)
  // Set by the Data tab's "Train on this"; opens New run with that dataset.
  const [trainDataset, setTrainDataset] = useState<{ id: string; key: number } | null>(null)
  // Set by the Train tab's "Try" on a conversion engine's checkpoint (RVC): the Convert tab selects it.
  const [convertPick, setConvertPick] = useState<{ id: string; key: number } | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [loras, setLoras] = useState<Lora[]>([])
  const [lorasLoading, setLorasLoading] = useState(true)
  const [engines, setEngines] = useState<Engine[]>([])
  const [voice, setVoice] = useState<VoiceChoice>(BASE_VOICE)
  const [seed, setSeed] = useState<number | null>(null)
  const [clips, setClips] = useState<Clip[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const nextClipId = useRef(1)
  const showError = useCallback((message: string) => setError(message), [])

  const refreshStatus = useCallback(() => {
    getStatus().then(setStatus, (e: Error) => setError(e.message))
  }, [])

  const refreshLoras = useCallback(() => {
    setLorasLoading(true)
    listLoras()
      .then(
        (list) => {
          setLoras(list)
          // A checkpoint that disappeared leaves its engine's base model selected.
          setVoice((v) => (v.lora && !list.some((l) => l.id === v.lora) ? { engine: v.engine, lora: null } : v))
        },
        (e: Error) => setError(e.message),
      )
      .finally(() => setLorasLoading(false))
  }, [])

  useEffect(() => {
    refreshStatus()
    refreshLoras()
    listEngines().then(setEngines, (e: Error) => setError(e.message))
    // The server asks GitHub for a newer release in the background at startup: look again soon.
    const later = setTimeout(refreshStatus, UPDATE_LOOK_MS)
    return () => clearTimeout(later)
  }, [refreshStatus, refreshLoras])

  const [dismissedUpdate, setDismissedUpdate] = useState<string | null>(() => {
    try {
      return localStorage.getItem(DISMISSED_UPDATE_KEY)
    } catch {
      return null
    }
  })
  function dismissUpdate(version: string) {
    setDismissedUpdate(version) // until a newer one is out
    try {
      localStorage.setItem(DISMISSED_UPDATE_KEY, version)
    } catch {
      // Hidden for this session only.
    }
  }

  // How a take names the voice it was spoken with: its name (and checkpoint), or the base model.
  function voiceLabel(): string {
    const lora = loras.find((l) => l.id === voice.lora)
    if (!lora) return `${t.baseModel} · ${engineName(voice.engine, engines)}`
    return lora.step !== null ? `${lora.voice} · ${t.stepLabel(formatStep(lora.step))}` : lora.voice
  }

  async function generate(params: Omit<SynthesisParams, 'loraId' | 'engine'>) {
    setBusy(true)
    setError(null)
    const started = performance.now()
    try {
      const out = await synthesize({ ...params, engine: voice.engine, loraId: voice.lora })
      const clip: Clip = {
        id: nextClipId.current++,
        url: URL.createObjectURL(out.audio),
        text: params.text.trim(),
        voice: voiceLabel(),
        seed: out.seed,
        elapsedMs: performance.now() - started,
        createdAt: new Date(),
      }
      setClips((prev) => [clip, ...prev])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
      refreshStatus()
    }
  }

  // Engines that convert recordings (RVC) have their own tab; Synthesize lists the others and their voices.
  const converter = engines.find((e) => e.capabilities.conversion) ?? null
  const ttsEngines = engines.filter((e) => !e.capabilities.conversion)
  const ttsLoras = loras.filter((l) => !engines.some((e) => e.id === l.engine && e.capabilities.conversion))

  // The engine that edits recorded speech (any take: it keeps the voice from the audio itself).
  const editor = engines.find((e) => e.capabilities.speech_edit && !e.unavailable) ?? null

  async function editClip(clip: Clip, newText: string): Promise<boolean> {
    if (!editor) return false
    setBusy(true)
    setError(null)
    const started = performance.now()
    try {
      const audio = await (await fetch(clip.url)).blob()
      const out = await editSpeech(editor.id, audio, clip.text, newText)
      const edited: Clip = {
        id: nextClipId.current++,
        url: URL.createObjectURL(out.audio),
        text: newText,
        voice: t.editedVoice(clip.voice),
        seed: out.seed,
        elapsedMs: performance.now() - started,
        createdAt: new Date(),
        edited: out.span ?? undefined,
      }
      setClips((prev) => [edited, ...prev])
      return true
    } catch (e) {
      setError((e as Error).message)
      return false
    } finally {
      setBusy(false)
      refreshStatus()
    }
  }

  function removeClip(id: number) {
    setClips((prev) => {
      const clip = prev.find((c) => c.id === id)
      if (clip) {
        URL.revokeObjectURL(clip.url)
        forgetPeaks(clip.url)
      }
      return prev.filter((c) => c.id !== id)
    })
  }

  async function unload() {
    try {
      setStatus(await unloadModel())
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function switchTab(next: Tab) {
    setTab(next)
    setTrainDataset(null)
    history.replaceState(null, '', next === 'synthesize' ? location.pathname : `#${next}`)
    // Training writes checkpoints and may unload the inference model.
    if (next === 'synthesize' || next === 'convert') {
      refreshLoras()
      refreshStatus()
    }
  }

  async function importPackage(file: File) {
    try {
      const { id, engine } = await importLora(file)
      refreshLoras()
      setVoice({ engine, lora: id })
    } catch (e) {
      setError((e as Error).message)
    }
  }

  // A voice made here (GPT-SoVITS weights imported, a reference saved): list it and speak with it.
  function voiceMade(id: string, engine: string) {
    refreshLoras()
    setVoice({ engine, lora: id })
  }

  function tryCheckpoint(loraId: string) {
    const engine = loras.find((l) => l.id === loraId)?.engine ?? DEFAULT_ENGINE
    if (engines.some((e) => e.id === engine && e.capabilities.conversion)) {
      setConvertPick({ id: loraId, key: Date.now() })
      switchTab('convert')
      return
    }
    setVoice({ engine, lora: loraId })
    switchTab('synthesize')
  }

  function trainOn(datasetId: string) {
    switchTab('train')
    setTrainDataset({ id: datasetId, key: Date.now() })
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          Polyvox <span>Studio</span>
        </h1>
        <nav className="tabs" aria-label={t.sections}>
          {TABS.map((name) => (
            <button
              key={name}
              type="button"
              className={tab === name ? 'active' : ''}
              aria-current={tab === name ? 'page' : undefined}
              onClick={() => switchTab(name)}
            >
              {t.tabs[name]}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        <StatusBar status={status} onUnload={unload} />
        <LanguageToggle />
        <ThemeToggle />
      </header>

      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button type="button" className="ghost small" onClick={() => setError(null)}>
            {t.dismiss}
          </button>
        </div>
      )}

      {tab !== 'models' && status?.no_speech_model && (
        <div className="notice">
          <span>{t.noSpeechModel}</span>
          <button type="button" className="ghost small" onClick={() => switchTab('models')}>
            {t.openModels}
          </button>
        </div>
      )}

      {status?.update && dismissedUpdate !== status.update.version && (
        <div className="notice">
          <span>{t.updateAvailable(status.update.version)}</span>
          {status.update.url && (
            <a className="button ghost small" href={status.update.url} target="_blank" rel="noreferrer">
              {t.releaseNotes}
            </a>
          )}
          <button type="button" className="ghost small" onClick={() => dismissUpdate(status.update!.version)}>
            {t.dismiss}
          </button>
        </div>
      )}

      {/* Both views stay mounted so form input and clips survive tab switches. */}
      <main hidden={tab !== 'synthesize'}>
        <SynthesisPanel
          engine={engines.find((e) => e.id === voice.engine) ?? null}
          busy={busy}
          seed={seed}
          onSeedChange={setSeed}
          onGenerate={generate}
          onError={showError}
          onTranscribed={refreshStatus}
          voiceReference={loras.find((l) => l.id === voice.lora)?.reference_text ?? null}
          isBase={!voice.lora}
          voiceVersion={loras.find((l) => l.id === voice.lora)?.version ?? null}
          voicePicker={
            <VoicePicker
              engines={ttsEngines}
              loras={ttsLoras}
              selected={voice}
              modelId={status?.model_id ?? null}
              onSelect={setVoice}
              onRefresh={refreshLoras}
              onImport={importPackage}
              onImported={voiceMade}
              onError={showError}
              loading={lorasLoading}
            />
          }
          voiceDetails={
            <VoiceDetails
              engines={ttsEngines}
              loras={ttsLoras}
              selected={voice}
              modelId={status?.model_id ?? null}
              onSelect={setVoice}
              onRefresh={refreshLoras}
              onImported={voiceMade}
              onError={showError}
            />
          }
        >
          <History
            clips={clips}
            onUseSeed={setSeed}
            onRemove={removeClip}
            editor={editor ? editor.name : null}
            busy={busy}
            onEdit={editClip}
          />
        </SynthesisPanel>
      </main>
      <ConvertView
        hidden={tab !== 'convert'}
        pick={convertPick}
        engine={converter}
        loras={loras}
        status={status}
        onVoicesChanged={refreshLoras}
        onError={showError}
        onOpenModels={() => switchTab('models')}
      />
      {tab === 'prepare' && <PrepareView onError={showError} onTrain={trainOn} />}
      {tab === 'models' && <ModelsView onError={showError} onChanged={refreshStatus} />}
      {tab === 'train' && (
        <TrainingView
          key={trainDataset?.key}
          loras={loras}
          onError={showError}
          onTryCheckpoint={tryCheckpoint}
          newRunDataset={trainDataset?.id}
        />
      )}
    </div>
  )
}
