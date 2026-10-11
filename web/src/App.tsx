import { useCallback, useEffect, useRef, useState } from 'react'
import { DEFAULT_ENGINE, getStatus, importLora, listEngines, listLoras, synthesize, unloadModel } from './api'
import type { Engine, Lora, Status, SynthesisParams } from './api'
import { BASE_VOICE, engineName } from './engines'
import type { VoiceChoice } from './engines'
import { forgetPeaks } from './peaks'
import { History } from './components/History'
import { ModelsView } from './components/ModelsView'
import type { Clip } from './components/History'
import { PrepareView } from './components/PrepareView'
import { StatusBar } from './components/StatusBar'
import { ThemeToggle } from './components/ThemeToggle'
import { LanguageToggle } from './components/LanguageToggle'
import { useT } from './i18n'
import { SynthesisPanel } from './components/SynthesisPanel'
import { TrainingView } from './components/TrainingView'
import { VoicePanel } from './components/VoicePanel'

// Workflow order: raw audio -> dataset -> LoRA -> speech.
const TABS = ['synthesize', 'prepare', 'train', 'models'] as const
type Tab = (typeof TABS)[number]

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
  }, [refreshStatus, refreshLoras])

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
        voice: voice.lora ?? `${t.baseModel} · ${engineName(voice.engine, engines)}`,
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
    if (next === 'synthesize') {
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

  function tryCheckpoint(loraId: string) {
    setVoice({ engine: loras.find((l) => l.id === loraId)?.engine ?? DEFAULT_ENGINE, lora: loraId })
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

      {tab !== 'models' && (status?.missing_required.length ?? 0) > 0 && (
        <div className="notice">
          <span>{t.missingModels(status!.missing_required.join(', '))}</span>
          <button type="button" className="ghost small" onClick={() => switchTab('models')}>
            {t.openModels}
          </button>
        </div>
      )}

      {/* Both views stay mounted so form input and clips survive tab switches. */}
      <main className="layout" hidden={tab !== 'synthesize'}>
        <div className="main-col">
          <SynthesisPanel
            engine={engines.find((e) => e.id === voice.engine) ?? null}
            busy={busy}
            seed={seed}
            onSeedChange={setSeed}
            onGenerate={generate}
            onError={showError}
            onTranscribed={refreshStatus}
          />
          <History clips={clips} onUseSeed={setSeed} onRemove={removeClip} />
        </div>
        <aside className="side-col">
          <VoicePanel
            engines={engines}
            loras={loras}
            selected={voice}
            modelId={status?.model_id ?? null}
            onSelect={setVoice}
            onRefresh={refreshLoras}
            onImport={importPackage}
            onError={showError}
            loading={lorasLoading}
          />
        </aside>
      </main>
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
