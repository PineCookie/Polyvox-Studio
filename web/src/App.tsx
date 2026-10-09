import { useCallback, useEffect, useRef, useState } from 'react'
import { getStatus, importLora, listLoras, synthesize, unloadModel } from './api'
import type { Lora, Status, SynthesisParams } from './api'
import { History } from './components/History'
import type { Clip } from './components/History'
import { PrepareView } from './components/PrepareView'
import { StatusBar } from './components/StatusBar'
import { SynthesisPanel } from './components/SynthesisPanel'
import { TrainingView } from './components/TrainingView'
import { VoicePanel } from './components/VoicePanel'

// Workflow order: raw audio -> dataset -> LoRA -> speech.
const TABS = { synthesize: 'Synthesize', prepare: 'Prepare', train: 'Train' } as const
type Tab = keyof typeof TABS

function tabFromHash(): Tab {
  const t = location.hash.slice(1)
  if (t === 'data') return 'prepare' // the tab's earlier name
  return t in TABS ? (t as Tab) : 'synthesize'
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash)
  // Set by the Data tab's "Train on this"; opens New run with that dataset.
  const [trainDataset, setTrainDataset] = useState<{ id: string; key: number } | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [loras, setLoras] = useState<Lora[]>([])
  const [lorasLoading, setLorasLoading] = useState(true)
  const [selectedLora, setSelectedLora] = useState<string | null>(null)
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
      .then((list) => {
        setLoras(list)
        // Drop a selection whose checkpoint disappeared.
        setSelectedLora((sel) => (sel && list.some((l) => l.id === sel) ? sel : null))
      }, (e: Error) => setError(e.message))
      .finally(() => setLorasLoading(false))
  }, [])

  useEffect(() => {
    refreshStatus()
    refreshLoras()
  }, [refreshStatus, refreshLoras])

  async function generate(params: Omit<SynthesisParams, 'loraId'>) {
    setBusy(true)
    setError(null)
    const started = performance.now()
    try {
      const out = await synthesize({ ...params, loraId: selectedLora })
      const clip: Clip = {
        id: nextClipId.current++,
        url: URL.createObjectURL(out.audio),
        text: params.text.trim(),
        voice: selectedLora ?? 'Base model',
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
      if (clip) URL.revokeObjectURL(clip.url)
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
      const id = await importLora(file)
      refreshLoras()
      setSelectedLora(id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function tryCheckpoint(loraId: string) {
    setSelectedLora(loraId)
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
          VoxCPM <span>Studio</span>
        </h1>
        <nav className="tabs" aria-label="Sections">
          {(Object.keys(TABS) as Tab[]).map((t) => (
            <button
              key={t}
              type="button"
              className={tab === t ? 'active' : ''}
              aria-current={tab === t ? 'page' : undefined}
              onClick={() => switchTab(t)}
            >
              {TABS[t]}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        <StatusBar status={status} onUnload={unload} />
      </header>

      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button type="button" className="ghost small" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {/* Both views stay mounted so form input and clips survive tab switches. */}
      <main className="layout" hidden={tab !== 'synthesize'}>
        <div className="main-col">
          <SynthesisPanel
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
            loras={loras}
            selected={selectedLora}
            modelId={status?.model_id ?? null}
            onSelect={setSelectedLora}
            onRefresh={refreshLoras}
            onImport={importPackage}
            onError={showError}
            loading={lorasLoading}
          />
        </aside>
      </main>
      {tab === 'prepare' && <PrepareView onError={showError} onTrain={trainOn} />}
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
