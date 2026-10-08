import { useCallback, useEffect, useRef, useState } from 'react'
import { getStatus, importLora, listLoras, synthesize, unloadModel } from './api'
import type { Lora, Status, SynthesisParams } from './api'
import { History } from './components/History'
import type { Clip } from './components/History'
import { StatusBar } from './components/StatusBar'
import { SynthesisPanel } from './components/SynthesisPanel'
import { TrainingView } from './components/TrainingView'
import { VoicePanel } from './components/VoicePanel'

type Tab = 'synthesize' | 'train'

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash === '#train' ? 'train' : 'synthesize'))
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
    history.replaceState(null, '', next === 'train' ? '#train' : location.pathname)
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

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          VoxCPM <span>Studio</span>
        </h1>
        <nav className="tabs" aria-label="Sections">
          {(['synthesize', 'train'] as const).map((t) => (
            <button
              key={t}
              type="button"
              className={tab === t ? 'active' : ''}
              aria-current={tab === t ? 'page' : undefined}
              onClick={() => switchTab(t)}
            >
              {t === 'synthesize' ? 'Synthesize' : 'Train'}
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
          <SynthesisPanel busy={busy} seed={seed} onSeedChange={setSeed} onGenerate={generate} />
          <History clips={clips} onUseSeed={setSeed} onRemove={removeClip} />
        </div>
        <aside className="side-col">
          <VoicePanel
            loras={loras}
            selected={selectedLora}
            onSelect={setSelectedLora}
            onRefresh={refreshLoras}
            onImport={importPackage}
            onError={showError}
            loading={lorasLoading}
          />
        </aside>
      </main>
      {tab === 'train' && <TrainingView loras={loras} onError={showError} onTryCheckpoint={tryCheckpoint} />}
    </div>
  )
}
