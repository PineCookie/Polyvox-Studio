import { useCallback, useEffect, useRef, useState } from 'react'
import { getStatus, listLoras, synthesize, unloadModel } from './api'
import type { Lora, Status, SynthesisParams } from './api'
import { History } from './components/History'
import type { Clip } from './components/History'
import { StatusBar } from './components/StatusBar'
import { SynthesisPanel } from './components/SynthesisPanel'
import { VoicePanel } from './components/VoicePanel'

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [loras, setLoras] = useState<Lora[]>([])
  const [lorasLoading, setLorasLoading] = useState(true)
  const [selectedLora, setSelectedLora] = useState<string | null>(null)
  const [seed, setSeed] = useState<number | null>(null)
  const [clips, setClips] = useState<Clip[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const nextClipId = useRef(1)

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

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          VoxCPM <span>Studio</span>
        </h1>
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

      <main className="layout">
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
            loading={lorasLoading}
          />
        </aside>
      </main>
    </div>
  )
}
