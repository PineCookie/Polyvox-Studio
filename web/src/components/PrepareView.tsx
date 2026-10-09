import { useCallback, useEffect, useState } from 'react'
import { listDatasets } from '../api'
import type { ImportJob } from '../api'
import { datasetName } from '../format'
import { ImportForm } from './ImportForm'

interface Props {
  onError: (message: string) => void
  // Opens the Train tab with this dataset selected for review.
  onTrain: (datasetId: string) => void
}

/** Raw recordings and lists in, datasets out (import, slice, transcribe; later align and audit). */
export function PrepareView({ onError, onTrain }: Props) {
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [prepared, setPrepared] = useState<ImportJob['result'] | null>(null)
  const [formKey, setFormKey] = useState(0)

  const refresh = useCallback(async () => {
    try {
      setDatasets(await listDatasets())
    } catch (e) {
      onError((e as Error).message)
    }
  }, [onError])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function imported(result: ImportJob['result']) {
    setPrepared(result)
    await refresh()
  }

  return (
    <main className="layout">
      <div className="main-col">
        {prepared && (
          <section className="panel prepared">
            <p>
              <strong>{datasetName(prepared.id)}</strong> is ready: {prepared.imported.toLocaleString()} clips
              {prepared.sliced > 0 && `, sliced from ${prepared.sliced} recordings`}
              {prepared.transcribed > 0 && `, ${prepared.transcribed.toLocaleString()} transcribed by ASR`}.
            </p>
            <div className="run-actions">
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  setPrepared(null)
                  setFormKey((k) => k + 1)
                }}
              >
                Prepare another
              </button>
              <button type="button" className="primary" onClick={() => onTrain(prepared.id)}>
                Review and train
              </button>
            </div>
          </section>
        )}
        <ImportForm key={formKey} datasetNames={(datasets ?? []).map(datasetName)} onImported={imported} />
      </div>

      <aside className="side-col">
        <section className="panel voice-panel">
          <header className="panel-head">
            <h2>Datasets</h2>
          </header>
          <div className="voice-list">
            {(datasets ?? []).map((id) => (
              <button
                key={id}
                type="button"
                className="voice-row"
                onClick={() => onTrain(id)}
                title="Review in the Train tab"
              >
                <span className="voice-row-name">{datasetName(id)}</span>
                <span className="muted">review →</span>
              </button>
            ))}
            {datasets?.length === 0 && <p className="muted empty">None yet. Prepare one here.</p>}
          </div>
        </section>
      </aside>
    </main>
  )
}
