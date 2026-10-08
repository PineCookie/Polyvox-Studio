import { useCallback, useEffect, useState } from 'react'
import { inspectDataset, listDatasets } from '../api'
import type { DatasetReport } from '../api'
import { datasetName } from '../format'
import { DatasetSummary } from './DatasetSummary'
import { ImportForm } from './ImportForm'

interface Props {
  onError: (message: string) => void
  onTrain: (datasetId: string) => void
}

const IMPORT = '\0import'

export function DataView({ onError, onTrain }: Props) {
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [report, setReport] = useState<DatasetReport | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const list = await listDatasets()
      setDatasets(list)
      setSelected((sel) => sel ?? list[0] ?? IMPORT)
    } catch (e) {
      onError((e as Error).message)
    }
  }, [onError])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    setReport(null)
    setReportError(null)
    if (!selected || selected === IMPORT) return
    let cancelled = false
    inspectDataset(selected).then(
      (r) => !cancelled && setReport(r),
      (e: Error) => !cancelled && setReportError(e.message),
    )
    return () => {
      cancelled = true
    }
  }, [selected])

  async function imported(id: string) {
    await refresh()
    setSelected(id)
  }

  const trainable = !!report && report.valid > 0 && report.errors.length === 0

  return (
    <main className="layout">
      <div className="main-col">
        {selected === IMPORT ? (
          <ImportForm datasetNames={(datasets ?? []).map(datasetName)} onImported={imported} />
        ) : selected ? (
          <section className="panel">
            <header className="panel-head">
              <h2>{datasetName(selected)}</h2>
              <button type="button" className="primary" onClick={() => onTrain(selected)} disabled={!trainable}>
                Train on this
              </button>
            </header>
            <p className="muted hint path">data/{selected}</p>
            <DatasetSummary report={report} error={reportError} />
          </section>
        ) : null}
      </div>

      <aside className="side-col">
        <section className="panel voice-panel">
          <header className="panel-head">
            <h2>Datasets</h2>
            <button
              type="button"
              className={`small ${selected === IMPORT ? 'primary' : 'ghost'}`}
              onClick={() => setSelected(IMPORT)}
            >
              Import
            </button>
          </header>
          <div className="voice-list">
            {(datasets ?? []).map((id) => (
              <button
                key={id}
                type="button"
                className={`voice-row ${id === selected ? 'active' : ''}`}
                aria-current={id === selected}
                onClick={() => setSelected(id)}
              >
                <span className="voice-row-name">{datasetName(id)}</span>
              </button>
            ))}
            {datasets?.length === 0 && <p className="muted empty">No datasets yet. Import one.</p>}
          </div>
        </section>
      </aside>
    </main>
  )
}
