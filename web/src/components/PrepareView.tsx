import { useCallback, useEffect, useState } from 'react'
import { listDatasets } from '../api'
import type { ImportJob } from '../api'
import { datasetName } from '../format'
import { useT } from '../i18n'
import { DatasetPage } from './DatasetPage'
import { ImportForm } from './ImportForm'

interface Props {
  onError: (message: string) => void
  // Opens the Train tab with this dataset selected for a new run.
  onTrain: (datasetId: string) => void
}

const IMPORT = '\0import'

/** Raw recordings and lists in, checked datasets out: import (slice, transcribe, align), then review. */
export function PrepareView({ onError, onTrain }: Props) {
  const t = useT()
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [selected, setSelected] = useState(IMPORT)
  const [imported, setImported] = useState<ImportJob['result'] | null>(null)
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

  async function done(result: ImportJob['result']) {
    await refresh()
    setImported(result)
    setSelected(result.id)
    setFormKey((k) => k + 1) // a fresh form for the next import
  }

  return (
    <main className="layout">
      <div className="main-col">
        {/* Stays mounted while a dataset is open, so a running import keeps its progress. */}
        <div hidden={selected !== IMPORT}>
          <ImportForm key={formKey} datasetNames={(datasets ?? []).map(datasetName)} onImported={done} />
        </div>
        {selected !== IMPORT && (
          <DatasetPage
            key={selected}
            id={selected}
            imported={imported?.id === selected ? imported : null}
            onTrain={onTrain}
          />
        )}
      </div>

      <aside className="side-col">
        <section className="panel voice-panel">
          <header className="panel-head">
            <h2>{t.datasets}</h2>
            <button
              type="button"
              className={`small ${selected === IMPORT ? 'primary' : 'ghost'}`}
              onClick={() => setSelected(IMPORT)}
            >
              {t.import}
            </button>
          </header>
          <div className="voice-list">
            {(datasets ?? []).map((id) => (
              <button
                key={id}
                type="button"
                className={`voice-row ${id === selected ? 'active' : ''}`}
                onClick={() => setSelected(id)}
                aria-current={id === selected}
              >
                <span className="voice-row-name">{datasetName(id)}</span>
              </button>
            ))}
            {datasets?.length === 0 && <p className="muted empty">{t.noDatasets}</p>}
          </div>
        </section>
      </aside>
    </main>
  )
}
