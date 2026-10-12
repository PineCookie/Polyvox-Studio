import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { listDatasets, newDataset } from '../api'
import { datasetName } from '../format'
import { useT } from '../i18n'
import { DatasetPage } from './DatasetPage'

interface Props {
  onError: (message: string) => void
  // Opens the Train tab with this dataset selected for a new run.
  onTrain: (datasetId: string) => void
}

// Matches importer.DATASET_NAME.
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

/** Datasets: pick one, or start a new one; each dataset's page adds sources and runs its steps. */
export function PrepareView({ onError, onTrain }: Props) {
  const t = useT()
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [created, setCreated] = useState<string | null>(null) // just made: Add source opens
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)

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

  const names = (datasets ?? []).map(datasetName)
  const nameError = !name
    ? null
    : !NAME_PATTERN.test(name)
      ? t.nameChars
      : names.includes(name)
        ? t.datasetExists
        : null

  async function create(e: FormEvent) {
    e.preventDefault()
    setError(null)
    try {
      const id = await newDataset(name)
      await refresh()
      setCreated(id)
      setSelected(id)
      setNaming(false)
      setName('')
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <main className="layout">
      <div className="main-col">
        {selected ? (
          <DatasetPage
            key={selected}
            id={selected}
            isNew={selected === created}
            onTrain={onTrain}
            onDeleted={() => {
              setSelected(null)
              refresh()
            }}
          />
        ) : (
          <section className="panel">
            <p className="muted empty">{datasets?.length ? t.pickDataset : t.noDatasets}</p>
          </section>
        )}
      </div>

      <aside className="side-col">
        <section className="panel voice-panel">
          <header className="panel-head">
            <h2>{t.datasets}</h2>
            <button
              type="button"
              className={`small ${naming ? 'primary' : 'ghost'}`}
              onClick={() => setNaming((n) => !n)}
            >
              {t.newDataset}
            </button>
          </header>
          {naming && (
            <form className="new-dataset" onSubmit={create}>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t.datasetName}
                aria-label={t.datasetName}
                autoFocus
              />
              <button type="submit" className="primary small" disabled={!name || !!nameError}>
                {t.create}
              </button>
              {(nameError || error) && <span className="field-error">{nameError ?? error}</span>}
            </form>
          )}
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
          </div>
        </section>
      </aside>
    </main>
  )
}
