import { useState } from 'react'
import type { CreateJob } from '../api'
import { datasetName, formatTimings } from '../format'
import { useT } from '../i18n'
import { DatasetReview } from './DatasetReview'
import { DatasetSteps } from './DatasetSteps'
import { ExportTool } from './ExportTool'

interface Props {
  id: string
  // The import that just made this dataset, if any.
  imported: CreateJob['result'] | null
  onTrain: (id: string) => void
}

/** A dataset: its processing steps, review, then train. */
export function DatasetPage({ id, imported, onTrain }: Props) {
  const t = useT()
  // Bumped when a step finishes, so the review reloads.
  const [version, setVersion] = useState(0)

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h2 className="run-title">{datasetName(id)}</h2>
          <div className="run-actions">
            <button type="button" className="primary" onClick={() => onTrain(id)}>
              {t.trainOnIt}
            </button>
          </div>
        </header>
        {imported && (
          <p className="notice">
            {t.importedNew(imported.clips, imported.recordings, Object.keys(imported.steps).length > 0)}
            {imported.timings && Object.keys(imported.timings).length > 0 && (
              <span className="muted"> {formatTimings(imported.timings)}.</span>
            )}
          </p>
        )}
        <DatasetSteps id={id} onChanged={() => setVersion((v) => v + 1)} />
        <div className="tools">
          <ExportTool id={id} disabled={false} />
        </div>
      </section>

      <section className="panel">
        <DatasetReview key={version} source={{ id }} />
      </section>
    </>
  )
}
