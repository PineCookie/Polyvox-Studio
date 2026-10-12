import { useState } from 'react'
import { datasetName } from '../format'
import { useT } from '../i18n'
import { DatasetReview } from './DatasetReview'
import { DatasetSteps } from './DatasetSteps'
import { ExportTool } from './ExportTool'

interface Props {
  id: string
  // Just created: open Add source at once.
  isNew: boolean
  onTrain: (id: string) => void
}

/** A dataset: its processing steps, review, then train. */
export function DatasetPage({ id, isNew, onTrain }: Props) {
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
        <DatasetSteps id={id} startAdding={isNew} onChanged={() => setVersion((v) => v + 1)} />
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
