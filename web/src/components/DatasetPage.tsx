import { useState } from 'react'
import { checkDataset, tidyFillers } from '../api'
import type { Aligner, CheckJob, ImportJob } from '../api'
import { ALIGNER_LABEL, datasetName } from '../format'
import { useJob } from '../useJob'
import { DatasetReview } from './DatasetReview'
import { JobProgress } from './JobProgress'

interface Props {
  id: string
  // The import that just made this dataset, if any.
  imported: ImportJob['result'] | null
  onTrain: (id: string) => void
}

/** A prepared dataset: check its clips against their text, review, then train. */
export function DatasetPage({ id, imported, onTrain }: Props) {
  const [aligner, setAligner] = useState<Aligner>('qwen3')
  const [error, setError] = useState<string | null>(null)
  // Bumped when a check finishes, so the review reloads the results.
  const [version, setVersion] = useState(0)
  const [job, setJob] = useJob<CheckJob>((j) => {
    if (j.state === 'done') setVersion((v) => v + 1)
    else if (j.state === 'failed') setError(j.error ?? 'Check failed.')
  })
  const running = job?.state === 'running'
  const [tidied, setTidied] = useState<number | null>(null) // clips the last tidy changed

  async function check() {
    setError(null)
    try {
      setJob(await checkDataset(id, aligner))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function tidy() {
    setError(null)
    try {
      const { changed } = await tidyFillers(id)
      setTidied(changed)
      if (changed > 0) setVersion((v) => v + 1)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h2 className="run-title">{datasetName(id)}</h2>
          <div className="run-actions">
            <button type="button" className="primary" onClick={() => onTrain(id)}>
              Train on it
            </button>
          </div>
        </header>
        {imported && (
          <p className="notice">
            Imported {imported.imported.toLocaleString()} clips
            {imported.sliced > 0 && `, sliced from ${imported.sliced} recordings`}
            {imported.transcribed > 0 && `; ${imported.transcribed.toLocaleString()} transcribed by ASR`}
            {imported.trimmed > 0 && `; ${imported.trimmed.toLocaleString()} trimmed`}.
          </p>
        )}
        <div className="inline-field check-action">
          <button type="button" className="ghost" onClick={check} disabled={running}>
            {running ? 'Checking…' : 'Check clips'}
          </button>
          <span className="muted hint">against their text with</span>
          <select
            value={aligner}
            onChange={(e) => setAligner(e.target.value as Aligner)}
            disabled={running}
            aria-label="Aligner"
          >
            {(Object.keys(ALIGNER_LABEL) as Aligner[]).map((a) => (
              <option key={a} value={a}>
                {ALIGNER_LABEL[a]}
              </option>
            ))}
          </select>
        </div>
        <p className="muted hint">
          Forced alignment finds text that does not match the audio, untranscribed speech, words cut at the file edge
          and long silence. Qwen3 also handles mixed languages; wav2vec2 is sharper on single wrong words (Chinese model
          by default).
        </p>
        <div className="inline-field check-action">
          <button
            type="button"
            className="ghost"
            onClick={tidy}
            disabled={running}
            title="Shorten runs of fillers to two (哈哈哈哈 → 哈哈, 嗯嗯嗯 → 嗯嗯). The original manifest is kept as train.jsonl.orig."
          >
            Tidy fillers
          </button>
          {tidied !== null && (
            <span className="muted hint">
              {tidied === 0
                ? 'Nothing to tidy.'
                : `${tidied.toLocaleString()} transcripts tidied; check again to refresh their flags.`}
            </span>
          )}
        </div>
        {job && <JobProgress job={job} what="Check" />}
        {error && <p className="field-error">{error}</p>}
      </section>

      <section className="panel">
        <DatasetReview key={version} source={{ id }} />
      </section>
    </>
  )
}
