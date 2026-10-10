import { useState } from 'react'
import { DEFAULT_LUFS, checkDataset, normalizeDataset, tidyFillers } from '../api'
import type { Aligner, CheckJob, ImportJob, NormalizeJob } from '../api'
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

function spread(levels: [number, number, number]): string {
  return `${levels[0]} … ${levels[2]} LUFS (median ${levels[1]})`
}

function NormalizeSummary({ result }: { result: NormalizeJob['result'] }) {
  const parts = [`${result.normalized.toLocaleString()} clips normalized`]
  if (result.unchanged) parts.push(`${result.unchanged.toLocaleString()} already there`)
  if (result.peak_limited) parts.push(`${result.peak_limited.toLocaleString()} kept quieter by their peaks`)
  if (result.skipped) parts.push(`${result.skipped.toLocaleString()} skipped`)
  return (
    <span className="muted hint">
      {parts.join(', ')}.
      {result.before && result.after && ` Before ${spread(result.before)}; now ${spread(result.after)}.`}
    </span>
  )
}

/** A prepared dataset: check its clips against their text, review, then train. */
export function DatasetPage({ id, imported, onTrain }: Props) {
  const [aligner, setAligner] = useState<Aligner>('qwen3')
  const [compareAsr, setCompareAsr] = useState(true)
  const [detectSinging, setDetectSinging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Bumped when a check finishes, so the review reloads the results.
  const [version, setVersion] = useState(0)
  const [job, setJob] = useJob<CheckJob>((j) => {
    if (j.state === 'done') setVersion((v) => v + 1)
    else if (j.state === 'failed') setError(j.error ?? 'Check failed.')
  })
  const [lufs, setLufs] = useState(DEFAULT_LUFS)
  const [normalizing, setNormalizing] = useJob<NormalizeJob>((j) => {
    if (j.state === 'done') setVersion((v) => v + 1)
    else if (j.state === 'failed') setError(j.error ?? 'Normalizing failed.')
  })
  const running = job?.state === 'running' || normalizing?.state === 'running'
  const [tidied, setTidied] = useState<number | null>(null) // clips the last tidy changed

  async function check() {
    setError(null)
    try {
      setJob(await checkDataset(id, aligner, compareAsr, detectSinging))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function normalize() {
    setError(null)
    try {
      setNormalizing(await normalizeDataset(id, lufs))
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
          <label className="check">
            <input
              type="checkbox"
              checked={compareAsr}
              onChange={(e) => setCompareAsr(e.target.checked)}
              disabled={running}
            />{' '}
            and compare with ASR
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={detectSinging}
              onChange={(e) => setDetectSinging(e.target.checked)}
              disabled={running}
            />{' '}
            detect singing
          </label>
        </div>
        <p className="muted hint">
          Forced alignment finds text that does not match the audio, untranscribed speech, words cut at the file edge
          and long silence; comparing with ASR finds words the text leaves out. Texts made by ASR are not compared, and
          get no mismatch flag (there it mostly marks laughs or fast speech); worst-first still sorts them. Qwen3 also
          handles mixed languages; wav2vec2 is sharper on single wrong words (Chinese model by default).
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
        <div className="inline-field check-action">
          <button
            type="button"
            className="ghost"
            onClick={normalize}
            disabled={running}
            title="One gain per clip (EBU R128 loudness), lowered where the peak would pass -1 dBFS. Originals are kept in originals/; checks stay valid."
          >
            {normalizing?.state === 'running' ? 'Normalizing…' : 'Normalize loudness'}
          </button>
          <span className="muted hint">to</span>
          <input
            type="number"
            min={-40}
            max={-10}
            step={1}
            value={lufs}
            onChange={(e) => setLufs(Number(e.target.value))}
            disabled={running}
            aria-label="Target loudness (LUFS)"
          />
          <span className="muted hint">LUFS</span>
          {normalizing?.state === 'done' && <NormalizeSummary result={normalizing.result} />}
        </div>
        {job && <JobProgress job={job} what="Check" />}
        {normalizing && <JobProgress job={normalizing} what="Normalizing" />}
        {error && <p className="field-error">{error}</p>}
      </section>

      <section className="panel">
        <DatasetReview key={version} source={{ id }} />
      </section>
    </>
  )
}
