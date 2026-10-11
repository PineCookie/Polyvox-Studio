import { useState } from 'react'
import { DEFAULT_LUFS, checkDataset, normalizeDataset, tidyFillers } from '../api'
import type { Aligner, CheckJob, ImportJob, NormalizeJob } from '../api'
import { ALIGNER_LABEL, datasetName, formatTimings } from '../format'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { DatasetReview } from './DatasetReview'
import { ExportTool } from './ExportTool'
import { JobProgress } from './JobProgress'

interface Props {
  id: string
  // The import that just made this dataset, if any.
  imported: ImportJob['result'] | null
  onTrain: (id: string) => void
}

function NormalizeSummary({ result }: { result: NormalizeJob['result'] }) {
  const t = useT()
  const count = (n: number) => (n ? n.toLocaleString() : '')
  const spread = ([low, median, high]: [number, number, number]) => t.lufsSpread(low, high, median)
  return (
    <span className="muted hint">
      {t.normalizedSummary(
        result.normalized.toLocaleString(),
        count(result.unchanged),
        count(result.peak_limited),
        count(result.skipped),
      )}
      {result.before && result.after && t.beforeAfter(spread(result.before), spread(result.after))}
    </span>
  )
}

/** A prepared dataset: check its clips against their text, review, then train. */
export function DatasetPage({ id, imported, onTrain }: Props) {
  const t = useT()
  const [aligner, setAligner] = useState<Aligner>('qwen3')
  const [compareAsr, setCompareAsr] = useState(true)
  const [detectSinging, setDetectSinging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Bumped when a check finishes, so the review reloads the results.
  const [version, setVersion] = useState(0)
  const [job, setJob] = useJob<CheckJob>((j) => {
    if (j.state === 'done') setVersion((v) => v + 1)
    else if (j.state === 'failed') setError(j.error ?? t.checkFailed)
  })
  const [lufs, setLufs] = useState(DEFAULT_LUFS)
  const [normalizing, setNormalizing] = useJob<NormalizeJob>((j) => {
    if (j.state === 'done') setVersion((v) => v + 1)
    else if (j.state === 'failed') setError(j.error ?? t.normalizeFailed)
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
              {t.trainOnIt}
            </button>
          </div>
        </header>
        {imported && (
          <p className="notice">
            {t.importedSummary(
              imported.imported.toLocaleString(),
              imported.sliced,
              imported.transcribed ? imported.transcribed.toLocaleString() : '',
              imported.trimmed ? imported.trimmed.toLocaleString() : '',
            )}
            {imported.timings && <span className="muted"> {formatTimings(imported.timings)}.</span>}
          </p>
        )}
        <div className="tools">
          <div className="tool-row">
            <div className="tool-info">
              <strong>{t.checkClipsTitle}</strong>
              <span className="muted hint">{t.checkClipsHint}</span>
              <details className="tool-help">
                <summary>{t.howChecking}</summary>
                <p className="muted hint">{t.checkingExplained}</p>
              </details>
            </div>
            <div className="tool-controls">
              <select
                value={aligner}
                onChange={(e) => setAligner(e.target.value as Aligner)}
                disabled={running}
                aria-label={t.aligner}
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
                {t.compareWithAsr}
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={detectSinging}
                  onChange={(e) => setDetectSinging(e.target.checked)}
                  disabled={running}
                />{' '}
                {t.detectSinging}
              </label>
              <button type="button" className="ghost" onClick={check} disabled={running}>
                {job?.state === 'running' ? t.checking : t.checkButton}
              </button>
            </div>
          </div>

          <div className="tool-row">
            <div className="tool-info">
              <strong>{t.tidyTitle}</strong>
              <span className="muted hint">
                {tidied === null ? t.tidyHint : tidied === 0 ? t.nothingToTidy : t.tidied(tidied.toLocaleString())}
              </span>
            </div>
            <div className="tool-controls">
              <button type="button" className="ghost" onClick={tidy} disabled={running}>
                {t.tidyButton}
              </button>
            </div>
          </div>

          <div className="tool-row">
            <div className="tool-info">
              <strong>{t.normalizeTitle}</strong>
              {normalizing?.state === 'done' ? (
                <NormalizeSummary result={normalizing.result} />
              ) : (
                <span className="muted hint">{t.normalizeHint}</span>
              )}
            </div>
            <div className="tool-controls">
              <input
                type="number"
                min={-40}
                max={-10}
                step={1}
                value={lufs}
                onChange={(e) => setLufs(Number(e.target.value))}
                disabled={running}
                aria-label={t.targetLufs}
              />
              <span className="muted hint">LUFS</span>
              <button type="button" className="ghost" onClick={normalize} disabled={running}>
                {normalizing?.state === 'running' ? t.normalizing : t.normalizeButton}
              </button>
            </div>
          </div>

          <ExportTool id={id} disabled={running} />
        </div>
        {job && <JobProgress job={job} what={t.checkWhat} />}
        {job?.state === 'done' && job.result.timings && (
          <p className="muted hint">{t.checkDone(formatTimings(job.result.timings))}</p>
        )}
        {normalizing && <JobProgress job={normalizing} what={t.normalizeWhat} />}
        {error && <p className="field-error">{error}</p>}
      </section>

      <section className="panel">
        <DatasetReview key={version} source={{ id }} />
      </section>
    </>
  )
}
