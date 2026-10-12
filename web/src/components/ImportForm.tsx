import { useState } from 'react'
import type { FormEvent } from 'react'
import { createDataset, previewImport } from '../api'
import type { CreateJob, ImportPreview } from '../api'
import { formatDuration, formatRates } from '../format'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { JobProgress } from './JobProgress'

interface Props {
  datasetNames: string[]
  onImported: (result: CreateJob['result']) => void
}

// Matches importer.DATASET_NAME.
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

/** Bring material in: a list or folder becomes a dataset; its processing steps run on the dataset page. */
export function ImportForm({ datasetNames, onImported }: Props) {
  const t = useT()
  const [source, setSource] = useState('')
  const [audioDir, setAudioDir] = useState('')
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [name, setName] = useState('')
  const [copyAudio, setCopyAudio] = useState(true)
  const [runSteps, setRunSteps] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useJob<CreateJob>((j) => {
    if (j.state === 'done') onImported(j.result)
    else if (j.state === 'failed') setError(j.error ?? t.importFailed)
  })

  const running = job?.state === 'running'

  async function scan(e: FormEvent) {
    e.preventDefault()
    if (!source.trim()) return
    setScanning(true)
    setError(null)
    setJob(null)
    try {
      const p = await previewImport(source, audioDir)
      setPreview(p)
      setName(p.suggested_name)
    } catch (err) {
      setPreview(null)
      setError((err as Error).message)
    } finally {
      setScanning(false)
    }
  }

  async function runImport() {
    setError(null)
    try {
      setJob(await createDataset(source, audioDir, name, copyAudio, runSteps))
    } catch (err) {
      setError((err as Error).message)
    }
  }

  // Any change to the inputs invalidates the scan.
  function edit(setter: (v: string) => void) {
    return (v: string) => {
      setter(v)
      setPreview(null)
      setJob(null)
    }
  }

  const nameError = !name
    ? t.required
    : !NAME_PATTERN.test(name)
      ? t.nameChars
      : datasetNames.includes(name)
        ? t.datasetExists
        : null
  const withAudio = preview ? preview.found + preview.untranscribed : 0
  const recordings = preview?.long_recordings ?? 0
  const clips = withAudio - recordings

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{t.importTitle}</h2>
      </header>

      <form className="import-paths" onSubmit={scan}>
        <label className="field">
          <span>{t.source}</span>
          <input
            value={source}
            onChange={(e) => edit(setSource)(e.target.value)}
            placeholder={t.sourcePlaceholder}
            spellCheck={false}
            disabled={running}
          />
        </label>
        <label className="field">
          <span>
            {t.audioFolder} <span className="muted">{t.audioFolderHint}</span>
          </span>
          <input
            value={audioDir}
            onChange={(e) => edit(setAudioDir)(e.target.value)}
            placeholder="D:\voices\speaker\clips"
            spellCheck={false}
            disabled={running}
          />
        </label>
        <p className="muted hint">{t.sourceHint}</p>
        <div className="actions">
          <button
            type="submit"
            className={preview ? 'ghost' : 'primary'}
            disabled={!source.trim() || scanning || running}
          >
            {scanning ? t.scanning : t.scan}
          </button>
        </div>
      </form>

      {error && <p className="field-error">{error}</p>}

      {preview && (
        <div className="import-preview">
          <p className="import-stats">
            <span className="tag">{t.kindLabel[preview.kind]}</span>
            <span>
              {t.clipsWithAudio(withAudio, preview.rows)}
              {preview.untranscribed > 0 && (
                <span className="muted">{t.withAndWithoutText(preview.found, preview.untranscribed)}</span>
              )}
            </span>
            {withAudio > 0 && (
              <span className="muted">
                {t.previewLength(
                  formatDuration(preview.total_seconds),
                  preview.median_seconds?.toFixed(1) ?? '-',
                  formatRates(preview.sample_rates),
                )}
              </span>
            )}
          </p>

          {(preview.missing > 0 || preview.duplicates > 0 || preview.unreadable > 0) && (
            <ul className="issues">
              {preview.missing > 0 && (
                <li>
                  {t.missingAudio(preview.missing)}
                  {preview.missing_examples.length > 0 && (
                    <span className="muted">{t.forExample(preview.missing_examples.map(fileName).join(', '))}</span>
                  )}
                </li>
              )}
              {preview.duplicates > 0 && <li>{t.duplicates(preview.duplicates)}</li>}
              {preview.unreadable > 0 && <li>{t.unreadable(preview.unreadable)}</li>}
            </ul>
          )}

          {withAudio === 0 ? (
            <p className="notice">
              <strong>{t.transcriptsOnlyTitle}</strong> {t.transcriptsOnly}
            </p>
          ) : (
            <>
              {preview.examples.length > 0 && (
                <table className="metrics examples">
                  <tbody>
                    {preview.examples.map((ex) => (
                      <tr key={ex.audio}>
                        <td className="path" title={ex.audio}>
                          {fileName(ex.audio)}
                        </td>
                        <td>{ex.text}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              <p className="muted hint">{t.importBrings(clips, recordings)}</p>

              <label className="field">
                <span>
                  {t.datasetName} {nameError && !running && <span className="field-error">{nameError}</span>}
                </span>
                <input value={name} onChange={(e) => setName(e.target.value)} disabled={running} />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={copyAudio}
                  onChange={(e) => setCopyAudio(e.target.checked)}
                  disabled={running}
                />{' '}
                {t.copyAudio(name || '…')}
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={runSteps}
                  onChange={(e) => setRunSteps(e.target.checked)}
                  disabled={running}
                />{' '}
                {t.runStepsAfter}
              </label>
              <p className="muted hint">{t.runStepsAfterHint}</p>

              {job && <JobProgress job={job} what={t.importWhat} />}

              <div className="actions">
                <button type="button" className="primary" onClick={runImport} disabled={!!nameError || running}>
                  {running ? t.importing : t.importButton}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
