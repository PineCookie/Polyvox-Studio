import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { cancelJob, getJob, importDataset, previewImport } from '../api'
import type { ImportJob, ImportPreview } from '../api'
import { formatDuration, formatRates } from '../format'
import { ProgressBar } from './RunList'

interface Props {
  datasetNames: string[]
  onImported: (id: string) => void
}

const KIND_LABEL: Record<ImportPreview['kind'], string> = {
  gsv: 'GPT-SoVITS list',
  csv: 'CSV (file|text)',
  jsonl: 'JSONL manifest',
  folder: 'Folder of audio + transcripts',
}

// ISO codes the ASR accepts (Qwen3-ASR supports ~30; these cover the usual data).
const LANGUAGES: [string, string][] = [
  ['', 'Detect per clip'],
  ['zh', 'Chinese'],
  ['yue', 'Cantonese'],
  ['en', 'English'],
  ['ja', 'Japanese'],
  ['ko', 'Korean'],
]

const PHASE_LABEL: Record<string, string> = {
  scanning: 'Scanning',
  transcribing: 'Transcribing',
  copying: 'Copying audio',
}

const JOB_POLL_MS = 700

// Matches importer.DATASET_NAME.
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

export function ImportForm({ datasetNames, onImported }: Props) {
  const [source, setSource] = useState('')
  const [audioDir, setAudioDir] = useState('')
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [name, setName] = useState('')
  const [copyAudio, setCopyAudio] = useState(true)
  const [transcribe, setTranscribe] = useState(true)
  const [language, setLanguage] = useState('')
  const [prompt, setPrompt] = useState('')
  const [scanning, setScanning] = useState(false)
  const [job, setJob] = useState<ImportJob | null>(null)
  const [error, setError] = useState<string | null>(null)

  const running = job?.state === 'running'
  const onImportedRef = useRef(onImported)
  useEffect(() => {
    onImportedRef.current = onImported
  })

  // Follow the import job until it ends.
  useEffect(() => {
    if (!job || job.state !== 'running') return
    const timer = setTimeout(async () => {
      try {
        const next = await getJob<ImportJob>(job.id)
        setJob(next)
        if (next.state === 'done') onImportedRef.current(next.result.id)
        else if (next.state === 'failed') setError(next.error ?? 'Import failed.')
      } catch (e) {
        setError((e as Error).message)
        setJob(null)
      }
    }, JOB_POLL_MS)
    return () => clearTimeout(timer)
  }, [job])

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
      setJob(await importDataset({ source, audioDir, name, copyAudio, transcribe, language, prompt }))
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
    ? 'Required'
    : !NAME_PATTERN.test(name)
      ? 'Letters, digits, ".", "_" and "-" only.'
      : datasetNames.includes(name)
        ? 'A dataset with this name exists.'
        : null
  const willTranscribe = !!preview && transcribe && preview.untranscribed > 0
  const importCount = preview ? preview.found + (willTranscribe ? preview.untranscribed : 0) : 0
  const withAudio = preview ? preview.found + preview.untranscribed : 0

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Import a dataset</h2>
      </header>

      <form className="import-paths" onSubmit={scan}>
        <label className="field">
          <span>Source</span>
          <input
            value={source}
            onChange={(e) => edit(setSource)(e.target.value)}
            placeholder="D:\voices\speaker.list   or a folder of clips (with or without .lab/.txt transcripts)"
            spellCheck={false}
            disabled={running}
          />
        </label>
        <label className="field">
          <span>
            Audio folder <span className="muted">optional · where the clips are, if the list's paths are stale</span>
          </span>
          <input
            value={audioDir}
            onChange={(e) => edit(setAudioDir)(e.target.value)}
            placeholder="D:\voices\speaker\clips"
            spellCheck={false}
            disabled={running}
          />
        </label>
        <p className="muted hint">
          GPT-SoVITS <code>.list</code>, <code>file|text</code> CSV, <code>.jsonl</code>, or a folder. Paths are on
          this computer; Explorer's “Copy as path” works.
        </p>
        <div className="actions">
          <button
            type="submit"
            className={preview ? 'ghost' : 'primary'}
            disabled={!source.trim() || scanning || running}
          >
            {scanning ? 'Scanning…' : 'Scan'}
          </button>
        </div>
      </form>

      {error && <p className="field-error">{error}</p>}

      {preview && (
        <div className="import-preview">
          <p className="import-stats">
            <span className="tag">{KIND_LABEL[preview.kind]}</span>
            <span>
              <strong>{withAudio}</strong> of {preview.rows} clips have audio
              {preview.untranscribed > 0 && (
                <span className="muted">
                  {' '}
                  · {preview.found} with text, {preview.untranscribed} without
                </span>
              )}
            </span>
            {withAudio > 0 && (
              <span className="muted">
                {formatDuration(preview.total_seconds)} · median {preview.median_seconds?.toFixed(1)} s ·{' '}
                {formatRates(preview.sample_rates)}
              </span>
            )}
          </p>

          {(preview.missing > 0 || preview.duplicates > 0 || preview.unreadable > 0) && (
            <ul className="issues">
              {preview.missing > 0 && (
                <li>
                  {preview.missing} without audio (skipped)
                  {preview.missing_examples.length > 0 && (
                    <span className="muted"> · e.g. {preview.missing_examples.map(fileName).join(', ')}</span>
                  )}
                </li>
              )}
              {preview.duplicates > 0 && <li>{preview.duplicates} duplicates (skipped)</li>}
              {preview.unreadable > 0 && <li>{preview.unreadable} unreadable audio files (skipped)</li>}
            </ul>
          )}

          {withAudio === 0 ? (
            <p className="notice">
              <strong>Transcripts only.</strong> No audio was found for this source. If you have the clips, set Audio
              folder to where they are and scan again; otherwise it can't be imported yet.
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

              {preview.untranscribed > 0 && (
                <fieldset className="field asr-options" disabled={running}>
                  <label className="check">
                    <input type="checkbox" checked={transcribe} onChange={(e) => setTranscribe(e.target.checked)} />{' '}
                    Transcribe the {preview.untranscribed} clips without text (Qwen3-ASR)
                  </label>
                  {transcribe && (
                    <div className="form-grid">
                      <label className="field">
                        <span>Language</span>
                        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                          {LANGUAGES.map(([code, label]) => (
                            <option key={code} value={code}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field">
                        <span>
                          Names and terms <span className="muted">optional</span>
                        </span>
                        <input
                          value={prompt}
                          onChange={(e) => setPrompt(e.target.value)}
                          placeholder="Speaker and place names the ASR should spell right"
                        />
                      </label>
                    </div>
                  )}
                  {!transcribe && <p className="muted hint">Those clips will be left out.</p>}
                </fieldset>
              )}

              <label className="field">
                <span>
                  Dataset name {nameError && !running && <span className="field-error">{nameError}</span>}
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
                Copy audio into <code>data/{name || '…'}/wavs</code> (self-contained; otherwise the files stay where
                they are)
              </label>

              {job && job.state !== 'done' ? (
                <div className="job-progress">
                  {job.state === 'running' ? (
                    <>
                      <span>
                        {PHASE_LABEL[job.phase] ?? 'Starting'}
                        {job.total > 0 && (
                          <span className="muted">
                            {' '}
                            {job.done} / {job.total}
                          </span>
                        )}
                      </span>
                      <ProgressBar value={job.done} max={job.total || 1} />
                      <button type="button" className="ghost small" onClick={() => cancelJob(job.id)}>
                        Cancel
                      </button>
                    </>
                  ) : (
                    <span className="muted">{job.state === 'cancelled' ? 'Import cancelled.' : 'Import failed.'}</span>
                  )}
                </div>
              ) : null}

              <div className="actions">
                <button
                  type="button"
                  className="primary"
                  onClick={runImport}
                  disabled={!!nameError || running || importCount === 0}
                >
                  {running ? 'Importing…' : `Import ${importCount} clips`}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
