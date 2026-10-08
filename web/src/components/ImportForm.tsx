import { useState } from 'react'
import type { FormEvent } from 'react'
import { importDataset, previewImport } from '../api'
import type { ImportPreview } from '../api'
import { formatDuration, formatRates } from '../format'

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

// Matches importer.DATASET_NAME.
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

export function ImportForm({ datasetNames, onImported }: Props) {
  const [source, setSource] = useState('')
  const [audioDir, setAudioDir] = useState('')
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [name, setName] = useState('')
  const [copyAudio, setCopyAudio] = useState(true)
  const [busy, setBusy] = useState<'scan' | 'import' | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function scan(e: FormEvent) {
    e.preventDefault()
    if (!source.trim()) return
    setBusy('scan')
    setError(null)
    try {
      const p = await previewImport(source, audioDir)
      setPreview(p)
      setName(p.suggested_name)
    } catch (err) {
      setPreview(null)
      setError((err as Error).message)
    } finally {
      setBusy(null)
    }
  }

  async function runImport() {
    if (!preview) return
    setBusy('import')
    setError(null)
    try {
      onImported((await importDataset(source, audioDir, name, copyAudio)).id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(null)
    }
  }

  // Any change to the inputs invalidates the scan.
  function edit(setter: (v: string) => void) {
    return (v: string) => {
      setter(v)
      setPreview(null)
    }
  }

  const nameError = !name
    ? 'Required'
    : !NAME_PATTERN.test(name)
      ? 'Letters, digits, ".", "_" and "-" only.'
      : datasetNames.includes(name)
        ? 'A dataset with this name exists.'
        : null

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
            placeholder="D:\voices\speaker.list   or a folder of clips with .lab/.txt transcripts"
            spellCheck={false}
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
          />
        </label>
        <p className="muted hint">
          GPT-SoVITS <code>.list</code>, <code>file|text</code> CSV, <code>.jsonl</code>, or a folder. Paths are on
          this computer; Explorer's “Copy as path” works.
        </p>
        <div className="actions">
          <button type="submit" className={preview ? 'ghost' : 'primary'} disabled={!source.trim() || busy !== null}>
            {busy === 'scan' ? 'Scanning…' : 'Scan'}
          </button>
        </div>
      </form>

      {error && <p className="field-error">{error}</p>}

      {preview && (
        <div className="import-preview">
          <p className="import-stats">
            <span className="tag">{KIND_LABEL[preview.kind]}</span>
            <span>
              <strong>{preview.found}</strong> of {preview.rows} clips found
            </span>
            {preview.found > 0 && (
              <span className="muted">
                {formatDuration(preview.total_seconds)} · median {preview.median_seconds?.toFixed(1)} s ·{' '}
                {formatRates(preview.sample_rates)}
              </span>
            )}
          </p>

          {(preview.missing > 0 || preview.skipped_empty > 0 || preview.duplicates > 0 || preview.unreadable > 0) && (
            <ul className="issues">
              {preview.missing > 0 && (
                <li>
                  {preview.missing} without audio (skipped)
                  {preview.missing_examples.length > 0 && (
                    <span className="muted"> · e.g. {preview.missing_examples.map(fileName).join(', ')}</span>
                  )}
                </li>
              )}
              {preview.skipped_empty > 0 && <li>{preview.skipped_empty} without a transcript (skipped)</li>}
              {preview.duplicates > 0 && <li>{preview.duplicates} duplicates (skipped)</li>}
              {preview.unreadable > 0 && <li>{preview.unreadable} unreadable audio files (skipped)</li>}
            </ul>
          )}

          {preview.found === 0 ? (
            <p className="notice">
              <strong>Transcripts only.</strong> No audio was found for this source. If you have the clips, set Audio
              folder to where they are and scan again; otherwise it can't be imported yet.
            </p>
          ) : (
            <>
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

              <label className="field">
                <span>
                  Dataset name {nameError && <span className="field-error">{nameError}</span>}
                </span>
                <input value={name} onChange={(e) => setName(e.target.value)} />
              </label>
              <label className="check">
                <input type="checkbox" checked={copyAudio} onChange={(e) => setCopyAudio(e.target.checked)} /> Copy
                audio into <code>data/{name || '…'}/wavs</code> (self-contained; otherwise the files stay where they
                are)
              </label>
              <div className="actions">
                <button type="button" className="primary" onClick={runImport} disabled={!!nameError || busy !== null}>
                  {busy === 'import' ? 'Importing…' : `Import ${preview.found} clips`}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
