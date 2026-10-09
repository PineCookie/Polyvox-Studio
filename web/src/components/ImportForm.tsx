import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { getStatus, importDataset, previewImport } from '../api'
import type { Aligner, CleanupOption, ImportJob, ImportPreview } from '../api'
import { ALIGNER_LABEL, formatDuration, formatRates } from '../format'
import { useJob } from '../useJob'
import { JobProgress } from './JobProgress'

interface Props {
  datasetNames: string[]
  onImported: (result: ImportJob['result']) => void
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

// Clean-up options in the order they run: [option, label, what it is for].
const CLEANUP: [CleanupOption, string, string][] = [
  ['separate', 'Separate the voice from music and background', 'Vocal separation (bleedless Mel-Band RoFormer)'],
  ['dereverb', 'Remove reverb', 'For echoey rooms or reverb added in a stream mix'],
  ['denoise', 'Remove noise', 'For hiss, hum or fan noise'],
]
const NO_MODEL = 'No model installed; see the README (Configuration) to add one.'

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
  const [slice, setSlice] = useState(true)
  const [sliceMin, setSliceMin] = useState(2)
  const [sliceMax, setSliceMax] = useState(15)
  const [check, setCheck] = useState(true)
  const [aligner, setAligner] = useState<Aligner>('qwen3')
  const [trim, setTrim] = useState(true)
  const [tidy, setTidy] = useState(true)
  const [detectSinging, setDetectSinging] = useState(true)
  // Optional clean-up, off by default; which ones have a model installed.
  const [cleanup, setCleanup] = useState<CleanupOption[]>([])
  const [available, setAvailable] = useState<CleanupOption[]>([])
  useEffect(() => {
    getStatus().then(
      (s) => setAvailable(s.cleanup),
      () => setAvailable([]),
    )
  }, [])
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useJob<ImportJob>((j) => {
    if (j.state === 'done') onImported(j.result)
    else if (j.state === 'failed') setError(j.error ?? 'Import failed.')
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
      // Slices are new files without text: they need copying and transcription.
      setJob(
        await importDataset({
          source,
          audioDir,
          name,
          copyAudio: copyAudio || willSlice,
          transcribe: transcribe || willSlice,
          language,
          prompt,
          slice: willSlice,
          sliceMin,
          sliceMax,
          check,
          aligner,
          // Trimming rewrites the copies; files imported in place stay as they are.
          trim: check && trim && (copyAudio || willSlice),
          cleanup: copyAudio || willSlice ? cleanup : [],
          tidyFillers: tidy,
          detectSinging: check && detectSinging,
        }),
      )
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
  const willSlice = !!preview && slice && preview.long_recordings > 0
  const willTranscribe = !!preview && (willSlice || (transcribe && preview.untranscribed > 0))
  const importCount = preview ? preview.found + (willTranscribe ? preview.untranscribed : 0) : 0
  const withAudio = preview ? preview.found + preview.untranscribed : 0
  const sliceRangeOk = sliceMin > 0 && sliceMax > sliceMin

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
          GPT-SoVITS <code>.list</code>, <code>file|text</code> CSV, <code>.jsonl</code>, or a folder. Paths are on this
          computer; Explorer's “Copy as path” works.
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

              <fieldset className="field asr-options" disabled={running || !(copyAudio || willSlice)}>
                <span className="check-title">
                  Clean up the audio first <span className="muted">optional · slower · needs copying</span>
                </span>
                {CLEANUP.map(([option, label, hint]) => (
                  <label key={option} className="check" title={available.includes(option) ? hint : NO_MODEL}>
                    <input
                      type="checkbox"
                      checked={cleanup.includes(option)}
                      disabled={!available.includes(option)}
                      onChange={(e) =>
                        setCleanup((c) => (e.target.checked ? [...c, option] : c.filter((x) => x !== option)))
                      }
                    />{' '}
                    {label}
                    {!available.includes(option) && <span className="muted"> · no model installed</span>}
                  </label>
                ))}
                <p className="muted hint">
                  RoFormer models (as in UVR / MSST-WebUI) on the GPU. Separation keeps the voice and drops music and
                  background; de-reverb and denoise are for echoey or noisy recordings.
                </p>
              </fieldset>

              {preview.long_recordings > 0 && (
                <fieldset className="field asr-options" disabled={running}>
                  <label className="check">
                    <input type="checkbox" checked={slice} onChange={(e) => setSlice(e.target.checked)} /> Slice the{' '}
                    {preview.long_recordings} long recordings into clips
                  </label>
                  {slice ? (
                    <>
                      <div className="inline-field">
                        <span className="muted hint">Clips from</span>
                        <input
                          type="number"
                          min={0.5}
                          step={0.5}
                          value={sliceMin}
                          onChange={(e) => setSliceMin(e.target.valueAsNumber)}
                          aria-label="Minimum clip length (seconds)"
                        />
                        <span className="muted hint">to</span>
                        <input
                          type="number"
                          min={3}
                          step={1}
                          value={sliceMax}
                          onChange={(e) => setSliceMax(e.target.valueAsNumber)}
                          aria-label="Maximum clip length (seconds)"
                        />
                        <span className="muted hint">seconds</span>
                        {!sliceRangeOk && <span className="field-error">min must be below max</span>}
                      </div>
                      <p className="muted hint">
                        Each recording is transcribed and aligned first, then cut between words, preferably at sentence
                        ends; each clip keeps its part of the transcript. Any transcript of a whole recording is not
                        kept.
                      </p>
                    </>
                  ) : (
                    <p className="muted hint">Long recordings are imported as they are (too long to train well).</p>
                  )}
                </fieldset>
              )}

              {(preview.untranscribed > 0 || willSlice) && (
                <fieldset className="field asr-options" disabled={running}>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={transcribe || willSlice}
                      disabled={willSlice}
                      onChange={(e) => setTranscribe(e.target.checked)}
                    />{' '}
                    {willSlice
                      ? 'Transcribe the slices and clips without text (Qwen3-ASR)'
                      : `Transcribe the ${preview.untranscribed} clips without text (Qwen3-ASR)`}
                  </label>
                  {(transcribe || willSlice) && (
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
                  {!transcribe && !willSlice && <p className="muted hint">Those clips will be left out.</p>}
                </fieldset>
              )}

              <fieldset className="field asr-options" disabled={running}>
                <label className="check">
                  <input type="checkbox" checked={check} onChange={(e) => setCheck(e.target.checked)} /> Check each clip
                  against its text (forced alignment)
                </label>
                {check && (
                  <>
                    <div className="inline-field">
                      <span className="muted hint">Aligner</span>
                      <select
                        value={aligner}
                        onChange={(e) => setAligner(e.target.value as Aligner)}
                        aria-label="Aligner"
                      >
                        {(Object.keys(ALIGNER_LABEL) as Aligner[]).map((a) => (
                          <option key={a} value={a}>
                            {ALIGNER_LABEL[a]}
                          </option>
                        ))}
                      </select>
                    </div>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={trim && (copyAudio || willSlice)}
                        disabled={!(copyAudio || willSlice)}
                        onChange={(e) => setTrim(e.target.checked)}
                      />{' '}
                      Trim the clip edges: long silence, and untranscribed sounds (a laugh, 嗯) a pause separates from
                      the words; keeps 0.15 s
                    </label>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={detectSinging}
                        onChange={(e) => setDetectSinging(e.target.checked)}
                      />{' '}
                      Flag sung clips (karaoke, covers): drop them in review with one click
                    </label>
                    <p className="muted hint">
                      Flags clips whose text does not match, with untranscribed speech, cut at the edge or with long
                      silence, for review after the import.
                    </p>
                  </>
                )}
              </fieldset>

              <label className="check">
                <input type="checkbox" checked={tidy} onChange={(e) => setTidy(e.target.checked)} disabled={running} />{' '}
                Tidy fillers: shorten runs to two (哈哈哈哈 → 哈哈, 嗯嗯嗯 → 嗯嗯); counts are unreliable and only need
                to be consistent
              </label>

              <label className="field">
                <span>Dataset name {nameError && !running && <span className="field-error">{nameError}</span>}</span>
                <input value={name} onChange={(e) => setName(e.target.value)} disabled={running} />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={copyAudio || willSlice}
                  onChange={(e) => setCopyAudio(e.target.checked)}
                  disabled={running || willSlice}
                />{' '}
                Copy audio into <code>data/{name || '…'}/wavs</code> (self-contained; otherwise the files stay where
                they are)
              </label>

              {job && <JobProgress job={job} what="Import" />}

              <div className="actions">
                <button
                  type="button"
                  className="primary"
                  onClick={runImport}
                  disabled={!!nameError || running || (importCount === 0 && !willSlice) || (willSlice && !sliceRangeOk)}
                >
                  {running ? 'Importing…' : willSlice ? 'Slice and import' : `Import ${importCount} clips`}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
