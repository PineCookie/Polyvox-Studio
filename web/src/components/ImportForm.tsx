import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { DEFAULT_LUFS, getStatus, importDataset, previewImport } from '../api'
import type { Aligner, CleanupOption, ImportJob, ImportPreview } from '../api'
import { ALIGNER_LABEL, formatDuration, formatRates } from '../format'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { JobProgress } from './JobProgress'

interface Props {
  datasetNames: string[]
  onImported: (result: ImportJob['result']) => void
}

// ISO codes the ASR accepts (Qwen3-ASR supports ~30; these cover the usual data).
// 'auto': detect until the first clips agree, then force that language (much faster).
const LANGUAGES = ['auto', '', 'zh', 'yue', 'en', 'ja', 'ko'] // labels: t.languages

// Clean-up options in the order they run (labels and hints: t.cleanupLabel).
const CLEANUP: CleanupOption[] = ['separate', 'dereverb', 'denoise']

// Matches importer.DATASET_NAME.
const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

export function ImportForm({ datasetNames, onImported }: Props) {
  const t = useT()
  const [source, setSource] = useState('')
  const [audioDir, setAudioDir] = useState('')
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [name, setName] = useState('')
  const [copyAudio, setCopyAudio] = useState(true)
  const [transcribe, setTranscribe] = useState(true)
  const [language, setLanguage] = useState('auto')
  const [prompt, setPrompt] = useState('')
  const [slice, setSlice] = useState(true)
  const [sliceMin, setSliceMin] = useState(2)
  const [sliceMax, setSliceMax] = useState(15)
  const [check, setCheck] = useState(true)
  const [aligner, setAligner] = useState<Aligner>('qwen3')
  const [trim, setTrim] = useState(true)
  const [tidy, setTidy] = useState(true)
  const [normalize, setNormalize] = useState(true)
  const [lufs, setLufs] = useState(DEFAULT_LUFS)
  const [detectSinging, setDetectSinging] = useState(false)
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
          normalizeLufs: normalize && (copyAudio || willSlice) && Number.isFinite(lufs) ? lufs : null,
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
    ? t.required
    : !NAME_PATTERN.test(name)
      ? t.nameChars
      : datasetNames.includes(name)
        ? t.datasetExists
        : null
  const willSlice = !!preview && slice && preview.long_recordings > 0
  const willTranscribe = !!preview && (willSlice || (transcribe && preview.untranscribed > 0))
  const importCount = preview ? preview.found + (willTranscribe ? preview.untranscribed : 0) : 0
  const withAudio = preview ? preview.found + preview.untranscribed : 0
  const sliceRangeOk = sliceMin > 0 && sliceMax > sliceMin

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

              <fieldset className="field asr-options" disabled={running || !(copyAudio || willSlice)}>
                <span className="check-title">
                  {t.cleanupTitle} <span className="muted">{t.cleanupOptional}</span>
                </span>
                {CLEANUP.map((option) => (
                  <label
                    key={option}
                    className="check"
                    title={available.includes(option) ? t.cleanupLabel[option][1] : t.noModel}
                  >
                    <input
                      type="checkbox"
                      checked={cleanup.includes(option)}
                      disabled={!available.includes(option)}
                      onChange={(e) =>
                        setCleanup((c) => (e.target.checked ? [...c, option] : c.filter((x) => x !== option)))
                      }
                    />{' '}
                    {t.cleanupLabel[option][0]}
                    {!available.includes(option) && <span className="muted">{t.noModelInstalled}</span>}
                  </label>
                ))}
                <p className="muted hint">{t.cleanupHint}</p>
              </fieldset>

              {preview.long_recordings > 0 && (
                <fieldset className="field asr-options" disabled={running}>
                  <label className="check">
                    <input type="checkbox" checked={slice} onChange={(e) => setSlice(e.target.checked)} />{' '}
                    {t.sliceLong(preview.long_recordings)}
                  </label>
                  {slice ? (
                    <>
                      <div className="inline-field">
                        <span className="muted hint">{t.clipsFrom}</span>
                        <input
                          type="number"
                          min={0.5}
                          step={0.5}
                          value={sliceMin}
                          onChange={(e) => setSliceMin(e.target.valueAsNumber)}
                          aria-label={t.minClipLength}
                        />
                        <span className="muted hint">{t.to}</span>
                        <input
                          type="number"
                          min={3}
                          step={1}
                          value={sliceMax}
                          onChange={(e) => setSliceMax(e.target.valueAsNumber)}
                          aria-label={t.maxClipLength}
                        />
                        <span className="muted hint">{t.secondsUnit}</span>
                        {!sliceRangeOk && <span className="field-error">{t.minBelowMax}</span>}
                      </div>
                      <p className="muted hint">{t.sliceHint}</p>
                    </>
                  ) : (
                    <p className="muted hint">{t.noSliceHint}</p>
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
                    {willSlice ? t.transcribeSlices : t.transcribeClips(preview.untranscribed)}
                  </label>
                  {(transcribe || willSlice) && (
                    <div className="form-grid">
                      <label className="field">
                        <span>{t.language}</span>
                        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                          {LANGUAGES.map((code) => (
                            <option key={code} value={code}>
                              {t.languages[code]}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field">
                        <span>
                          {t.namesAndTerms} <span className="muted">{t.optional}</span>
                        </span>
                        <input
                          value={prompt}
                          onChange={(e) => setPrompt(e.target.value)}
                          placeholder={t.namesPlaceholder}
                        />
                      </label>
                    </div>
                  )}
                  {!transcribe && !willSlice && <p className="muted hint">{t.leftOut}</p>}
                </fieldset>
              )}

              <fieldset className="field asr-options" disabled={running}>
                <label className="check">
                  <input type="checkbox" checked={check} onChange={(e) => setCheck(e.target.checked)} />{' '}
                  {t.checkEachClip}
                </label>
                {check && (
                  <>
                    <div className="inline-field">
                      <span className="muted hint">{t.aligner}</span>
                      <select
                        value={aligner}
                        onChange={(e) => setAligner(e.target.value as Aligner)}
                        aria-label={t.aligner}
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
                      {t.trimEdges}
                    </label>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={detectSinging}
                        onChange={(e) => setDetectSinging(e.target.checked)}
                      />{' '}
                      {t.flagSung}
                    </label>
                    <p className="muted hint">{t.checkHint}</p>
                  </>
                )}
              </fieldset>

              <label className="check">
                <input type="checkbox" checked={tidy} onChange={(e) => setTidy(e.target.checked)} disabled={running} />{' '}
                {t.tidyOption}
              </label>

              <div className="inline-field">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={normalize && (copyAudio || willSlice)}
                    disabled={running || !(copyAudio || willSlice)}
                    onChange={(e) => setNormalize(e.target.checked)}
                  />{' '}
                  {t.normalizeTo}
                </label>
                <input
                  type="number"
                  min={-40}
                  max={-10}
                  step={1}
                  value={lufs}
                  onChange={(e) => setLufs(e.target.valueAsNumber)}
                  disabled={running || !normalize || !(copyAudio || willSlice)}
                  aria-label={t.targetLufs}
                />
                <span className="muted hint">{t.lufsHint}</span>
              </div>

              <label className="field">
                <span>
                  {t.datasetName} {nameError && !running && <span className="field-error">{nameError}</span>}
                </span>
                <input value={name} onChange={(e) => setName(e.target.value)} disabled={running} />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={copyAudio || willSlice}
                  onChange={(e) => setCopyAudio(e.target.checked)}
                  disabled={running || willSlice}
                />{' '}
                {t.copyAudio(name || '…')}
              </label>

              {job && <JobProgress job={job} what={t.importWhat} />}

              <div className="actions">
                <button
                  type="button"
                  className="primary"
                  onClick={runImport}
                  disabled={!!nameError || running || (importCount === 0 && !willSlice) || (willSlice && !sliceRangeOk)}
                >
                  {running ? t.importing : willSlice ? t.sliceAndImport : t.importClips(importCount)}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
