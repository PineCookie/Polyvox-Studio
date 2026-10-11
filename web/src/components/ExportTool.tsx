import { useEffect, useState } from 'react'
import { EXPORT_LANGUAGES, exportDataset, exportDir, listExportFormats } from '../api'
import type { ExportFormat, ExportJob } from '../api'
import { formatDuration } from '../format'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { JobProgress } from './JobProgress'

const RATES = [16000, 22050, 24000, 32000, 44100, 48000]

const num = (value: string) => (value.trim() === '' ? null : Number(value))

/** Dataset tool: write the training clips in another TTS tool's layout. */
export function ExportTool({ id, disabled }: { id: string; disabled: boolean }) {
  const t = useT()
  const [formats, setFormats] = useState<ExportFormat[]>([])
  const [format, setFormat] = useState('')
  const [outDir, setOutDir] = useState('')
  const [minSeconds, setMinSeconds] = useState('')
  const [maxSeconds, setMaxSeconds] = useState('')
  const [rate, setRate] = useState('')
  const [language, setLanguage] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useJob<ExportJob>((j) => {
    if (j.state === 'failed') setError(j.error ?? t.dsExportFailed)
  })
  const running = job?.state === 'running'
  const chosen = formats.find((f) => f.id === format)

  useEffect(() => {
    listExportFormats().then(
      (list) => {
        setFormats(list)
        if (list[0]) choose(list[0])
      },
      (e: Error) => setError(e.message),
    )
    // Once per dataset; choose() only sets state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  function choose(f: ExportFormat) {
    setFormat(f.id)
    setMinSeconds(f.min_seconds === null ? '' : String(f.min_seconds))
    setMaxSeconds(f.max_seconds === null ? '' : String(f.max_seconds))
    exportDir(id, f.id).then(setOutDir, () => setOutDir(''))
  }

  async function start() {
    setError(null)
    try {
      setJob(
        await exportDataset(id, {
          format,
          out_dir: outDir,
          min_seconds: num(minSeconds),
          max_seconds: num(maxSeconds),
          sample_rate: num(rate),
          language: format === 'gpt_sovits' && language ? language : null,
        }),
      )
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const result = job?.state === 'done' ? job.result : null
  const skipped = result ? result.skipped.short + result.skipped.long + result.skipped.unreadable : 0
  return (
    <>
      <div className="tool-row">
        <div className="tool-info">
          <strong>{t.dsExportTitle}</strong>
          {result ? (
            <span className="muted hint">
              {t.dsExportedSummary(result.clips.toLocaleString(), formatDuration(result.seconds), result.dir)}
              {skipped > 0 &&
                ` ${t.dsExportSkipped(result.skipped.short, result.skipped.long, result.skipped.unreadable)}`}
            </span>
          ) : (
            <span className="muted hint">{chosen?.description ?? t.dsExportHint}</span>
          )}
        </div>
        <div className="tool-controls export-controls">
          <select
            value={format}
            onChange={(e) => {
              const f = formats.find((x) => x.id === e.target.value)
              if (f) choose(f)
            }}
            disabled={disabled || running}
            aria-label={t.dsExportFormat}
          >
            {formats.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
          <button type="button" className="ghost" onClick={start} disabled={disabled || running || !format}>
            {running ? t.dsExporting : t.dsExportButton}
          </button>
        </div>
      </div>
      <details className="tool-help export-options">
        <summary>{t.dsExportOptions}</summary>
        <div className="form-grid">
          <label className="field export-dir">
            <span>{t.dsExportFolder}</span>
            <input value={outDir} onChange={(e) => setOutDir(e.target.value)} disabled={running} spellCheck={false} />
          </label>
          <label className="field">
            <span>{t.dsExportMinSeconds}</span>
            <input
              type="number"
              min={0}
              step="any"
              value={minSeconds}
              onChange={(e) => setMinSeconds(e.target.value)}
              placeholder={t.noLimit}
              disabled={running}
            />
          </label>
          <label className="field">
            <span>{t.dsExportMaxSeconds}</span>
            <input
              type="number"
              min={0}
              step="any"
              value={maxSeconds}
              onChange={(e) => setMaxSeconds(e.target.value)}
              placeholder={t.noLimit}
              disabled={running}
            />
          </label>
          <label className="field">
            <span>{t.dsExportRate}</span>
            <select value={rate} onChange={(e) => setRate(e.target.value)} disabled={running}>
              <option value="">{t.keepRate}</option>
              {RATES.map((r) => (
                <option key={r} value={r}>
                  {(r / 1000).toLocaleString()} kHz
                </option>
              ))}
            </select>
          </label>
          {format === 'gpt_sovits' && (
            <label className="field">
              <span>{t.dsExportLanguage}</span>
              <select value={language} onChange={(e) => setLanguage(e.target.value)} disabled={running}>
                <option value="">{t.languageFromText}</option>
                {EXPORT_LANGUAGES.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </details>
      {job && <JobProgress job={job} what={t.dsExportWhat} />}
      {error && <p className="field-error">{error}</p>}
    </>
  )
}
