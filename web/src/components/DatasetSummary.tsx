import type { DatasetReport } from '../api'
import { formatDuration, formatRates } from '../format'
import { useT } from '../i18n'

export function DatasetSummary({ report, error }: { report: DatasetReport | null; error: string | null }) {
  const t = useT()
  if (error) return <p className="field-error">{error}</p>
  if (!report) return <p className="muted hint">{t.checkingAudio}</p>

  return (
    <div className="dataset-summary">
      <p>
        <strong>{report.valid}</strong>
        {report.valid !== report.samples && <span className="muted">{t.ofSamples(report.samples)}</span>}
        {t.clipsWord}
        {report.dropped > 0 && <span className="muted">{t.droppedNotCounted(report.dropped)}</span>}
        {report.valid > 0 && (
          <>
            {' · '}
            {t.totalMedian(formatDuration(report.total_seconds), report.median_seconds!.toFixed(1))}
            <span title={t.resampledTitle}>{formatRates(report.sample_rates)}</span>
          </>
        )}
      </p>
      {report.errors.length > 0 && (
        <ul className="issues errors">
          {report.errors.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      )}
      {report.warnings.length > 0 && (
        <ul className="issues">
          {report.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
