import type { DatasetReport } from '../api'
import { formatDuration, formatRates } from '../format'

export function DatasetSummary({ report, error }: { report: DatasetReport | null; error: string | null }) {
  if (error) return <p className="field-error">{error}</p>
  if (!report) return <p className="muted hint">Checking audio files…</p>

  return (
    <div className="dataset-summary">
      <p>
        <strong>{report.valid}</strong>
        {report.valid !== report.samples && <span className="muted"> of {report.samples}</span>} clips
        {report.valid > 0 && (
          <>
            {' · '}
            {formatDuration(report.total_seconds)} total · median {report.median_seconds!.toFixed(1)} s ·{' '}
            <span title="Resampled to the model's rate during training">{formatRates(report.sample_rates)}</span>
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
