import type { ImportPreview } from '../api'
import { formatDuration, formatRates } from '../format'
import { useT } from '../i18n'

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

/** What a scanned source holds: counts, problems, a few examples, and what adding it brings. */
export function SourcePreview({ preview }: { preview: ImportPreview }) {
  const t = useT()
  const withAudio = preview.found + preview.untranscribed
  const recordings = preview.long_recordings
  return (
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
          <p className="muted hint">{t.importBrings(withAudio - recordings, recordings)}</p>
        </>
      )}
    </div>
  )
}
