import { useEffect, useRef, useState } from 'react'
import { datasetAudioUrl, getDatasetRows } from '../api'
import type { ClipRow } from '../api'

interface Props {
  datasetId: string
}

const PAGE = 50
const SEARCH_DELAY_MS = 250

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

/** Browse a dataset's clips: play, read the transcript, spot ASR-made texts. */
export function DatasetClips({ datasetId }: Props) {
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<{ total: number; rows: ClipRow[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState<number | null>(null)
  const audio = useRef<HTMLAudioElement>(null)

  // Debounce typing into the search box; a new search starts at page 1.
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(query)
      setOffset(0)
    }, SEARCH_DELAY_MS)
    return () => clearTimeout(timer)
  }, [query])

  useEffect(() => {
    let cancelled = false
    getDatasetRows(datasetId, offset, PAGE, search).then(
      (p) => !cancelled && (setPage(p), setError(null)),
      (e: Error) => !cancelled && setError(e.message),
    )
    return () => {
      cancelled = true
    }
  }, [datasetId, offset, search])

  function play(index: number) {
    const el = audio.current
    if (!el) return
    if (playing === index && !el.paused) {
      el.pause()
      return
    }
    el.src = datasetAudioUrl(datasetId, index)
    el.play().then(
      () => setPlaying(index),
      (e: Error) => setError(e.message),
    )
  }

  const total = page?.total ?? 0
  const asrOnPage = page?.rows.filter((r) => r.asr).length ?? 0

  return (
    <div className="clips">
      <div className="clips-head">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search text or file name"
          aria-label="Search clips"
        />
        <span className="muted hint">
          {page ? `${total.toLocaleString()} clips` : 'Loading…'}
          {asrOnPage > 0 && ' · ASR = transcript made by speech recognition'}
        </span>
      </div>
      {error && <p className="field-error">{error}</p>}

      {page && page.rows.length > 0 && (
        <div className="table-wrap">
          <table className="metrics clip-table">
            <tbody>
              {page.rows.map((row) => (
                <tr key={row.index} className={row.index === playing ? 'playing' : ''}>
                  <td className="clip-play">
                    <button
                      type="button"
                      className="ghost small"
                      onClick={() => play(row.index)}
                      disabled={row.seconds === null}
                      aria-label={`Play ${fileName(row.file)}`}
                      title={row.file}
                    >
                      {row.index === playing ? '❚❚' : '▶'}
                    </button>
                  </td>
                  <td className="clip-text">
                    {row.text}
                    {row.asr && <span className="tag subtle">ASR</span>}
                  </td>
                  <td className="clip-seconds">{row.seconds === null ? 'missing' : `${row.seconds.toFixed(1)} s`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {page && page.rows.length === 0 && <p className="muted empty">No clips match.</p>}

      {total > PAGE && (
        <div className="pager">
          <button type="button" className="ghost small" onClick={() => setOffset(offset - PAGE)} disabled={offset === 0}>
            Previous
          </button>
          <span className="muted hint">
            {offset + 1}–{Math.min(offset + PAGE, total)} of {total.toLocaleString()}
          </span>
          <button
            type="button"
            className="ghost small"
            onClick={() => setOffset(offset + PAGE)}
            disabled={offset + PAGE >= total}
          >
            Next
          </button>
        </div>
      )}
      <audio ref={audio} onEnded={() => setPlaying(null)} onPause={() => setPlaying(null)} hidden />
    </div>
  )
}
