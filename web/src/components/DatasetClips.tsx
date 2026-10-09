import { Fragment, useEffect, useRef, useState } from 'react'
import { datasetAudioUrl, dropFlagged, getDatasetRows, TRIMMABLE_FLAGS, trimFlagged } from '../api'
import type { CheckFlag, CheckSummary, ClipPage, DatasetSource, FlagFilter } from '../api'
import { ALIGNER_LABEL, FLAG_LABEL } from '../format'
import { ClipView } from './ClipView'
import { Waveform } from './Waveform'

interface Props {
  source: DatasetSource
}

const PAGE = 50
const SEARCH_DELAY_MS = 250

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path

/** Browse a dataset's clips: play, read the transcript, spot ASR-made texts and flagged clips. */
export function DatasetClips({ source }: Props) {
  // Effects follow the source's value, not the object passed in each render.
  const sourceKey = JSON.stringify(source)
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [flag, setFlag] = useState<FlagFilter>('')
  const [worstFirst, setWorstFirst] = useState(false)
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<ClipPage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null) // outcome of a bulk action
  const [playing, setPlaying] = useState<number | null>(null)
  const [open, setOpen] = useState<number | null>(null) // clip shown with its words
  // Bumped after an edit or drop, so the page reloads.
  const [version, setVersion] = useState(0)
  // Clips edited under the current filter: they stay listed, so the result can be seen.
  const [stay, setStay] = useState<number[]>([])
  const audio = useRef<HTMLAudioElement>(null)
  const loaded = useRef<number | null>(null) // clip index the player holds
  const openPos = useRef(0) // the open clip's row on the page

  // Debounce typing into the search box; a new search starts at page 1.
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(query)
      setOffset(0)
      setStay([])
    }, SEARCH_DELAY_MS)
    return () => clearTimeout(timer)
  }, [query])

  const stayKey = stay.join(',')
  useEffect(() => {
    let cancelled = false
    const keep = stayKey ? stayKey.split(',').map(Number) : []
    getDatasetRows(JSON.parse(sourceKey), offset, PAGE, { query: search, flag, worstFirst, keep }).then(
      (p) => {
        if (cancelled) return
        setPage(p)
        setError(null)
        // A fixed clip that left a filtered list: open the one now in its place, to review on.
        setOpen((o) => (o === null || p.rows.some((r) => r.index === o) ? o : (p.rows[openPos.current]?.index ?? null)))
      },
      (e: Error) => !cancelled && setError(e.message),
    )
    return () => {
      cancelled = true
    }
  }, [sourceKey, offset, search, flag, worstFirst, stayKey, version])

  // `at`: start from this second (a click on the waveform); otherwise play / pause.
  function play(index: number, at?: number) {
    const el = audio.current
    if (!el) return
    if (at === undefined && playing === index && !el.paused) {
      el.pause()
      return
    }
    if (loaded.current !== index) {
      el.src = datasetAudioUrl(source, index, version)
      loaded.current = index
    }
    if (at !== undefined) el.currentTime = at
    el.play().then(
      () => setPlaying(index),
      (e: Error) => setError(e.message),
    )
  }

  async function dropAll(f: CheckFlag, count: number) {
    if (!('id' in source)) return
    const label = FLAG_LABEL[f][0]
    if (!window.confirm(`Drop all ${count} clips flagged “${label}”? They stay listed and can be restored one by one.`))
      return
    try {
      await dropFlagged(source.id, f)
      changed()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function trimAll(f: CheckFlag, count: number) {
    if (!('id' in source)) return
    const label = FLAG_LABEL[f][0]
    const ask =
      `Trim the edges of the ${count} clips flagged “${label}”? Each keeps 0.15 s around its transcribed speech; ` +
      'clips also flagged Mismatch or Missing words?, or starting / ending with a filler the aligner could not place, are ' +
      'left for review. The originals are kept in the ' +
      "dataset's originals folder."
    if (!window.confirm(ask)) return
    try {
      const r = await trimFlagged(source.id, f)
      const notes = [
        r.review > 0 && `${r.review} left for review`,
        r.skipped > 0 && `${r.skipped} can't be rewritten in place`,
      ].filter(Boolean)
      setNotice(`Trimmed ${r.trimmed} clips${notes.length ? ` (${notes.join(', ')})` : ''}.`)
      changed()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  // After an edit, drop, or trim: reload the page and fetch audio afresh. An edited clip (`index`)
  // stays listed even if it no longer matches the filter.
  function changed(index?: number) {
    if (index !== undefined) setStay((s) => (s.includes(index) ? s : [...s, index]))
    loaded.current = null
    setVersion((v) => v + 1)
  }

  function toggle(index: number) {
    openPos.current = page?.rows.findIndex((r) => r.index === index) ?? 0
    setOpen((o) => (o === index ? null : index))
  }

  function filter(apply: () => void) {
    apply()
    setOffset(0)
    setStay([])
  }

  const total = page?.total ?? 0
  const asrOnPage = page?.rows.filter((r) => r.asr).length ?? 0

  return (
    <div className="clips">
      {page && (page.checks || page.dropped > 0) && (
        <CheckBar
          checks={page.checks}
          dropped={page.dropped}
          flag={flag}
          worstFirst={worstFirst}
          onFlag={(v) => filter(() => setFlag(v))}
          onWorstFirst={(v) => filter(() => setWorstFirst(v))}
          onDropAll={'id' in source ? dropAll : undefined}
          onTrimAll={'id' in source ? trimAll : undefined}
        />
      )}
      {notice && <p className="muted hint">{notice}</p>}
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
                <Fragment key={row.index}>
                  <tr
                    className={[
                      row.index === playing && 'playing',
                      row.index === open && 'open',
                      row.dropped && 'dropped',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                  >
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
                    <td className="clip-wave">
                      {row.peaks && row.seconds !== null && (
                        <Waveform
                          peaks={row.peaks}
                          seconds={row.seconds}
                          speech={row.speech}
                          // Greyed: what Trim all would cut from this clip.
                          trim={row.flags.some((f) => TRIMMABLE_FLAGS.includes(f)) ? row.trim : null}
                          audio={row.index === playing ? audio.current : null}
                          onSeek={(at) => play(row.index, at)}
                        />
                      )}
                    </td>
                    <td className="clip-text" onClick={() => toggle(row.index)}>
                      {row.text}
                      {row.dropped && <span className="tag subtle">Dropped</span>}
                      {row.asr && <span className="tag subtle">ASR</span>}
                      {row.flags.map((f) => (
                        <FlagTag key={f} flag={f} score={row.score} />
                      ))}
                    </td>
                    <td className="clip-seconds">
                      <button
                        type="button"
                        className="ghost small"
                        onClick={() => toggle(row.index)}
                        aria-expanded={row.index === open}
                        title="Words and a larger waveform"
                      >
                        {row.seconds === null ? 'missing' : `${row.seconds.toFixed(1)} s`}{' '}
                        {row.index === open ? '▴' : '▾'}
                      </button>
                    </td>
                  </tr>
                  {row.index === open && (
                    <tr className="clip-detail">
                      <td colSpan={4}>
                        <ClipView
                          source={source}
                          index={row.index}
                          audio={row.index === playing ? audio.current : null}
                          onSeek={(at) => play(row.index, at)}
                          onChanged={() => changed(row.index)}
                        />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {page && page.rows.length === 0 && <p className="muted empty">No clips match.</p>}

      {total > PAGE && (
        <div className="pager">
          <button
            type="button"
            className="ghost small"
            onClick={() => setOffset(offset - PAGE)}
            disabled={offset === 0}
          >
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

function FlagTag({ flag, score }: { flag: CheckFlag; score: number | null }) {
  const [label, meaning] = FLAG_LABEL[flag]
  const detail = flag === 'mismatch' && score !== null ? ` (score ${score.toPrecision(2)})` : ''
  return (
    <span className="tag warn" title={meaning + detail}>
      {label}
    </span>
  )
}

interface CheckBarProps {
  checks: CheckSummary | null
  dropped: number
  flag: FlagFilter
  worstFirst: boolean
  onFlag: (value: FlagFilter) => void
  onWorstFirst: (value: boolean) => void
  // Drop every clip with one flag (a dataset, not a run's copy).
  onDropAll?: (flag: CheckFlag, count: number) => void
  // Trim every clip with one flag (TRIMMABLE_FLAGS only).
  onTrimAll?: (flag: CheckFlag, count: number) => void
}

/** What the alignment check found, with the review filters. */
function CheckBar({ checks, dropped, flag, worstFirst, onFlag, onWorstFirst, onDropAll, onTrimAll }: CheckBarProps) {
  const counts = checks ? (Object.keys(FLAG_LABEL) as CheckFlag[]).filter((f) => checks.flags[f]) : []
  const one = flag !== '' && flag !== 'any' && flag !== 'dropped' ? flag : null // a single flag shown
  const oneCount = one ? (checks?.flags[one] ?? 0) : 0
  return (
    <div className="check-bar">
      <p>
        {checks ? (
          <>
            <strong>{checks.flagged.toLocaleString()}</strong> of {checks.checked.toLocaleString()} checked clips
            flagged
            {checks.aligner && <span className="muted"> · {ALIGNER_LABEL[checks.aligner]}</span>}
          </>
        ) : (
          'Not checked yet'
        )}
        {dropped > 0 && <span className="muted"> · {dropped.toLocaleString()} dropped</span>}
      </p>
      <div className="check-filters">
        <select value={flag} onChange={(e) => onFlag(e.target.value as FlagFilter)} aria-label="Clips shown">
          <option value="">All clips</option>
          {checks && <option value="any">Any flag ({checks.flagged.toLocaleString()})</option>}
          {counts.map((f) => (
            <option key={f} value={f}>
              {FLAG_LABEL[f][0]} ({checks!.flags[f]!.toLocaleString()})
            </option>
          ))}
          {dropped > 0 && <option value="dropped">Dropped ({dropped.toLocaleString()})</option>}
        </select>
        {checks && (
          <label className="check">
            <input type="checkbox" checked={worstFirst} onChange={(e) => onWorstFirst(e.target.checked)} /> Worst match
            first
          </label>
        )}
        {one && onTrimAll && oneCount > 0 && TRIMMABLE_FLAGS.includes(one) && (
          <button type="button" className="ghost small" onClick={() => onTrimAll(one, oneCount)}>
            Trim all {oneCount.toLocaleString()}
          </button>
        )}
        {one && onDropAll && oneCount > 0 && (
          <button type="button" className="ghost small" onClick={() => onDropAll(one, oneCount)}>
            Drop all {oneCount.toLocaleString()}
          </button>
        )}
      </div>
    </div>
  )
}
