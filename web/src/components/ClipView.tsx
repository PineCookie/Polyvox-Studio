import { useEffect, useRef, useState } from 'react'
import { getDatasetClip, hearClipPart, setClipDropped, setClipText, TRIMMABLE_FLAGS, trimClip } from '../api'
import type { ClipDetail, DatasetSource, Heard } from '../api'
import { ALIGNER_LABEL } from '../format'
import { Waveform } from './Waveform'

// Untranscribed sound at one end of a clip, in seconds.
interface Edge {
  side: 'before' | 'after'
  start: number
  end: number
}

interface Props {
  source: DatasetSource
  index: number
  // The player, while it plays this clip.
  audio: HTMLAudioElement | null
  onSeek: (seconds: number) => void
  // After the text was edited or the clip dropped / restored.
  onChanged?: () => void
}

/** A closer look at one clip: a larger waveform with each checked word where the aligner put it,
 * what ASR heard, and (for a dataset, not a run's copy) editing its text or dropping it. */
export function ClipView({ source, index, audio, onSeek, onChanged }: Props) {
  const sourceKey = JSON.stringify(source)
  const [clip, setClip] = useState<ClipDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState<string | null>(null) // the text being edited
  const [saving, setSaving] = useState(false)
  const [busy, setBusy] = useState<'trim' | Edge['side'] | null>(null)
  const track = useRef<HTMLDivElement>(null)
  const datasetId = 'id' in source ? source.id : null // a run's copy is read-only

  async function save(text: string) {
    if (!datasetId) return
    setSaving(true)
    setError(null)
    try {
      setClip(await setClipText(datasetId, index, text))
      setDraft(null)
      onChanged?.()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function trimEdges() {
    if (!datasetId) return
    setBusy('trim')
    setError(null)
    try {
      setClip(await trimClip(datasetId, index))
      onChanged?.()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  // Put what ASR hears in an edge sound into the editor, on that side of the text.
  async function addHeard(edge: Edge) {
    if (!datasetId || !clip) return
    setBusy(edge.side)
    setError(null)
    try {
      const heard = (await hearClipPart(datasetId, index, edge.start, edge.end)).trim()
      if (!heard) setError('ASR heard nothing there; type it in, or trim it off.')
      else setDraft(edge.side === 'before' ? heard + clip.text : clip.text + heard)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  async function drop(dropped: boolean) {
    if (!datasetId || !clip) return
    try {
      await setClipDropped(datasetId, index, dropped)
      setClip({ ...clip, dropped })
      onChanged?.()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    let cancelled = false
    getDatasetClip(JSON.parse(sourceKey), index).then(
      (c) => !cancelled && setClip(c),
      (e: Error) => !cancelled && setError(e.message),
    )
    return () => {
      cancelled = true
    }
  }, [sourceKey, index])

  // Light up the word being spoken (on the DOM: no re-render per frame).
  const words = clip?.words
  useEffect(() => {
    const el = track.current
    if (!audio || !el || !words?.length) return
    const buttons = el.querySelectorAll<HTMLElement>('.word')
    let frame = 0
    let current = -1
    const tick = () => {
      // The last word begun, until the next begins (CTC marks only onsets, so spans are thin).
      const t = audio.currentTime
      let started = -1
      while (started + 1 < words.length && words[started + 1].start <= t) started++
      const next = started >= 0 && t < (words[started + 1]?.start ?? words[started].end + 0.3) ? started : -1
      if (next !== current) {
        buttons[current]?.classList.remove('current')
        buttons[next]?.classList.add('current')
        current = next
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      buttons[current]?.classList.remove('current')
    }
  }, [audio, words])

  if (!clip && error) return <p className="field-error">{error}</p>
  if (!clip) return <p className="muted hint">Loading…</p>
  if (!clip.peaks || clip.seconds === null) return <p className="muted hint">The audio file can't be read.</p>
  const seconds = clip.seconds
  const weak = clip.words.filter((w) => w.weak)
  const edges: Edge[] = [
    ...(clip.sound_before !== null ? [{ side: 'before' as const, start: 0, end: clip.sound_before }] : []),
    ...(clip.sound_after !== null ? [{ side: 'after' as const, start: clip.sound_after, end: seconds }] : []),
  ]
  // Trimming is offered for the flags it fixes; it would shorten most clips' silence a little.
  const trim = clip.trim && clip.flags.some((f) => TRIMMABLE_FLAGS.includes(f)) ? clip.trim : null
  const cuts: [number, number][] = trim
    ? (
        [
          [0, trim[0]],
          [trim[1], seconds],
        ] as [number, number][]
      ).filter(([a, b]) => b - a > 0.01)
    : []

  return (
    <div className="clip-view">
      {datasetId && (
        <div className="clip-edit">
          {draft === null ? (
            <>
              <p className="clip-edit-text">{clip.text}</p>
              <button type="button" className="ghost small" onClick={() => setDraft(clip.text)}>
                Edit
              </button>
              <button type="button" className="ghost small" onClick={() => drop(!clip.dropped)}>
                {clip.dropped ? 'Restore' : 'Drop'}
              </button>
            </>
          ) : (
            <form
              className="clip-edit-form"
              onSubmit={(e) => {
                e.preventDefault()
                save(draft)
              }}
            >
              <input value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Transcript" autoFocus />
              <button type="submit" className="primary small" disabled={saving || !draft.trim()}>
                {saving ? 'Saving…' : 'Save'}
              </button>
              <button type="button" className="ghost small" onClick={() => setDraft(null)} disabled={saving}>
                Cancel
              </button>
            </form>
          )}
        </div>
      )}
      {clip.dropped && <p className="muted hint">Dropped: left out of training runs started from now on.</p>}
      {error && <p className="field-error">{error}</p>}
      <Waveform
        peaks={clip.peaks}
        seconds={seconds}
        speech={clip.speech}
        audio={audio}
        onSeek={onSeek}
        marks={weak.map((w) => [w.start, w.end])}
        cuts={cuts}
        large
      />
      {clip.words.length > 0 && (
        <div className="word-track" ref={track}>
          {clip.words.map((w, i) => (
            <button
              key={i}
              type="button"
              className={`word ${w.weak ? 'weak' : ''} ${i % 2 ? 'lower' : ''}`}
              style={{ left: `${(w.start / seconds) * 100}%`, width: `${((w.end - w.start) / seconds) * 100}%` }}
              title={`${w.text} · ${w.start.toFixed(2)}–${w.end.toFixed(2)} s · confidence ${w.confidence.toPrecision(2)}`}
              onClick={() => onSeek(w.start)}
            >
              {w.text}
            </button>
          ))}
        </div>
      )}
      <p className="muted hint">
        {clip.words.length > 0 ? (
          <>
            Shaded: the speech the text covers. Words sit where {ALIGNER_LABEL[clip.aligner ?? 'qwen3']} placed them
            {weak.length > 0 && (
              <>
                ; <span className="weak-legend">{weak.length} unsure</span> (likely not what was said)
              </>
            )}
            . Click a word or the waveform to play from there.
          </>
        ) : clip.speech ? (
          'Checked before word timings were kept: check the dataset again to see each word.'
        ) : (
          'Not checked yet: check the dataset to see where each word is.'
        )}
      </p>
      {datasetId && (edges.length > 0 || trim) && (
        <div className="edge-sound">
          {edges.map((edge) => (
            <span key={edge.side} className="edge-sound-part">
              <span className="muted">Sound {edge.side} the words, not in the text:</span>
              <button type="button" className="ghost small" onClick={() => onSeek(edge.start)}>
                ▶ Play
              </button>
              <button
                type="button"
                className="ghost small"
                onClick={() => addHeard(edge)}
                disabled={busy !== null}
                title="Transcribe just this sound and add it to the text in the editor"
              >
                {busy === edge.side ? 'Listening…' : 'Add to text'}
              </button>
            </span>
          ))}
          {trim && (
            <button
              type="button"
              className="ghost small"
              onClick={trimEdges}
              disabled={busy !== null}
              title={`Keep ${trim[0].toFixed(2)}–${trim[1].toFixed(2)} s: 0.15 s around the transcribed speech. The original is kept in the dataset's originals folder.`}
            >
              {busy === 'trim' ? 'Trimming…' : 'Trim edges'}
            </button>
          )}
          {cuts.length > 0 && <span className="muted cut-legend">grey: what trimming cuts</span>}
        </div>
      )}
      {clip.heard && <HeardLine heard={clip.heard} onUse={datasetId ? () => setDraft(clip.heard!.text) : undefined} />}
    </div>
  )
}

/** What ASR heard, with the stretches that differ from the transcript marked. */
function HeardLine({ heard, onUse }: { heard: Heard; onUse?: () => void }) {
  const same = heard.diff.every(([op]) => op === 'equal')
  return (
    <div className="heard">
      <span className="muted">ASR heard:</span>{' '}
      {heard.diff.map(([op, mine, theirs], i) =>
        op === 'equal' ? (
          <span key={i}>{theirs} </span>
        ) : op === 'missing' ? (
          <del key={i} title="In the transcript, not heard">
            {mine}{' '}
          </del>
        ) : (
          <ins key={i} className={op} title={op === 'extra' ? 'Heard, not in the transcript' : `Transcript: ${mine}`}>
            {theirs}{' '}
          </ins>
        ),
      )}
      {same ? (
        <span className="muted">(same as the transcript)</span>
      ) : (
        onUse && (
          <button type="button" className="ghost small" onClick={onUse} title="Put ASR's text in the editor">
            Use ASR text
          </button>
        )
      )}
    </div>
  )
}
