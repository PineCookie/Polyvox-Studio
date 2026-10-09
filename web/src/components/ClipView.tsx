import { useEffect, useRef, useState } from 'react'
import { getDatasetClip } from '../api'
import type { ClipDetail, DatasetSource } from '../api'
import { ALIGNER_LABEL } from '../format'
import { Waveform } from './Waveform'

interface Props {
  source: DatasetSource
  index: number
  // The player, while it plays this clip.
  audio: HTMLAudioElement | null
  onSeek: (seconds: number) => void
}

/** A closer look at one clip: a larger waveform with each checked word where the aligner put it. */
export function ClipView({ source, index, audio, onSeek }: Props) {
  const sourceKey = JSON.stringify(source)
  const [clip, setClip] = useState<ClipDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const track = useRef<HTMLDivElement>(null)

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

  if (error) return <p className="field-error">{error}</p>
  if (!clip) return <p className="muted hint">Loading…</p>
  if (!clip.peaks || clip.seconds === null) return <p className="muted hint">The audio file can't be read.</p>
  const seconds = clip.seconds
  const weak = clip.words.filter((w) => w.weak)

  return (
    <div className="clip-view">
      <Waveform
        peaks={clip.peaks}
        seconds={seconds}
        speech={clip.speech}
        audio={audio}
        onSeek={onSeek}
        marks={weak.map((w) => [w.start, w.end])}
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
    </div>
  )
}
