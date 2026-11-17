import { useRef, useState } from 'react'
import { formatClock } from '../format'
import { useT } from '../i18n'
import { usePeaks } from '../peaks'
import { PauseIcon, PlayIcon } from './icons'
import { Waveform } from './Waveform'

/** A player in the app's own look: play / pause, the waveform (click to play from a point, a
 * playhead while it plays) and the time. */
export function AudioPlayer({ url }: { url: string }) {
  const t = useT()
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [at, setAt] = useState(0)
  const shape = usePeaks(url)

  function toggle() {
    const el = audio.current
    if (!el) return
    if (el.paused) el.play().catch(() => {})
    else el.pause()
  }

  function seek(seconds: number) {
    const el = audio.current
    if (!el) return
    el.currentTime = seconds
    el.play().catch(() => {})
  }

  return (
    <div className="audio-player">
      <button
        type="button"
        className={`play-button${playing ? ' playing' : ''}`}
        onClick={toggle}
        aria-label={playing ? t.pause : t.playLabel}
      >
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>
      {shape ? (
        <Waveform
          peaks={shape.peaks}
          seconds={shape.seconds}
          speech={null}
          audio={playing ? audio.current : null}
          onSeek={seek}
          ruler={false}
        />
      ) : (
        <span className="waveform-placeholder" />
      )}
      <span className="audio-time">
        {playing || at > 0 ? `${formatClock(at)} / ` : ''}
        {shape ? formatClock(shape.seconds) : '–:––'}
      </span>
      <audio
        ref={audio}
        src={url}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false)
          setAt(0)
        }}
        onTimeUpdate={(e) => setAt(e.currentTarget.currentTime)}
        hidden
      />
    </div>
  )
}
