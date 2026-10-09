import { useRef, useState } from 'react'
import { usePeaks } from '../peaks'
import { Waveform } from './Waveform'

/** An audio player with its waveform: playhead while playing, click to play from a point. */
export function AudioPlayer({ url }: { url: string }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const shape = usePeaks(url)

  function seek(at: number) {
    const el = audio.current
    if (!el) return
    el.currentTime = at
    el.play().catch(() => {})
  }

  return (
    <div className="audio-player">
      {shape && (
        <Waveform
          peaks={shape.peaks}
          seconds={shape.seconds}
          speech={null}
          audio={playing ? audio.current : null}
          onSeek={seek}
          large
        />
      )}
      <audio
        ref={audio}
        controls
        src={url}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
    </div>
  )
}
