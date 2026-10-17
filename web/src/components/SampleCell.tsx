import { useT } from '../i18n'
import { usePeaks } from '../peaks'
import { Waveform } from './Waveform'

/** Play button and waveform of one short clip (click the waveform to play from there); the
 * list it is in shares one <audio> (`audio`: that element while this clip plays). */
export function SampleCell({
  url,
  playing,
  audio,
  score,
  onPlay,
}: {
  url: string
  playing: boolean
  audio: HTMLAudioElement | null
  score?: number
  onPlay: (at?: number) => void
}) {
  const t = useT()
  const shape = usePeaks(url)
  return (
    <div className="sample-cell">
      <button
        type="button"
        className="ghost small"
        onClick={() => onPlay()}
        aria-label={playing ? t.pause : t.playLabel}
      >
        {playing ? '❚❚' : '▶'}
      </button>
      {shape ? (
        <Waveform peaks={shape.peaks} seconds={shape.seconds} speech={null} audio={audio} onSeek={onPlay} />
      ) : (
        <span className="sample-wave-placeholder" />
      )}
      {shape && <span className="muted hint">{t.seconds(shape.seconds.toFixed(1))}</span>}
      {score !== undefined && (
        <span className="muted hint sample-score" title={t.similarityCellTitle}>
          {score.toFixed(2)}
        </span>
      )}
    </div>
  )
}
