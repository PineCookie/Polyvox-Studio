import { useEffect, useRef } from 'react'
import type { MouseEvent } from 'react'

interface Props {
  peaks: number[]
  seconds: number
  // Where the aligned words are; shaded when the clip was checked.
  speech: [number, number] | null
  // The player, while it plays this clip: drives the playhead.
  audio: HTMLAudioElement | null
  onSeek: (seconds: number) => void
  // Spans to mark (unsure words), in seconds.
  marks?: [number, number][]
  // Spans trimming would cut away, in seconds.
  cuts?: [number, number][]
  large?: boolean
}

const MAX_GAIN = 8 // quiet clips are scaled up for visibility, near-silence stays flat

/** A clip's waveform outline; click to play from a point. */
export function Waveform({ peaks, seconds, speech, audio, onSeek, marks = [], cuts = [], large = false }: Props) {
  const playhead = useRef<SVGLineElement>(null)
  const width = peaks.length

  // Moved directly on the DOM: a re-render per frame would redraw the whole table.
  useEffect(() => {
    const line = playhead.current
    if (!audio || !line) return
    let frame = 0
    const tick = () => {
      const x = String(Math.min(1, audio.currentTime / seconds) * width)
      line.setAttribute('x1', x)
      line.setAttribute('x2', x)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [audio, seconds, width])

  const gain = Math.min(MAX_GAIN, 1 / Math.max(...peaks, 1e-6))
  const top = peaks.map((p, i) => `${i},${1 - Math.min(1, p * gain) * 0.95}`)
  const bottom = peaks.map((p, i) => `${i},${1 + Math.min(1, p * gain) * 0.95}`).reverse()
  const outline = `M0,1 L${top.join(' ')} L${width},1 L${bottom.join(' ')} Z`

  function seek(e: MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    onSeek(Math.max(0, ((e.clientX - box.left) / box.width) * seconds))
  }

  return (
    <svg
      className={large ? 'waveform large' : 'waveform'}
      viewBox={`0 0 ${width} 2`}
      preserveAspectRatio="none"
      onClick={seek}
      role="img"
      aria-label="Waveform; click to play from a point"
    >
      {speech && (
        <rect
          className="waveform-speech"
          x={(speech[0] / seconds) * width}
          y={0}
          width={((speech[1] - speech[0]) / seconds) * width}
          height={2}
        />
      )}
      {marks.map(([start, end]) => (
        <rect
          key={start}
          className="waveform-mark"
          x={(start / seconds) * width}
          y={0}
          width={Math.max(((end - start) / seconds) * width, width / 400)}
          height={2}
        />
      ))}
      {cuts.map(([start, end]) => (
        <rect
          key={`cut-${start}`}
          className="waveform-cut"
          x={(start / seconds) * width}
          y={0}
          width={((end - start) / seconds) * width}
          height={2}
        />
      ))}
      <path className="waveform-outline" d={outline} />
      {audio && <line ref={playhead} className="waveform-playhead" x1={0} x2={0} y1={0} y2={2} />}
    </svg>
  )
}
