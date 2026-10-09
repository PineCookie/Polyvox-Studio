import { useEffect, useRef } from 'react'
import type { KeyboardEvent, MouseEvent, PointerEvent } from 'react'

// The part of a clip to keep when trimming, set with two handles.
export interface KeepRange {
  value: [number, number] // seconds
  onChange: (value: [number, number]) => void
  // A handle was let go (0: start, 1: end), e.g. to play from there.
  onDone?: (handle: 0 | 1, value: [number, number]) => void
}

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
  // With handles: what trimming keeps; the rest is greyed.
  keep?: KeepRange
  large?: boolean
}

const MAX_GAIN = 8 // quiet clips are scaled up for visibility, near-silence stays flat
const MIN_KEEP_S = 0.3 // matches the server's minimum
const KEY_STEP_S = 0.05

/** A clip's waveform outline; click to play from a point. */
export function Waveform({ peaks, seconds, speech, audio, onSeek, marks = [], keep, large = false }: Props) {
  const playhead = useRef<SVGLineElement>(null)
  const wrap = useRef<HTMLDivElement>(null)
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
  const cuts: [number, number][] = keep
    ? (
        [
          [0, keep.value[0]],
          [keep.value[1], seconds],
        ] as [number, number][]
      ).filter(([a, b]) => b - a > 0.005)
    : []

  function seek(e: MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    onSeek(Math.max(0, ((e.clientX - box.left) / box.width) * seconds))
  }

  // One handle moved to `at` seconds, kept apart from the other.
  function moved(handle: 0 | 1, at: number): [number, number] {
    const [start, end] = keep!.value
    return handle === 0
      ? [Math.min(Math.max(0, at), end - MIN_KEEP_S), end]
      : [start, Math.max(Math.min(seconds, at), start + MIN_KEEP_S)]
  }

  function drag(handle: 0 | 1, e: PointerEvent<HTMLDivElement>) {
    if (!keep || !(e.buttons & 1) || !wrap.current) return
    const box = wrap.current.getBoundingClientRect()
    keep.onChange(moved(handle, ((e.clientX - box.left) / box.width) * seconds))
  }

  function nudge(handle: 0 | 1, e: KeyboardEvent<HTMLDivElement>) {
    const step = e.key === 'ArrowLeft' ? -KEY_STEP_S : e.key === 'ArrowRight' ? KEY_STEP_S : 0
    if (!keep || !step) return
    e.preventDefault()
    const value = moved(handle, keep.value[handle] + step)
    keep.onChange(value)
    keep.onDone?.(handle, value)
  }

  const svg = (
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
  if (!keep) return svg

  return (
    <div className="waveform-wrap" ref={wrap}>
      {svg}
      {([0, 1] as const).map((handle) => (
        <div
          key={handle}
          className="trim-handle"
          style={{ left: `${(keep.value[handle] / seconds) * 100}%` }}
          role="slider"
          tabIndex={0}
          aria-label={handle === 0 ? 'Keep from' : 'Keep until'}
          aria-valuemin={0}
          aria-valuemax={seconds}
          aria-valuenow={keep.value[handle]}
          aria-valuetext={`${keep.value[handle].toFixed(2)} s`}
          title={`${handle === 0 ? 'Keep from' : 'Keep until'} ${keep.value[handle].toFixed(2)} s: drag, or use the arrow keys`}
          onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
          onPointerMove={(e) => drag(handle, e)}
          onPointerUp={() => keep.onDone?.(handle, keep.value)}
          onKeyDown={(e) => nudge(handle, e)}
        />
      ))}
    </div>
  )
}
