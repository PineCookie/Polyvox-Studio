// Shared by the charts (LossChart, SimilarityChart).
import { useLayoutEffect, useRef, useState } from 'react'

// Height follows the width (wider charts get taller), within these bounds.
const MIN_HEIGHT = 180
const MAX_HEIGHT = 340
export const chartHeight = (width: number) => Math.round(Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, width * 0.3)))
export const MARGIN = { top: 10, right: 12, bottom: 24, left: 48 }

export function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min || Math.abs(max) || 1
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw) ?? raw
  const ticks = []
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) ticks.push(Number(t.toPrecision(12)))
  return ticks
}

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}
