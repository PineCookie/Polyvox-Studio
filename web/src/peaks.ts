// Waveform outlines of audio the browser already holds (generated clips, uploads).
import { useEffect, useState } from 'react'

export interface Peaks {
  peaks: number[] // peak level 0..1 per equal slice
  seconds: number
}

const BINS = 600
const cache = new Map<string, Promise<Peaks>>()
let context: AudioContext | null = null

async function compute(url: string): Promise<Peaks> {
  const data = await (await fetch(url)).arrayBuffer()
  context ??= new AudioContext()
  const audio = await context.decodeAudioData(data)
  const channels = Array.from({ length: audio.numberOfChannels }, (_, c) => audio.getChannelData(c))
  const size = audio.length / BINS
  const peaks = Array.from({ length: BINS }, (_, b) => {
    let max = 0
    for (const ch of channels)
      for (let i = Math.floor(b * size); i < Math.floor((b + 1) * size); i++) max = Math.max(max, Math.abs(ch[i]))
    return max
  })
  return { peaks, seconds: audio.duration }
}

/** Peaks of the audio at `url` (an object URL or same-origin path); null while decoding or if it fails. */
export function usePeaks(url: string | null): Peaks | null {
  const [result, setResult] = useState<{ url: string; peaks: Peaks } | null>(null)
  useEffect(() => {
    if (!url) return
    let cancelled = false
    if (!cache.has(url)) cache.set(url, compute(url))
    cache.get(url)!.then(
      (peaks) => !cancelled && setResult({ url, peaks }),
      () => cache.delete(url),
    )
    return () => {
      cancelled = true
    }
  }, [url])
  return url && result?.url === url ? result.peaks : null
}

/** Forget a URL's peaks (when its object URL is revoked). */
export function forgetPeaks(url: string): void {
  cache.delete(url)
}
