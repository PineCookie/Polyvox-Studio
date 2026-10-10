import type { Aligner, CheckFlag } from './api'
import { messages } from './i18n'

export const ALIGNER_LABEL: Record<Aligner, string> = {
  qwen3: 'Qwen3-ForcedAligner',
  ctc: 'wav2vec2 (CTC)',
}

// Alignment check flags (checks.py), in display order.
export const FLAGS: CheckFlag[] = [
  'singing',
  'mismatch',
  'asr_more',
  'filler',
  'edge_sound',
  'extra_speech',
  'cut_edge',
  'long_silence',
]

/** A flag's short tag and what it means, in the UI language. */
export function flagLabel(flag: CheckFlag): [string, string] {
  return messages().flags[flag]
}

export function formatDuration(seconds: number): string {
  const t = messages()
  const s = Math.round(seconds)
  if (s < 60) return t.seconds(s)
  const m = Math.round(s / 60)
  if (m < 60) return t.minutes(m)
  return t.hoursMinutes(Math.floor(m / 60), m % 60)
}

// { "32000": 650, "44100": 6 } -> "32 kHz, 44.1 kHz"
export function formatRates(rates: Record<string, number>): string {
  return Object.keys(rates)
    .map((r) => `${Number(r) / 1000} kHz`)
    .join(', ')
}

// "speaker_a/train.jsonl" -> "speaker_a"; other manifests keep their path.
export function datasetName(id: string): string {
  return id.endsWith('/train.jsonl') ? id.slice(0, -'/train.jsonl'.length) : id
}

export function formatStep(step: number): string {
  return step.toLocaleString()
}

// Compact loss / lr values: 0.7123, 0.00012 -> "1.2e-4".
export function formatValue(v: number): string {
  if (v === 0) return '0'
  return Math.abs(v) < 0.001 ? v.toExponential(1) : v.toFixed(4)
}

// { separation: 250, asr: 130, vad: 0.4 } -> "Took 6 min: separation 4 min, ASR 2 min"
export function formatTimings(timings: Record<string, number>): string {
  const t = messages()
  const entries = Object.entries(timings)
  const total = entries.reduce((sum, [, s]) => sum + s, 0)
  const parts = entries.filter(([, s]) => s >= 1).map(([stage, s]) => `${t.stage[stage] ?? stage} ${formatDuration(s)}`)
  return t.tookTotal(formatDuration(total), parts.join(', '))
}
