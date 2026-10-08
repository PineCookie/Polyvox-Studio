export function formatDuration(seconds: number): string {
  const s = Math.round(seconds)
  if (s < 60) return `${s} s`
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min`
  return `${Math.floor(m / 60)} h ${m % 60} min`
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
