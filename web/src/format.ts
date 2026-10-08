export function formatDuration(seconds: number): string {
  const s = Math.round(seconds)
  if (s < 60) return `${s} s`
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min`
  return `${Math.floor(m / 60)} h ${m % 60} min`
}

export function formatStep(step: number): string {
  return step.toLocaleString()
}

// Compact loss / lr values: 0.7123, 0.00012 -> "1.2e-4".
export function formatValue(v: number): string {
  if (v === 0) return '0'
  return Math.abs(v) < 0.001 ? v.toExponential(1) : v.toFixed(4)
}
