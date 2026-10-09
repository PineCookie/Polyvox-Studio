import type { Aligner, CheckFlag } from './api'

export const ALIGNER_LABEL: Record<Aligner, string> = {
  qwen3: 'Qwen3-ForcedAligner',
  ctc: 'wav2vec2 (CTC)',
}

// Alignment check flags (checks.py): short tag, and what it means.
export const FLAG_LABEL: Record<CheckFlag, [string, string]> = {
  singing: ['Singing', 'A sung clip (karaoke, a cover): not wanted in training data'],
  mismatch: ['Mismatch', 'Part of the text does not match the audio'],
  asr_more: ['Missing words?', 'Speech recognition hears words the transcript does not have'],
  filler: ['Filler?', 'A filler or laugh (嗯, 哈哈, 呵呵, ...) may not be in the audio, or not as written'],
  edge_sound: [
    'Edge sound',
    'Sound before or after the transcribed words, a pause apart (嗯, a laugh, or words the text misses): trim it off or add it to the text',
  ],
  extra_speech: [
    'Untranscribed',
    'Speech right next to the transcribed words, too close to cut off: the text may miss something',
  ],
  cut_edge: ['Cut edge', 'The sound runs into the start or end of the file: a word may be cut'],
  long_silence: [
    'Long silence',
    'Over 0.5 s of silence after the speech (the model learns when to stop from it), or over 1 s before it',
  ],
}

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
