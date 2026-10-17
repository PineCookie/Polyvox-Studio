// What Synthesize holds for each of an engine's reference inputs, and the audio it sends.
import type { ReferenceAudio, ReferenceClip, ReferenceInput } from './api'

export type InputSource = 'none' | 'file' | 'dataset' | 'same'

export interface ClipChoice {
  dataset: string
  clips: ReferenceClip[]
}

export interface InputState {
  source: InputSource
  file: File | null
  text: string // the uploaded clip's transcript (or the reused one's, with 'same')
  clips: ClipChoice
}

export function emptyInput(source: InputSource = 'none'): InputState {
  return { source, file: null, text: '', clips: { dataset: '', clips: [] } }
}

/** Clips the dataset picker offers: one where the input takes one short clip. */
export function clipCount(input: ReferenceInput): number {
  return input.max_seconds && !input.several ? 1 : 3
}

/** The audio an input sends, or null (none chosen yet). `other`: the state of the input
 * it reuses with source 'same'. */
export function inputAudio(state: InputState, other: InputState | null): ReferenceAudio | null {
  const from = state.source === 'same' ? other : state
  if (!from) return null
  if (from.source === 'file' && from.file) return { file: from.file, text: state.text }
  if (from.source === 'dataset' && from.clips.clips.length)
    return { clips: { dataset: from.clips.dataset, indices: from.clips.clips.map((c) => c.index) } }
  return null
}

/** An uploaded clip (own or reused) still lacks the transcript the input requires. */
export function lacksTranscript(input: ReferenceInput, state: InputState, other: InputState | null): boolean {
  const audio = inputAudio(state, other)
  return input.transcript === 'required' && !!audio && 'file' in audio && !audio.text.trim()
}
