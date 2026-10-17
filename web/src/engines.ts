// TTS engines in the UI: synthesis defaults and the voice a request uses.
import { DEFAULT_ENGINE } from './api'
import type { Engine, EngineOptions } from './api'

// Synthesis defaults that differ from an engine's own (the API's).
const UI_DEFAULTS: Record<string, EngineOptions> = {
  // Below VoxCPM's 2.0: with a voice LoRA, 1.6 sounded more natural and closer to the
  // voice, and the occasional added sound at 1.5 was gone.
  voxcpm: { cfg_value: 1.6 },
}

export function defaultOptions(engine: Engine): EngineOptions {
  return { ...Object.fromEntries(engine.params.map((p) => [p.name, p.default])), ...UI_DEFAULTS[engine.id] }
}

// What Synthesize uses: an engine's base model, or one of its LoRAs.
export interface VoiceChoice {
  engine: string
  lora: string | null
}

export const BASE_VOICE: VoiceChoice = { engine: DEFAULT_ENGINE, lora: null }

// Names for lists that don't load /api/engines (the server's names and order are the same).
const NAMES: Record<string, string> = {
  voxcpm: 'VoxCPM',
  dots_tts: 'dots.tts',
  gpt_sovits: 'GPT-SoVITS',
  fish_s2: 'Fish Audio S2-Pro',
}

export function engineName(id: string, engines: Engine[] = []): string {
  return engines.find((e) => e.id === id)?.name ?? NAMES[id] ?? id
}

/** Sort key for engine ids: the server's order, unknown engines last. */
export function engineRank(id: string): number {
  const i = Object.keys(NAMES).indexOf(id)
  return i < 0 ? Infinity : i
}
