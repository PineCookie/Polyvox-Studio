// TTS engines in the UI: their settings' defaults, names and order (as /api/engines lists them),
// and the voice a request uses.
import { DEFAULT_ENGINE } from './api'
import type { Engine, EngineOptions } from './api'

export function defaultOptions(engine: Engine): EngineOptions {
  return Object.fromEntries(engine.params.map((p) => [p.name, p.default]))
}

// What Synthesize uses: an engine's base model, or one of its LoRAs.
export interface VoiceChoice {
  engine: string
  lora: string | null
}

export const BASE_VOICE: VoiceChoice = { engine: DEFAULT_ENGINE, lora: null }

/** An engine's name (its id when this server doesn't have it). */
export function engineName(id: string, engines: Engine[]): string {
  return engines.find((e) => e.id === id)?.name ?? id
}

/** Sort key for engine ids: the server's order, unknown engines last. */
export function engineRank(id: string, engines: Engine[]): number {
  const i = engines.findIndex((e) => e.id === id)
  return i < 0 ? Infinity : i
}
