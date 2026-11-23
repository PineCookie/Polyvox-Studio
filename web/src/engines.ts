// TTS engines in the UI: their settings' defaults, names and order (as /api/engines lists them),
// and the voice a request uses.
import type { Engine, EngineOptions } from './api'

export function defaultOptions(engine: Engine): EngineOptions {
  return Object.fromEntries(engine.params.map((p) => [p.name, p.default]))
}

// What Synthesize uses: an engine's base model, or one of its LoRAs.
export interface VoiceChoice {
  engine: string
  lora: string | null
}

/** The base model Synthesize starts with: the first speech engine's. */
export function firstVoice(engines: Engine[]): VoiceChoice {
  return { engine: engines.find((e) => !e.capabilities.conversion)?.id ?? '', lora: null }
}

/** An engine's name (its id when this server doesn't have it). */
export function engineName(id: string, engines: Engine[]): string {
  return engines.find((e) => e.id === id)?.name ?? id
}

/** Sort key for engine ids: the server's order, unknown engines last. */
export function engineRank(id: string, engines: Engine[]): number {
  const i = engines.findIndex((e) => e.id === id)
  return i < 0 ? Infinity : i
}
