// Voices for Synthesize: checkpoints grouped by run / voice folder, as the picker lists them.
import type { Engine, Lora } from './api'
import { formatStep } from './format'
import type { Messages } from './i18n'

export interface Checkpoint {
  lora: Lora
  step: number | null
  label: string
}

export interface Voice {
  key: string // its folder (Lora.voice_path): names may repeat across engines
  name: string
  engine: string
  checkpoints: Checkpoint[] // best (highest step) first
  baseModelId: string | null
}

// Groups checkpoints by run / voice folder (Lora.voice_path) and drops
// copies of the final weights: the script saves the last step twice (N-1 and N), and
// "latest" duplicates whichever step it was copied from.
export function groupVoices(loras: Lora[], t: Messages): Voice[] {
  const byFolder = new Map<string, Lora[]>()
  for (const lora of loras) byFolder.set(lora.voice_path, [...(byFolder.get(lora.voice_path) ?? []), lora])

  return [...byFolder.entries()]
    .map(([key, list]) => {
      const stepDirs = list.filter((l) => /step_\d+$/.test(l.id))
      const steps = new Set(stepDirs.map((l) => l.step))
      const max = Math.max(-1, ...stepDirs.map((l) => l.step ?? -1))
      const checkpoints = list
        .filter((l) => {
          if (/step_\d+$/.test(l.id)) return !(l.step === max - 1 && steps.has(max))
          return l.step === null || !steps.has(l.step) // "latest" or an imported package
        })
        .map((lora) => ({
          lora,
          step: lora.step,
          label: lora.step !== null ? t.stepLabel(formatStep(lora.step)) : t.latest,
        }))
        .sort((a, b) => (b.step ?? -1) - (a.step ?? -1))
      const best = checkpoints[0]?.lora
      const name = list[0].voice
      return {
        key,
        name,
        engine: (best ?? list[0]).engine,
        checkpoints,
        baseModelId: best?.base_model_id ?? null,
      }
    })
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
}

export function shortModel(id: string): string {
  return id.split('/').at(-1) ?? id
}

/** The model a voice's engine runs here, when its voices are LoRAs of it (their recorded base must match). */
export function engineModel(v: Voice, engines: Engine[]): string | null {
  const engine = engines.find((e) => e.id === v.engine)
  return engine?.capabilities.package === 'lora' ? engine.model_id : null
}

/** A LoRA trained on another base model than its engine runs here. */
export function incompatible(v: Voice, engines: Engine[]): boolean {
  const model = engineModel(v, engines)
  return !!model && !!v.baseModelId && v.baseModelId !== model
}

export function unknownEngine(v: Voice, engines: Engine[]): boolean {
  return engines.length > 0 && !engines.some((e) => e.id === v.engine)
}

/** A steady colour for a voice's initial, from its name. */
export function voiceHue(name: string): number {
  let h = 0
  for (const ch of name) h = (h * 31 + ch.codePointAt(0)!) % 360
  return h
}
