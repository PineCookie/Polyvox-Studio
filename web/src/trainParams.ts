// The New-run form: an engine's fields (Engine.train_form) and how they depend on each other.
// What a field means and its defaults for this GPU come from the server; the run's size from
// POST /api/training/plan.
import type { Condition, EngineParam, FormValue, FormValues } from './api'

// Fields a run's size gives (Engine.plan) unless set by hand.
export const SCHEDULE_KEYS = ['num_iters', 'warmup_steps', 'save_interval', 'valid_interval', 'log_interval']

/** The form's values before anything is changed. */
export function formDefaults(form: EngineParam[]): FormValues {
  return Object.fromEntries(form.map((p) => [p.name, p.default]))
}

/** Each named setting has one of its values (no condition: true). */
export function holds(condition: Condition | null, values: FormValues): boolean {
  return !condition || Object.entries(condition).every(([name, allowed]) => allowed.includes(values[name]))
}

export const isShown = (p: EngineParam, values: FormValues) => holds(p.shown_if, values)

/** It can't be changed now: never here (Param.disabled), or not while its condition holds. */
export const isLocked = (p: EngineParam, values: FormValues) =>
  !!p.disabled || (!!p.disabled_if && holds(p.disabled_if, values))

/** The key of its label now (Param.labels: the first condition that holds), else its name. */
export const labelKey = (p: EngineParam, values: FormValues) => p.labels.find(([c]) => holds(c, values))?.[1] ?? p.name

/** ``values`` with ``p`` set to ``value`` and the settings that go with that value (Param.implies). */
export function withValue(values: FormValues, p: EngineParam, value: FormValue): FormValues {
  return { ...values, [p.name]: value, ...(p.implies?.[String(value)] ?? {}) }
}

/** Every number is filled in. */
export const allNumbers = (values: FormValues) =>
  Object.values(values).every((v) => typeof v !== 'number' || Number.isFinite(v))
