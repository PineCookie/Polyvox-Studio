import type { ReactNode } from 'react'
import { useT } from '../i18n'
import { DEFAULT_ENGINE } from '../api'
import type { Engine, EngineParam, FormValue, FormValues } from '../api'
import { isLocked, isShown, labelKey } from '../trainParams'
import { Info } from './Info'

interface Props {
  // The engine's training form (Engine.train_form) and its values.
  form: EngineParam[]
  values: FormValues
  // Omitted: read-only (an existing run shows what it was started with).
  onChange?: (param: EngineParam, value: FormValue) => void
  engine: string
  // The models a run can train (the choice shows when there is more than one).
  engines?: Engine[]
  onEngine?: (id: string) => void
  // Above the main fields (e.g. the run name).
  header?: ReactNode
  // Under the main fields: what the run will do.
  hint?: ReactNode
  // The validation section differs between a new and an existing run.
  validation: ReactNode
}

/** Training parameters of any engine, drawn from its form: editable for a new run, read-only
 * for an existing one. Labels: t.params[key]; hints and reasons: message keys (a key the UI
 * doesn't know is shown as it is). */
export function ParamsFields({
  form,
  values,
  onChange,
  engine,
  engines = [],
  onEngine,
  header,
  hint,
  validation,
}: Props) {
  const t = useT()
  const readOnly = !onChange
  const message = (key: string) => {
    const text = (t as unknown as Record<string, unknown>)[key]
    return typeof text === 'string' ? text : key
  }
  const label = (key: string) => (t.params as Record<string, string>)[key] ?? key
  const choiceLabel = (p: EngineParam, value: string | number) =>
    (t.trainChoices as Record<string, Record<string, string>>)[p.name]?.[String(value)] ?? String(value)
  // Sizing-only fields (epochs) aren't kept with a run.
  const shown = form.filter((p) => isShown(p, values) && !(readOnly && p.scope === 'plan'))
  const main = shown.filter((p) => p.group === 'main')
  const advanced = shown.filter((p) => p.group === 'advanced')

  const showEngine = engines.length > 1 || engine !== DEFAULT_ENGINE
  const engineOptions = engines.some((e) => e.id === engine)
    ? engines
    : [...engines, { id: engine, name: engine, unavailable: null } as Engine]
  const cantTrain = (e: Engine) => (e.training_unavailable === undefined ? e.unavailable : e.training_unavailable)

  function control(p: EngineParam) {
    const value = values[p.name]
    const locked = isLocked(p, values)
    const title = p.disabled ? message(p.disabled) : undefined
    const name = label(labelKey(p, values))
    if (p.kind === 'bool') {
      return (
        <label key={p.name} className="check" title={title}>
          <input
            type="checkbox"
            checked={!!value}
            disabled={locked}
            onChange={(e) => onChange?.(p, e.target.checked)}
          />{' '}
          {name}
        </label>
      )
    }
    return (
      <label key={p.name} className="field" title={title}>
        <span>
          {name}
          {p.hint && <Info text={message(p.hint)} />}
        </span>
        {p.kind === 'choice' ? (
          <select
            value={String(value)}
            disabled={locked}
            onChange={(e) => onChange?.(p, p.choices?.find((c) => String(c) === e.target.value) ?? e.target.value)}
          >
            {(p.choices ?? []).map((c) => (
              <option key={String(c)} value={String(c)}>
                {choiceLabel(p, c)}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="number"
            value={typeof value === 'number' && Number.isFinite(value) ? value : ''}
            min={p.min ?? undefined}
            max={p.max ?? undefined}
            step={p.step ?? (p.kind === 'int' ? 1 : 'any')}
            disabled={locked}
            onChange={(e) => onChange?.(p, e.target.valueAsNumber)}
          />
        )}
      </label>
    )
  }

  // A switch with its note under it (main fields: a choice that changes the run).
  const withNote = (p: EngineParam) => (
    <div key={p.name} className="field">
      {control(p)}
      {p.hint && <span className="muted hint">{message(p.hint)}</span>}
    </div>
  )

  return (
    <fieldset className="params" disabled={readOnly}>
      {header}
      <div className="form-grid">
        {showEngine && (
          <label className="field">
            <span>{t.params.engine}</span>
            <select value={engine} onChange={(e) => onEngine?.(e.target.value)}>
              {engineOptions.map((e) => (
                <option key={e.id} value={e.id} disabled={!!cantTrain(e)} title={cantTrain(e) ?? undefined}>
                  {e.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {main.filter((p) => p.kind !== 'bool').map(control)}
      </div>
      {main.filter((p) => p.kind === 'bool').map(withNote)}
      {hint}
      {validation}
      {/* Read-only shows everything; a new run keeps the rarely changed ones folded. */}
      {advanced.length > 0 && (
        <details className="advanced" open={readOnly || undefined}>
          <summary>{t.advanced}</summary>
          <div className="form-grid">
            {advanced.filter((p) => p.kind !== 'bool').map(control)}
            {advanced.some((p) => p.kind === 'bool') && (
              <div className="toggles">{advanced.filter((p) => p.kind === 'bool').map(control)}</div>
            )}
          </div>
          {advanced
            .filter((p) => p.kind === 'bool' && p.hint)
            .map((p) => (
              <p key={p.name} className="muted hint">
                {message(p.hint as string)}
              </p>
            ))}
        </details>
      )}
    </fieldset>
  )
}
