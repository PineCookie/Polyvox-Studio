import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { getTtsProgress } from '../api'
import type { Engine, EngineOptions, EngineParam, ReferenceInput, SynthesisParams, TtsProgress } from '../api'
import { defaultOptions } from '../engines'
import { formatDuration } from '../format'
import { useT } from '../i18n'
import { emptyInput, inputAudio, lacksTranscript } from '../referenceInputs'
import type { InputState } from '../referenceInputs'
import { ReferenceInputField } from './ReferenceInputField'
import { ProgressBar } from './RunList'

// VoxCPM2's non-verbal tags, as its cookbook spells them ([laughing], not
// [Laughter]); best used sparingly, not several in one sentence.
const TAGS: [string, string][] = [
  ['[laughing]', 'laugh'],
  ['[sigh]', 'sigh'],
  ['[Uhm]', 'uhm'],
  ['[Shh]', 'shh'],
] // [tag, its name in t.tagNames]
const MORE_TAGS = [
  '[Question-ah]',
  '[Question-ei]',
  '[Question-en]',
  '[Question-oh]',
  '[Surprise-wa]',
  '[Surprise-yo]',
  '[Dissatisfaction-hnn]',
]
const POLL_MS = 400

/** Phase and an estimated share of the generation, polled while it runs. */
function SynthesisProgress() {
  const t = useT()
  const [progress, setProgress] = useState<TtsProgress | null>(null)
  useEffect(() => {
    let alive = true
    const poll = () =>
      getTtsProgress().then(
        (p) => alive && setProgress(p),
        () => {},
      )
    poll()
    const timer = setInterval(poll, POLL_MS)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])
  if (!progress) return null
  const { phase, step, expected, attempt, elapsed } = progress
  const label =
    phase === 'loading'
      ? t.progressLoading
      : phase === 'compiling'
        ? t.progressCompiling
        : phase === 'generating'
          ? t.progressGenerating + (attempt > 1 ? t.progressRetry(attempt) : '')
          : t.progressPreparing
  // The estimate can be short or long; never show done before it is.
  const share = phase === 'generating' && expected ? Math.min(step / expected, 0.97) : 0
  return (
    <div className="synthesis-progress">
      <span className="muted hint">
        {label}
        {phase === 'generating' && ` · ${Math.round(share * 100)}%`} · {formatDuration(elapsed)}
      </span>
      <ProgressBar value={share} max={1} />
    </div>
  )
}

/** One numeric synthesis setting of an engine, as a slider. */
function ParamSlider({ param, value, onChange }: { param: EngineParam; value: number; onChange: (v: number) => void }) {
  const t = useT()
  const step = param.step ?? (param.kind === 'int' ? 1 : 0.1)
  const decimals = param.kind === 'int' ? 0 : Math.max(0, -Math.floor(Math.log10(step)))
  return (
    <label className="field">
      <span>
        {t.paramLabels[param.name] ?? param.name} <output>{value.toFixed(decimals)}</output>
      </span>
      <input
        type="range"
        min={param.min ?? 0}
        max={param.max ?? 10}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {t.paramHints[param.name] && <span className="muted hint">{t.paramHints[param.name]}</span>}
    </label>
  )
}

interface Props {
  // The engine of the chosen voice: its settings and what it can do.
  engine: Engine | null
  busy: boolean
  seed: number | null
  onSeedChange: (seed: number | null) => void
  onGenerate: (params: Omit<SynthesisParams, 'loraId' | 'engine'>) => void
  onError: (message: string) => void
  // Called after a transcription (it loads the ASR model).
  onTranscribed: () => void
  // Transcript of the chosen voice's own reference clip (GPT-SoVITS voices); null: it has none.
  voiceReference: string | null
  // The base model is chosen (no voice): settings only it uses show.
  isBase: boolean
  // The chosen voice's model version (null: unknown, or the base model, whose version is an option).
  voiceVersion: string | null
}

export function SynthesisPanel({
  engine,
  busy,
  seed,
  onSeedChange,
  onGenerate,
  onError,
  onTranscribed,
  voiceReference,
  isBase,
  voiceVersion,
}: Props) {
  const t = useT()
  const [text, setText] = useState('')
  const [control, setControl] = useState('')
  // Each engine's reference inputs as set here, by '<engine>.<input>' (kept while switching voices).
  const [inputStates, setInputStates] = useState<Record<string, InputState>>({})
  // Settings changed by hand, per engine (each keeps its own while switching voices).
  const [changed, setChanged] = useState<Record<string, EngineOptions>>({})
  const caps = engine?.capabilities
  const options: EngineOptions = engine ? { ...defaultOptions(engine), ...changed[engine.id] } : {}
  const setOption = (name: string, value: number | boolean | string) =>
    engine && setChanged((c) => ({ ...c, [engine.id]: { ...c[engine.id], [name]: value } }))
  // Settings shown for this choice: base-only ones with the base model, version-bound ones for their versions.
  const version = isBase ? String(options.base_version ?? '') : voiceVersion
  const shown = (engine?.params ?? []).filter(
    (p) => (isBase || !p.base_only) && (!p.versions.length || (version !== null && p.versions.includes(version))),
  )
  const textArea = useRef<HTMLTextAreaElement>(null)

  // The engine's reference inputs for this model version, each with its state. An input that may
  // reuse another's audio starts out doing so with the base model (it cloned best that way) and
  // empty with a trained voice (there it added nothing and could carry over how the clip ends).
  const inputs = (caps?.reference_inputs ?? []).filter(
    (i) => !i.versions.length || (version !== null && i.versions.includes(version)),
  )
  const stateOf = (i: ReferenceInput): InputState =>
    inputStates[`${engine?.id}.${i.name}`] ?? emptyInput(i.same_as && isBase ? 'same' : 'none')
  const setStateOf = (i: ReferenceInput, state: InputState) =>
    setInputStates((s) => ({ ...s, [`${engine?.id}.${i.name}`]: state }))
  const otherOf = (i: ReferenceInput) => {
    const other = i.same_as ? inputs.find((o) => o.name === i.same_as) : undefined
    return other ? { input: other, state: stateOf(other) } : null
  }
  const references: SynthesisParams['references'] = {}
  for (const i of inputs) {
    const audio = inputAudio(stateOf(i), otherOf(i)?.state ?? null)
    if (audio) references[i.slot] = audio
  }
  const untranscribed = inputs.find((i) => lacksTranscript(i, stateOf(i), otherOf(i)?.state ?? null))
  const inputLabel = (i: ReferenceInput) => t.referenceInputs[`${engine?.id}.${i.name}`]?.[0] ?? i.name

  // Engines that always clone a clip need one: chosen here, or the voice's own.
  const needsReference = !!caps?.reference_required && !Object.keys(references).length && voiceReference === null
  const canSubmit =
    !busy && !!engine && !engine.unavailable && text.trim().length > 0 && !needsReference && !untranscribed

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (!canSubmit) return
    onGenerate({ text, control: caps?.style ? control : '', references, options, seed })
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit()
  }

  /** Put a tag at the cursor (or over the selection), then keep typing after it. */
  function insertTag(tag: string) {
    const el = textArea.current
    const start = el?.selectionStart ?? text.length
    const end = el?.selectionEnd ?? text.length
    setText(text.slice(0, start) + tag + text.slice(end))
    requestAnimationFrame(() => {
      el?.focus()
      el?.setSelectionRange(start + tag.length, start + tag.length)
    })
  }

  return (
    <form className="panel" onSubmit={submit} onKeyDown={onKeyDown}>
      <header className="panel-head">
        <h2>{t.synthesize}</h2>
      </header>

      <label className="field">
        <span>{t.text}</span>
        <textarea
          ref={textArea}
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={t.textPlaceholder}
          autoFocus
        />
      </label>
      {caps?.tags && (
        <div className="tag-row" role="group" aria-label={t.tagGroup}>
          {TAGS.map(([tag, label]) => (
            <button
              key={tag}
              type="button"
              className="ghost small"
              onClick={() => insertTag(tag)}
              title={t.insertTag(tag)}
            >
              {t.tagNames[label]}
            </button>
          ))}
          <select
            value=""
            onChange={(e) => e.target.value && insertTag(e.target.value)}
            aria-label={t.moreTagsLabel}
            className="small"
          >
            <option value="">{t.moreTags}</option>
            {MORE_TAGS.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
          <span className="muted hint">{t.tagsHint}</span>
        </div>
      )}

      {caps?.style && (
        <label className="field">
          <span>
            {t.style} <span className="muted">{t.optional}</span>
          </span>
          <input value={control} onChange={(e) => setControl(e.target.value)} placeholder={t.stylePlaceholder} />
        </label>
      )}

      {inputs.length > 0 && (
        <fieldset className="field reference">
          <legend>
            {t.referenceAudio}{' '}
            <span className="muted">
              {!caps?.reference_required
                ? t.referenceOptional
                : voiceReference !== null
                  ? t.referenceVoiceDefault
                  : t.referenceRequired}
            </span>
          </legend>
          {caps?.reference_required && !references.prompt && voiceReference !== null && (
            <p className="muted hint">{t.usesVoiceReference(voiceReference)}</p>
          )}
          {inputs.map((i) => (
            <ReferenceInputField
              key={`${engine?.id}.${i.name}`}
              engineId={engine?.id ?? ''}
              input={i}
              value={stateOf(i)}
              onChange={(state) => setStateOf(i, state)}
              other={otherOf(i)}
              onError={onError}
              onTranscribed={onTranscribed}
            />
          ))}
        </fieldset>
      )}

      <details className="advanced">
        <summary>{t.advanced}</summary>
        <div className="advanced-grid">
          {shown
            .filter((p) => p.kind === 'choice')
            .map((p) => (
              <label key={`${engine?.id}.${p.name}`} className="field">
                <span>{t.paramLabels[p.name] ?? p.name}</span>
                <select value={String(options[p.name])} onChange={(e) => setOption(p.name, e.target.value)}>
                  {(p.choices ?? []).map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                {t.paramHints[p.name] && <span className="muted hint">{t.paramHints[p.name]}</span>}
              </label>
            ))}
          {shown
            .filter((p) => p.kind === 'float' || p.kind === 'int')
            .map((p) => (
              <ParamSlider
                key={`${engine?.id}.${p.name}`}
                param={p}
                value={Number(options[p.name])}
                onChange={(v) => setOption(p.name, v)}
              />
            ))}
          <label className="field">
            <span>{t.seed}</span>
            <input
              type="number"
              value={seed ?? ''}
              placeholder={t.random}
              onChange={(e) => onSeedChange(e.target.value === '' ? null : Number(e.target.value))}
            />
          </label>
          <div className="toggles">
            {shown
              .filter((p) => p.kind === 'bool')
              .map((p) => (
                <label key={`${engine?.id}.${p.name}`}>
                  <input
                    type="checkbox"
                    checked={Boolean(options[p.name])}
                    onChange={(e) => setOption(p.name, e.target.checked)}
                  />{' '}
                  {t.paramLabels[p.name] ?? p.name}
                </label>
              ))}
          </div>
        </div>
      </details>

      {busy && <SynthesisProgress />}
      <div className="actions">
        {engine?.unavailable ? (
          <span className="field-error">{t.engineUnavailable(engine.name, engine.unavailable)}</span>
        ) : needsReference ? (
          <span className="field-error">{t.needsReference(engine!.name)}</span>
        ) : untranscribed ? (
          <span className="field-error">{t.needsTranscript(inputLabel(untranscribed))}</span>
        ) : (
          <span className="muted hint">{t.ctrlEnter}</span>
        )}
        <button type="submit" className="primary" disabled={!canSubmit}>
          {busy ? t.generating : t.generate}
        </button>
      </div>
    </form>
  )
}
