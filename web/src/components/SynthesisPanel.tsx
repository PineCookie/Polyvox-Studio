import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import type { FormEvent, KeyboardEvent, ReactNode } from 'react'
import { getTtsProgress } from '../api'
import type { Engine, EngineOptions, EngineParam, ReferenceInput, SynthesisParams, TtsProgress } from '../api'
import { defaultOptions } from '../engines'
import { formatDuration, formatSize } from '../format'
import { useT } from '../i18n'
import { emptyInput, inputAudio, lacksTranscript } from '../referenceInputs'
import type { InputState } from '../referenceInputs'
import { scoped, unknownTags } from '../tags'
import { useTakes } from '../useTakes'
import { CloseIcon, SparkIcon, SplitIcon, TagIcon } from './icons'
import { Info } from './Info'
import { ReferenceInputField } from './ReferenceInputField'
import { ProgressBar } from './RunList'
import { TagBar } from './TagBar'
import { TaggedText } from './TaggedText'

const POLL_MS = 400
const TAGS_KEY = 'polyvox.showTags' // "0" once the tag strip was hidden; shown otherwise

function tagsShown(): boolean {
  try {
    return localStorage.getItem(TAGS_KEY) !== '0'
  } catch {
    return true
  }
}

/** Phase and an estimated share of the generation (or of a model it needs, downloading first),
 * polled while it runs; also shown while History edits a clip. */
export function SynthesisProgress() {
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
  if (phase === 'downloading') {
    return (
      <div className="synthesis-progress">
        <span className="muted hint">
          {t.downloadingNeeded(progress.model ?? '', formatSize(step), formatSize(expected ?? 0))}
        </span>
        <ProgressBar value={step} max={expected || 1} />
      </div>
    )
  }
  const label =
    phase === 'loading'
      ? t.progressLoading
      : phase === 'compiling'
        ? t.progressCompiling
        : phase === 'generating'
          ? t.progressGenerating + (attempt > 1 ? t.progressRetry(attempt) : '')
          : t.progressPreparing
  // Loading and compiling: against how long they took last time on this computer.
  const oneOff = phase === 'loading' || phase === 'compiling'
  const inPhase = progress.phase_elapsed ?? null
  const last = progress.last_time ?? null
  // The estimate can be short or long; never show done before it is.
  const share =
    phase === 'generating' && expected
      ? Math.min(step / expected, 0.97)
      : oneOff && last && inPhase !== null
        ? Math.min(inPhase / last, 0.97)
        : 0
  const time = oneOff && inPhase !== null ? inPhase : elapsed
  return (
    <div className="synthesis-progress">
      <span className="muted hint">
        {label}
        {phase === 'generating' && ` · ${Math.round(share * 100)}%`} · {formatDuration(time)}
        {oneOff && (last ? ` · ${t.progressLastTime(formatDuration(last))}` : ` · ${t.progressFirstTime}`)}
      </span>
      <ProgressBar value={share} max={1} />
    </div>
  )
}

/** One numeric synthesis setting of an engine, as a slider; its explanation behind ⓘ. */
export function ParamSlider({
  param,
  value,
  onChange,
}: {
  param: EngineParam
  value: number
  onChange: (v: number) => void
}) {
  const t = useT()
  const step = param.step ?? (param.kind === 'int' ? 1 : 0.1)
  const decimals = param.kind === 'int' ? 0 : Math.max(0, -Math.floor(Math.log10(step)))
  const hint = t.paramHints[param.name]
  return (
    <label className="field slider-field">
      <span>
        <span className="label-text">
          {t.paramLabels[param.name] ?? param.name}
          {hint && <Info text={hint} />}
        </span>
        <output>{value.toFixed(decimals)}</output>
      </span>
      <input
        type="range"
        min={param.min ?? 0}
        max={param.max ?? 10}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
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
  // Transcript of the chosen voice's own reference clip (GPT-SoVITS, S2-Pro voices); null: it has none.
  voiceReference: string | null
  // The base model is chosen (no voice): settings only it uses show.
  isBase: boolean
  // The chosen voice's model version (null: unknown, or the base model, whose version is an option).
  voiceVersion: string | null
  // The voice button at the top of the editor, and what the voice is at the top of the settings.
  voicePicker: ReactNode
  voiceDetails: ReactNode
  // Under the editor: the takes.
  children: ReactNode
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
  voicePicker,
  voiceDetails,
  children,
}: Props) {
  const t = useT()
  const [text, setText] = useState('')
  const [control, setControl] = useState('')
  const [showTags, setShowTags] = useState(tagsShown)
  function toggleTags(show: boolean) {
    setShowTags(show)
    try {
      localStorage.setItem(TAGS_KEY, show ? '1' : '0')
    } catch {
      // Not remembered (storage blocked); the choice still applies now.
    }
  }
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

  const unknown = unknownTags(text, engine)
  // Long texts in separate takes (engines that do): where, with a voice to keep or without.
  const cloning = !isBase || Object.keys(references).length > 0
  const takes = useTakes(caps?.takes && engine ? engine.id : null, text, cloning)

  /** A blank line at the cursor: the text after it (or typed next, at the end) starts a new take. */
  function newTake() {
    const el = textArea.current
    const at = el?.selectionStart ?? text.length
    const before = text.slice(0, at).replace(/\s+$/, '')
    const after = text.slice(el?.selectionEnd ?? at).replace(/^\s+/, '')
    const gap = before ? '\n\n' : ''
    replaceText(before + gap + after, before.length + gap.length)
  }

  /** New text with the cursor at `cursor`, the box focused at once (so typing goes on there). */
  function replaceText(next: string, cursor: number) {
    flushSync(() => setText(next))
    textArea.current?.focus()
    textArea.current?.setSelectionRange(cursor, cursor)
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      submit()
    }
  }

  /** Put a tag at the cursor (or over the selection), then keep typing after it. A scoped tag
   * ("<whisper></whisper>") goes around the selection, or around the cursor to type into. */
  function insertTag(tag: string) {
    const el = textArea.current
    const start = el?.selectionStart ?? text.length
    const end = el?.selectionEnd ?? text.length
    const pair = scoped(tag)
    let next: string
    let cursor: number
    if (pair) {
      const [, open, close] = pair
      next = text.slice(0, start) + open + text.slice(start, end) + close + text.slice(end)
      cursor = start === end ? start + open.length : end + open.length + close.length
    } else {
      next = text.slice(0, start) + tag + text.slice(end)
      cursor = start + tag.length
    }
    replaceText(next, cursor)
  }

  // Why Generate can't run (shown beside it).
  const blocker = engine?.unavailable
    ? t.engineUnavailable(engine.name, engine.unavailable)
    : needsReference
      ? t.needsReference(engine!.name)
      : untranscribed
        ? t.needsTranscript(inputLabel(untranscribed))
        : null
  const choices = shown.filter((p) => p.kind === 'choice')
  const sliders = shown.filter((p) => p.kind === 'float' || p.kind === 'int')
  const toggles = shown.filter((p) => p.kind === 'bool')
  const ownReference = !!caps?.voice_reference && voiceReference !== null
  const referenceHint = ownReference
    ? t.referenceInfoVoice
    : caps?.reference_required
      ? t.referenceInfoRequired
      : t.referenceInfoOptional

  return (
    <div className="synth">
      <div className="synth-main">
        <form className="composer" onSubmit={submit} onKeyDown={onKeyDown}>
          <div className="composer-head">{voicePicker}</div>
          <TaggedText
            ref={textArea}
            value={text}
            onChange={setText}
            placeholder={(engine && t.textPlaceholders[engine.id]) ?? t.textPlaceholder}
            unknown={unknown}
            cuts={takes.cuts}
            rows={7}
            autoFocus
          />
          {unknown.length > 0 && (
            <p className="composer-warning">{t.unknownTags(engine!.name, unknown.join(' '), unknown.length)}</p>
          )}
          {caps?.tags && engine && showTags && <TagBar key={engine.id} engine={engine} onInsert={insertTag} />}
          <div className="composer-foot">
            {caps?.tags && engine && (
              <button
                type="button"
                className={`ghost small tag-button${showTags ? ' active' : ''}`}
                aria-pressed={showTags}
                onClick={() => toggleTags(!showTags)}
                title={showTags ? t.hideTags : t.showTags}
              >
                <TagIcon /> {t.tags}
              </button>
            )}
            {caps?.takes && (
              <button type="button" className="ghost small tag-button" onClick={newTake} title={t.newTakeTitle}>
                <SplitIcon /> {t.newTake}
              </button>
            )}
            <span className="char-count">
              {t.characters(text.length)}
              {takes.count > 1 && (
                <>
                  {' · '}
                  {t.takes(takes.count)} <Info text={t.takesInfo} />
                </>
              )}
            </span>
            <span className="spacer" />
            {blocker ? (
              <span className="field-error composer-blocker">{blocker}</span>
            ) : (
              !busy && <kbd className="shortcut">{t.ctrlEnter}</kbd>
            )}
            <button type="submit" className="primary generate" disabled={!canSubmit}>
              {busy ? <span className="spinner" aria-hidden /> : <SparkIcon />}
              {busy ? t.generating : t.generate}
            </button>
          </div>
          {busy && (
            <div className="composer-progress">
              <SynthesisProgress />
            </div>
          )}
        </form>
        {children}
      </div>

      <aside className="synth-side" onKeyDown={onKeyDown}>
        <div className="settings">
          {voiceDetails}

          {caps?.style && (
            <section className="settings-section">
              <label className="field">
                <span>
                  <span className="label-text">
                    {t.style}
                    {engine && t.styleHints[engine.id] && <Info text={t.styleHints[engine.id]} />}
                  </span>
                  <span className="muted">{t.optional}</span>
                </span>
                <input
                  value={control}
                  onChange={(e) => setControl(e.target.value)}
                  placeholder={(engine && t.stylePlaceholders[engine.id]) ?? t.stylePlaceholder}
                />
              </label>
            </section>
          )}

          {inputs.length > 0 && (
            <section className="settings-section">
              <div className="settings-head">
                <h3>
                  {t.referenceAudio}
                  <Info text={referenceHint} />
                </h3>
                <span className="muted settings-note">
                  {caps?.reference_required && !ownReference ? t.required : t.optional}
                </span>
              </div>
              {ownReference && !references.prompt && (
                <p className="muted hint uses-voice-reference">{t.usesVoiceReference}</p>
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
                  search={!!engine?.capabilities.reference_search}
                />
              ))}
            </section>
          )}

          <section className="settings-section">
            <div className="settings-head">
              <h3>{t.settingsTitle}</h3>
            </div>
            {choices.map((p) => (
              <label key={`${engine?.id}.${p.name}`} className="field">
                <span>
                  <span className="label-text">
                    {t.paramLabels[p.name] ?? p.name}
                    {t.paramHints[p.name] && <Info text={t.paramHints[p.name]} />}
                  </span>
                </span>
                <select value={String(options[p.name])} onChange={(e) => setOption(p.name, e.target.value)}>
                  {(p.choices ?? []).map((c) => (
                    <option key={c} value={c}>
                      {t.choiceLabels[c] ?? c}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            {sliders.map((p) => (
              <ParamSlider
                key={`${engine?.id}.${p.name}`}
                param={p}
                value={Number(options[p.name])}
                onChange={(v) => setOption(p.name, v)}
              />
            ))}
            <label className="field">
              <span>
                <span className="label-text">
                  {t.seed}
                  <Info text={t.seedHint} />
                </span>
              </span>
              <span className="seed-row">
                <input
                  type="number"
                  value={seed ?? ''}
                  placeholder={t.random}
                  onChange={(e) => onSeedChange(e.target.value === '' ? null : Number(e.target.value))}
                />
                {seed !== null && (
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => onSeedChange(null)}
                    title={t.randomSeed}
                    aria-label={t.randomSeed}
                  >
                    <CloseIcon />
                  </button>
                )}
              </span>
            </label>
            {toggles.length > 0 && (
              <div className="toggles">
                {toggles.map((p) => (
                  <label key={`${engine?.id}.${p.name}`} className="switch">
                    <span className="label-text">
                      {t.paramLabels[p.name] ?? p.name}
                      {t.paramHints[p.name] && <Info text={t.paramHints[p.name]} />}
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      checked={Boolean(options[p.name])}
                      onChange={(e) => setOption(p.name, e.target.checked)}
                    />
                  </label>
                ))}
              </div>
            )}
          </section>
        </div>
      </aside>
    </div>
  )
}
