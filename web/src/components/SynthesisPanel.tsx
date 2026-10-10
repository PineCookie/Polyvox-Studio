import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { getTtsProgress, transcribe } from '../api'
import type { SynthesisParams, TtsProgress } from '../api'
import { formatDuration } from '../format'
import { useT } from '../i18n'
import { forgetPeaks } from '../peaks'
import { AudioPlayer } from './AudioPlayer'
import { FilePicker } from './FilePicker'
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

interface Props {
  busy: boolean
  seed: number | null
  onSeedChange: (seed: number | null) => void
  onGenerate: (params: Omit<SynthesisParams, 'loraId'>) => void
  onError: (message: string) => void
  // Called after a transcription (it loads the ASR model).
  onTranscribed: () => void
}

export function SynthesisPanel({ busy, seed, onSeedChange, onGenerate, onError, onTranscribed }: Props) {
  const t = useT()
  const [text, setText] = useState('')
  const [control, setControl] = useState('')
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null)
  const [promptText, setPromptText] = useState('')
  // Below VoxCPM's 2.0: with a voice LoRA, 1.6 sounded more natural and closer to the
  // voice, and the occasional added sound at 1.5 was gone.
  const [cfgValue, setCfgValue] = useState(1.6)
  const [steps, setSteps] = useState(10)
  const [normalize, setNormalize] = useState(true)
  const [denoise, setDenoise] = useState(true)
  const [transcribing, setTranscribing] = useState(false)
  const textArea = useRef<HTMLTextAreaElement>(null)
  // A playable URL for the chosen reference, released when it changes.
  const referenceUrl = useMemo(() => (referenceAudio ? URL.createObjectURL(referenceAudio) : null), [referenceAudio])
  useEffect(
    () => () => {
      if (referenceUrl) {
        URL.revokeObjectURL(referenceUrl)
        forgetPeaks(referenceUrl)
      }
    },
    [referenceUrl],
  )

  const canSubmit = !busy && text.trim().length > 0

  async function transcribeReference() {
    if (!referenceAudio) return
    setTranscribing(true)
    try {
      setPromptText((await transcribe(referenceAudio)).text)
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setTranscribing(false)
      onTranscribed()
    }
  }

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (!canSubmit) return
    onGenerate({ text, control, referenceAudio, promptText, cfgValue, steps, normalize, denoise, seed })
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

  function clearReference() {
    setReferenceAudio(null)
    setPromptText('')
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

      <label className="field">
        <span>
          {t.style} <span className="muted">{t.optional}</span>
        </span>
        <input value={control} onChange={(e) => setControl(e.target.value)} placeholder={t.stylePlaceholder} />
      </label>

      <fieldset className="field reference">
        <legend>
          {t.referenceAudio} <span className="muted">{t.referenceOptional}</span>
        </legend>
        <FilePicker
          file={referenceAudio}
          onChange={(f) => (f ? setReferenceAudio(f) : clearReference())}
          accept="audio/*"
          label={t.chooseAudio}
          empty={t.orDropFile}
        >
          {referenceUrl && <AudioPlayer url={referenceUrl} />}
        </FilePicker>
        {referenceAudio && (
          <div className="file-row">
            <input
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder={t.referenceTranscript}
            />
            <button
              type="button"
              className="ghost small"
              onClick={transcribeReference}
              disabled={transcribing}
              title={t.transcribeTitle}
            >
              {transcribing ? t.transcribing : t.transcribe}
            </button>
          </div>
        )}
      </fieldset>

      <details className="advanced">
        <summary>{t.advanced}</summary>
        <div className="advanced-grid">
          <label className="field">
            <span>
              {t.guidance} <output>{cfgValue.toFixed(1)}</output>
            </span>
            <input
              type="range"
              min={0.5}
              max={5}
              step={0.1}
              value={cfgValue}
              onChange={(e) => setCfgValue(Number(e.target.value))}
            />
            <span className="muted hint">{t.guidanceHint}</span>
          </label>
          <label className="field">
            <span>
              {t.steps} <output>{steps}</output>
            </span>
            <input type="range" min={1} max={50} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
            <span className="muted hint">{t.stepsHint}</span>
          </label>
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
            <label>
              <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} />{' '}
              {t.normalizeText}
            </label>
            <label>
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />{' '}
              {t.denoiseReference}
            </label>
          </div>
        </div>
      </details>

      {busy && <SynthesisProgress />}
      <div className="actions">
        <span className="muted hint">{t.ctrlEnter}</span>
        <button type="submit" className="primary" disabled={!canSubmit}>
          {busy ? t.generating : t.generate}
        </button>
      </div>
    </form>
  )
}
