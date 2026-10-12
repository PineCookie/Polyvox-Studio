import { useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { DEFAULT_LUFS, STEP_DEFAULTS, addSource, getStatus, getStepStatus, runRemaining, runStep } from '../api'
import type { AddSourceJob, Aligner, CleanupOption, StepJob, StepName, StepSettings, StepStatus } from '../api'
import { ALIGNER_LABEL, formatDuration, formatTimings } from '../format'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { JobProgress } from './JobProgress'

// ISO codes the ASR accepts (labels: t.languages); 'auto': detect until the clips agree, then force that.
const LANGUAGES = ['auto', '', 'zh', 'yue', 'en', 'ja', 'ko']
const CLEANUP: CleanupOption[] = ['separate', 'dereverb', 'denoise']

interface Props {
  id: string
  // A step changed clips or their checks: the review reloads.
  onChanged: () => void
}

type Need = 'needed' | 'optional' | 'done'

/** Settings as last run on this dataset, else the defaults. */
function savedSettings(status: StepStatus): StepSettings {
  const out = { ...STEP_DEFAULTS }
  for (const step of Object.keys(out) as StepName[]) {
    const saved = Object.fromEntries(Object.entries(status.settings[step] ?? {}).filter(([, v]) => v != null))
    out[step] = { ...STEP_DEFAULTS[step], ...saved } as never
  }
  out.transcribe = { ...out.transcribe, prompt: out.transcribe.prompt ?? '' }
  return out
}

/** One line on what a finished job did. */
function useSummary() {
  const t = useT()
  return (step: string, r: Record<string, unknown>): string => {
    const n = (key: string) => Number(r[key] ?? 0)
    switch (step) {
      case 'add':
        return t.stepAdded(n('clips'), n('recordings'))
      case 'clean_up':
        return t.stepCleaned(n('cleaned'), n('skipped'))
      case 'slice':
        return t.stepSliced(n('sliced'), n('clips'), n('missing'))
      case 'transcribe':
        return t.stepTranscribed(n('transcribed'), n('dropped'), n('tidied'))
      case 'check':
        return t.stepChecked(n('checked'), n('flagged'))
      case 'tidy':
        return t.stepTidied(n('trimmed'), n('normalized'))
      default:
        return ''
    }
  }
}

/** One step: number, title, what it still has to do, its settings and actions. */
function Row(p: { n: number; need: Need; title: string; state: string; children?: ReactNode; actions: ReactNode }) {
  return (
    <div className={`tool-row step-row step-${p.need}`}>
      <div className="tool-info">
        <strong>
          <span className="step-number">{p.n}</span> {p.title}
        </strong>
        <span className={`step-state ${p.need}`}>{p.state}</span>
        {p.children}
      </div>
      <div className="tool-controls">{p.actions}</div>
    </div>
  )
}

/** The dataset's processing steps in order: what each still has to do, its settings, Run. */
export function DatasetSteps({ id, onChanged }: Props) {
  const t = useT()
  const summary = useSummary()
  const [status, setStatus] = useState<StepStatus | null>(null)
  const [settings, setSettings] = useState<StepSettings>(STEP_DEFAULTS)
  const [available, setAvailable] = useState<CleanupOption[]>([])
  const [error, setError] = useState<string | null>(null)
  const [what, setWhat] = useState('') // the running / last job: a step, 'add' or 'remaining'
  const [done, setDone] = useState<string | null>(null) // what the last job did
  const [adding, setAdding] = useState(false)
  const [source, setSource] = useState('')
  const [audioDir, setAudioDir] = useState('')
  const [copyAudio, setCopyAudio] = useState(true)

  const refresh = useCallback(
    async (fresh = false) => {
      try {
        const s = await getStepStatus(id)
        setStatus(s)
        if (fresh) setSettings(savedSettings(s))
      } catch (e) {
        setError((e as Error).message)
      }
    },
    [id],
  )
  useEffect(() => {
    refresh(true)
    getStatus().then(
      (s) => setAvailable(s.cleanup),
      () => setAvailable([]),
    )
  }, [refresh])

  const [job, setJob] = useJob<StepJob | AddSourceJob>((j) => {
    refresh()
    onChanged()
    if (j.state === 'failed') setError(j.error ?? t.stepFailed)
    if (j.state !== 'done') return
    const result = j.result as Record<string, unknown>
    const lines =
      what === 'remaining'
        ? Object.entries(result)
            .filter(([k]) => k !== 'timings')
            .map(([k, v]) => summary(k, v as Record<string, unknown>))
        : [summary(what, result)]
    const timings = (result.timings as Record<string, number> | undefined) ?? {}
    const times = Object.keys(timings).length ? ` ${formatTimings(timings)}.` : ''
    setDone(lines.filter(Boolean).join(' ') + times)
  })
  const running = job?.state === 'running'

  async function start(name: string, call: () => Promise<StepJob | AddSourceJob>) {
    setError(null)
    setDone(null)
    setWhat(name)
    try {
      setJob(await call())
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const set = <S extends StepName>(step: S, patch: Partial<StepSettings[S]>) =>
    setSettings((s) => ({ ...s, [step]: { ...s[step], ...patch } }))

  if (!status) return error ? <p className="field-error">{error}</p> : null

  const need: Record<'sources' | StepName, Need> = {
    sources: 'done',
    clean_up: 'optional',
    slice: status.slice.recordings ? 'needed' : 'done',
    transcribe: status.transcribe.clips ? 'needed' : 'done',
    check: status.check.clips ? 'needed' : 'done',
    tidy: status.tidy.trimmable ? 'optional' : 'done',
  }
  const remaining = (['slice', 'transcribe', 'check'] as const).filter((s) => need[s] === 'needed')
  const sliceOk = settings.slice.min_seconds > 0 && settings.slice.max_seconds > settings.slice.min_seconds
  const run = <S extends StepName>(step: S, everything = false) =>
    start(step, () => runStep(id, step, settings[step], everything))

  const clips = status.sources.clips
  return (
    <div className="steps">
      <div className="steps-head">
        <h3>{t.stepsTitle}</h3>
        <button
          type="button"
          className={remaining.length ? 'primary' : 'ghost'}
          disabled={running || !remaining.length || !sliceOk}
          onClick={() => start('remaining', () => runRemaining(id, settings))}
          title={remaining.map((s) => t.stepNames[s]).join(' → ')}
        >
          {remaining.length ? t.runRemaining(remaining.length) : t.nothingRemaining}
        </button>
      </div>

      <div className="tools">
        <Row
          n={1}
          need={need.sources}
          title={t.stepNames.sources}
          state={t.sourcesState(clips, status.sources.recordings, status.slice.recordings, status.sources.dropped)}
          actions={
            <button type="button" className="ghost" onClick={() => setAdding((a) => !a)} disabled={running}>
              {t.addSource}
            </button>
          }
        >
          {adding && (
            <form
              className="add-source"
              onSubmit={(e) => {
                e.preventDefault()
                start('add', () => addSource(id, source, audioDir, copyAudio))
              }}
            >
              <input
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder={t.sourcePlaceholder}
                spellCheck={false}
                aria-label={t.source}
              />
              <input
                value={audioDir}
                onChange={(e) => setAudioDir(e.target.value)}
                placeholder={`${t.audioFolder} (${t.optional})`}
                spellCheck={false}
                aria-label={t.audioFolder}
              />
              <label className="check">
                <input type="checkbox" checked={copyAudio} onChange={(e) => setCopyAudio(e.target.checked)} />{' '}
                {t.copyClips}
              </label>
              <button type="submit" className="primary small" disabled={running || !source.trim()}>
                {t.add}
              </button>
            </form>
          )}
        </Row>

        <Row
          n={2}
          need={need.clean_up}
          title={t.stepNames.clean_up}
          state={t.cleanUpState(status.clean_up.clips, status.slice.recordings)}
          actions={
            <button
              type="button"
              className="ghost"
              disabled={running || !settings.clean_up.models.length}
              onClick={() => run('clean_up')}
            >
              {t.run}
            </button>
          }
        >
          <div className="step-settings">
            {CLEANUP.map((option) => (
              <label
                key={option}
                className="check"
                title={available.includes(option) ? t.cleanupLabel[option][1] : t.noModel}
              >
                <input
                  type="checkbox"
                  checked={settings.clean_up.models.includes(option)}
                  disabled={!available.includes(option) || running}
                  onChange={(e) => {
                    const models = settings.clean_up.models.filter((m) => m !== option)
                    set('clean_up', {
                      models: e.target.checked ? CLEANUP.filter((m) => [...models, option].includes(m)) : models,
                    })
                  }}
                />{' '}
                {t.cleanupLabel[option][0]}
                {!available.includes(option) && <span className="muted">{t.noModelInstalled}</span>}
              </label>
            ))}
          </div>
        </Row>

        <Row
          n={3}
          need={need.slice}
          title={t.stepNames.slice}
          state={
            status.slice.recordings
              ? t.sliceState(status.slice.recordings, formatDuration(status.slice.seconds))
              : t.notNeeded(status.sources.recordings ? t.allSliced : t.nothingToSlice)
          }
          actions={
            <button
              type="button"
              className="ghost"
              disabled={running || !status.slice.recordings || !sliceOk}
              onClick={() => run('slice')}
            >
              {t.run}
            </button>
          }
        >
          <div className="step-settings inline-field">
            <span className="muted hint">{t.clipsFrom}</span>
            <input
              type="number"
              min={0.5}
              step={0.5}
              value={settings.slice.min_seconds}
              onChange={(e) => set('slice', { min_seconds: e.target.valueAsNumber })}
              aria-label={t.minClipLength}
            />
            <span className="muted hint">{t.to}</span>
            <input
              type="number"
              min={3}
              step={1}
              value={settings.slice.max_seconds}
              onChange={(e) => set('slice', { max_seconds: e.target.valueAsNumber })}
              aria-label={t.maxClipLength}
            />
            <span className="muted hint">{t.secondsUnit}</span>
            {!sliceOk && <span className="field-error">{t.minBelowMax}</span>}
          </div>
          <span className="muted hint">{t.sliceStepHint}</span>
        </Row>

        <Row
          n={4}
          need={need.transcribe}
          title={t.stepNames.transcribe}
          state={status.transcribe.clips ? t.transcribeState(status.transcribe.clips) : t.notNeeded(t.allHaveText)}
          actions={
            <button
              type="button"
              className="ghost"
              disabled={running || (!status.transcribe.clips && !settings.transcribe.tidy_fillers)}
              onClick={() => run('transcribe')}
              title={status.transcribe.clips ? undefined : t.tidyOnlyTitle}
            >
              {t.run}
            </button>
          }
        >
          <div className="step-settings">
            <select
              value={settings.transcribe.language}
              onChange={(e) => set('transcribe', { language: e.target.value })}
              aria-label={t.language}
            >
              {LANGUAGES.map((code) => (
                <option key={code} value={code}>
                  {t.languages[code]}
                </option>
              ))}
            </select>
            <input
              value={settings.transcribe.prompt}
              onChange={(e) => set('transcribe', { prompt: e.target.value })}
              placeholder={t.namesPlaceholder}
              aria-label={t.namesAndTerms}
            />
            <label className="check" title={t.tidyOption}>
              <input
                type="checkbox"
                checked={settings.transcribe.tidy_fillers}
                onChange={(e) => set('transcribe', { tidy_fillers: e.target.checked })}
              />{' '}
              {t.tidyFillersShort}
            </label>
          </div>
        </Row>

        <Row
          n={5}
          need={need.check}
          title={t.stepNames.check}
          state={status.check.clips ? t.checkState(status.check.clips) : t.notNeeded(t.allChecked)}
          actions={
            <>
              <button
                type="button"
                className="ghost"
                disabled={running || !status.check.clips}
                onClick={() => run('check')}
              >
                {t.run}
              </button>
              <button
                type="button"
                className="ghost small"
                disabled={running || !clips}
                onClick={() => run('check', true)}
                title={t.checkAllTitle}
              >
                {t.checkAll}
              </button>
            </>
          }
        >
          <div className="step-settings">
            <select
              value={settings.check.aligner}
              onChange={(e) => set('check', { aligner: e.target.value as Aligner })}
              aria-label={t.aligner}
            >
              {(Object.keys(ALIGNER_LABEL) as Aligner[]).map((a) => (
                <option key={a} value={a}>
                  {ALIGNER_LABEL[a]}
                </option>
              ))}
            </select>
            <label className="check">
              <input
                type="checkbox"
                checked={settings.check.compare_asr}
                onChange={(e) => set('check', { compare_asr: e.target.checked })}
              />{' '}
              {t.compareWithAsr}
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={settings.check.detect_singing}
                onChange={(e) => set('check', { detect_singing: e.target.checked })}
              />{' '}
              {t.detectSinging}
            </label>
          </div>
          <details className="tool-help">
            <summary>{t.howChecking}</summary>
            <p className="muted hint">{t.checkingExplained}</p>
          </details>
        </Row>

        <Row
          n={6}
          need={need.tidy}
          title={t.stepNames.tidy}
          state={status.tidy.trimmable ? t.tidyState(status.tidy.trimmable) : t.tidyOptional}
          actions={
            <button
              type="button"
              className="ghost"
              disabled={
                running || !((settings.tidy.trim && status.tidy.trimmable) || settings.tidy.normalize_lufs !== null)
              }
              onClick={() => run('tidy')}
            >
              {t.run}
            </button>
          }
        >
          <div className="step-settings">
            <label className="check" title={t.trimEdges}>
              <input
                type="checkbox"
                checked={settings.tidy.trim}
                onChange={(e) => set('tidy', { trim: e.target.checked })}
              />{' '}
              {t.trimEdgesShort}
            </label>
            <label className="check" title={t.normalizeHint}>
              <input
                type="checkbox"
                checked={settings.tidy.normalize_lufs !== null}
                onChange={(e) => set('tidy', { normalize_lufs: e.target.checked ? DEFAULT_LUFS : null })}
              />{' '}
              {t.normalizeTo}
            </label>
            <input
              type="number"
              min={-40}
              max={-10}
              step={1}
              value={settings.tidy.normalize_lufs ?? DEFAULT_LUFS}
              disabled={settings.tidy.normalize_lufs === null}
              onChange={(e) => set('tidy', { normalize_lufs: e.target.valueAsNumber })}
              aria-label={t.targetLufs}
            />
            <span className="muted hint">LUFS</span>
          </div>
        </Row>
      </div>

      {job && <JobProgress job={job} what={t.stepWhat} />}
      {done && <p className="muted hint">{done}</p>}
      {error && <p className="field-error">{error}</p>}
    </div>
  )
}
