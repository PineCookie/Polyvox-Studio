import { useEffect, useMemo, useRef, useState } from 'react'
import { convertFileUrl, convertRecording } from '../api'
import type { ConvertJob, Engine, EngineOptions, Lora, Status } from '../api'
import { formatDuration, formatStep } from '../format'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { AudioPlayer } from './AudioPlayer'
import { FilePicker } from './FilePicker'
import { DownloadIcon } from './icons'
import { Info } from './Info'
import { JobProgress } from './JobProgress'
import { RvcImportDialog } from './RvcImportDialog'
import { ParamSlider } from './SynthesisPanel'

interface Result {
  job: ConvertJob
  recording: string // the file's name
  voice: string // the voice's name
}

interface Props {
  engine: Engine | null // the conversion engine (RVC); null while loading
  loras: Lora[]
  status: Status | null
  onVoicesChanged: () => void
  onError: (message: string) => void
  onOpenModels: () => void
  hidden: boolean
}

const VOICE_KEY = 'polyvox.convertVoice' // the voice chosen last

function defaults(engine: Engine | null): EngineOptions {
  return Object.fromEntries((engine?.params ?? []).map((p) => [p.name, p.default]))
}

/** Convert tab: a song or vocal recording sung / spoken again in an RVC voice. */
export function ConvertView({ engine, loras, status, onVoicesChanged, onError, onOpenModels, hidden }: Props) {
  const t = useT()
  const voices = useMemo(() => loras.filter((l) => l.engine === engine?.id), [loras, engine])
  const [voice, setVoice] = useState<string>(() => localStorage.getItem(VOICE_KEY) ?? '')
  const [file, setFile] = useState<File | null>(null)
  const [song, setSong] = useState(true)
  const [dereverb, setDereverb] = useState(true)
  const [shiftMusic, setShiftMusic] = useState(true)
  const [options, setOptions] = useState<EngineOptions>(() => defaults(engine))
  const [importing, setImporting] = useState(false)
  const [results, setResults] = useState<Result[]>([])
  const pending = useRef<Omit<Result, 'job'> | null>(null)
  const [job, setJob] = useJob<ConvertJob>((j) => {
    if (j.state === 'done' && pending.current) setResults((prev) => [{ job: j, ...pending.current! }, ...prev])
    if (j.state === 'failed') onError(j.error ?? t.jobFailed(t.convertWhat))
  })
  const running = job?.state === 'running'

  useEffect(() => setOptions((o) => ({ ...defaults(engine), ...o })), [engine])
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => void (preview && URL.revokeObjectURL(preview)), [preview])

  const chosen = voices.find((v) => v.id === voice) ?? null
  const cleanup = status?.cleanup ?? []
  const needsModels = song && (!cleanup.includes('separate') || (dereverb && !cleanup.includes('dereverb')))
  const blocker = engine?.unavailable
    ? t.engineUnavailable(engine.name, engine.unavailable)
    : needsModels
      ? t.convertNeedsSeparation
      : null
  const canConvert = !!engine && !!file && !!chosen && !blocker && !running

  function choose(id: string) {
    setVoice(id)
    try {
      localStorage.setItem(VOICE_KEY, id)
    } catch {
      // Remembered for this session only.
    }
  }

  async function start() {
    if (!canConvert || !file || !chosen) return
    try {
      pending.current = { recording: file.name, voice: chosen.voice }
      setJob(
        await convertRecording({
          audio: file,
          voice: chosen.id,
          options,
          separate: song,
          dereverb,
          shiftInstrumental: shiftMusic,
          seed: null,
        }),
      )
    } catch (e) {
      onError((e as Error).message)
    }
  }

  const params = engine?.params ?? []
  const sliders = params.filter((p) => p.kind !== 'bool')
  const toggles = params.filter((p) => p.kind === 'bool')

  return (
    <main className="synth convert" hidden={hidden}>
      <div className="synth-main">
        <section className="composer convert-input">
          <p className="muted hint">{t.convertIntro}</p>
          <FilePicker
            file={file}
            onChange={setFile}
            accept="audio/*,video/*"
            label={t.convertChoose}
            empty={t.convertNoFile}
          >
            {preview && <AudioPlayer url={preview} />}
          </FilePicker>
          <div className="segmented" role="radiogroup" aria-label={t.convertKind}>
            <label>
              <input type="radio" checked={song} onChange={() => setSong(true)} /> {t.convertSong}
            </label>
            <label>
              <input type="radio" checked={!song} onChange={() => setSong(false)} /> {t.convertVocals}
            </label>
          </div>
          <div className="composer-foot">
            {blocker && <span className="field-error composer-blocker">{blocker}</span>}
            {needsModels && (
              <button type="button" className="ghost small" onClick={onOpenModels}>
                {t.openModels}
              </button>
            )}
            <span className="spacer" />
            <button type="button" className="primary generate" disabled={!canConvert} onClick={start}>
              {running && <span className="spinner" aria-hidden />}
              {running ? t.converting : t.convertButton}
            </button>
          </div>
          {job && job.state !== 'done' && <JobProgress job={job} what={t.convertWhat} />}
        </section>

        <section className="takes" aria-label={t.convertResults}>
          <header className="takes-head">
            <h2>{t.convertResults}</h2>
            {results.length > 0 && <span className="muted">{results.length}</span>}
          </header>
          {results.length === 0 ? (
            <p className="takes-empty muted">{t.convertResultsEmpty}</p>
          ) : (
            <ol className="take-list">
              {results.map((r) => (
                <li key={r.job.id} className="take">
                  <div className="take-top">
                    <p className="take-text">{r.recording}</p>
                  </div>
                  {r.job.result.files.map((name) => (
                    <div key={name} className="convert-file">
                      <span className="convert-file-name">{t.convertFiles[name] ?? name}</span>
                      <AudioPlayer url={convertFileUrl(r.job.id, name)} />
                      <a
                        className="icon-button"
                        href={convertFileUrl(r.job.id, name)}
                        download={`${r.recording.replace(/\.[^.]+$/, '')}_${r.voice}_${name}.wav`}
                        title={t.download}
                        aria-label={t.download}
                      >
                        <DownloadIcon />
                      </a>
                    </div>
                  ))}
                  <p className="take-meta">
                    <span className="take-voice">{r.voice}</span>
                    <span>
                      {t.convertMeta(formatDuration(r.job.result.seconds), formatDuration(r.job.result.took))}
                    </span>
                  </p>
                </li>
              ))}
            </ol>
          )}
        </section>
      </div>

      <aside className="synth-side">
        <div className="settings">
          <section className="settings-section">
            <div className="settings-head">
              <h3>{t.convertVoice}</h3>
            </div>
            {voices.length === 0 ? (
              <p className="muted hint">{t.convertNoVoices}</p>
            ) : (
              <select value={chosen?.id ?? ''} onChange={(e) => choose(e.target.value)} aria-label={t.convertVoice}>
                <option value="">{t.convertChooseVoice}</option>
                {voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.step !== null ? `${v.voice} · ${t.stepLabel(formatStep(v.step))}` : v.voice}
                  </option>
                ))}
              </select>
            )}
            <button type="button" className="ghost small" onClick={() => setImporting(true)} disabled={!engine}>
              {t.rvcImportButton}
            </button>
          </section>

          <section className="settings-section">
            <div className="settings-head">
              <h3>{t.settingsTitle}</h3>
            </div>
            {sliders.map((p) => (
              <ParamSlider
                key={p.name}
                param={{ ...p, ...(p.name === 'shift' ? { min: -12, max: 12 } : {}) }}
                value={Number(options[p.name] ?? p.default)}
                onChange={(v) => setOptions((o) => ({ ...o, [p.name]: v }))}
              />
            ))}
            <div className="toggles">
              {toggles.map((p) => (
                <label key={p.name} className="switch">
                  <span className="label-text">
                    {t.paramLabels[p.name] ?? p.name}
                    {t.paramHints[p.name] && <Info text={t.paramHints[p.name]} />}
                  </span>
                  <input
                    type="checkbox"
                    role="switch"
                    checked={Boolean(options[p.name] ?? p.default)}
                    onChange={(e) => setOptions((o) => ({ ...o, [p.name]: e.target.checked }))}
                  />
                </label>
              ))}
              {song && (
                <>
                  <label className="switch">
                    <span className="label-text">
                      {t.convertDereverb}
                      <Info text={t.convertDereverbHint} />
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      checked={dereverb}
                      onChange={(e) => setDereverb(e.target.checked)}
                    />
                  </label>
                  <label className="switch">
                    <span className="label-text">
                      {t.convertShiftMusic}
                      <Info text={t.convertShiftMusicHint} />
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      checked={shiftMusic}
                      onChange={(e) => setShiftMusic(e.target.checked)}
                    />
                  </label>
                </>
              )}
            </div>
          </section>
        </div>
      </aside>

      {importing && (
        <RvcImportDialog
          onImported={(id) => {
            onVoicesChanged()
            choose(id)
          }}
          onError={onError}
          onClose={() => setImporting(false)}
        />
      )}
    </main>
  )
}
