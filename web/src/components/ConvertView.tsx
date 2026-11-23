import { useEffect, useMemo, useRef, useState } from 'react'
import { convertFileUrl, convertRecording, importLora } from '../api'
import type { BackingMode, ConvertJob, Engine, EngineOptions, Lora, Status } from '../api'
import { formatDuration, formatStep } from '../format'
import { importDialog } from '../importDialogs'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { AudioPlayer } from './AudioPlayer'
import { ExportDialog } from './ExportDialog'
import { FilePicker } from './FilePicker'
import { DownloadIcon, PlusIcon } from './icons'
import { Info } from './Info'
import { JobProgress } from './JobProgress'
import { ParamSlider } from './SynthesisPanel'

interface Result {
  job: ConvertJob
  recording: string // the file's name
  voice: string // the voice's name
  how: string // what was converted (lead only, lead and backing, all vocals)
}

interface Props {
  engine: Engine | null // the conversion engine (RVC); null while loading
  loras: Lora[]
  status: Status | null
  onVoicesChanged: () => void
  onError: (message: string) => void
  onOpenModels: () => void
  hidden: boolean
  // A voice to select (a run's checkpoint from the Train tab); key: a new request each time.
  pick?: { id: string; key: number } | null
}

const VOICE_KEY = 'polyvox.convertVoice' // the voice chosen last

function defaults(engine: Engine | null): EngineOptions {
  return Object.fromEntries((engine?.params ?? []).map((p) => [p.name, p.default]))
}

/** Convert tab: a song or vocal recording sung / spoken again in an RVC voice. */
export function ConvertView({ engine, loras, status, onVoicesChanged, onError, onOpenModels, hidden, pick }: Props) {
  const t = useT()
  const voices = useMemo(() => loras.filter((l) => l.engine === engine?.id), [loras, engine])
  const importer = importDialog(engine)
  const [voice, setVoice] = useState<string>(() => localStorage.getItem(VOICE_KEY) ?? '')
  const [file, setFile] = useState<File | null>(null)
  const [song, setSong] = useState(true)
  const [dereverb, setDereverb] = useState(true)
  // Lead and backing vocals apart: on for songs (harmonies would blur into the one converted voice),
  // off for a vocal or speech recording unless asked.
  const [split, setSplit] = useState({ song: true, vocals: false })
  const [backing, setBacking] = useState<BackingMode>('keep')
  const [shiftMusic, setShiftMusic] = useState(true)
  const [options, setOptions] = useState<EngineOptions>(() => defaults(engine))
  const [importing, setImporting] = useState(false)
  const [unpacking, setUnpacking] = useState(false) // a package being imported
  const [exporting, setExporting] = useState(false)
  const packageInput = useRef<HTMLInputElement>(null)
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

  useEffect(() => {
    if (pick) choose(pick.id)
  }, [pick])
  const chosen = voices.find((v) => v.id === voice) ?? null
  const cleanup = status?.cleanup ?? []
  const canSplit = !!status?.karaoke
  const splitting = canSplit && split[song ? 'song' : 'vocals']
  const needsModels = (song && !cleanup.includes('separate')) || (dereverb && !cleanup.includes('dereverb'))
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

  async function importPackage(file: File | undefined) {
    if (!file) return
    setUnpacking(true)
    try {
      const { id } = await importLora(file)
      onVoicesChanged()
      choose(id)
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setUnpacking(false)
      if (packageInput.current) packageInput.current.value = ''
    }
  }

  async function start() {
    if (!canConvert || !file || !chosen) return
    try {
      const how = !splitting ? t.convertHowAll : backing === 'keep' ? t.convertHowLead : t.convertHowBoth
      pending.current = { recording: file.name, voice: chosen.voice, how }
      setJob(
        await convertRecording({
          audio: file,
          voice: chosen.id,
          options,
          separate: song,
          dereverb,
          split: splitting,
          backing,
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
                    <span>{r.how}</span>
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
              {chosen && (
                <button type="button" className="ghost small" onClick={() => setExporting(true)}>
                  {t.export}
                </button>
              )}
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
            <div className="convert-imports">
              {importer && (
                <button
                  type="button"
                  className="ghost small"
                  onClick={() => setImporting(true)}
                  title={importer.title?.(t)}
                >
                  <PlusIcon /> {importer.label(t)}
                </button>
              )}
              <input
                ref={packageInput}
                type="file"
                accept=".zip,application/zip"
                hidden
                onChange={(e) => importPackage(e.target.files?.[0])}
              />
              <button
                type="button"
                className="ghost small"
                onClick={() => packageInput.current?.click()}
                disabled={!engine || unpacking}
                title={t.rvcPackageTitle}
              >
                <PlusIcon /> {unpacking ? t.importing : t.importPackage}
              </button>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-head">
              <h3>{t.convertStepsTitle}</h3>
            </div>
            <ol className="convert-steps">
              {song && (
                <li>
                  <div className="switch">
                    <span className="label-text">
                      {t.convertStepSeparate}
                      <Info text={t.convertStepSeparateHint} />
                    </span>
                    <span className="muted">{cleanup.includes('separate') ? t.convertAlways : t.notDownloaded}</span>
                  </div>
                </li>
              )}
              <li>
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
              </li>
              <li>
                <label className="switch">
                  <span className="label-text">
                    {t.convertSplit}
                    <Info text={t.convertSplitHint} />
                  </span>
                  <input
                    type="checkbox"
                    role="switch"
                    checked={splitting}
                    disabled={!canSplit}
                    onChange={(e) => setSplit((s) => ({ ...s, [song ? 'song' : 'vocals']: e.target.checked }))}
                  />
                </label>
                {!canSplit && (
                  <div className="convert-step-note">
                    <span className="muted hint">{t.convertSplitMissing}</span>
                    <button type="button" className="ghost small" onClick={onOpenModels}>
                      {t.openModels}
                    </button>
                  </div>
                )}
                {splitting && (
                  <div className="segmented" role="radiogroup" aria-label={t.convertBacking}>
                    <label>
                      <input type="radio" checked={backing === 'keep'} onChange={() => setBacking('keep')} />{' '}
                      {t.convertBackingKeep}
                    </label>
                    <label>
                      <input type="radio" checked={backing === 'convert'} onChange={() => setBacking('convert')} />{' '}
                      {t.convertBackingConvert}
                    </label>
                  </div>
                )}
              </li>
              <li>
                <div className="switch">
                  <span className="label-text">{splitting ? t.convertStepLead : t.convertStepVocals}</span>
                  <span className="muted">{t.convertStepSettingsBelow}</span>
                </div>
              </li>
              {song && (
                <li>
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
                </li>
              )}
            </ol>
          </section>

          <section className="settings-section">
            <div className="settings-head">
              <h3>{t.convertVoiceSettings}</h3>
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
            </div>
          </section>
        </div>
      </aside>

      {importing && importer && (
        <importer.Dialog
          onImported={(id) => {
            onVoicesChanged()
            choose(id)
          }}
          onError={onError}
          onClose={() => setImporting(false)}
        />
      )}
      {exporting && chosen && (
        <ExportDialog
          lora={chosen}
          engine={engine}
          // A run's last checkpoint is named after the run alone.
          final={!voices.some((v) => v.voice_path === chosen.voice_path && (v.step ?? -1) > (chosen.step ?? -1))}
          onError={onError}
          onClose={() => setExporting(false)}
        />
      )}
    </main>
  )
}
