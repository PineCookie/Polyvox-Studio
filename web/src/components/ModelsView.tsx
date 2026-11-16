import { useCallback, useEffect, useState } from 'react'
import { cancelJob, downloadModels, listModels, setDownloadSource } from '../api'
import type { DownloadJob, DownloadWayId, ModelGroup, ModelInfo, ModelList } from '../api'
import { formatDuration } from '../format'
import { useT } from '../i18n'
import type { Messages } from '../i18n'
import { useJob } from '../useJob'
import { ProgressBar } from './RunList'

interface Props {
  onError: (message: string) => void
  // After a download: features that needed a model may be available now.
  onChanged: () => void
}

const SOURCES = ['', 'https://hf-mirror.com'] // labels: t.sources
const CUSTOM = 'custom'
// Where earlier versions kept the source in the browser; it moves to the server once.
const OLD_SOURCE_KEY = 'polyvox.hf-endpoint'

function takeOldSource(): string | null {
  try {
    const old = localStorage.getItem(OLD_SOURCE_KEY)
    localStorage.removeItem(OLD_SOURCE_KEY)
    return old
  } catch {
    return null
  }
}

const formatSize = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`)
const GROUPS: ModelGroup[] = ['tts', 'dataset', 'cleanup'] // the server's models.GROUPS

/** What downloading ``m`` the chosen way fetches, in MB. */
function downloadSize(m: ModelInfo, way: DownloadWayId | undefined): number {
  return m.ways.find((w) => w.id === (way ?? 'prebuilt'))?.size_mb ?? m.size_mb
}
const formatSpeed = (mbPerS: number) => (mbPerS >= 10 ? `${Math.round(mbPerS)} MB/s` : `${mbPerS.toFixed(1)} MB/s`)

function stateText(m: ModelInfo, t: Messages): [string, string] {
  if (m.state === 'downloaded') return [t.downloaded, 'ok']
  if (m.state === 'custom') return [t.setByEnv, 'subtle']
  return m.fallback ? [t.notDownloaded, 'warn'] : [t.notDownloadedUnavailable, 'warn']
}

/** The models the studio uses: what is downloaded, and downloading the rest (from Hugging Face or a mirror). */
export function ModelsView({ onError, onChanged }: Props) {
  const t = useT()
  const [list, setList] = useState<ModelList | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [ways, setWays] = useState<Record<string, DownloadWayId>>({}) // by setting, for models with a choice
  const [source, setSource] = useState<string | null>(null) // as edited here; else the server's
  const [job, setJob] = useJob<DownloadJob>((j) => {
    refresh()
    onChanged()
    if (j.state === 'failed') onError(t.downloadFailed(j.error ?? ''))
  })

  const refresh = useCallback(() => {
    listModels().then(
      (l) => {
        setList(l)
        // Keep the ticks of what is still missing (a finished download drops its own).
        const missing = new Set(l.models.filter((m) => m.state === 'missing').map((m) => m.setting))
        setSelected((s) => s.filter((setting) => missing.has(setting)))
      },
      (e: Error) => onError(e.message),
    )
  }, [onError])

  useEffect(refresh, [refresh])

  useEffect(() => {
    const old = takeOldSource()
    if (old !== null) setDownloadSource(old).then(refresh, () => {})
  }, [refresh])

  if (!list) return <section className="panel muted">{t.loading}</section>

  const chosen = source ?? list.endpoint
  const isPreset = SOURCES.includes(chosen)
  const missing = list.models.filter((m) => m.state === 'missing')
  const total = list.models
    .filter((m) => selected.includes(m.setting))
    .reduce((sum, m) => sum + downloadSize(m, ways[m.setting]), 0)
  const running = job?.state === 'running'
  const current = running ? list.models.find((m) => m.setting === job.phase) : undefined

  // The server keeps the source (also for models that download themselves when first used).
  function saveSource(endpoint: string) {
    setDownloadSource(endpoint).then(setSource, (e: Error) => onError(e.message))
  }

  function chooseSource(value: string) {
    if (value === CUSTOM) {
      setSource('https://') // saved once typed (on leaving the field, or with a download)
    } else {
      setSource(value)
      saveSource(value)
    }
  }

  async function start() {
    try {
      const picked = Object.fromEntries(Object.entries(ways).filter(([setting]) => selected.includes(setting)))
      setJob(await downloadModels(selected, chosen, picked))
    } catch (e) {
      onError((e as Error).message)
    }
  }

  const toggle = (setting: string) =>
    setSelected((s) => (s.includes(setting) ? s.filter((x) => x !== setting) : [...s, setting]))

  return (
    <section className="panel models">
      <header className="panel-head">
        <h2>{t.models}</h2>
        <span className="muted hint" title="POLYVOX_MODELS_DIR">
          {list.models_dir}
        </span>
      </header>
      <p className="muted hint">{t.modelsIntro}</p>

      <div className="table-wrap">
        <table className="metrics models-table">
          {GROUPS.map((group) => {
            const rows = list.models.filter((m) => m.group === group)
            if (!rows.length) return null
            return (
              <tbody key={group}>
                <tr className="models-group">
                  <th colSpan={5}>{t.modelGroups[group] ?? group}</th>
                </tr>
                {rows.map((m) => {
                  const [label, tone] = stateText(m, t)
                  const page =
                    m.hub === 'modelscope'
                      ? `https://modelscope.cn/models/${m.repo}`
                      : `https://huggingface.co/${m.repo}`
                  const choosing = m.ways.length > 0 && m.state === 'missing'
                  return (
                    <tr key={m.setting}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selected.includes(m.setting)}
                          disabled={m.state !== 'missing' || running}
                          onChange={() => toggle(m.setting)}
                          aria-label={t.downloadModel(m.name)}
                        />
                      </td>
                      <td>
                        <strong>{m.name}</strong>
                        <div className="muted hint">{t.modelPurpose[m.setting] ?? m.purpose}</div>
                        {choosing && (
                          <div className="models-way">
                            <select
                              value={ways[m.setting] ?? 'prebuilt'}
                              onChange={(e) => setWays((w) => ({ ...w, [m.setting]: e.target.value as DownloadWayId }))}
                              disabled={running}
                              aria-label={t.downloadModel(m.name)}
                            >
                              {m.ways.map((w) => (
                                <option key={w.id} value={w.id}>
                                  {t.downloadWays[w.id]?.(formatSize(w.size_mb)) ?? w.id}
                                </option>
                              ))}
                            </select>
                            <div className="muted hint">{t.downloadWaysHint}</div>
                          </div>
                        )}
                      </td>
                      <td className="num">{formatSize(choosing ? downloadSize(m, ways[m.setting]) : m.size_mb)}</td>
                      <td>
                        <span className={`tag ${tone}`} title={m.state === 'missing' ? undefined : m.path}>
                          {label}
                        </span>
                      </td>
                      <td className="muted hint">
                        <a href={page} target="_blank" rel="noreferrer">
                          {m.repo}
                        </a>
                        {m.hub === 'modelscope' && ' (ModelScope)'}
                        {` · ${m.license || t.licenseNotStated}`}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            )
          })}
        </table>
      </div>

      <div className="models-actions">
        <label className="field">
          <span>{t.downloadFrom}</span>
          <select value={isPreset ? chosen : CUSTOM} onChange={(e) => chooseSource(e.target.value)} disabled={running}>
            {SOURCES.map((s) => (
              <option key={s} value={s}>
                {t.sources[s]}
              </option>
            ))}
            <option value={CUSTOM}>{t.anotherMirror}</option>
          </select>
          <span className="muted hint">{t.downloadSourceHint}</span>
        </label>
        {!isPreset && (
          <label className="field">
            <span>{t.mirrorUrl}</span>
            <input
              value={chosen}
              onChange={(e) => setSource(e.target.value)}
              onBlur={() => /^https?:\/\/[^/]+/.test(chosen) && saveSource(chosen)}
              placeholder="https://"
              disabled={running}
            />
          </label>
        )}
        <div className="models-buttons">
          {missing.length > selected.length && (
            <button
              type="button"
              className="ghost small"
              onClick={() => setSelected(missing.map((m) => m.setting))}
              disabled={running}
            >
              {t.selectAllMissing}
            </button>
          )}
          <button type="button" className="primary" onClick={start} disabled={running || selected.length === 0}>
            {selected.length ? t.downloadSelected(selected.length, formatSize(total)) : t.nothingSelected}
          </button>
        </div>
      </div>

      {job && running && (
        <div className="job-progress">
          <span>
            {t.downloadingModel(current?.name ?? '…')}
            {job.total > 0 && (
              <span className="muted">
                {' '}
                {formatSize(job.done)} / {formatSize(job.total)}
                {job.rate != null && job.rate > 0 && ` · ${t.downloadSpeed(formatSpeed(job.rate))}`}
                {job.seconds_left != null && job.seconds_left > 0 && t.timeLeft(formatDuration(job.seconds_left))}
              </span>
            )}
          </span>
          <ProgressBar value={job.done} max={job.total || 1} />
          <button type="button" className="ghost small" onClick={() => cancelJob(job.id)}>
            {t.cancel}
          </button>
        </div>
      )}
      {job?.state === 'cancelled' && <p className="muted hint">{t.downloadCancelled}</p>}
      {job?.state === 'done' && <p className="muted hint">{t.downloadedDone}</p>}
    </section>
  )
}
