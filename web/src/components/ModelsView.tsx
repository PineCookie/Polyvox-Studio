import { useCallback, useEffect, useState } from 'react'
import { cancelJob, downloadModels, listModels } from '../api'
import type { DownloadJob, ModelInfo, ModelList } from '../api'
import { useJob } from '../useJob'
import { ProgressBar } from './RunList'

interface Props {
  onError: (message: string) => void
  // After a download: features that needed a model may be available now.
  onChanged: () => void
}

const SOURCES = [
  { value: '', label: 'Hugging Face (huggingface.co)' },
  { value: 'https://hf-mirror.com', label: 'hf-mirror.com (a mirror, e.g. for mainland China)' },
]
const CUSTOM = 'custom'
const SOURCE_KEY = 'voxcpm-studio.hf-endpoint'

function storedSource(): string | null {
  try {
    return localStorage.getItem(SOURCE_KEY)
  } catch {
    return null
  }
}

const formatSize = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`)

function stateText(m: ModelInfo): [string, string] {
  if (m.state === 'downloaded') return ['Downloaded', 'ok']
  if (m.state === 'custom') return ['Set by environment', 'subtle']
  return m.fallback ? ['Not downloaded', 'warn'] : ['Not downloaded · option unavailable', 'warn']
}

/** The models the studio uses: what is downloaded, and downloading the rest (from Hugging Face or a mirror). */
export function ModelsView({ onError, onChanged }: Props) {
  const [list, setList] = useState<ModelList | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [source, setSource] = useState<string | null>(storedSource)
  const [job, setJob] = useJob<DownloadJob>((j) => {
    refresh()
    onChanged()
    if (j.state === 'failed') onError(`Download failed: ${j.error}`)
  })

  const refresh = useCallback(() => {
    listModels().then(
      (l) => {
        setList(l)
        // Preselect what is required and missing.
        setSelected(l.models.filter((m) => m.state === 'missing' && m.required).map((m) => m.setting))
      },
      (e: Error) => onError(e.message),
    )
  }, [onError])

  useEffect(refresh, [refresh])

  if (!list) return <section className="panel muted">Loading…</section>

  // The server's default unless one was chosen in this browser.
  const chosen = source ?? list.endpoint
  const isPreset = SOURCES.some((s) => s.value === chosen)
  const missing = list.models.filter((m) => m.state === 'missing')
  const total = list.models.filter((m) => selected.includes(m.setting)).reduce((sum, m) => sum + m.size_mb, 0)
  const running = job?.state === 'running'
  const current = running ? list.models.find((m) => m.setting === job.phase) : undefined

  function chooseSource(value: string) {
    const next = value === CUSTOM ? 'https://' : value
    setSource(next)
    try {
      localStorage.setItem(SOURCE_KEY, next)
    } catch {
      // Not remembered; still used now.
    }
  }

  async function start() {
    try {
      setJob(await downloadModels(selected, chosen))
    } catch (e) {
      onError((e as Error).message)
    }
  }

  const toggle = (setting: string) =>
    setSelected((s) => (s.includes(setting) ? s.filter((x) => x !== setting) : [...s, setting]))

  return (
    <section className="panel models">
      <header className="panel-head">
        <h2>Models</h2>
        <span className="muted hint" title="VOXCPM_STUDIO_MODELS_DIR">
          {list.models_dir}
        </span>
      </header>
      <p className="muted hint">
        Downloaded models live in the models folder and are used from there. A Hugging Face model that is not downloaded
        still loads, downloading to Hugging Face's cache the first time (with no progress shown); the import's clean-up
        options need their model downloaded here.
      </p>

      <div className="table-wrap">
        <table className="metrics models-table">
          <tbody>
            {list.models.map((m) => {
              const [label, tone] = stateText(m)
              const page =
                m.hub === 'modelscope' ? `https://modelscope.cn/models/${m.repo}` : `https://huggingface.co/${m.repo}`
              return (
                <tr key={m.setting}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.includes(m.setting)}
                      disabled={m.state !== 'missing' || running}
                      onChange={() => toggle(m.setting)}
                      aria-label={`Download ${m.name}`}
                    />
                  </td>
                  <td>
                    <strong>{m.name}</strong>
                    {m.required && <span className="tag subtle">Required</span>}
                    <div className="muted hint">{m.purpose}</div>
                  </td>
                  <td className="num">{formatSize(m.size_mb)}</td>
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
                    {m.license && ` · ${m.license}`}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="models-actions">
        <label className="field">
          <span>Download from</span>
          <select value={isPreset ? chosen : CUSTOM} onChange={(e) => chooseSource(e.target.value)} disabled={running}>
            {SOURCES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
            <option value={CUSTOM}>Another mirror…</option>
          </select>
        </label>
        {!isPreset && (
          <label className="field">
            <span>Mirror URL</span>
            <input
              value={chosen}
              onChange={(e) => chooseSource(e.target.value)}
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
              Select all missing
            </button>
          )}
          <button type="button" className="primary" onClick={start} disabled={running || selected.length === 0}>
            {selected.length ? `Download ${selected.length} (${formatSize(total)})` : 'Nothing selected'}
          </button>
        </div>
      </div>

      {job && running && (
        <div className="job-progress">
          <span>
            Downloading {current?.name ?? '…'}
            {job.total > 0 && (
              <span className="muted">
                {' '}
                {formatSize(job.done)} / {formatSize(job.total)}
              </span>
            )}
          </span>
          <ProgressBar value={job.done} max={job.total || 1} />
          <button type="button" className="ghost small" onClick={() => cancelJob(job.id)}>
            Cancel
          </button>
        </div>
      )}
      {job?.state === 'cancelled' && (
        <p className="muted hint">Cancelled; a file in flight finishes in the background, and the rest resumes.</p>
      )}
      {job?.state === 'done' && <p className="muted hint">Downloaded.</p>}
    </section>
  )
}
