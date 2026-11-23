import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { importVoice, listDatasets } from '../api'
import { useT } from '../i18n'

interface Props {
  onImported: (id: string, engine: string) => void
  onError: (message: string) => void
  onClose: () => void
}

const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

// "X:\rvc\assets\weights\voice_e100_s4200.pth" -> "voice" (the WebUI appends epoch and step).
function nameFromPath(path: string): string {
  const file = path.replace(/^"|"$/g, '').split(/[\\/]/).at(-1) ?? ''
  return file.replace(/\.pth$/i, '').replace(/[_-]e\d+(_s\d+)?$/i, '')
}

/** Import a voice the RVC WebUI trained; its index is built from a dataset or the WebUI's logs folder. */
export function RvcImportDialog({ onImported, onError, onClose }: Props) {
  const t = useT()
  const dialog = useRef<HTMLDialogElement>(null)
  const [path, setPath] = useState('')
  const [name, setName] = useState<string | null>(null) // null: follows the file name
  const [source, setSource] = useState<'dataset' | 'logs' | 'none'>('dataset')
  const [datasets, setDatasets] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const [logs, setLogs] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    dialog.current?.showModal()
    listDatasets().then(setDatasets, () => {})
  }, [])

  const voiceName = name ?? nameFromPath(path)
  const nameOk = NAME_PATTERN.test(voiceName)
  const indexOk = source === 'none' || (source === 'dataset' ? !!dataset : !!logs.trim())
  const ready = nameOk && !!path.trim() && indexOk

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!ready) return
    setBusy(true)
    try {
      const index =
        source === 'dataset' ? { index_dataset: dataset } : source === 'logs' ? { index_logs: logs.trim() } : {}
      const { id, engine } = await importVoice('rvc', { name: voiceName, path: path.trim(), ...index })
      onImported(id, engine)
      onClose()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <dialog
      ref={dialog}
      className="dialog"
      onClose={onClose}
      onClick={(e) => e.target === dialog.current && onClose()} // backdrop
    >
      <form className="dialog-body" onSubmit={submit}>
        <header className="panel-head">
          <h2>{t.rvcImportTitle}</h2>
        </header>
        <p className="muted hint">{t.rvcImportIntro}</p>

        <label className="field">
          <span>{t.rvcVoiceFile}</span>
          <input value={path} onChange={(e) => setPath(e.target.value)} placeholder={t.rvcVoiceExample} autoFocus />
        </label>
        <label className="field">
          <span>
            {t.name} {voiceName && !nameOk && <span className="field-error">{t.nameRule}</span>}
          </span>
          <input value={voiceName} onChange={(e) => setName(e.target.value)} />
        </label>

        <fieldset className="field">
          <legend>{t.rvcIndex}</legend>
          <div className="segmented" role="radiogroup">
            <label>
              <input type="radio" checked={source === 'dataset'} onChange={() => setSource('dataset')} />{' '}
              {t.rvcIndexDataset}
            </label>
            <label>
              <input type="radio" checked={source === 'logs'} onChange={() => setSource('logs')} /> {t.rvcIndexLogs}
            </label>
            <label>
              <input type="radio" checked={source === 'none'} onChange={() => setSource('none')} /> {t.rvcIndexNone}
            </label>
          </div>
          {source === 'dataset' ? (
            <>
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                <option value="">{t.chooseDatasetOption}</option>
                {datasets.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
              <span className="muted hint">{t.rvcIndexDatasetHint}</span>
            </>
          ) : source === 'logs' ? (
            <>
              <input value={logs} onChange={(e) => setLogs(e.target.value)} placeholder={t.rvcLogsExample} />
              <span className="muted hint">{t.rvcIndexLogsHint}</span>
            </>
          ) : (
            <span className="muted hint">{t.rvcIndexNoneHint}</span>
          )}
        </fieldset>

        <div className="actions">
          <button type="button" className="ghost" onClick={onClose}>
            {t.cancel}
          </button>
          <button type="submit" className="primary" disabled={busy || !ready}>
            {busy ? t.importing : t.import}
          </button>
        </div>
      </form>
    </dialog>
  )
}
