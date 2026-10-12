import { useEffect, useRef, useState } from 'react'
import { deleteDataset, getDeletion } from '../api'
import type { DatasetDeletion } from '../api'
import { datasetName } from '../format'
import { useT } from '../i18n'

interface Props {
  id: string
  onDeleted: () => void
  onClose: () => void
}

const formatBytes = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${Math.round(n / 1e6)} MB` : `${Math.ceil(n / 1e3)} KB`

/** Asks before deleting a dataset: what goes, which runs it affects; nothing happens without the second click. */
export function DeleteDatasetDialog({ id, onDeleted, onClose }: Props) {
  const t = useT()
  const dialog = useRef<HTMLDialogElement>(null)
  const [info, setInfo] = useState<DatasetDeletion | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    dialog.current?.showModal()
    getDeletion(id).then(setInfo, (e) => setError((e as Error).message))
  }, [id])

  async function confirm() {
    setBusy(true)
    setError(null)
    try {
      await deleteDataset(id)
      onDeleted()
    } catch (e) {
      setError((e as Error).message)
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
      <div className="dialog-body">
        <header className="panel-head">
          <h2>{t.deleteDatasetTitle(datasetName(id))}</h2>
        </header>
        {info ? (
          <>
            <p>{t.deleteDatasetWhat(info.folder, info.clips, formatBytes(info.bytes))}</p>
            {info.runs.length > 0 && (
              <p className="delete-warning">{t.deleteDatasetRuns(info.runs.length, info.runs.join(', '))}</p>
            )}
            {info.outside_recordings > 0 && (
              <p className="muted hint">{t.deleteDatasetOutside(info.outside_recordings)}</p>
            )}
            <p className="delete-warning">
              <strong>{t.cannotUndo}</strong>
            </p>
            {info.blocked && <p className="field-error">{t.deleteBlocked[info.blocked]}</p>}
          </>
        ) : (
          !error && <p className="muted hint">{t.checkingAudio}</p>
        )}
        {error && <p className="field-error">{error}</p>}
        <div className="actions">
          <button type="button" className="ghost" onClick={onClose} autoFocus>
            {t.cancel}
          </button>
          <button type="button" className="danger" onClick={confirm} disabled={busy || !info || !!info.blocked}>
            {busy ? t.deleting : t.deleteForGood}
          </button>
        </div>
      </div>
    </dialog>
  )
}
