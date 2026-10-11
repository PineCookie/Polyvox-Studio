import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { importGptSovits, listDatasets } from '../api'
import { useT } from '../i18n'

interface Props {
  onImported: (id: string, engine: string) => void
  onError: (message: string) => void
  onClose: () => void
}

const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

// "D:\x\SoVITS_weights_v2ProPlus\voice_e8_s200.pth" -> "voice" (the webui appends epoch and step).
function nameFromPath(path: string): string {
  const file = path.replace(/^"|"$/g, '').split(/[\\/]/).at(-1) ?? ''
  return file.replace(/\.pth$/i, '').replace(/[_-]e\d+(_s\d+)?$/i, '')
}

/** Import a voice the GPT-SoVITS webui trained: its two weight files and a default reference clip. */
export function GptSovitsImportDialog({ onImported, onError, onClose }: Props) {
  const t = useT()
  const dialog = useRef<HTMLDialogElement>(null)
  const [sovits, setSovits] = useState('')
  const [gpt, setGpt] = useState('')
  const [name, setName] = useState<string | null>(null) // null: follows the SoVITS file name
  const [source, setSource] = useState<'dataset' | 'clip'>('dataset')
  const [datasets, setDatasets] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const [clip, setClip] = useState('')
  const [clipText, setClipText] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    dialog.current?.showModal()
    listDatasets().then(setDatasets, () => {})
  }, [])

  const voiceName = name ?? nameFromPath(sovits)
  const nameOk = NAME_PATTERN.test(voiceName)
  const referenceOk = source === 'dataset' ? !!dataset : !!clip.trim()
  const ready = nameOk && !!sovits.trim() && !!gpt.trim() && referenceOk

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!ready) return
    setBusy(true)
    try {
      const reference = source === 'dataset' ? { dataset } : { reference: clip.trim(), reference_text: clipText.trim() }
      const { id, engine } = await importGptSovits({ name: voiceName, sovits, gpt, ...reference })
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
          <h2>{t.gsvImportTitle}</h2>
        </header>
        <p className="muted hint">{t.gsvImportIntro}</p>

        <label className="field">
          <span>{t.gsvSovitsFile}</span>
          <input
            value={sovits}
            onChange={(e) => setSovits(e.target.value)}
            placeholder={t.gsvSovitsExample}
            autoFocus
          />
        </label>
        <label className="field">
          <span>{t.gsvGptFile}</span>
          <input value={gpt} onChange={(e) => setGpt(e.target.value)} placeholder={t.gsvGptExample} />
        </label>
        <label className="field">
          <span>
            {t.name} {voiceName && !nameOk && <span className="field-error">{t.nameRule}</span>}
          </span>
          <input value={voiceName} onChange={(e) => setName(e.target.value)} />
        </label>

        <fieldset className="field">
          <legend>{t.gsvReference}</legend>
          <div className="segmented" role="radiogroup">
            <label>
              <input type="radio" checked={source === 'dataset'} onChange={() => setSource('dataset')} />{' '}
              {t.gsvReferenceFromDataset}
            </label>
            <label>
              <input type="radio" checked={source === 'clip'} onChange={() => setSource('clip')} /> {t.gsvReferenceClip}
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
              <span className="muted hint">{t.gsvReferencePickHint}</span>
            </>
          ) : (
            <>
              <input value={clip} onChange={(e) => setClip(e.target.value)} placeholder={t.gsvClipExample} />
              <input
                value={clipText}
                onChange={(e) => setClipText(e.target.value)}
                placeholder={t.referenceTranscript}
              />
            </>
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
