import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { createReferenceVoice, listDatasets } from '../api'
import type { Engine } from '../api'
import { useT } from '../i18n'
import { FilePicker } from './FilePicker'

interface Props {
  engine: Engine
  onCreated: (id: string, engine: string) => void
  onError: (message: string) => void
  onClose: () => void
}

const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

/** Save a voice for an engine whose voices are references only (Drama 3): a name and its clips. */
export function ReferenceVoiceDialog({ engine, onCreated, onError, onClose }: Props) {
  const t = useT()
  const dialog = useRef<HTMLDialogElement>(null)
  const [name, setName] = useState('')
  const [source, setSource] = useState<'dataset' | 'clip'>('dataset')
  const [datasets, setDatasets] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const [audio, setAudio] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    dialog.current?.showModal()
    listDatasets().then(setDatasets, () => {})
  }, [])

  const nameOk = NAME_PATTERN.test(name.trim())
  const ready = nameOk && (source === 'dataset' ? !!dataset : !!audio && !!text.trim())

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!ready) return
    setBusy(true)
    try {
      const { id, engine: made } = await createReferenceVoice(
        engine.id,
        name.trim(),
        source === 'dataset' ? { dataset } : { audio: audio!, text: text.trim() },
      )
      onCreated(id, made)
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
          <h2>{t.referenceVoiceTitle(engine.name)}</h2>
        </header>
        <p className="muted hint">{t.referenceVoiceIntro}</p>

        <label className="field">
          <span>
            {t.name} {name.trim() && !nameOk && <span className="field-error">{t.nameRule}</span>}
          </span>
          <input value={name} onChange={(e) => setName(e.target.value)} autoFocus />
        </label>

        <fieldset className="field">
          <legend>{t.gsvReference}</legend>
          <div className="segmented" role="radiogroup">
            <label>
              <input type="radio" checked={source === 'dataset'} onChange={() => setSource('dataset')} />{' '}
              {t.referenceClipsFromDataset}
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
              <span className="muted hint">{t.referenceVoicePickHint}</span>
            </>
          ) : (
            <>
              <FilePicker
                file={audio}
                onChange={setAudio}
                accept="audio/*"
                label={t.chooseAudio}
                empty={t.orDropFile}
              />
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder={t.referenceTranscript} />
            </>
          )}
        </fieldset>

        <div className="actions">
          <button type="button" className="ghost" onClick={onClose}>
            {t.cancel}
          </button>
          <button type="submit" className="primary" disabled={busy || !ready}>
            {busy ? t.saving : t.save}
          </button>
        </div>
      </form>
    </dialog>
  )
}
