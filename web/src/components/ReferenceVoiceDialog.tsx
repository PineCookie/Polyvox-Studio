import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { createReferenceVoice } from '../api'
import type { Engine } from '../api'
import { useT } from '../i18n'
import type { ClipChoice } from '../referenceInputs'
import { FilePicker } from './FilePicker'
import { ReferenceClips } from './ReferenceClips'

interface Props {
  engine: Engine
  onCreated: (id: string, engine: string) => void
  onError: (message: string) => void
  onClose: () => void
}

const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u

/** Save a voice for an engine whose voices are references only (Drama 3): a name and its clips,
 * chosen as in Synthesize's dataset clips (shown, so the voice is saved with exactly those). */
export function ReferenceVoiceDialog({ engine, onCreated, onError, onClose }: Props) {
  const t = useT()
  const dialog = useRef<HTMLDialogElement>(null)
  const [name, setName] = useState('')
  const [source, setSource] = useState<'dataset' | 'clip'>('dataset')
  const [clips, setClips] = useState<ClipChoice>({ dataset: '', clips: [] })
  const [testing, setTesting] = useState(false)
  const [audio, setAudio] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    dialog.current?.showModal()
  }, [])

  const nameOk = NAME_PATTERN.test(name.trim())
  const ready = nameOk && !testing && (source === 'dataset' ? clips.clips.length > 0 : !!audio && !!text.trim())

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!ready) return
    setBusy(true)
    try {
      const chosen = { dataset: clips.dataset, indices: clips.clips.map((c) => c.index) }
      const { id, engine: made } = await createReferenceVoice(
        engine.id,
        name.trim(),
        source === 'dataset' ? { clips: chosen } : { audio: audio!, text: text.trim() },
      )
      onCreated(id, made)
      onClose()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const sources = [
    ['dataset', t.sourceDataset],
    ['clip', t.gsvReferenceClip],
  ] as const
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
          <div className="seg" role="radiogroup" aria-label={t.gsvReference}>
            {sources.map(([way, label]) => (
              <button
                key={way}
                type="button"
                role="radio"
                aria-checked={source === way}
                className={source === way ? 'active' : ''}
                disabled={testing || busy}
                onClick={() => setSource(way)}
              >
                {label}
              </button>
            ))}
          </div>
          {source === 'dataset' ? (
            <ReferenceClips
              engineId={engine.id}
              count={engine.capabilities.voice_reference_clips}
              value={clips}
              onChange={setClips}
              onError={onError}
              search={engine.capabilities.reference_search}
              disabled={busy}
              onTesting={setTesting}
            />
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
