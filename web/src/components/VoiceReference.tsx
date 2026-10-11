import { useState } from 'react'
import type { FormEvent } from 'react'
import { listDatasets, setVoiceReference, voiceReferenceUrl } from '../api'
import type { Lora } from '../api'
import { useT } from '../i18n'
import { AudioPlayer } from './AudioPlayer'
import { FilePicker } from './FilePicker'

interface Props {
  lora: Lora
  onChanged: () => void
  onError: (message: string) => void
}

/** A GPT-SoVITS voice's default reference clip: listen, read its transcript, replace it. */
export function VoiceReference({ lora, onChanged, onError }: Props) {
  const t = useT()
  const [editing, setEditing] = useState(false)
  const [source, setSource] = useState<'dataset' | 'clip'>('dataset')
  const [datasets, setDatasets] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const [audio, setAudio] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [version, setVersion] = useState(0) // a fresh URL after a change

  function edit() {
    setEditing(true)
    listDatasets().then(setDatasets, () => {})
  }

  async function save(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      await setVoiceReference(lora.id, source === 'dataset' ? { dataset } : { audio: audio!, text })
      setEditing(false)
      setVersion((v) => v + 1)
      onChanged()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const ready = source === 'dataset' ? !!dataset : !!audio
  return (
    <div className="voice-reference">
      <div className="voice-current-head">
        <span className="muted">{t.gsvReference}</span>
        {!editing && (
          <button type="button" className="ghost small" onClick={edit}>
            {lora.reference_text === null ? t.gsvAddReference : t.gsvChangeReference}
          </button>
        )}
      </div>
      {lora.reference_text !== null && (
        <>
          <AudioPlayer url={voiceReferenceUrl(lora.id, String(version))} />
          <p className="voice-meta">{lora.reference_text || t.gsvNoTranscript}</p>
        </>
      )}
      {editing && (
        <form className="voice-reference-edit" onSubmit={save}>
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
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="">{t.chooseDatasetOption}</option>
              {datasets.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
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
          <div className="actions">
            <button type="button" className="ghost small" onClick={() => setEditing(false)}>
              {t.cancel}
            </button>
            <button type="submit" className="primary small" disabled={busy || !ready}>
              {busy ? t.saving : t.save}
            </button>
          </div>
        </form>
      )}
    </div>
  )
}
