import { useState } from 'react'
import type { FormEvent } from 'react'
import { setVoiceReference, voiceReferenceUrl } from '../api'
import type { Lora } from '../api'
import { useT } from '../i18n'
import type { ClipChoice } from '../referenceInputs'
import { AudioPlayer } from './AudioPlayer'
import { FilePicker } from './FilePicker'
import { ReferenceClips } from './ReferenceClips'

interface Props {
  lora: Lora
  // How many dataset clips its reference joins (capabilities.voice_reference_clips; GPT-SoVITS: one).
  count: number
  // Its engine can test sets of clips by speaking them (capabilities.reference_search).
  search?: boolean
  onChanged: () => void
  onError: (message: string) => void
}

/** A voice's default reference clip (GPT-SoVITS, S2-Pro, Drama 3): listen, read its transcript, and
 * replace it with dataset clips (chosen as in Synthesize) or an uploaded clip. */
export function VoiceReference({ lora, count, search = false, onChanged, onError }: Props) {
  const t = useT()
  const [editing, setEditing] = useState(false)
  const [source, setSource] = useState<'dataset' | 'clip'>('dataset')
  const [clips, setClips] = useState<ClipChoice>({ dataset: '', clips: [] })
  const [testing, setTesting] = useState(false)
  const [audio, setAudio] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [version, setVersion] = useState(0) // a fresh URL after a change

  async function save(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      const chosen = { dataset: clips.dataset, indices: clips.clips.map((c) => c.index) }
      await setVoiceReference(lora.id, source === 'dataset' ? { clips: chosen } : { audio: audio!, text })
      setEditing(false)
      setVersion((v) => v + 1)
      onChanged()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const ready = !testing && (source === 'dataset' ? clips.clips.length > 0 : !!audio)
  const sources = [
    ['dataset', t.sourceDataset],
    ['clip', t.gsvReferenceClip],
  ] as const
  return (
    <div className="voice-reference">
      <div className="voice-current-head">
        <span className="muted">{t.gsvReference}</span>
        {!editing && (
          <button type="button" className="ghost small" onClick={() => setEditing(true)}>
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
              engineId={lora.engine}
              count={count}
              value={clips}
              onChange={setClips}
              onError={onError}
              search={search}
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
          <div className="actions">
            <button type="button" className="ghost small" disabled={busy} onClick={() => setEditing(false)}>
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
