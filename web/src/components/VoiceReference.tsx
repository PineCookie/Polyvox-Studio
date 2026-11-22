import { useState } from 'react'
import type { FormEvent } from 'react'
import { listDatasets, searchReferences, setVoiceReference, voiceReferenceUrl } from '../api'
import type { Lora, ReferenceSearchJob } from '../api'
import { useT } from '../i18n'
import { useJob } from '../useJob'
import { AudioPlayer } from './AudioPlayer'
import { FilePicker } from './FilePicker'
import { JobProgress } from './JobProgress'
import { ReferencePickSelect, ReferenceSearchResult } from './ReferenceSearch'
import type { VoicePick } from './ReferenceSearch'

interface Props {
  lora: Lora
  // The engine joins a dataset's best few clips (capabilities.voice_reference_clips > 1), else takes one.
  joined: boolean
  // A dataset's clips can be chosen by comparing a few sets (capabilities.reference_search).
  search?: boolean
  onChanged: () => void
  onError: (message: string) => void
}

/** A voice's default reference clip (GPT-SoVITS, S2-Pro, Drama 3): listen, read its transcript, replace it. */
export function VoiceReference({ lora, joined, search: canSearch = false, onChanged, onError }: Props) {
  const t = useT()
  const [editing, setEditing] = useState(false)
  const [source, setSource] = useState<'dataset' | 'clip'>('dataset')
  const [datasets, setDatasets] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const [audio, setAudio] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [version, setVersion] = useState(0) // a fresh URL after a change
  const [pickBy, setPickBy] = useState<VoicePick>(canSearch ? 'voice' : 'text')
  const [job, setJob] = useJob<ReferenceSearchJob>((j) => {
    if (j.state === 'done') changed()
    else if (j.state === 'failed') onError(j.error ?? t.jobFailed(t.referenceSearchWhat))
  })
  const running = job?.state === 'running'
  // The last search's sets, shown until the reference is edited again (only for this voice).
  const searched = job?.state === 'done' && job.result.id === lora.id ? job.result : null

  function edit() {
    setEditing(true)
    setJob(null)
    listDatasets().then(setDatasets, () => {})
  }

  function changed() {
    setEditing(false)
    setVersion((v) => v + 1)
    onChanged()
  }

  async function save(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      if (source === 'dataset' && pickBy === 'test') {
        setJob(await searchReferences({ dataset, id: lora.id }))
        return
      }
      await setVoiceReference(
        lora.id,
        source === 'dataset' ? { dataset, by: pickBy === 'test' ? 'voice' : pickBy } : { audio: audio!, text },
      )
      changed()
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
      {searched && !editing && <ReferenceSearchResult result={searched} />}
      {editing && (
        <form className="voice-reference-edit" onSubmit={save}>
          <fieldset className="field-group" disabled={running}>
            <div className="segmented" role="radiogroup">
              <label>
                <input type="radio" checked={source === 'dataset'} onChange={() => setSource('dataset')} />{' '}
                {joined ? t.referenceClipsFromDataset : t.gsvReferenceFromDataset}
              </label>
              <label>
                <input type="radio" checked={source === 'clip'} onChange={() => setSource('clip')} />{' '}
                {t.gsvReferenceClip}
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
                <ReferencePickSelect value={pickBy} test={canSearch} onChange={setPickBy} />
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
          {job && <JobProgress job={job} what={t.referenceSearchWhat} />}
          <div className="actions">
            <button type="button" className="ghost small" disabled={running} onClick={() => setEditing(false)}>
              {t.cancel}
            </button>
            <button type="submit" className="primary small" disabled={busy || running || !ready}>
              {busy || running ? t.saving : t.save}
            </button>
          </div>
        </form>
      )}
    </div>
  )
}
