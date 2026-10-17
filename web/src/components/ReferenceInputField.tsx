import { useEffect, useMemo, useState } from 'react'
import { transcribe } from '../api'
import type { ReferenceInput } from '../api'
import { useT } from '../i18n'
import { forgetPeaks } from '../peaks'
import { clipCount, emptyInput } from '../referenceInputs'
import type { InputSource, InputState } from '../referenceInputs'
import { AudioPlayer } from './AudioPlayer'
import { FilePicker } from './FilePicker'
import { ReferenceClips } from './ReferenceClips'

interface Props {
  engineId: string
  input: ReferenceInput
  value: InputState
  onChange: (state: InputState) => void
  // The input this one may reuse (input.same_as), with its state.
  other: { input: ReferenceInput; state: InputState } | null
  onError: (message: string) => void
  onTranscribed: () => void
}

/** One of an engine's reference inputs, under the engine's own name for it. */
export function ReferenceInputField({ engineId, input, value, onChange, other, onError, onTranscribed }: Props) {
  const t = useT()
  const [transcribing, setTranscribing] = useState(false)
  const label = (i: ReferenceInput) => t.referenceInputs[`${engineId}.${i.name}`]?.[0] ?? i.name
  const hint = t.referenceInputs[`${engineId}.${input.name}`]?.[1]
  const file = value.source === 'file' ? value.file : null
  // A playable URL for the uploaded clip, released when it changes.
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(
    () => () => {
      if (url) {
        URL.revokeObjectURL(url)
        forgetPeaks(url)
      }
    },
    [url],
  )

  const sources: [InputSource, string][] = [
    ['none', t.sourceNone],
    ['file', t.sourceFile],
    ['dataset', t.sourceDataset],
    ...(other ? [['same', t.sourceSame(label(other.input))] as [InputSource, string]] : []),
  ]
  // An uploaded clip (its own, or the reused input's) takes a transcript typed here; dataset clips have theirs.
  const audioFile = value.source === 'same' ? (other?.state.source === 'file' ? other.state.file : null) : file
  const showTranscript = input.transcript !== 'none' && !!audioFile

  async function fillTranscript() {
    if (!audioFile) return
    setTranscribing(true)
    try {
      onChange({ ...value, text: (await transcribe(audioFile)).text })
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setTranscribing(false)
      onTranscribed()
    }
  }

  return (
    <div className="reference-input">
      <div className="reference-input-head">
        <strong>{label(input)}</strong>
        <div className="segmented" role="radiogroup" aria-label={label(input)}>
          {sources.map(([source, name]) => (
            <label key={source}>
              <input type="radio" checked={value.source === source} onChange={() => onChange({ ...value, source })} />{' '}
              {name}
            </label>
          ))}
        </div>
      </div>
      {hint && <p className="muted hint">{hint}</p>}
      {value.source === 'file' && (
        <FilePicker
          file={file}
          onChange={(f) => onChange(f ? { ...value, file: f } : emptyInput('file'))}
          accept="audio/*"
          label={t.chooseAudio}
          empty={t.orDropFile}
        >
          {url && <AudioPlayer url={url} />}
        </FilePicker>
      )}
      {value.source === 'dataset' && (
        <ReferenceClips
          count={clipCount(input)}
          value={value.clips}
          onChange={(clips) => onChange({ ...value, clips })}
          onError={onError}
        />
      )}
      {showTranscript && (
        <div className="file-row">
          <input
            value={value.text}
            onChange={(e) => onChange({ ...value, text: e.target.value })}
            placeholder={t.referenceTranscript}
          />
          <button
            type="button"
            className="ghost small"
            onClick={fillTranscript}
            disabled={transcribing}
            title={t.transcribeTitle}
          >
            {transcribing ? t.transcribing : t.transcribe}
          </button>
        </div>
      )}
    </div>
  )
}
