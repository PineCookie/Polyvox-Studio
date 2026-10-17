import { useEffect, useRef, useState } from 'react'
import { datasetAudioUrl, listDatasets, pickReferenceClips } from '../api'
import type { ReferenceClip } from '../api'
import { useT } from '../i18n'
import { SampleCell } from './SampleCell'

// Three clips (~20 s) cloned a voice clearly better than one in every engine.
const COUNT = 3
const STORE = 'polyvox.referenceDataset'

export interface ClipChoice {
  dataset: string
  clips: ReferenceClip[]
}

function remembered(): string {
  try {
    return localStorage.getItem(STORE) ?? ''
  } catch {
    return ''
  }
}

function remember(dataset: string) {
  try {
    localStorage.setItem(STORE, dataset)
  } catch {
    // private window or blocked storage: the choice just isn't remembered
  }
}

interface Props {
  value: ClipChoice
  onChange: (choice: ClipChoice) => void
  onError: (message: string) => void
}

/** Reference clips from a dataset, picked for it (best checked clips of 4-10 s); each can be
 * heard or removed, and "Other clips" brings the next best. */
export function ReferenceClips({ value, onChange, onError }: Props) {
  const t = useT()
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [seen, setSeen] = useState<number[]>([]) // clips shown for this dataset: "Other clips" skips them
  const [busy, setBusy] = useState(false)
  // One player for the list: starting a clip stops the one playing.
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState<number | null>(null)

  function play(index: number, at?: number) {
    const el = audio.current
    if (!el) return
    if (at === undefined && playing === index && !el.paused) {
      el.pause()
      return
    }
    const url = datasetAudioUrl({ id: value.dataset }, index)
    if (!el.src.endsWith(url)) el.src = url
    if (at !== undefined) el.currentTime = at
    el.play().then(
      () => setPlaying(index),
      () => setPlaying(null),
    )
  }

  async function pick(dataset: string, exclude: number[]) {
    setBusy(true)
    try {
      let clips = await pickReferenceClips(dataset, COUNT, exclude).catch((e) => {
        if (!exclude.length) throw e
        return null // every clip was shown: start over
      })
      if (clips === null) {
        exclude = []
        clips = await pickReferenceClips(dataset, COUNT, exclude)
      }
      onChange({ dataset, clips })
      setSeen([...exclude, ...clips.map((c) => c.index)])
    } catch (e) {
      onChange({ dataset, clips: [] })
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    let alive = true
    listDatasets().then(
      (ids) => {
        if (!alive) return
        setDatasets(ids)
        const last = remembered()
        if (!value.dataset && ids.includes(last)) pick(last, [])
      },
      () => alive && setDatasets([]),
    )
    return () => {
      alive = false
    }
    // Once, when the dataset choice opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function chooseDataset(dataset: string) {
    remember(dataset)
    setSeen([])
    if (dataset) pick(dataset, [])
    else onChange({ dataset: '', clips: [] })
  }

  const remove = (index: number) => onChange({ ...value, clips: value.clips.filter((c) => c.index !== index) })
  const seconds = value.clips.reduce((sum, c) => sum + c.seconds, 0)

  return (
    <div className="reference-clips">
      <div className="file-row">
        <select value={value.dataset} onChange={(e) => chooseDataset(e.target.value)} disabled={busy}>
          <option value="">{datasets?.length === 0 ? t.noDatasets : t.chooseDatasetOption}</option>
          {(datasets ?? []).map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        {value.dataset && (
          <button type="button" className="ghost small" onClick={() => pick(value.dataset, seen)} disabled={busy}>
            {t.otherClips}
          </button>
        )}
      </div>
      {value.clips.map((c) => (
        <div key={c.index} className="reference-clip">
          <SampleCell
            url={datasetAudioUrl({ id: value.dataset }, c.index)}
            playing={playing === c.index}
            audio={playing === c.index ? audio.current : null}
            onPlay={(at) => play(c.index, at)}
          />
          <button
            type="button"
            className="ghost small"
            onClick={() => remove(c.index)}
            title={t.removeReferenceClip}
            aria-label={t.removeReferenceClip}
          >
            ×
          </button>
          <p className="voice-meta">{c.text}</p>
        </div>
      ))}
      {value.clips.length > 0 && <p className="muted hint">{t.referenceClipsHint(value.clips.length, seconds)}</p>}
      <audio ref={audio} onPause={() => setPlaying(null)} onEnded={() => setPlaying(null)} hidden />
    </div>
  )
}
