import { useEffect, useRef, useState } from 'react'
import { datasetAudioUrl, listDatasets, pickReferenceClips, searchReferences } from '../api'
import type { ReferencePick, ReferenceSearchJob } from '../api'
import { useT } from '../i18n'
import type { ClipChoice } from '../referenceInputs'
import { useJob } from '../useJob'
import { SparkIcon } from './icons'
import { JobProgress } from './JobProgress'
import { ReferenceSearchResult } from './ReferenceSearch'
import { SampleCell } from './SampleCell'

const STORE = 'polyvox.referenceDataset'
const PICK_STORE = 'polyvox.referencePick.' // + engine id

function remembered(key: string): string {
  try {
    return localStorage.getItem(key) ?? ''
  } catch {
    return ''
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // private window or blocked storage: the choice just isn't remembered
  }
}

interface Props {
  engineId: string
  // How many clips to pick (several clips of ~20 s clone a voice better than one).
  count: number
  value: ClipChoice
  onChange: (choice: ClipChoice) => void
  onError: (message: string) => void
  // The engine can test a few sets by speaking (capabilities.reference_search); it also picks the
  // clips most like the voice by default (measured on it), the others the cleanest transcripts.
  search?: boolean
}

/** Reference clips from a dataset, picked for it (clips of 4-10 s: the cleanest transcripts, or the
 * most like the voice); each can be heard or removed, and "Other clips" brings the next best. With
 * `search`, "Test by speaking" lets the engine try a few sets and keeps the one that sounds most like
 * the speaker. */
export function ReferenceClips({ engineId, count, value, onChange, onError, search: canSearch = false }: Props) {
  const t = useT()
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [seen, setSeen] = useState<number[]>([]) // clips shown for this dataset: "Other clips" skips them
  const [busy, setBusy] = useState(false)
  const [by, setBy] = useState<ReferencePick>(() => {
    const kept = remembered(PICK_STORE + engineId)
    return kept === 'text' || kept === 'voice' ? kept : canSearch ? 'voice' : 'text'
  })
  const [searched, setSearched] = useState<ReferenceSearchJob['result'] | null>(null)
  const [job, setJob] = useJob<ReferenceSearchJob>((j) => {
    if (j.state === 'done') {
      onChange({ dataset: value.dataset, clips: j.result.clips })
      setSeen((s) => [...new Set([...s, ...j.result.clips.map((c) => c.index)])])
      setSearched(j.result)
    } else if (j.state === 'failed') onError(j.error ?? t.jobFailed(t.referenceSearchWhat))
  })
  const searching = job?.state === 'running'
  const locked = busy || searching
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

  async function search() {
    setSearched(null)
    try {
      setJob(await searchReferences({ dataset: value.dataset, engine: engineId, count }))
    } catch (e) {
      onError((e as Error).message)
    }
  }

  async function pick(dataset: string, exclude: number[], how: ReferencePick = by) {
    setSearched(null)
    setBusy(true)
    try {
      let clips = await pickReferenceClips(dataset, count, exclude, how).catch((e) => {
        if (!exclude.length) throw e
        return null // every clip was shown: start over
      })
      if (clips === null) {
        exclude = []
        clips = await pickReferenceClips(dataset, count, exclude, how)
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
        const last = remembered(STORE)
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
    remember(STORE, dataset)
    setSeen([])
    setSearched(null)
    if (dataset) pick(dataset, [])
    else onChange({ dataset: '', clips: [] })
  }

  function choosePick(how: ReferencePick) {
    remember(PICK_STORE + engineId, how)
    setBy(how)
    setSeen([])
    if (value.dataset) pick(value.dataset, [], how)
  }

  function remove(index: number) {
    setSearched(null)
    onChange({ ...value, clips: value.clips.filter((c) => c.index !== index) })
  }
  const seconds = value.clips.reduce((sum, c) => sum + c.seconds, 0)

  return (
    <div className="reference-clips">
      <div className="file-row">
        <select value={value.dataset} onChange={(e) => chooseDataset(e.target.value)} disabled={locked}>
          <option value="">{datasets?.length === 0 ? t.noDatasets : t.chooseDatasetOption}</option>
          {(datasets ?? []).map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        {value.dataset && (
          <button type="button" className="ghost small" onClick={() => pick(value.dataset, seen)} disabled={locked}>
            {t.otherClips}
          </button>
        )}
      </div>
      {value.dataset && (
        <div className="pick-row">
          <label className="pick-by">
            <span className="muted">{t.referencePickBy}</span>
            <select value={by} onChange={(e) => choosePick(e.target.value as ReferencePick)} disabled={locked}>
              <option value="voice">{t.referencePickVoice}</option>
              <option value="text">{t.referencePickText}</option>
            </select>
          </label>
          {canSearch && (
            <button
              type="button"
              className="ghost small find-best"
              onClick={search}
              disabled={locked}
              title={t.findBestClipsHint}
            >
              <SparkIcon /> {t.findBestClips}
            </button>
          )}
        </div>
      )}
      {job && <JobProgress job={job} what={t.referenceSearchWhat} />}
      {searched && <ReferenceSearchResult result={searched} />}
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
