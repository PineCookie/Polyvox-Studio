import { useEffect, useRef, useState } from 'react'
import { cancelJob, datasetAudioUrl, listDatasets, pickReferenceClips, searchReferences } from '../api'
import type { ReferencePick, ReferenceSearchJob } from '../api'
import { useT } from '../i18n'
import type { ClipChoice } from '../referenceInputs'
import { useJob } from '../useJob'
import { SparkIcon } from './icons'
import { Info } from './Info'
import { JobProgress } from './JobProgress'
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
  // The engine can test a few sets by speaking them (capabilities.reference_search). It also picks
  // the clips most like the voice by default (measured on it); the others the cleanest transcripts.
  search?: boolean
  // While the clips are in use (saving): nothing can be changed.
  disabled?: boolean
  // Tells the parent when a test by speaking starts / ends (nothing else should use the clips meanwhile).
  onTesting?: (testing: boolean) => void
}

/** Clips of a dataset to clone a voice from, the same wherever they are chosen (Synthesize, saving a
 * voice, changing its reference): the dataset, how to choose (most like the voice / cleanest
 * transcripts), the clips themselves (hear, remove), other clips, and with `search` a test by
 * speaking that puts in the set the engine makes sound most like the speaker. */
export function ReferenceClips({
  engineId,
  count,
  value,
  onChange,
  onError,
  search = false,
  disabled,
  onTesting,
}: Props) {
  const t = useT()
  const [datasets, setDatasets] = useState<string[] | null>(null)
  const [seen, setSeen] = useState<number[]>([]) // clips shown for this dataset: "Other clips" skips them
  const [busy, setBusy] = useState(false)
  const [by, setBy] = useState<ReferencePick>(() => {
    const kept = remembered(PICK_STORE + engineId)
    return kept === 'text' || kept === 'voice' ? kept : search ? 'voice' : 'text'
  })
  // The last test's outcome, shown while its clips are the ones listed.
  const [tested, setTested] = useState<ReferenceSearchJob['result'] | null>(null)
  const [job, setJob] = useJob<ReferenceSearchJob>((j) => {
    onTesting?.(false)
    if (j.state === 'done') {
      onChange({ dataset: value.dataset, clips: j.result.clips })
      setSeen((s) => [...new Set([...s, ...j.result.clips.map((c) => c.index)])])
      setTested(j.result)
    } else if (j.state === 'failed') onError(j.error ?? t.jobFailed(t.referenceSearchWhat))
  })
  const testing = job?.state === 'running'
  const locked = busy || testing || !!disabled
  // A test still running when the clips go away (a dialog closed) is stopped.
  const runningTest = useRef<string | null>(null)
  useEffect(() => {
    runningTest.current = job?.state === 'running' ? job.id : null
  }, [job])
  useEffect(
    () => () => {
      if (runningTest.current) cancelJob(runningTest.current).catch(() => {})
    },
    [],
  )
  // Only the latest pick reports its clips, and none after the chooser has gone (another source
  // chosen meanwhile): the answer would bring back the state it started from.
  const pickRun = useRef(0)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
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

  async function pick(dataset: string, exclude: number[], how: ReferencePick = by) {
    const run = ++pickRun.current
    const current = () => mounted.current && run === pickRun.current
    setTested(null)
    setBusy(true)
    if (dataset !== value.dataset) onChange({ dataset, clips: [] }) // the menu shows it while its clips load
    try {
      let clips = await pickReferenceClips(dataset, count, exclude, how).catch((e) => {
        if (!exclude.length) throw e
        return null // every clip was shown: start over
      })
      if (clips === null && current()) {
        exclude = []
        clips = await pickReferenceClips(dataset, count, exclude, how)
      }
      if (!current() || clips === null) return
      onChange({ dataset, clips })
      setSeen([...exclude, ...clips.map((c) => c.index)])
    } catch (e) {
      if (!current()) return
      onChange({ dataset, clips: [] })
      onError((e as Error).message)
    } finally {
      if (current()) setBusy(false)
    }
  }

  async function test() {
    setTested(null)
    try {
      setJob(await searchReferences({ dataset: value.dataset, engine: engineId, count }))
      onTesting?.(true)
    } catch (e) {
      onError((e as Error).message)
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
    if (dataset) pick(dataset, [])
    else {
      setTested(null)
      onChange({ dataset: '', clips: [] })
    }
  }

  function choosePick(how: ReferencePick) {
    remember(PICK_STORE + engineId, how)
    setBy(how)
    setSeen([])
    if (value.dataset) pick(value.dataset, [], how)
  }

  function remove(index: number) {
    setTested(null)
    onChange({ ...value, clips: value.clips.filter((c) => c.index !== index) })
  }

  const seconds = value.clips.reduce((sum, c) => sum + c.seconds, 0)
  const ways: [ReferencePick, string][] = [
    ['voice', t.referencePickVoice],
    ['text', t.referencePickText],
  ]
  return (
    <div className="reference-clips">
      <select
        value={value.dataset}
        onChange={(e) => chooseDataset(e.target.value)}
        disabled={locked}
        aria-label={t.chooseDatasetOption}
      >
        <option value="">{datasets?.length === 0 ? t.noDatasets : t.chooseDatasetOption}</option>
        {(datasets ?? []).map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
      {value.dataset && (
        <div className="seg" role="radiogroup" aria-label={t.referencePickBy}>
          {ways.map(([way, name]) => (
            <button
              key={way}
              type="button"
              role="radio"
              aria-checked={by === way}
              className={by === way ? 'active' : ''}
              disabled={locked}
              onClick={() => by !== way && choosePick(way)}
            >
              {name}
            </button>
          ))}
        </div>
      )}
      {testing && job ? (
        <JobProgress job={job} what={t.referenceSearchWhat} />
      ) : busy ? (
        <p className="clips-loading muted" role="status">
          <span className="spinner" aria-hidden /> {by === 'voice' ? t.findingAlikeClips : t.findingClips}
        </p>
      ) : (
        value.clips.map((c) => (
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
              disabled={locked}
              title={t.removeReferenceClip}
              aria-label={t.removeReferenceClip}
            >
              ×
            </button>
            <p className="voice-meta">{c.text}</p>
          </div>
        ))
      )}
      {tested && !testing && <TestedNote result={tested} />}
      {value.dataset && (
        <div className="reference-clips-foot">
          <span className="muted">{value.clips.length > 0 && t.referenceClipsHint(value.clips.length, seconds)}</span>
          <button type="button" className="ghost small" onClick={() => pick(value.dataset, seen)} disabled={locked}>
            {t.otherClips}
          </button>
          {search && (
            <button
              type="button"
              className="ghost small with-icon"
              onClick={test}
              disabled={locked}
              title={t.findBestClipsHint}
            >
              <SparkIcon /> {t.findBestClips}
            </button>
          )}
        </div>
      )}
      <audio ref={audio} onPause={() => setPlaying(null)} onEnded={() => setPlaying(null)} hidden />
    </div>
  )
}

/** One line on the last test by speaking: which set was kept; every set's scores in its ⓘ. */
function TestedNote({ result }: { result: ReferenceSearchJob['result'] }) {
  const t = useT()
  const signed = (n: number) => (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n).toFixed(1)
  const name = (set: string) => t.referenceSearchSets[set] ?? set
  const rows = result.tried.map(
    (s) =>
      `${s.name === result.chosen ? '✓ ' : ''}${name(s.name)}: ` +
      t.referenceSearchRow(s.likeness.toFixed(2), s.pitch == null ? '–' : t.semitones(signed(s.pitch))),
  )
  return (
    <p className="tested-note muted">
      <SparkIcon /> {t.testedNote(name(result.chosen))}
      <Info text={rows.join('\n')} />
    </p>
  )
}
