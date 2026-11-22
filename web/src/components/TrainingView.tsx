import { useCallback, useEffect, useState } from 'react'
import { appendNew, getRun, listDatasets, listEngines, listRuns, resumeRun, startRun, stopTraining } from '../api'
import type { Engine, Lora, RunDetail as Run, RunSummary, TrainRequest } from '../api'
import { useT } from '../i18n'
import { NewRunForm } from './NewRunForm'
import { RunDetail } from './RunDetail'
import { NEW_RUN, RunList } from './RunList'

interface Props {
  loras: Lora[]
  onError: (message: string) => void
  onTryCheckpoint: (loraId: string) => void
  // Open the New run form with this dataset (from the Data tab).
  newRunDataset?: string
}

const LIST_POLL_MS = 4000
const DETAIL_POLL_MS = 2000

const isLive = (run: { status: string }) => run.status === 'running' || run.status === 'stopping'

export function TrainingView({ loras, onError, onTryCheckpoint, newRunDataset }: Props) {
  const t = useT()
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runsLoaded, setRunsLoaded] = useState(false)
  const [datasets, setDatasets] = useState<string[]>([])
  const [selected, setSelected] = useState<string | null>(newRunDataset ? NEW_RUN : null)
  const [detail, setDetail] = useState<Run | null>(null)
  // Their training forms (fields, this GPU's defaults): New run and the runs' Parameters tab.
  const [engines, setEngines] = useState<Engine[]>([])
  useEffect(() => {
    listEngines().then(setEngines, (e: Error) => onError(e.message))
  }, [onError])

  const activeRun = runs.find(isLive)?.name ?? null

  const refreshRuns = useCallback(async () => {
    try {
      const list = await listRuns()
      setRuns(list)
      setRunsLoaded(true)
      // First load: show the job in progress, else the most recent run, else the form.
      setSelected((sel) => sel ?? (list.find(isLive) ?? list[0])?.name ?? NEW_RUN)
    } catch (e) {
      onError((e as Error).message)
    }
  }, [onError])

  useEffect(() => {
    refreshRuns()
    listDatasets().then(setDatasets, (e: Error) => onError(e.message))
    const timer = setInterval(refreshRuns, LIST_POLL_MS)
    return () => clearInterval(timer)
  }, [refreshRuns, onError])

  // The selected run's detail, polled while it trains and refetched when its summary changes.
  const summary = runs.find((r) => r.name === selected)
  const summaryKey = summary && `${summary.status}:${summary.saved_step}`
  const live = !!summary && isLive(summary)
  useEffect(() => {
    if (!selected || selected === NEW_RUN) {
      setDetail(null)
      return
    }
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    // The first load is in full; polls while training ask only for what is new (metric
    // points, console lines) and add it to what they have.
    let known: Run | undefined
    const load = async () => {
      try {
        const fetched = await getRun(selected, known)
        if (cancelled) return
        const run = known ? appendNew(known, fetched) : fetched
        known = run
        setDetail(run)
        if (isLive(run)) timer = setTimeout(load, DETAIL_POLL_MS)
      } catch (e) {
        if (cancelled) return
        onError((e as Error).message)
        // A request that failed once (server busy, restarting) is tried again while the run trains.
        if (known ? isLive(known) : live) timer = setTimeout(load, DETAIL_POLL_MS * 3)
      }
    }
    load()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [selected, summaryKey, live, onError])

  async function act(action: () => Promise<unknown>) {
    try {
      await action()
    } catch (e) {
      onError((e as Error).message)
    }
    await refreshRuns()
  }

  async function start(params: TrainRequest) {
    await act(async () => {
      const run = await startRun(params)
      setDetail(run)
      setSelected(run.name)
    })
  }

  const shown = detail && detail.name === selected ? detail : null

  return (
    <main className="layout training">
      <div className="main-col">
        {selected === NEW_RUN ? (
          <NewRunForm
            datasets={datasets}
            runNames={runs.map((r) => r.name)}
            activeRun={activeRun}
            onStart={start}
            initialDataset={newRunDataset}
            engines={engines}
          />
        ) : shown ? (
          <RunDetail
            key={shown.name}
            run={shown}
            loras={loras}
            activeRun={activeRun}
            onStop={() => act(stopTraining)}
            onResume={(numIters) => act(() => resumeRun(shown.name, numIters))}
            onTryCheckpoint={onTryCheckpoint}
            engines={engines}
          />
        ) : (
          <section className="panel">
            <p className="muted empty">{t.loading}</p>
          </section>
        )}
      </div>
      <aside className="side-col">
        <RunList
          runs={runs}
          selected={selected}
          onSelect={setSelected}
          onNew={() => setSelected(NEW_RUN)}
          loading={!runsLoaded}
        />
      </aside>
    </main>
  )
}
