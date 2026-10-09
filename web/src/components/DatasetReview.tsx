import { useEffect, useState } from 'react'
import { getDatasetReport } from '../api'
import type { DatasetReport, DatasetSource } from '../api'
import { DatasetClips } from './DatasetClips'
import { DatasetSummary } from './DatasetSummary'

interface Props {
  source: DatasetSource
  // Called with each loaded report, e.g. to gate starting a run on it.
  onReport?: (report: DatasetReport | null) => void
}

/** Stats, warnings and the clips of a dataset or of the copy a run trains on. */
export function DatasetReview({ source, onReport }: Props) {
  const sourceKey = JSON.stringify(source)
  const [report, setReport] = useState<DatasetReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setReport(null)
    setError(null)
    onReport?.(null)
    let cancelled = false
    getDatasetReport(JSON.parse(sourceKey)).then(
      (r) => {
        if (cancelled) return
        setReport(r)
        onReport?.(r)
      },
      (e: Error) => !cancelled && setError(e.message),
    )
    return () => {
      cancelled = true
    }
    // onReport is a notification; a new function identity must not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey])

  return (
    <>
      <DatasetSummary report={report} error={error} />
      <DatasetClips key={sourceKey} source={source} />
    </>
  )
}
