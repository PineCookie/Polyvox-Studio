// Follow a background job (/api/jobs/{id}) until it ends.
import { useEffect, useRef, useState } from 'react'
import { getJob } from './api'
import type { Job } from './api'

const JOB_POLL_MS = 700

/** The job being followed (set it to a freshly started one); `onEnd` runs once when it stops running. */
export function useJob<J extends Job>(onEnd: (job: J) => void) {
  const [job, setJob] = useState<J | null>(null)
  const onEndRef = useRef(onEnd)
  useEffect(() => {
    onEndRef.current = onEnd
  })

  useEffect(() => {
    if (!job || job.state !== 'running') return
    const timer = setTimeout(async () => {
      let next: J
      try {
        next = await getJob<J>(job.id)
      } catch (e) {
        next = { ...job, state: 'failed', error: (e as Error).message }
      }
      setJob(next)
      if (next.state !== 'running') onEndRef.current(next)
    }, JOB_POLL_MS)
    return () => clearTimeout(timer)
  }, [job])

  return [job, setJob] as const
}
