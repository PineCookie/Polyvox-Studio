// Follow a background job (/api/jobs/{id}) until it ends.
import { useEffect, useRef, useState } from 'react'
import { getJob } from './api'
import type { Job } from './api'

const JOB_POLL_MS = 700
// Polls that may fail in a row (server busy or restarting) before the job is shown as failed; each
// waits a little longer (about 15 s in all).
const MAX_FAILURES = 6

/** The job being followed (set it to a freshly started one); `onEnd` runs once when it stops running. */
export function useJob<J extends Job>(onEnd: (job: J) => void) {
  const [job, setJob] = useState<J | null>(null)
  const [retry, setRetry] = useState(0)
  const failures = useRef(0)
  const onEndRef = useRef(onEnd)
  useEffect(() => {
    onEndRef.current = onEnd
  })

  useEffect(() => {
    if (!job || job.state !== 'running') return
    let stale = false // another job followed meanwhile, or the component gone
    const timer = setTimeout(
      async () => {
        let next: J
        try {
          next = await getJob<J>(job.id)
          failures.current = 0
        } catch (e) {
          if (stale) return
          if (++failures.current < MAX_FAILURES) {
            setRetry((n) => n + 1)
            return
          }
          failures.current = 0
          next = { ...job, state: 'failed', error: (e as Error).message }
        }
        if (stale) return
        setJob(next)
        if (next.state !== 'running') onEndRef.current(next)
      },
      JOB_POLL_MS * (failures.current + 1),
    )
    return () => {
      stale = true
      clearTimeout(timer)
    }
  }, [job, retry])

  return [job, setJob] as const
}
