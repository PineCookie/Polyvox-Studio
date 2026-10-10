import { cancelJob } from '../api'
import type { Job } from '../api'
import { formatDuration } from '../format'
import { useT } from '../i18n'
import { ProgressBar } from './RunList'

// Phases that count seconds of audio; the others count clips.
const AUDIO_PHASES = new Set(['slicing', 'separating_slicing'])

/** Phase, count, bar and Cancel for a running job; a short note once it failed or was cancelled. */
export function JobProgress({ job, what }: { job: Job; what: string }) {
  const t = useT()
  if (job.state === 'done') return null
  return (
    <div className="job-progress">
      {job.state === 'running' ? (
        <>
          <span>
            {t.phase[job.phase] ?? t.starting}
            {job.total > 0 && (
              <span className="muted">
                {' '}
                {AUDIO_PHASES.has(job.phase)
                  ? t.audioProgress(formatDuration(job.done), formatDuration(job.total))
                  : `${job.done} / ${job.total}`}
                {job.seconds_left != null && job.seconds_left > 0 && t.timeLeft(formatDuration(job.seconds_left))}
              </span>
            )}
          </span>
          <ProgressBar value={job.done} max={job.total || 1} />
          <button type="button" className="ghost small" onClick={() => cancelJob(job.id)}>
            {t.cancel}
          </button>
        </>
      ) : (
        <span className="muted">{job.state === 'cancelled' ? t.jobCancelled(what) : t.jobFailed(what)}</span>
      )}
    </div>
  )
}
