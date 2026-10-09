import { cancelJob } from '../api'
import type { Job } from '../api'
import { formatDuration } from '../format'
import { ProgressBar } from './RunList'

const PHASE_LABEL: Record<string, string> = {
  scanning: 'Scanning',
  separating: 'Separating the voice',
  slicing: 'Transcribing and slicing recordings',
  transcribing: 'Transcribing',
  aligning: 'Checking clips against their text',
  copying: 'Copying audio',
}

/** Phase, count, bar and Cancel for a running job; a short note once it failed or was cancelled. */
export function JobProgress({ job, what }: { job: Job; what: string }) {
  if (job.state === 'done') return null
  return (
    <div className="job-progress">
      {job.state === 'running' ? (
        <>
          <span>
            {PHASE_LABEL[job.phase] ?? 'Starting'}
            {job.total > 0 && (
              <span className="muted">
                {' '}
                {/* Slicing counts seconds of audio; the other phases count clips. */}
                {job.phase === 'slicing'
                  ? `${formatDuration(job.done)} / ${formatDuration(job.total)}`
                  : `${job.done} / ${job.total}`}
              </span>
            )}
          </span>
          <ProgressBar value={job.done} max={job.total || 1} />
          <button type="button" className="ghost small" onClick={() => cancelJob(job.id)}>
            Cancel
          </button>
        </>
      ) : (
        <span className="muted">{job.state === 'cancelled' ? `${what} cancelled.` : `${what} failed.`}</span>
      )}
    </div>
  )
}
