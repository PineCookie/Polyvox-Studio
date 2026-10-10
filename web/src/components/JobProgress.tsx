import { cancelJob } from '../api'
import type { Job } from '../api'
import { formatDuration } from '../format'
import { ProgressBar } from './RunList'

const PHASE_LABEL: Record<string, string> = {
  scanning: 'Scanning',
  separating: 'Separating the voice',
  slicing: 'Transcribing and slicing recordings',
  separating_slicing: 'Separating, transcribing and slicing recordings',
  transcribing: 'Transcribing',
  aligning: 'Checking clips against their text',
  copying: 'Copying audio',
  normalizing: 'Normalizing loudness',
}

// Phases that count seconds of audio; the others count clips.
const AUDIO_PHASES = new Set(['slicing', 'separating_slicing'])

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
                {AUDIO_PHASES.has(job.phase)
                  ? `${formatDuration(job.done)} of ${formatDuration(job.total)} audio`
                  : `${job.done} / ${job.total}`}
                {job.seconds_left != null && job.seconds_left > 0 && ` · ~${formatDuration(job.seconds_left)} left`}
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
