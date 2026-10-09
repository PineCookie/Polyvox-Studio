import { AudioPlayer } from './AudioPlayer'

export interface Clip {
  id: number
  url: string
  text: string
  voice: string
  seed: number | null
  elapsedMs: number
  createdAt: Date
}

interface Props {
  clips: Clip[]
  onUseSeed: (seed: number) => void
  onRemove: (id: number) => void
}

function fileName(clip: Clip): string {
  const slug = clip.text
    .slice(0, 32)
    .replace(/[^\p{L}\p{N}]+/gu, '_')
    .replace(/^_|_$/g, '')
  return `${slug || 'voxcpm'}_${clip.seed ?? clip.id}.wav`
}

export function History({ clips, onUseSeed, onRemove }: Props) {
  if (clips.length === 0) {
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>History</h2>
        </header>
        <p className="muted empty">Generated clips from this session appear here.</p>
      </section>
    )
  }

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>History</h2>
        <span className="muted">{clips.length}</span>
      </header>
      <ol className="history">
        {clips.map((clip) => (
          <HistoryClip key={clip.id} clip={clip} onUseSeed={onUseSeed} onRemove={onRemove} />
        ))}
      </ol>
    </section>
  )
}

function HistoryClip({ clip, onUseSeed, onRemove }: { clip: Clip } & Omit<Props, 'clips'>) {
  return (
    <li className="clip">
      <p className="clip-text">{clip.text}</p>
      <AudioPlayer url={clip.url} />
      <div className="clip-meta">
        <span className="tag">{clip.voice}</span>
        {clip.seed !== null && <span className="muted">seed {clip.seed}</span>}
        <span className="muted">{(clip.elapsedMs / 1000).toFixed(1)} s</span>
        <span className="spacer" />
        {clip.seed !== null && (
          <button type="button" className="ghost small" onClick={() => onUseSeed(clip.seed!)}>
            Use seed
          </button>
        )}
        <a className="button ghost small" href={clip.url} download={fileName(clip)}>
          Download
        </a>
        <button type="button" className="ghost small" onClick={() => onRemove(clip.id)} aria-label="Remove clip">
          Remove
        </button>
      </div>
    </li>
  )
}
