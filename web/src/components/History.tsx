import { useT } from '../i18n'
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
  const t = useT()
  if (clips.length === 0) {
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>{t.history}</h2>
        </header>
        <p className="muted empty">{t.historyEmpty}</p>
      </section>
    )
  }

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{t.history}</h2>
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
  const t = useT()
  return (
    <li className="clip">
      <p className="clip-text">{clip.text}</p>
      <AudioPlayer url={clip.url} />
      <div className="clip-meta">
        <span className="tag">{clip.voice}</span>
        {clip.seed !== null && <span className="muted">{t.seedValue(clip.seed)}</span>}
        <span className="muted">{t.seconds((clip.elapsedMs / 1000).toFixed(1))}</span>
        <span className="spacer" />
        {clip.seed !== null && (
          <button type="button" className="ghost small" onClick={() => onUseSeed(clip.seed!)}>
            {t.useSeed}
          </button>
        )}
        <a className="button ghost small" href={clip.url} download={fileName(clip)}>
          {t.download}
        </a>
        <button type="button" className="ghost small" onClick={() => onRemove(clip.id)} aria-label={t.removeClip}>
          {t.remove}
        </button>
      </div>
    </li>
  )
}
