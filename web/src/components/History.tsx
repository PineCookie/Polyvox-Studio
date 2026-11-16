import { useState } from 'react'
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
  edited?: [number, number] // an edit's new audio, in seconds
}

interface Props {
  clips: Clip[]
  onUseSeed: (seed: number) => void
  onRemove: (id: number) => void
  // The engine that edits speech (null: none can here), and the edit itself: resolves true when done.
  editor: string | null
  busy: boolean
  onEdit: (clip: Clip, newText: string) => Promise<boolean>
}

function fileName(clip: Clip): string {
  const slug = clip.text
    .slice(0, 32)
    .replace(/[^\p{L}\p{N}]+/gu, '_')
    .replace(/^_|_$/g, '')
  return `${slug || 'speech'}_${clip.seed ?? clip.id}.wav`
}

export function History({ clips, ...props }: Props) {
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
          <HistoryClip key={clip.id} clip={clip} {...props} />
        ))}
      </ol>
    </section>
  )
}

function HistoryClip({ clip, onUseSeed, onRemove, editor, busy, onEdit }: { clip: Clip } & Omit<Props, 'clips'>) {
  const t = useT()
  const [draft, setDraft] = useState<string | null>(null) // the new text while editing
  const [editing, setEditing] = useState(false)
  const changed = draft !== null && draft.trim() !== '' && draft.trim() !== clip.text

  async function apply() {
    setEditing(true)
    try {
      if (await onEdit(clip, draft!.trim())) setDraft(null)
    } finally {
      setEditing(false)
    }
  }

  return (
    <li className="clip">
      <p className="clip-text">{clip.text}</p>
      <AudioPlayer url={clip.url} />
      <div className="clip-meta">
        <span className="tag">{clip.voice}</span>
        {clip.edited && (
          <span className="muted">{t.editedSpan(clip.edited[0].toFixed(2), clip.edited[1].toFixed(2))}</span>
        )}
        {!clip.edited && clip.seed !== null && <span className="muted">{t.seedValue(clip.seed)}</span>}
        <span className="muted">{t.seconds((clip.elapsedMs / 1000).toFixed(1))}</span>
        <span className="spacer" />
        {editor && draft === null && (
          <button type="button" className="ghost small" onClick={() => setDraft(clip.text)}>
            {t.editWords}
          </button>
        )}
        {!clip.edited && clip.seed !== null && (
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
      {draft !== null && editor && (
        <div className="take-edit">
          <textarea
            value={draft}
            rows={2}
            onChange={(e) => setDraft(e.target.value)}
            aria-label={t.editWords}
            disabled={editing}
          />
          <span className="muted hint">{t.editHint(editor)}</span>
          <div className="actions">
            <button type="button" className="small" disabled={!changed || busy} onClick={apply}>
              {editing ? t.editingWords : t.applyEdit}
            </button>
            <button type="button" className="ghost small" disabled={editing} onClick={() => setDraft(null)}>
              {t.cancel}
            </button>
          </div>
        </div>
      )}
    </li>
  )
}
