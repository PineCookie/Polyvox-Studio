import { useState } from 'react'
import { useT } from '../i18n'
import { AudioPlayer } from './AudioPlayer'
import { DownloadIcon, MoreIcon } from './icons'
import { Popover } from './Popover'
import { SynthesisProgress } from './SynthesisPanel'

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

/** The takes of this session, newest first, under the editor. */
export function History({ clips, ...props }: Props) {
  const t = useT()
  return (
    <section className="takes" aria-label={t.history}>
      <header className="takes-head">
        <h2>{t.history}</h2>
        {clips.length > 0 && <span className="muted">{clips.length}</span>}
      </header>
      {clips.length === 0 ? (
        <p className="takes-empty muted">{t.historyEmpty}</p>
      ) : (
        <ol className="take-list">
          {clips.map((clip) => (
            <Take key={clip.id} clip={clip} {...props} />
          ))}
        </ol>
      )}
    </section>
  )
}

function Take({ clip, onUseSeed, onRemove, editor, busy, onEdit }: { clip: Clip } & Omit<Props, 'clips'>) {
  const t = useT()
  const [draft, setDraft] = useState<string | null>(null) // the new text while editing
  const [editing, setEditing] = useState(false)
  const [menu, setMenu] = useState(false)
  const changed = draft !== null && draft.trim() !== '' && draft.trim() !== clip.text

  async function apply() {
    setEditing(true)
    try {
      if (await onEdit(clip, draft!.trim())) setDraft(null)
    } finally {
      setEditing(false)
    }
  }

  const act = (action: () => void) => () => {
    setMenu(false)
    action()
  }

  return (
    <li className="take">
      <div className="take-top">
        <p className="take-text">{clip.text}</p>
        <div className="take-actions">
          {editor && draft === null && (
            <button type="button" className="ghost small" onClick={() => setDraft(clip.text)}>
              {t.editWords}
            </button>
          )}
          <a
            className="icon-button"
            href={clip.url}
            download={fileName(clip)}
            title={t.download}
            aria-label={t.download}
          >
            <DownloadIcon />
          </a>
          <Popover
            open={menu}
            onClose={() => setMenu(false)}
            align="end"
            className="menu"
            anchor={
              <button
                type="button"
                className="icon-button"
                aria-label={t.moreActions}
                aria-haspopup="menu"
                aria-expanded={menu}
                onClick={() => setMenu((m) => !m)}
              >
                <MoreIcon />
              </button>
            }
          >
            {!clip.edited && clip.seed !== null && (
              <button type="button" className="menu-item" onClick={act(() => onUseSeed(clip.seed!))}>
                {t.useSeedLong(clip.seed)}
              </button>
            )}
            <button type="button" className="menu-item danger-item" onClick={act(() => onRemove(clip.id))}>
              {t.removeClip}
            </button>
          </Popover>
        </div>
      </div>
      <AudioPlayer url={clip.url} />
      <p className="take-meta">
        <span className="take-voice">{clip.voice}</span>
        {clip.edited && <span>{t.editedSpan(clip.edited[0].toFixed(2), clip.edited[1].toFixed(2))}</span>}
        {!clip.edited && clip.seed !== null && <span>{t.seedValue(clip.seed)}</span>}
        <span>{t.madeIn((clip.elapsedMs / 1000).toFixed(1))}</span>
      </p>
      {draft !== null && editor && (
        <div className="take-edit">
          <textarea
            value={draft}
            rows={2}
            onChange={(e) => setDraft(e.target.value)}
            aria-label={t.editWords}
            disabled={editing}
            autoFocus
          />
          <span className="muted hint">{t.editHint(editor)}</span>
          {editing && <SynthesisProgress />}
          <div className="actions">
            <button type="button" className="ghost small" disabled={editing} onClick={() => setDraft(null)}>
              {t.cancel}
            </button>
            <button type="button" className="primary small" disabled={!changed || busy} onClick={apply}>
              {editing ? t.editingWords : t.applyEdit}
            </button>
          </div>
        </div>
      )}
    </li>
  )
}
