import { useState } from 'react'
import type { Engine } from '../api'
import { useT } from '../i18n'
import { tagWord } from '../tags'

interface Props {
  engine: Engine
  onInsert: (tag: string) => void
}

/** An engine's inline tags: one group open at a time, its tags as chips; a custom
 * [description] where the model takes any. Shows how the engine's tags act. */
export function TagBar({ engine, onInsert }: Props) {
  const t = useT()
  const caps = engine.capabilities
  const groups = caps.tag_groups
  const [open, setOpen] = useState<string>(groups[0]?.name ?? '')
  const [custom, setCustom] = useState('')
  const shown = groups.find((g) => g.name === open)

  function insertCustom() {
    const words = custom.trim().replace(/^\[|\]$/g, '')
    if (!words) return
    onInsert(`[${words}]`)
    setCustom('')
  }

  return (
    <div className="tag-bar" role="group" aria-label={t.tagGroup}>
      <div className="tag-groups">
        <span className="muted">{t.tags}</span>
        {groups.map((g) => (
          <button
            key={g.name}
            type="button"
            className={`tag-group${open === g.name ? ' active' : ''}`}
            aria-pressed={open === g.name}
            onClick={() => setOpen(open === g.name ? '' : g.name)}
          >
            {t.tagGroups[g.name] ?? g.name}
          </button>
        ))}
        {caps.free_tags && (
          <button
            type="button"
            className={`tag-group${open === 'custom' ? ' active' : ''}`}
            aria-pressed={open === 'custom'}
            onClick={() => setOpen(open === 'custom' ? '' : 'custom')}
          >
            {t.customTag}
          </button>
        )}
      </div>
      {shown && (
        <div className="tag-chips">
          {shown.tags.map((tag) => (
            <button key={tag} type="button" className="chip" onClick={() => onInsert(tag)} title={t.insertTag(tag)}>
              {t.tagNames[tagWord(tag)] ?? tagWord(tag)}
            </button>
          ))}
        </div>
      )}
      {open === 'custom' && (
        // Not a <form>: this sits inside Synthesize's form; Enter inserts instead of generating.
        <div className="file-row">
          <input
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            placeholder={t.customTagPlaceholder}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                e.stopPropagation()
                insertCustom()
              }
            }}
          />
          <button type="button" className="ghost small" onClick={insertCustom} disabled={!custom.trim()}>
            {t.insert}
          </button>
        </div>
      )}
      {t.tagHints[engine.id] && <p className="muted hint">{t.tagHints[engine.id]}</p>}
    </div>
  )
}
