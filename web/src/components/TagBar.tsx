import { useState } from 'react'
import type { Engine } from '../api'
import { useT } from '../i18n'
import { scoped, tagWord } from '../tags'
import { Info } from './Info'

interface Props {
  engine: Engine
  onInsert: (tag: string) => void
}

/** An engine's inline tags as a strip in the editor: its groups as tabs, the open group's tags as
 * chips drawn like the tags in the text (so the syntax shows too), and "Your own…" where the model
 * takes any description. How the engine's tags act is behind ⓘ. */
export function TagBar({ engine, onInsert }: Props) {
  const t = useT()
  const caps = engine.capabilities
  const groups = caps.tag_groups
  const [open, setOpen] = useState<string>(groups[0]?.name ?? (caps.free_tags ? 'custom' : ''))
  const [custom, setCustom] = useState('')
  const shown = groups.find((g) => g.name === open)

  function insertCustom() {
    const words = custom.trim().replace(/^\[|\]$/g, '')
    if (!words) return
    onInsert(`[${words}]`)
    setCustom('')
  }

  // A chip reads exactly as the tag goes into the text ("[sigh]", "<whisper>…"); a translated
  // name, where there is one, shows on hover.
  function chipLabel(tag: string): string {
    return scoped(tag) ? `<${tagWord(tag)}>…` : tag
  }
  function chipTitle(tag: string): string {
    const name = t.tagNames[tagWord(tag)]
    return name ? `${name} · ${t.insertTag(tag)}` : t.insertTag(tag)
  }

  return (
    <div className="tag-strip" role="group" aria-label={t.tagGroup}>
      <div className="tag-tabs" role="tablist">
        {groups.map((g) => (
          <button
            key={g.name}
            type="button"
            role="tab"
            aria-selected={open === g.name}
            className={open === g.name ? 'active' : ''}
            onClick={() => setOpen(g.name)}
          >
            {t.tagGroups[g.name] ?? g.name}
          </button>
        ))}
        {caps.free_tags && (
          <button
            type="button"
            role="tab"
            aria-selected={open === 'custom'}
            className={open === 'custom' ? 'active' : ''}
            onClick={() => setOpen('custom')}
          >
            {t.customTag}
          </button>
        )}
        {t.tagHints[engine.id] && <Info text={t.tagHints[engine.id]} />}
      </div>
      {shown && (
        <div className="tag-chips" role="tabpanel">
          {shown.tags.map((tag) => (
            <button key={tag} type="button" className="tag-chip" onClick={() => onInsert(tag)} title={chipTitle(tag)}>
              {chipLabel(tag)}
            </button>
          ))}
        </div>
      )}
      {open === 'custom' && (
        // Not a <form>: this sits inside Synthesize's form; Enter inserts instead of generating.
        <div className="tag-custom" role="tabpanel">
          <input
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            placeholder={t.customTagPlaceholder}
            autoFocus
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
    </div>
  )
}
