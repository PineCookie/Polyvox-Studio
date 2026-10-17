import { forwardRef, useRef } from 'react'
import type { KeyboardEvent } from 'react'
import { splitTags } from '../tags'

interface Props {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  unknown?: string[] // tags to mark as not understood
  rows?: number
  autoFocus?: boolean
  onKeyDown?: (e: KeyboardEvent<HTMLTextAreaElement>) => void
}

/** A text box that highlights inline tags: a backdrop with the same text (transparent, the tags
 * marked) under a transparent textarea, scrolled with it. */
export const TaggedText = forwardRef<HTMLTextAreaElement, Props>(function TaggedText(
  { value, onChange, placeholder, unknown = [], rows = 5, autoFocus, onKeyDown },
  ref,
) {
  const backdrop = useRef<HTMLDivElement>(null)
  return (
    <div className="tagged-text">
      <div className="tagged-backdrop" ref={backdrop} aria-hidden>
        {splitTags(value).map((piece, i) =>
          piece.tag ? (
            <mark key={i} className={unknown.includes(piece.text) ? 'unknown' : undefined}>
              {piece.text}
            </mark>
          ) : (
            <span key={i}>{piece.text}</span>
          ),
        )}
        {/* a final newline needs a line of its own here too */}
        {value.endsWith('\n') && ' '}
      </div>
      <textarea
        ref={ref}
        rows={rows}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={(e) => {
          if (backdrop.current) backdrop.current.scrollTop = e.currentTarget.scrollTop
        }}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        autoFocus={autoFocus}
        spellCheck={false}
      />
    </div>
  )
})
