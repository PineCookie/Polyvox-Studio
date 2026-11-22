import { forwardRef, Fragment, useRef } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { splitTags } from '../tags'

interface Props {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  unknown?: string[] // tags to mark as not understood
  cuts?: number[] // offsets to mark: a take ends there and another starts
  rows?: number
  autoFocus?: boolean
  onKeyDown?: (e: KeyboardEvent<HTMLTextAreaElement>) => void
}

/** The text with a mark (no width: the lines stay the textarea's) at each offset in ``cuts``. */
function withCuts(text: string, start: number, cuts: number[]): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const cut of cuts) {
    const at = cut - start
    if (at <= 0 || at > text.length) continue
    out.push(text.slice(last, at), <i key={cut} className="take-cut" />)
    last = at
  }
  out.push(text.slice(last))
  return out
}

/** A text box that highlights inline tags and marks where takes end: a backdrop with the same
 * text (transparent, the tags marked) under a transparent textarea, scrolled with it. */
export const TaggedText = forwardRef<HTMLTextAreaElement, Props>(function TaggedText(
  { value, onChange, placeholder, unknown = [], cuts = [], rows = 5, autoFocus, onKeyDown },
  ref,
) {
  const backdrop = useRef<HTMLDivElement>(null)
  let offset = 0
  return (
    <div className="tagged-text">
      <div className="tagged-backdrop" ref={backdrop} aria-hidden>
        {splitTags(value).map((piece, i) => {
          const start = offset
          offset += piece.text.length
          // A take can end right after a tag ("…</whisper>"): its mark goes after the tag (the
          // text that follows starts at the cut, and marks only go inside or at the end of text).
          return piece.tag ? (
            <Fragment key={i}>
              <mark className={unknown.includes(piece.text) ? 'unknown' : undefined}>{piece.text}</mark>
              {cuts.includes(offset) && <i className="take-cut" />}
            </Fragment>
          ) : (
            <span key={i}>{withCuts(piece.text, start, cuts)}</span>
          )
        })}
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
