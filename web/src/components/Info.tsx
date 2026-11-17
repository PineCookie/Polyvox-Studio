import { useCallback, useId, useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useT } from '../i18n'
import { useDismiss } from '../useDismiss'

const MARGIN = 12 // kept clear of the window's edges

/** A small ⓘ beside a label: its explanation shows on hover, focus or a tap. The bubble is placed
 * on the window (fixed), so a scrolling panel around the label can't cut it off. */
export function Info({ text }: { text: string }) {
  const t = useT()
  const id = useId()
  const wrap = useRef<HTMLSpanElement>(null)
  const bubble = useRef<HTMLSpanElement>(null)
  const [hover, setHover] = useState(false)
  const [pinned, setPinned] = useState(false) // opened by a click / tap: stays until dismissed
  const [place, setPlace] = useState<CSSProperties>({ visibility: 'hidden' })
  const open = hover || pinned
  const close = useCallback(() => setPinned(false), [])
  useDismiss(wrap, pinned, close)

  useLayoutEffect(() => {
    if (!open) return
    function position() {
      const anchor = wrap.current?.getBoundingClientRect()
      const own = bubble.current
      if (!anchor || !own) return
      const width = own.offsetWidth
      const height = own.offsetHeight
      const left = Math.min(Math.max(MARGIN, anchor.left - 6), window.innerWidth - width - MARGIN)
      // Below the ⓘ, or above it when there is no room below.
      const below = anchor.bottom + 6
      const top = below + height > window.innerHeight - MARGIN ? anchor.top - 6 - height : below
      setPlace({ left, top })
    }
    position()
    window.addEventListener('scroll', position, true)
    window.addEventListener('resize', position)
    return () => {
      window.removeEventListener('scroll', position, true)
      window.removeEventListener('resize', position)
      setPlace({ visibility: 'hidden' })
    }
  }, [open])

  return (
    <span className="info" ref={wrap} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <button
        type="button"
        className="info-button"
        aria-label={t.moreInfo}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onClick={(e) => {
          e.preventDefault() // inside a <label>: don't move focus to its control
          setPinned((p) => !p)
        }}
        onFocus={() => setHover(true)}
        onBlur={() => setHover(false)}
      >
        i
      </button>
      {open && (
        <span ref={bubble} id={id} role="tooltip" className="info-bubble" style={place}>
          {text}
        </span>
      )}
    </span>
  )
}
