import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { useT } from '../i18n'
import { useDismiss } from '../useDismiss'

const MARGIN = 10 // kept clear of the window's edges (scroll bars not counted as window)
const GAP = 8 // between the ⓘ and the bubble (the arrow sits in it)
const CLOSE_DELAY_MS = 120 // the pointer may cross the gap onto the bubble

interface Place {
  style: CSSProperties // the bubble's
  maxHeight: number // its text's (scrolls beyond)
  arrow: number // the arrow's distance from the bubble's left edge, under / over the ⓘ
  side: 'below' | 'above'
}

const HIDDEN: Place = { style: { visibility: 'hidden', left: 0, top: 0 }, maxHeight: 10_000, arrow: 0, side: 'below' }

/** Where the bubble goes for an ⓘ at ``anchor``: below it, or above when there is more room there;
 * always inside the window (shrunk to scroll when the window is too short for it); hidden while the
 * ⓘ is scrolled out of the window. */
function placeFor(anchor: DOMRect, width: number, height: number): Place {
  const vw = document.documentElement.clientWidth
  const vh = document.documentElement.clientHeight
  if (anchor.bottom < 0 || anchor.top > vh || anchor.right < 0 || anchor.left > vw) return HIDDEN // scrolled away
  const centre = anchor.left + anchor.width / 2
  const left = Math.min(Math.max(MARGIN, centre - Math.min(24, width / 2)), Math.max(MARGIN, vw - width - MARGIN))
  const roomBelow = vh - anchor.bottom - GAP - MARGIN
  const roomAbove = anchor.top - GAP - MARGIN
  const side = height <= roomBelow || roomBelow >= roomAbove ? 'below' : 'above'
  const room = Math.max(40, side === 'below' ? roomBelow : roomAbove)
  const shown = Math.min(height, room)
  const top = side === 'below' ? anchor.bottom + GAP : anchor.top - GAP - shown
  const arrow = Math.min(Math.max(10, centre - left), width - 10)
  return { style: { left, top }, maxHeight: room, arrow, side }
}

/** A small ⓘ beside a label: its explanation shows on hover, focus or a tap. The bubble is drawn
 * over the whole page or its dialog (so no panel around the label can cut it off or cover it), points at its ⓘ,
 * stays inside the window and follows the ⓘ while open. */
export function Info({ text }: { text: string }) {
  const t = useT()
  const id = useId()
  const wrap = useRef<HTMLSpanElement>(null)
  const bubble = useRef<HTMLSpanElement>(null)
  const inner = useRef<HTMLSpanElement>(null)
  const [hover, setHover] = useState(false)
  const [pinned, setPinned] = useState(false) // opened by a click / tap: stays until dismissed
  const [place, setPlace] = useState<Place>(HIDDEN)
  const closing = useRef<number | undefined>(undefined)
  const open = hover || pinned
  const close = useCallback(() => setPinned(false), [])
  const parts = useMemo(() => [wrap, bubble], [])
  useDismiss(parts, pinned, close)

  const enter = () => {
    window.clearTimeout(closing.current)
    setHover(true)
  }
  const leave = () => {
    window.clearTimeout(closing.current)
    closing.current = window.setTimeout(() => setHover(false), CLOSE_DELAY_MS)
  }
  useEffect(() => () => window.clearTimeout(closing.current), [])

  useLayoutEffect(() => {
    if (!open) return
    let frame = 0
    let last = ''
    // Every frame while open: the ⓘ can move (scrolling, text above it changing, the window).
    function position() {
      const anchor = wrap.current?.getBoundingClientRect()
      const own = bubble.current
      if (anchor && own && inner.current) {
        const next = placeFor(anchor, own.offsetWidth, inner.current.scrollHeight)
        const key = JSON.stringify(next)
        if (key !== last) {
          last = key
          setPlace(next)
        }
      }
      frame = requestAnimationFrame(position)
    }
    position()
    return () => {
      cancelAnimationFrame(frame)
      setPlace(HIDDEN)
    }
  }, [open, text])

  return (
    <span className="info" ref={wrap} onMouseEnter={enter} onMouseLeave={leave}>
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
        onFocus={enter}
        onBlur={leave}
      >
        i
      </button>
      {open &&
        createPortal(
          <span
            ref={bubble}
            id={id}
            role="tooltip"
            className={`info-bubble ${place.side}`}
            style={{ ...place.style, '--arrow': `${place.arrow}px` } as CSSProperties}
            onMouseEnter={enter}
            onMouseLeave={leave}
          >
            <span ref={inner} className="info-text" style={{ maxHeight: place.maxHeight }}>
              {text}
            </span>
          </span>,
          // A modal dialog is drawn in the browser's top layer, over everything in the page: an ⓘ
          // inside one puts its bubble in the dialog, else it would show behind it.
          wrap.current?.closest('dialog') ?? document.body,
        )}
    </span>
  )
}
