import { useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useDismiss } from '../useDismiss'

interface Props {
  open: boolean
  onClose: () => void
  // The control that opens it (rendered in place; the panel hangs below it).
  anchor: ReactNode
  children: ReactNode
  className?: string
  // Which edge the panel lines up with; 'auto' picks the side with room.
  align?: 'start' | 'end' | 'auto'
  // Open upwards (a control near the bottom of the screen).
  up?: boolean
}

/** A panel that opens under its anchor and closes on Escape or a press outside. */
export function Popover({ open, onClose, anchor, children, className = '', align = 'auto', up = false }: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const [side, setSide] = useState<'start' | 'end'>(align === 'end' ? 'end' : 'start')
  useDismiss(wrap, open, onClose)

  useLayoutEffect(() => {
    if (!open || align !== 'auto' || !wrap.current || !panel.current) return
    const box = wrap.current.getBoundingClientRect()
    const width = panel.current.offsetWidth
    setSide(box.left + width > window.innerWidth - 12 && box.right - width > 12 ? 'end' : 'start')
  }, [open, align])

  return (
    <div className="pop-anchor" ref={wrap}>
      {anchor}
      {open && (
        <div ref={panel} className={`popover ${side}${up ? ' up' : ''} ${className}`} role="dialog">
          {children}
        </div>
      )}
    </div>
  )
}
