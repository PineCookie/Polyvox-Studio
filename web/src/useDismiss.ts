import { useEffect } from 'react'
import type { RefObject } from 'react'

type Refs = RefObject<HTMLElement | null> | RefObject<HTMLElement | null>[]

/** Calls `onClose` on Escape or a press outside `refs` (one element or several, e.g. an ⓘ and
 * its bubble drawn elsewhere), while `open` (pop-ups, ⓘ explanations). */
export function useDismiss(refs: Refs, open: boolean, onClose: () => void) {
  useEffect(() => {
    if (!open) return
    const inside = (target: Node) => (Array.isArray(refs) ? refs : [refs]).some((r) => r.current?.contains(target))
    const press = (e: PointerEvent) => {
      if (!inside(e.target as Node)) onClose()
    }
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('pointerdown', press)
    document.addEventListener('keydown', key)
    return () => {
      document.removeEventListener('pointerdown', press)
      document.removeEventListener('keydown', key)
    }
  }, [refs, open, onClose])
}
