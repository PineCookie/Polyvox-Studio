import { useEffect } from 'react'
import type { RefObject } from 'react'

/** Calls `onClose` on Escape or a press outside `ref`, while `open` (pop-ups, ⓘ explanations). */
export function useDismiss(ref: RefObject<HTMLElement | null>, open: boolean, onClose: () => void) {
  useEffect(() => {
    if (!open) return
    const press = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
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
  }, [ref, open, onClose])
}
