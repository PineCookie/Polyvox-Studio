import { useEffect, useState } from 'react'
import { getTakes } from './api'

const WAIT_MS = 250 // after the last change, before asking

interface Takes {
  count: number // takes the text is spoken in (as last asked)
  cuts: number[] // offsets in the text where a take ends and another starts
}

/** The takes an engine would speak ``text`` in (``engine`` null: it speaks any text in one). While
 * the text changes, the cuts before the change stay shown until the answer for the new text. */
export function useTakes(engine: string | null, text: string, cloning: boolean): Takes {
  const [known, setKnown] = useState<{ engine: string; text: string; ends: number[] } | null>(null)
  useEffect(() => {
    if (!engine) return
    let stale = false
    const timer = setTimeout(() => {
      getTakes(engine, text, cloning)
        .then((ends) => !stale && setKnown({ engine, text, ends }))
        .catch(() => undefined) // (only a hint: nothing shown without an answer)
    }, WAIT_MS)
    return () => {
      stale = true
      clearTimeout(timer)
    }
  }, [engine, text, cloning])
  if (!engine || !known || known.engine !== engine) return { count: 0, cuts: [] }
  let same = 0 // how far the text is still the one asked about
  while (same < text.length && same < known.text.length && text[same] === known.text[same]) same++
  return { count: known.ends.length, cuts: known.ends.slice(0, -1).filter((end) => end <= same) }
}
