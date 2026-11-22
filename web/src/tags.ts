// Inline tags in a text to synthesize: where they are, and which an engine doesn't know.
// Point tags "[laughing]"; scoped tags "<whisper>...</whisper>" (Drama 3) around a span.
import type { Engine } from './api'

const TAG = /(\[[^[\]\n]*\]|<\/?[a-z][\w -]*>)/i

/** The text in pieces, tags ("[...]", "<...>", "</...>") on their own. */
export function splitTags(text: string): { text: string; tag: boolean }[] {
  return text
    .split(TAG)
    .filter((piece) => piece !== '')
    .map((piece) => ({ text: piece, tag: TAG.test(piece) && /^[[<]/.test(piece) }))
}

/** Tags in the text that the engine would read out as words: unknown to a fixed-vocabulary model. */
export function unknownTags(text: string, engine: Engine | null): string[] {
  const caps = engine?.capabilities
  if (!caps?.tags || caps.free_tags) return []
  const known = new Set(caps.tag_groups.flatMap((g) => g.tags))
  return [
    ...new Set(
      splitTags(text)
        .filter((p) => p.tag && !known.has(p.text))
        .map((p) => p.text),
    ),
  ]
}

/** A tag's name without its brackets, e.g. "laughing" ("[laughing]"), "whisper" ("<whisper></whisper>"). */
export const tagWord = (tag: string) => scoped(tag)?.[0] ?? tag.replace(/^\[|\]$/g, '')

/** A scoped tag's name and its opening and closing tags ("<whisper></whisper>"), else null. */
// What a cut between takes must not fall inside (as the server's spoken_parts): special tokens, a
// tag pair with all it holds, a [bracketed] tag.
const WHOLE = /<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>[\s\S]*?<\/\1>|\[[^[\]]*\]/g

/** ``at`` moved past the tag or tag pair it falls inside, if any: a new take can start there
 * (a blank line inside "<whisper>...</whisper>" doesn't cut, it would only widen the whisper). */
export function outsideTags(text: string, at: number): number {
  for (const m of text.matchAll(WHOLE)) {
    const start = m.index ?? 0
    if (start < at && at < start + m[0].length) return start + m[0].length
  }
  return at
}

export function scoped(tag: string): [string, string, string] | null {
  const m = /^<([a-z][\w -]*)><\/\1>$/i.exec(tag)
  return m ? [m[1], `<${m[1]}>`, `</${m[1]}>`] : null
}
