// Inline tags in a text to synthesize: where they are, and which an engine doesn't know.
import type { Engine } from './api'

const TAG = /(\[[^[\]\n]*\])/

/** The text in pieces, tags ("[...]") on their own. */
export function splitTags(text: string): { text: string; tag: boolean }[] {
  return text
    .split(TAG)
    .filter((piece) => piece !== '')
    .map((piece) => ({ text: piece, tag: TAG.test(piece) && piece.startsWith('[') }))
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

/** A tag's name without its brackets, e.g. "laughing". */
export const tagWord = (tag: string) => tag.replace(/^\[|\]$/g, '')
