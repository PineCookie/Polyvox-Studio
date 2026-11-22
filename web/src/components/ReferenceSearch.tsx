import type { ReferencePick, ReferenceSearchJob } from '../api'
import { useT } from '../i18n'
import { CheckIcon } from './icons'
import { Info } from './Info'

// How a voice's clips are picked from a dataset: as Synthesize's Pick, or tested by speaking (slow).
export type VoicePick = ReferencePick | 'test'

/** Pick for a voice's dataset clips: the same choices as Synthesize's dataset clips; `test`: the
 * engine can also try a few sets by speaking them (capabilities.reference_search). */
export function ReferencePickSelect({
  value,
  test,
  onChange,
}: {
  value: VoicePick
  test: boolean
  onChange: (pick: VoicePick) => void
}) {
  const t = useT()
  return (
    <label className="pick-by">
      <span className="muted">{t.referencePickBy}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as VoicePick)}>
        <option value="voice">{t.referencePickVoice}</option>
        <option value="text">{t.referencePickText}</option>
        {test && <option value="test">{t.referencePickTest}</option>}
      </select>
      {test && <Info text={t.findBestClipsHint} />}
    </label>
  )
}

/** The sets a finished search compared, the kept one ticked. */
export function ReferenceSearchResult({ result }: { result: ReferenceSearchJob['result'] }) {
  const t = useT()
  const signed = (n: number) => (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n).toFixed(1)
  return (
    <div className="reference-search-result">
      <span className="muted">{t.referenceSearchDone}</span>
      <ul>
        {result.tried.map((s) => {
          const kept = s.name === result.chosen
          return (
            <li key={s.name} className={kept ? 'kept' : undefined}>
              <span className="set-name">
                {t.referenceSearchSets[s.name] ?? s.name}
                {kept && (
                  <span className="kept-mark" title={t.referenceSearchKept}>
                    <CheckIcon />
                  </span>
                )}
              </span>
              <span className="muted num">
                {t.referenceSearchRow(s.likeness.toFixed(2), s.pitch == null ? '–' : t.semitones(signed(s.pitch)))}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
