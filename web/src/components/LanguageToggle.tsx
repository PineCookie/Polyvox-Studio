import { LANGS, useLang, useT } from '../i18n'
import type { Lang } from '../i18n'

/** UI language; remembered in this browser (default: the browser's language). */
export function LanguageToggle() {
  const { lang, setLang } = useLang()
  const t = useT()
  return (
    <select
      className="lang-select"
      value={lang}
      onChange={(e) => setLang(e.target.value as Lang)}
      aria-label={t.language}
      title={t.language}
    >
      {(Object.keys(LANGS) as Lang[]).map((l) => (
        <option key={l} value={l}>
          {LANGS[l].name}
        </option>
      ))}
    </select>
  )
}
