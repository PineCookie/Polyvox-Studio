import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { LANGS, LangContext, STORAGE_KEY, initialLang, setCurrentLang } from '.'
import type { Lang } from '.'

/** The UI language for everything inside: remembered in this browser, default the browser's. */
export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang)
  setCurrentLang(lang) // before the children render, for format.ts
  useEffect(() => {
    document.documentElement.lang = LANGS[lang].html
  }, [lang])

  function setLang(next: Lang) {
    setLangState(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // Not remembered; applies now.
    }
  }

  return <LangContext.Provider value={{ lang, setLang }}>{children}</LangContext.Provider>
}
