// UI languages: en.ts is the source; every other language must match its type
// (a missing or extra message is a build error). Messages with values are
// functions, since word order differs between languages. The provider is in
// I18nProvider.tsx.
import { createContext, useContext } from 'react'
import { en } from './en'
import type { Messages } from './en'
import { zh } from './zh'

export type Lang = 'en' | 'zh'
export type { Messages }

export const LANGS: Record<Lang, { messages: Messages; name: string; html: string }> = {
  en: { messages: en, name: 'English', html: 'en' },
  zh: { messages: zh, name: '简体中文', html: 'zh-CN' },
}

export const STORAGE_KEY = 'polyvox.lang'

// For code outside components (format.ts); set by the provider before its children render.
let current: Messages = en

/** The current language's messages, for helpers outside React components. */
export function messages(): Messages {
  return current
}

export function setCurrentLang(lang: Lang): void {
  current = LANGS[lang].messages
}

export function initialLang(): Lang {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'en' || stored === 'zh') return stored
  } catch {
    // Storage blocked: follow the browser.
  }
  return navigator.languages.some((l) => l.toLowerCase().startsWith('zh')) ? 'zh' : 'en'
}

export const LangContext = createContext<{ lang: Lang; setLang: (lang: Lang) => void }>({
  lang: 'en',
  setLang: () => {},
})

/** The current language's messages (re-renders when the language changes). */
export function useT(): Messages {
  return LANGS[useContext(LangContext).lang].messages
}

export function useLang() {
  return useContext(LangContext)
}
