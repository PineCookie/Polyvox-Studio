import { useState } from 'react'

type Theme = 'auto' | 'light' | 'dark'

// Also read in index.html, which applies it before the page renders.
const STORAGE_KEY = 'voxcpm-studio.theme'
const NEXT: Record<Theme, Theme> = { auto: 'light', light: 'dark', dark: 'auto' }
const LABEL: Record<Theme, string> = { auto: '◐ Auto', light: '☀ Light', dark: '☾ Dark' }

function stored(): Theme {
  try {
    const value = localStorage.getItem(STORAGE_KEY)
    return value === 'light' || value === 'dark' ? value : 'auto'
  } catch {
    return 'auto'
  }
}

/** Light / dark / follow the system; remembered in this browser. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(stored)

  function cycle() {
    const next = NEXT[theme]
    setTheme(next)
    if (next === 'auto') delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = next
    try {
      if (next === 'auto') localStorage.removeItem(STORAGE_KEY)
      else localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // Not remembered (storage blocked); the choice still applies now.
    }
  }

  return (
    <button
      type="button"
      className="ghost small"
      onClick={cycle}
      title={`Theme: ${theme === 'auto' ? 'follows the system' : theme}. Click for ${NEXT[theme]}.`}
    >
      {LABEL[theme]}
    </button>
  )
}
