// Which groups of a list are folded, remembered in localStorage under `key`.
import { useState } from 'react'

function read(key: string): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(key) ?? '[]') as string[])
  } catch {
    return new Set()
  }
}

function write(key: string, ids: Set<string>) {
  try {
    localStorage.setItem(key, JSON.stringify([...ids]))
  } catch {
    // not remembered (storage unavailable)
  }
}

export function useFolded(key: string): [Set<string>, (id: string) => void] {
  const [folded, setFolded] = useState<Set<string>>(() => read(key))
  function toggle(id: string) {
    setFolded((f) => {
      const next = new Set(f)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      write(key, next)
      return next
    })
  }
  return [folded, toggle]
}
