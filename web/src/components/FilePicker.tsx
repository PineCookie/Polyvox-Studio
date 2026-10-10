import { useRef, useState } from 'react'
import type { DragEvent, ReactNode } from 'react'

interface Props {
  file: File | null
  onChange: (file: File | null) => void
  accept?: string // e.g. "audio/*"
  label: string // the button
  empty: string // shown while no file is chosen
  children?: ReactNode // under the row (e.g. a player for the chosen file)
}

/** A file chooser in the app's own look (the native input shows the browser's
 * language), that also takes a file dropped anywhere on it. */
export function FilePicker({ file, onChange, accept, label, empty, children }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  function accepts(f: File): boolean {
    if (!accept) return true
    return accept.split(',').some((a) => {
      const pattern = a.trim()
      return pattern.endsWith('/*') ? f.type.startsWith(pattern.slice(0, -1)) : f.type === pattern
    })
  }

  function drop(e: DragEvent) {
    e.preventDefault()
    setOver(false)
    const f = e.dataTransfer.files[0]
    if (f && accepts(f)) onChange(f)
  }

  return (
    <div
      className={`file-picker${over ? ' over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <div className="file-row">
        <input ref={input} type="file" accept={accept} hidden onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
        <button type="button" className="ghost small" onClick={() => input.current?.click()}>
          {label}
        </button>
        <span className={file ? 'file-name' : 'muted hint'}>{file ? file.name : empty}</span>
        {file && (
          <button
            type="button"
            className="ghost small"
            onClick={() => {
              onChange(null)
              if (input.current) input.current.value = ''
            }}
          >
            ×
          </button>
        )}
      </div>
      {children}
    </div>
  )
}
