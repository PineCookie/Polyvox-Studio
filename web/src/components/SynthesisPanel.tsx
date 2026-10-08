import { useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import type { SynthesisParams } from '../api'

interface Props {
  busy: boolean
  seed: number | null
  onSeedChange: (seed: number | null) => void
  onGenerate: (params: Omit<SynthesisParams, 'loraId'>) => void
}

export function SynthesisPanel({ busy, seed, onSeedChange, onGenerate }: Props) {
  const [text, setText] = useState('')
  const [control, setControl] = useState('')
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null)
  const [promptText, setPromptText] = useState('')
  const [cfgValue, setCfgValue] = useState(2.0)
  const [steps, setSteps] = useState(10)
  const [normalize, setNormalize] = useState(true)
  const [denoise, setDenoise] = useState(true)
  const fileInput = useRef<HTMLInputElement>(null)

  const canSubmit = !busy && text.trim().length > 0

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (!canSubmit) return
    onGenerate({ text, control, referenceAudio, promptText, cfgValue, steps, normalize, denoise, seed })
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit()
  }

  function clearReference() {
    setReferenceAudio(null)
    setPromptText('')
    if (fileInput.current) fileInput.current.value = ''
  }

  return (
    <form className="panel" onSubmit={submit} onKeyDown={onKeyDown}>
      <header className="panel-head">
        <h2>Synthesize</h2>
      </header>

      <label className="field">
        <span>Text</span>
        <textarea
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="What should the voice say?"
          autoFocus
        />
      </label>

      <label className="field">
        <span>
          Style <span className="muted">optional</span>
        </span>
        <input
          value={control}
          onChange={(e) => setControl(e.target.value)}
          placeholder="e.g. a calm, warm female voice, speaking slowly"
        />
      </label>

      <fieldset className="field reference">
        <legend>
          Reference audio <span className="muted">optional · voice cloning</span>
        </legend>
        <div className="file-row">
          <input
            ref={fileInput}
            type="file"
            accept="audio/*"
            onChange={(e) => setReferenceAudio(e.target.files?.[0] ?? null)}
          />
          {referenceAudio && (
            <button type="button" className="ghost small" onClick={clearReference}>
              Clear
            </button>
          )}
        </div>
        {referenceAudio && (
          <input
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            placeholder="Transcript of the reference (improves cloning fidelity)"
          />
        )}
      </fieldset>

      <details className="advanced">
        <summary>Advanced</summary>
        <div className="advanced-grid">
          <label className="field">
            <span>
              Guidance <output>{cfgValue.toFixed(1)}</output>
            </span>
            <input
              type="range"
              min={0.5}
              max={5}
              step={0.1}
              value={cfgValue}
              onChange={(e) => setCfgValue(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>
              Steps <output>{steps}</output>
            </span>
            <input type="range" min={1} max={50} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>
          <label className="field">
            <span>Seed</span>
            <input
              type="number"
              value={seed ?? ''}
              placeholder="Random"
              onChange={(e) => onSeedChange(e.target.value === '' ? null : Number(e.target.value))}
            />
          </label>
          <div className="toggles">
            <label>
              <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> Normalize
              text
            </label>
            <label>
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} /> Denoise
              reference
            </label>
          </div>
        </div>
      </details>

      <div className="actions">
        <span className="muted hint">Ctrl + Enter</span>
        <button type="submit" className="primary" disabled={!canSubmit}>
          {busy ? 'Generating…' : 'Generate'}
        </button>
      </div>
    </form>
  )
}
