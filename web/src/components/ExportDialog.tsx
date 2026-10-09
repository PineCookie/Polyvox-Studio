import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { exportLora } from '../api'
import type { Lora } from '../api'
import { formatStep } from '../format'

interface Props {
  lora: Lora
  // The run's last checkpoint: named after the run alone.
  final: boolean
  onError: (message: string) => void
  onClose: () => void
}

const NAME_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}_.-]{0,79}$/u
const HUB_ID_PATTERN = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/

// "speaker_r64/checkpoints/step_0004000" -> "speaker_r64_step4000"; the run name for its final weights.
function defaultName(lora: Lora, final: boolean): string {
  const run = lora.id.split('/')[0]
  return !final && lora.step !== null ? `${run}_step${lora.step}` : run
}

function save(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

export function ExportDialog({ lora, final, onError, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [name, setName] = useState(() => defaultName(lora, final))
  const [baseModel, setBaseModel] = useState(lora.base_model_id ?? '')
  const [half, setHalf] = useState(true)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    dialog.current?.showModal()
  }, [])

  const nameOk = NAME_PATTERN.test(name)
  const baseOk = HUB_ID_PATTERN.test(baseModel)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!nameOk || !baseOk) return
    setBusy(true)
    try {
      save(await exportLora(lora.id, name, baseModel, half), `${name}.zip`)
      onClose()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <dialog
      ref={dialog}
      className="dialog"
      onClose={onClose}
      onClick={(e) => e.target === dialog.current && onClose()} // backdrop
    >
      <form className="dialog-body" onSubmit={submit}>
        <header className="panel-head">
          <h2>Export for sharing</h2>
          <span className="muted hint">
            {lora.id.split('/')[0]}
            {lora.step !== null && ` · step ${formatStep(lora.step)}`}
          </span>
        </header>

        <p className="muted hint">
          A zip with the weights, <code>lora_config.json</code> and a README on how to use it. Local paths, optimizer
          state, logs, transcripts and dataset audio are left out.
        </p>

        <label className="field">
          <span>Name {!nameOk && <span className="field-error">letters, digits, . _ -</span>}</span>
          <input value={name} onChange={(e) => setName(e.target.value)} autoFocus />
        </label>
        <label className="field">
          <span>Base model {!baseOk && <span className="field-error">a Hugging Face id</span>}</span>
          <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} placeholder="openbmb/VoxCPM2" />
        </label>
        <label className="check">
          <input type="checkbox" checked={half} onChange={(e) => setHalf(e.target.checked)} /> Half-size weights
          (bfloat16, same output)
        </label>

        <div className="actions">
          <button type="button" className="ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={busy || !nameOk || !baseOk}>
            {busy ? 'Exporting…' : 'Download zip'}
          </button>
        </div>
      </form>
    </dialog>
  )
}
