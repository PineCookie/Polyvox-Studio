import { useState } from 'react'
import type { FormEvent } from 'react'
import { exportLora } from '../api'
import type { Lora } from '../api'

interface Props {
  lora: Lora
  onError: (message: string) => void
  onClose: () => void
}

const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/
const HUB_ID_PATTERN = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/

// "speaker_r64/checkpoints/step_0005000" -> "speaker_r64_step5000"; ".../latest" -> "speaker_r64"
function defaultName(lora: Lora): string {
  const run = lora.id.split('/')[0]
  return /step_\d+$/.test(lora.id) && lora.step !== null ? `${run}_step${lora.step}` : run
}

function save(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

export function ExportForm({ lora, onError, onClose }: Props) {
  const [name, setName] = useState(() => defaultName(lora))
  const [baseModel, setBaseModel] = useState(lora.base_model_id ?? '')
  const [half, setHalf] = useState(true)
  const [busy, setBusy] = useState(false)

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
    <form className="export-form" onSubmit={submit}>
      <p className="muted hint">
        A zip with the weights, <code>lora_config.json</code> and a README. Local paths, optimizer state, logs,
        transcripts and dataset audio are left out.
      </p>
      <label className="field">
        <span>
          Name {!nameOk && <span className="field-error">letters, digits, . _ -</span>}
        </span>
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label className="field">
        <span>
          Base model {!baseOk && <span className="field-error">a Hugging Face id</span>}
        </span>
        <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} placeholder="openbmb/VoxCPM2" />
      </label>
      <label className="check">
        <input type="checkbox" checked={half} onChange={(e) => setHalf(e.target.checked)} /> Half-size weights
        (bfloat16, same output)
      </label>
      <div className="actions">
        <button type="button" className="ghost small" onClick={onClose}>
          Cancel
        </button>
        <button type="submit" className="primary small" disabled={busy || !nameOk || !baseOk}>
          {busy ? 'Exporting…' : 'Download'}
        </button>
      </div>
    </form>
  )
}
