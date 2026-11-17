import { useState } from 'react'
import type { Status } from '../api'
import { engineName } from '../engines'
import { useT } from '../i18n'
import { Popover } from './Popover'

interface Props {
  status: Status | null
  onUnload: () => void
}

/** Whether a model is loaded, as one pill; the machine's details and Unload in its pop-up. */
export function StatusBar({ status, onUnload }: Props) {
  const t = useT()
  const [open, setOpen] = useState(false)
  if (!status) return <span className="status-pill muted">{t.connecting}</span>
  // Any TTS engine (VoxCPM's own `loaded` predates engines).
  const ttsLoaded = (status.loaded_engines ?? []).length > 0 || status.loaded
  const loaded = [
    ...(status.loaded_engines ?? []).map((id) => engineName(id)),
    status.asr_loaded && t.asrLoaded,
    status.aligner_loaded && t.alignerLoaded,
    status.separation_loaded && t.separationLoaded,
    status.singing_loaded && t.singingLoaded,
  ].filter(Boolean) as string[]
  if (!loaded.length && ttsLoaded) loaded.push(t.modelLoaded)

  return (
    <Popover
      open={open}
      onClose={() => setOpen(false)}
      align="end"
      className="status-popover"
      anchor={
        <button
          type="button"
          className="status-pill"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
          title={t.statusTitle}
        >
          <span className={`dot ${ttsLoaded ? 'on' : ''}`} aria-hidden />
          {ttsLoaded ? t.modelLoaded : t.modelIdle}
        </button>
      }
    >
      <dl className="status-details">
        <dt>{t.statusDevice}</dt>
        <dd title={`PyTorch ${status.torch}`}>
          {status.device}
          {status.gpu?.total_gb ? ` · ${status.gpu.total_gb} GB` : ''}
        </dd>
        <dt>{t.statusCompile}</dt>
        <dd>{status.optimize ? t.tritonOn : t.tritonOff}</dd>
        <dt>{t.statusLoaded}</dt>
        <dd>{loaded.length ? loaded.join(', ') : t.statusNothing}</dd>
      </dl>
      {status.gpu && (
        <p className="muted hint status-note">{t.gpuTitle(status.gpu.name, status.gpu.exclusive, status.gpu.note)}</p>
      )}
      {loaded.length > 0 && (
        <div className="actions">
          <button
            type="button"
            className="ghost small"
            onClick={() => {
              setOpen(false)
              onUnload()
            }}
            title={t.unloadTitle}
          >
            {t.unload}
          </button>
        </div>
      )}
    </Popover>
  )
}
