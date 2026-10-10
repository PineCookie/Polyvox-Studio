import type { Status } from '../api'
import { useT } from '../i18n'

interface Props {
  status: Status | null
  onUnload: () => void
}

export function StatusBar({ status, onUnload }: Props) {
  const t = useT()
  if (!status) return <span className="status muted">{t.connecting}</span>

  return (
    <div className="status">
      <span className={`dot ${status.loaded ? 'on' : ''}`} aria-hidden />
      <span>{status.loaded ? t.modelLoaded : t.modelIdle}</span>
      <span className="muted" title={`PyTorch ${status.torch}`}>
        {status.device}
      </span>
      {status.gpu?.total_gb && (
        <span
          className="muted"
          title={t.gpuTitle(status.gpu.name, status.gpu.train_batch, status.gpu.exclusive, status.gpu.note)}
        >
          {status.gpu.total_gb} GB
        </span>
      )}
      <span className="muted">{status.optimize ? t.tritonOn : t.tritonOff}</span>
      {status.asr_loaded && <span className="muted">{t.asrLoaded}</span>}
      {status.aligner_loaded && <span className="muted">{t.alignerLoaded}</span>}
      {status.separation_loaded && <span className="muted">{t.separationLoaded}</span>}
      {status.singing_loaded && <span className="muted">{t.singingLoaded}</span>}
      {(status.loaded ||
        status.asr_loaded ||
        status.aligner_loaded ||
        status.separation_loaded ||
        status.singing_loaded) && (
        <button type="button" className="ghost small" onClick={onUnload} title={t.unloadTitle}>
          {t.unload}
        </button>
      )}
    </div>
  )
}
