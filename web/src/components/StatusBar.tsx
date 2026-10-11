import type { Status } from '../api'
import { useT } from '../i18n'

interface Props {
  status: Status | null
  onUnload: () => void
}

export function StatusBar({ status, onUnload }: Props) {
  const t = useT()
  if (!status) return <span className="status muted">{t.connecting}</span>
  // Any TTS engine (VoxCPM's own `loaded` predates engines).
  const ttsLoaded = (status.loaded_engines ?? []).length > 0 || status.loaded

  return (
    <div className="status">
      <span className={`dot ${ttsLoaded ? 'on' : ''}`} aria-hidden />
      <span title={status.loaded_engines?.join(', ')}>{ttsLoaded ? t.modelLoaded : t.modelIdle}</span>
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
      {(ttsLoaded ||
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
