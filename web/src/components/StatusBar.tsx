import type { Status } from '../api'

interface Props {
  status: Status | null
  onUnload: () => void
}

export function StatusBar({ status, onUnload }: Props) {
  if (!status) return <span className="status muted">Connecting…</span>

  return (
    <div className="status">
      <span className={`dot ${status.loaded ? 'on' : ''}`} aria-hidden />
      <span>{status.loaded ? 'Model loaded' : 'Model idle'}</span>
      <span className="muted" title={`PyTorch ${status.torch}`}>
        {status.device}
      </span>
      {status.gpu?.total_gb && (
        <span
          className="muted"
          title={`Settings for a ${status.gpu.name} GPU: training batch ${status.gpu.train_batch}${
            status.gpu.exclusive ? '; synthesis and dataset preparation take turns on the GPU' : ''
          }${status.gpu.note ? `. ${status.gpu.note}` : ''}`}
        >
          {status.gpu.total_gb} GB
        </span>
      )}
      <span className="muted">{status.optimize ? 'Triton on' : 'Triton off'}</span>
      {status.asr_loaded && <span className="muted">ASR loaded</span>}
      {status.aligner_loaded && <span className="muted">Aligner loaded</span>}
      {status.separation_loaded && <span className="muted">Separation loaded</span>}
      {status.singing_loaded && <span className="muted">Singing detector loaded</span>}
      {(status.loaded ||
        status.asr_loaded ||
        status.aligner_loaded ||
        status.separation_loaded ||
        status.singing_loaded) && (
        <button
          type="button"
          className="ghost small"
          onClick={onUnload}
          title="Free GPU memory (TTS, ASR, aligner, separation)"
        >
          Unload
        </button>
      )}
    </div>
  )
}
