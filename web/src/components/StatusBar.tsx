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
      <span className="muted">{status.device}</span>
      <span className="muted">{status.optimize ? 'Triton on' : 'Triton off'}</span>
      {status.loaded && (
        <button type="button" className="ghost small" onClick={onUnload} title="Free GPU memory">
          Unload
        </button>
      )}
    </div>
  )
}
