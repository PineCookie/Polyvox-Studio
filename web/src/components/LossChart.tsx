import { useMemo, useState } from 'react'
import type { PointerEvent } from 'react'
import type { MetricPoint } from '../api'
import { chartHeight, MARGIN as M, niceTicks, useWidth } from '../chartUtils'
import { formatStep, formatValue } from '../format'
import { useT } from '../i18n'

interface Props {
  title: string
  metric: string
  train: MetricPoint[]
  val: MetricPoint[]
}

const SMOOTHING = 0.85

interface Point {
  step: number
  value: number
}

function series(points: MetricPoint[], metric: string): Point[] {
  return points.filter((p) => Number.isFinite(p[metric])).map((p) => ({ step: p.step, value: p[metric] }))
}

// Debiased exponential moving average, as TensorBoard draws it.
function smooth(points: Point[]): Point[] {
  let last = 0
  return points.map((p, i) => {
    last = last * SMOOTHING + (1 - SMOOTHING) * p.value
    return { step: p.step, value: last / (1 - SMOOTHING ** (i + 1)) }
  })
}

function quantile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]
}

function nearest(points: Point[], step: number): Point | null {
  let best: Point | null = null
  for (const p of points) if (!best || Math.abs(p.step - step) < Math.abs(best.step - step)) best = p
  return best
}

export function LossChart({ title, metric, train, val }: Props) {
  const t = useT()
  const [ref, width] = useWidth<HTMLDivElement>()
  const height = chartHeight(width)
  const [hoverStep, setHoverStep] = useState<number | null>(null)

  const raw = useMemo(() => series(train, metric), [train, metric])
  const smoothed = useMemo(() => smooth(raw), [raw])
  const valPoints = useMemo(() => series(val, metric), [val, metric])

  const geometry = useMemo(() => {
    if (raw.length === 0 || width === 0) return null
    const steps = [...raw, ...valPoints].map((p) => p.step)
    const x0 = Math.min(...steps)
    const x1 = Math.max(...steps, x0 + 1)
    // Clip the y range to the bulk of the raw values so a few spikes don't flatten the curve.
    const sorted = raw.map((p) => p.value).sort((a, b) => a - b)
    const ys = [
      quantile(sorted, 0.02),
      quantile(sorted, 0.98),
      ...smoothed.map((p) => p.value),
      ...valPoints.map((p) => p.value),
    ]
    let y0 = Math.min(...ys)
    let y1 = Math.max(...ys)
    const pad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.1 || 1
    y0 = Math.max(0, y0 - pad)
    y1 += pad

    const plotW = width - M.left - M.right
    const plotH = chartHeight(width) - M.top - M.bottom
    const x = (step: number) => M.left + ((step - x0) / (x1 - x0)) * plotW
    const y = (v: number) => M.top + (1 - (Math.min(Math.max(v, y0), y1) - y0) / (y1 - y0)) * plotH
    const path = (pts: Point[]) =>
      pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.value).toFixed(1)}`).join('')
    return {
      x0,
      x1,
      y0,
      y1,
      x,
      y,
      plotW,
      path,
      yTicks: niceTicks(y0, y1, 4),
      xTicks: niceTicks(x0, x1, Math.max(2, Math.floor(plotW / 90))),
    }
  }, [raw, smoothed, valPoints, width])

  function onPointerMove(e: PointerEvent<SVGRectElement>) {
    if (!geometry) return
    const rect = e.currentTarget.getBoundingClientRect()
    const step = geometry.x0 + ((e.clientX - rect.left) / rect.width) * (geometry.x1 - geometry.x0)
    setHoverStep(nearest(raw, step)?.step ?? null)
  }

  const hoverIndex = hoverStep === null ? -1 : raw.findIndex((p) => p.step === hoverStep)
  const hoverVal = hoverStep === null ? null : nearest(valPoints, hoverStep)
  const latest = smoothed.at(-1)

  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{title}</span>
        {latest && <span className="muted">{t.smoothed(formatValue(latest.value))}</span>}
        <span className="spacer" />
        {raw.length > 0 && (
          <span className="legend">
            <span className="legend-item">
              <span className="swatch line train" aria-hidden /> {t.train}
            </span>
            {valPoints.length > 0 && (
              <span className="legend-item">
                <span className="swatch dot val" aria-hidden /> {t.validation}
              </span>
            )}
          </span>
        )}
      </figcaption>

      <div className="chart-body" ref={ref}>
        {!geometry ? (
          <p className="muted empty chart-empty">{raw.length === 0 ? t.noStepsYet : ''}</p>
        ) : (
          <svg width={width} height={height} role="img" aria-label={t.overSteps(title)}>
            {geometry.yTicks.map((t) => (
              <g key={`y${t}`}>
                <line className="grid" x1={M.left} x2={width - M.right} y1={geometry.y(t)} y2={geometry.y(t)} />
                <text className="tick" x={M.left - 8} y={geometry.y(t)} textAnchor="end" dominantBaseline="middle">
                  {formatValue(t)}
                </text>
              </g>
            ))}
            {geometry.xTicks.map((t) => (
              <text key={`x${t}`} className="tick" x={geometry.x(t)} y={height - 6} textAnchor="middle">
                {formatStep(t)}
              </text>
            ))}

            <path className="series-raw" d={geometry.path(raw)} />
            <path className="series-train" d={geometry.path(smoothed)} />
            {valPoints.map((p) => (
              <circle key={p.step} className="series-val" cx={geometry.x(p.step)} cy={geometry.y(p.value)} r={4} />
            ))}

            {hoverIndex >= 0 && (
              <g pointerEvents="none">
                <line
                  className="crosshair"
                  x1={geometry.x(raw[hoverIndex].step)}
                  x2={geometry.x(raw[hoverIndex].step)}
                  y1={M.top}
                  y2={height - M.bottom}
                />
                <circle
                  className="hover-dot"
                  cx={geometry.x(smoothed[hoverIndex].step)}
                  cy={geometry.y(smoothed[hoverIndex].value)}
                  r={4}
                />
              </g>
            )}
            <rect
              x={M.left}
              y={M.top}
              width={geometry.plotW}
              height={height - M.top - M.bottom}
              fill="transparent"
              onPointerMove={onPointerMove}
              onPointerLeave={() => setHoverStep(null)}
            />
          </svg>
        )}

        {geometry && hoverIndex >= 0 && (
          <div
            className="tooltip"
            style={{
              left: Math.min(geometry.x(raw[hoverIndex].step) + 12, width - 170),
              top: M.top,
            }}
          >
            <strong>{t.stepLabel(formatStep(raw[hoverIndex].step))}</strong>
            <span>
              <span className="swatch line train" aria-hidden /> {t.train} {formatValue(raw[hoverIndex].value)}
              <span className="muted">{t.smoothedValue(formatValue(smoothed[hoverIndex].value))}</span>
            </span>
            {hoverVal && (
              <span>
                <span className="swatch dot val" aria-hidden /> {t.validation} {formatValue(hoverVal.value)}
                {hoverVal.step !== raw[hoverIndex].step && (
                  <span className="muted"> @ {formatStep(hoverVal.step)}</span>
                )}
              </span>
            )}
          </div>
        )}
      </div>
    </figure>
  )
}
