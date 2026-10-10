import { useMemo, useState } from 'react'
import type { PointerEvent } from 'react'
import type { MetricPoint } from '../api'
import { chartHeight, MARGIN, niceTicks, useWidth } from '../chartUtils'
import { formatStep } from '../format'
import { useT } from '../i18n'

const M = { ...MARGIN, left: 60 } // room for 1.0e-4 style ticks
const COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']
const GROUP_PREFIX = 'lr_' // per-network rates (Prodigy-SF), else the single "lr"

const fmt = (v: number) => (v === 0 ? '0' : v.toExponential(1))

/** The learning rate over the steps: one line per parameter group when the run
 * logs them (Prodigy-SF's per-network step sizes), else the logged rate. */
export function LrChart({ train }: { train: MetricPoint[] }) {
  const t = useT()
  const [ref, width] = useWidth<HTMLDivElement>()
  const height = chartHeight(width)
  const [hoverStep, setHoverStep] = useState<number | null>(null)

  const keys = useMemo(() => {
    const groups = new Set<string>()
    for (const p of train) for (const k of Object.keys(p)) if (k.startsWith(GROUP_PREFIX)) groups.add(k)
    return groups.size > 0 ? [...groups] : ['lr']
  }, [train])
  const points = useMemo(() => train.filter((p) => keys.some((k) => Number.isFinite(p[k]))), [train, keys])

  const geometry = useMemo(() => {
    if (points.length === 0 || width === 0) return null
    const x0 = points[0].step
    const x1 = Math.max(points.at(-1)!.step, x0 + 1)
    const values = points.flatMap((p) => keys.map((k) => p[k]).filter(Number.isFinite))
    const y1 = Math.max(...values, 1e-12) * 1.08
    const plotW = width - M.left - M.right
    const plotH = height - M.top - M.bottom
    const x = (step: number) => M.left + ((step - x0) / (x1 - x0)) * plotW
    const y = (v: number) => M.top + (1 - v / y1) * plotH
    const path = (key: string) =>
      points
        .filter((p) => Number.isFinite(p[key]))
        .map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p[key]).toFixed(1)}`)
        .join('')
    return {
      x0,
      x1,
      x,
      y,
      plotW,
      path,
      yTicks: niceTicks(0, y1, 4),
      xTicks: niceTicks(x0, x1, Math.max(2, Math.floor(plotW / 90))),
    }
  }, [points, keys, width, height])

  function onPointerMove(e: PointerEvent<SVGRectElement>) {
    if (!geometry) return
    const rect = e.currentTarget.getBoundingClientRect()
    const step = geometry.x0 + ((e.clientX - rect.left) / rect.width) * (geometry.x1 - geometry.x0)
    let best: MetricPoint | null = null
    for (const p of points) if (!best || Math.abs(p.step - step) < Math.abs(best.step - step)) best = p
    setHoverStep(best?.step ?? null)
  }

  const label = (key: string) => (key === 'lr' ? t.lrSingle : key.slice(GROUP_PREFIX.length))
  const hover = hoverStep === null ? null : points.find((p) => p.step === hoverStep)
  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{t.lrTitle}</span>
        <span className="spacer" />
        <span className="legend">
          {keys.map((k, i) => (
            <span key={k} className="legend-item">
              <span className="swatch line" style={{ background: COLORS[i % COLORS.length] }} aria-hidden /> {label(k)}
            </span>
          ))}
        </span>
      </figcaption>
      <div className="chart-body" ref={ref}>
        {!geometry ? (
          <p className="muted empty chart-empty">{t.noStepsYet}</p>
        ) : (
          <svg width={width} height={height} role="img" aria-label={t.overSteps(t.lrTitle)}>
            {geometry.yTicks.map((v) => (
              <g key={`y${v}`}>
                <line className="grid" x1={M.left} x2={width - M.right} y1={geometry.y(v)} y2={geometry.y(v)} />
                <text className="tick" x={M.left - 8} y={geometry.y(v)} textAnchor="end" dominantBaseline="middle">
                  {fmt(v)}
                </text>
              </g>
            ))}
            {geometry.xTicks.map((v) => (
              <text key={`x${v}`} className="tick" x={geometry.x(v)} y={height - 6} textAnchor="middle">
                {formatStep(v)}
              </text>
            ))}
            {keys.map((k, i) => (
              <path key={k} className="series-lr" style={{ stroke: COLORS[i % COLORS.length] }} d={geometry.path(k)} />
            ))}
            {hover && (
              <line
                className="crosshair"
                pointerEvents="none"
                x1={geometry.x(hover.step)}
                x2={geometry.x(hover.step)}
                y1={M.top}
                y2={height - M.bottom}
              />
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
        {geometry && hover && (
          <div className="tooltip" style={{ left: Math.min(geometry.x(hover.step) + 12, width - 190), top: M.top }}>
            <strong>{t.stepLabel(formatStep(hover.step))}</strong>
            {keys.map((k, i) => (
              <span key={k}>
                <span className="swatch line" style={{ background: COLORS[i % COLORS.length] }} aria-hidden />{' '}
                {label(k)} {Number.isFinite(hover[k]) ? fmt(hover[k]) : '–'}
              </span>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}
