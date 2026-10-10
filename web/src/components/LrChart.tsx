import { useMemo, useState } from 'react'
import type { PointerEvent } from 'react'
import type { MetricPoint } from '../api'
import { chartHeight, MARGIN, niceTicks, useWidth } from '../chartUtils'
import { formatStep } from '../format'
import { useT } from '../i18n'

const M = { ...MARGIN, left: 60 } // room for 1.0e-4 style ticks
const COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']
// Prodigy-SF logs per network: lr_<name> (d x lr) and elr_<name> (the step of
// the training weights, smaller once Schedule-Free's average builds up).
const STEP_PREFIX = 'lr_'
const EFFECTIVE_PREFIX = 'elr_'

const fmt = (v: number) => (v === 0 ? '0' : v.toExponential(1))

interface Series {
  key: string
  label: string
  color: string
  dashed: boolean
}

/** The learning rate over the steps: per network when the run logs it
 * (Prodigy-SF: the effective rate solid, its d x lr dashed), else the logged rate. */
export function LrChart({ train }: { train: MetricPoint[] }) {
  const t = useT()
  const [ref, width] = useWidth<HTMLDivElement>()
  const height = chartHeight(width)
  const [hoverStep, setHoverStep] = useState<number | null>(null)

  const series = useMemo<Series[]>(() => {
    const names: string[] = []
    let effective = false
    for (const p of train)
      for (const k of Object.keys(p)) {
        if (k.startsWith(EFFECTIVE_PREFIX)) effective = true
        const name = k.startsWith(STEP_PREFIX) ? k.slice(STEP_PREFIX.length) : null
        if (name && !names.includes(name)) names.push(name)
      }
    if (names.length === 0) return [{ key: 'lr', label: t.lrSingle, color: COLORS[0], dashed: false }]
    return names.flatMap((name, i) => {
      const color = COLORS[i % COLORS.length]
      const step = { key: STEP_PREFIX + name, label: name, color, dashed: effective }
      return effective ? [{ key: EFFECTIVE_PREFIX + name, label: name, color, dashed: false }, step] : [step]
    })
  }, [train, t])
  const keys = useMemo(() => series.map((s) => s.key), [series])
  const dashed = series.some((s) => s.dashed)
  const points = useMemo(() => train.filter((p) => keys.some((k) => Number.isFinite(p[k]))), [train, keys])
  // Legend toggles: networks hidden by name; d x lr is off by default (it sets
  // the scale far above the effective rates otherwise).
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set())
  const [showStep, setShowStep] = useState(false)
  const visible = useMemo(
    () => series.filter((s) => !hidden.has(s.label) && (!s.dashed || showStep)),
    [series, hidden, showStep],
  )
  const toggle = (label: string) =>
    setHidden((h) => {
      const next = new Set(h)
      if (!next.delete(label)) next.add(label)
      return next
    })

  const geometry = useMemo(() => {
    if (points.length === 0 || width === 0) return null
    const x0 = points[0].step
    const x1 = Math.max(points.at(-1)!.step, x0 + 1)
    const scaleKeys = visible.length > 0 ? visible.map((s) => s.key) : keys
    const values = points.flatMap((p) => scaleKeys.map((k) => p[k]).filter(Number.isFinite))
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
  }, [points, keys, visible, width, height])

  function onPointerMove(e: PointerEvent<SVGRectElement>) {
    if (!geometry) return
    const rect = e.currentTarget.getBoundingClientRect()
    const step = geometry.x0 + ((e.clientX - rect.left) / rect.width) * (geometry.x1 - geometry.x0)
    let best: MetricPoint | null = null
    for (const p of points) if (!best || Math.abs(p.step - step) < Math.abs(best.step - step)) best = p
    setHoverStep(best?.step ?? null)
  }

  const hover = hoverStep === null ? null : points.find((p) => p.step === hoverStep)
  const solid = series.filter((s) => !s.dashed)
  const value = (p: MetricPoint, key: string) => (Number.isFinite(p[key]) ? fmt(p[key]) : '–')
  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{t.lrTitle}</span>
        <span className="spacer" />
        <span className="legend">
          {solid.map((s) =>
            solid.length > 1 ? (
              <button
                key={s.key}
                type="button"
                className={`legend-item legend-toggle${hidden.has(s.label) ? ' off' : ''}`}
                aria-pressed={!hidden.has(s.label)}
                title={t.legendToggle}
                onClick={() => toggle(s.label)}
              >
                <span className="swatch line" style={{ background: s.color }} aria-hidden /> {s.label}
              </button>
            ) : (
              <span key={s.key} className="legend-item">
                <span className="swatch line" style={{ background: s.color }} aria-hidden /> {s.label}
              </span>
            ),
          )}
          {dashed && (
            <button
              type="button"
              className={`legend-item legend-toggle${showStep ? '' : ' off'}`}
              aria-pressed={showStep}
              title={t.lrDashedTitle}
              onClick={() => setShowStep((v) => !v)}
            >
              <span className="swatch line dashed-swatch" aria-hidden /> {t.lrDashed}
            </button>
          )}
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
            {visible.map((s) => (
              <path
                key={s.key}
                className={s.dashed ? 'series-lr dashed' : 'series-lr'}
                style={{ stroke: s.color }}
                d={geometry.path(s.key)}
              />
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
            {solid
              .filter((s) => !hidden.has(s.label))
              .map((s) => {
                const step = series.find((d) => d.dashed && d.label === s.label)
                return (
                  <span key={s.key}>
                    <span className="swatch line" style={{ background: s.color }} aria-hidden /> {s.label}{' '}
                    {value(hover, s.key)}
                    {step && <span className="muted"> ({t.lrDashedValue(value(hover, step.key))})</span>}
                  </span>
                )
              })}
          </div>
        )}
      </div>
    </figure>
  )
}
