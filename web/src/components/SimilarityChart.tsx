import { useMemo, useState } from 'react'
import type { PointerEvent } from 'react'
import { chartHeight, MARGIN as M, niceTicks, useWidth } from '../chartUtils'
import { formatStep } from '../format'
import { useT } from '../i18n'

export interface SimilarityPoint {
  step: number
  mean: number
  scores: number[] // one per sample of that validation
}

interface Props {
  points: SimilarityPoint[] // by step
  target: number | null // real clips of the training set
}

const fmt = (v: number) => v.toFixed(2)

/** Speaker similarity per validation: the mean as a line, each sample as a dot,
 * and the level real clips reach as a dashed line. */
export function SimilarityChart({ points, target }: Props) {
  const t = useT()
  const [ref, width] = useWidth<HTMLDivElement>()
  const height = chartHeight(width)
  const [hover, setHover] = useState<SimilarityPoint | null>(null)

  const geometry = useMemo(() => {
    if (points.length === 0 || width === 0) return null
    const x0 = points[0].step
    const x1 = Math.max(points.at(-1)!.step, x0 + 1)
    const ys = [...points.flatMap((p) => p.scores), ...(target === null ? [] : [target])]
    const pad = 0.05
    const y0 = Math.max(-1, Math.min(...ys) - pad)
    const y1 = Math.min(1, Math.max(...ys) + pad)
    const plotW = width - M.left - M.right
    const plotH = height - M.top - M.bottom
    const x = (step: number) => M.left + ((step - x0) / (x1 - x0)) * plotW
    const y = (v: number) => M.top + (1 - (v - y0) / (y1 - y0 || 1)) * plotH
    return {
      x0,
      x1,
      x,
      y,
      plotW,
      line: points.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.mean).toFixed(1)}`).join(''),
      yTicks: niceTicks(y0, y1, 4),
      xTicks: niceTicks(x0, x1, Math.max(2, Math.floor(plotW / 90))),
    }
  }, [points, target, width, height])

  function onPointerMove(e: PointerEvent<SVGRectElement>) {
    if (!geometry) return
    const rect = e.currentTarget.getBoundingClientRect()
    const step = geometry.x0 + ((e.clientX - rect.left) / rect.width) * (geometry.x1 - geometry.x0)
    let best: SimilarityPoint | null = null
    for (const p of points) if (!best || Math.abs(p.step - step) < Math.abs(best.step - step)) best = p
    setHover(best)
  }

  const latest = points.at(-1)
  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{t.similarityTitle}</span>
        {latest && <span className="muted">{t.similarityLatest(fmt(latest.mean))}</span>}
        <span className="spacer" />
        <span className="legend">
          <span className="legend-item">
            <span className="swatch line train" aria-hidden /> {t.similaritySamples}
          </span>
          {target !== null && (
            <span className="legend-item">
              <span className="swatch line target" aria-hidden /> {t.similarityTarget(fmt(target))}
            </span>
          )}
        </span>
      </figcaption>

      <div className="chart-body" ref={ref}>
        {geometry && (
          <svg width={width} height={height} role="img" aria-label={t.overSteps(t.similarityTitle)}>
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
            {target !== null && (
              <line
                className="series-target"
                x1={M.left}
                x2={width - M.right}
                y1={geometry.y(target)}
                y2={geometry.y(target)}
              />
            )}
            {points.flatMap((p) =>
              p.scores.map((s, i) => (
                <circle
                  key={`${p.step}-${i}`}
                  className="series-sample"
                  cx={geometry.x(p.step)}
                  cy={geometry.y(s)}
                  r={3}
                />
              )),
            )}
            <path className="series-train" d={geometry.line} />
            {points.map((p) => (
              <circle key={p.step} className="hover-dot" cx={geometry.x(p.step)} cy={geometry.y(p.mean)} r={3.5} />
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
              onPointerLeave={() => setHover(null)}
            />
          </svg>
        )}
        {geometry && hover && (
          <div className="tooltip" style={{ left: Math.min(geometry.x(hover.step) + 12, width - 170), top: M.top }}>
            <strong>{t.stepLabel(formatStep(hover.step))}</strong>
            <span>{t.similarityMean(fmt(hover.mean))}</span>
            <span className="muted">{hover.scores.map(fmt).join(' · ')}</span>
          </div>
        )}
      </div>
    </figure>
  )
}
