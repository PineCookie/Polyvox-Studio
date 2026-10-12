import type { ReactNode } from 'react'

/** A folded or open section; the owner keeps the state (often remembered with useFolded). */
export interface Fold {
  open: boolean
  onToggle: () => void
}

/** The ▸ / ▾ arrow; <summary> elements draw the same one in CSS. */
export function Chevron({ open }: { open: boolean }) {
  return (
    <span className="chevron" aria-hidden>
      {open ? '▾' : '▸'}
    </span>
  )
}

/** A title that folds its section. */
export function FoldTitle({ fold, className = '', children }: { fold: Fold; className?: string; children: ReactNode }) {
  return (
    <button type="button" className={`fold-head ${className}`} aria-expanded={fold.open} onClick={fold.onToggle}>
      <Chevron open={fold.open} />
      {children}
    </button>
  )
}

/** A panel whose heading folds it; `extra` stays in the header either way. */
export function FoldPanel({
  title,
  extra,
  fold,
  children,
}: {
  title: string
  extra?: ReactNode
  fold: Fold
  children: ReactNode
}) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>
          <FoldTitle fold={fold}>{title}</FoldTitle>
        </h2>
        {extra}
      </header>
      {fold.open && children}
    </section>
  )
}
