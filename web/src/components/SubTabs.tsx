export interface SubTab<T extends string> {
  id: T
  label: string
  disabled?: boolean
  title?: string
}

interface Props<T extends string> {
  tabs: SubTab<T>[]
  value: T
  onChange: (id: T) => void
  label: string
}

/** Sections within a page, e.g. a run's Progress / Dataset / Parameters. */
export function SubTabs<T extends string>({ tabs, value, onChange, label }: Props<T>) {
  return (
    <nav className="subtabs" role="tablist" aria-label={label}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={tab.id === value}
          className={tab.id === value ? 'active' : ''}
          disabled={tab.disabled}
          title={tab.title}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  )
}
