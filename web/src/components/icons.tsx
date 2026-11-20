// Small line icons (16 px grid, current colour). Decorative: the buttons that hold them carry the label.
import type { ReactNode } from 'react'

function Icon({ children, size = 16 }: { children: ReactNode; size?: number }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {children}
    </svg>
  )
}

export const PlayIcon = ({ size }: { size?: number }) => (
  <Icon size={size}>
    <path
      d="M5 3.2v9.6c0 .4.4.6.7.4l7.2-4.8a.5.5 0 0 0 0-.8L5.7 2.8c-.3-.2-.7 0-.7.4Z"
      fill="currentColor"
      stroke="none"
    />
  </Icon>
)

export const PauseIcon = ({ size }: { size?: number }) => (
  <Icon size={size}>
    <rect x="4" y="3" width="2.6" height="10" rx="0.8" fill="currentColor" stroke="none" />
    <rect x="9.4" y="3" width="2.6" height="10" rx="0.8" fill="currentColor" stroke="none" />
  </Icon>
)

export const MoreIcon = () => (
  <Icon>
    <circle cx="3.5" cy="8" r="1.1" fill="currentColor" stroke="none" />
    <circle cx="8" cy="8" r="1.1" fill="currentColor" stroke="none" />
    <circle cx="12.5" cy="8" r="1.1" fill="currentColor" stroke="none" />
  </Icon>
)

export const DownloadIcon = () => (
  <Icon>
    <path d="M8 2.5v8M4.8 7.5 8 10.7l3.2-3.2M3 13.5h10" />
  </Icon>
)

export const ChevronDownIcon = () => (
  <Icon>
    <path d="m4 6 4 4 4-4" />
  </Icon>
)

export const SearchIcon = () => (
  <Icon>
    <circle cx="7" cy="7" r="4.2" />
    <path d="m10.2 10.2 3.3 3.3" />
  </Icon>
)

export const DiceIcon = () => (
  <Icon>
    <rect x="2.5" y="2.5" width="11" height="11" rx="2.5" />
    <circle cx="5.7" cy="5.7" r="0.9" fill="currentColor" stroke="none" />
    <circle cx="10.3" cy="10.3" r="0.9" fill="currentColor" stroke="none" />
    <circle cx="8" cy="8" r="0.9" fill="currentColor" stroke="none" />
  </Icon>
)

export const TagIcon = () => (
  <Icon>
    <path d="M2.5 3.5v3.6c0 .3.1.5.3.7l5.6 5.6c.4.4 1 .4 1.4 0l3.4-3.4c.4-.4.4-1 0-1.4L7.6 3c-.2-.2-.4-.3-.7-.3H3.3a.8.8 0 0 0-.8.8Z" />
    <circle cx="5.3" cy="5.5" r="0.9" fill="currentColor" stroke="none" />
  </Icon>
)

export const SparkIcon = () => (
  <Icon>
    <path d="M8 2.5 9.3 6.7 13.5 8l-4.2 1.3L8 13.5 6.7 9.3 2.5 8l4.2-1.3Z" fill="currentColor" stroke="none" />
  </Icon>
)

export const PlusIcon = () => (
  <Icon>
    <path d="M8 3.5v9M3.5 8h9" />
  </Icon>
)

export const RefreshIcon = () => (
  <Icon>
    <path d="M13 4.5V7.5H10M3 11.5V8.5H6" />
    <path d="M12.6 7.2A4.8 4.8 0 0 0 4 5.3M3.4 8.8A4.8 4.8 0 0 0 12 10.7" />
  </Icon>
)

export const CloseIcon = () => (
  <Icon>
    <path d="m4.5 4.5 7 7M11.5 4.5l-7 7" />
  </Icon>
)

export const SunIcon = () => (
  <Icon>
    <circle cx="8" cy="8" r="2.8" />
    <path d="M8 1.8v1.4M8 12.8v1.4M1.8 8h1.4M12.8 8h1.4M3.6 3.6l1 1M11.4 11.4l1 1M3.6 12.4l1-1M11.4 4.6l1-1" />
  </Icon>
)

export const MoonIcon = () => (
  <Icon>
    <path d="M13 9.6A5.5 5.5 0 0 1 6.4 3a5.5 5.5 0 1 0 6.6 6.6Z" />
  </Icon>
)

export const AutoThemeIcon = () => (
  <Icon>
    <circle cx="8" cy="8" r="5.5" />
    <path d="M8 2.5a5.5 5.5 0 0 1 0 11Z" fill="currentColor" stroke="none" />
  </Icon>
)

export const SplitIcon = () => (
  <Icon>
    <path d="M2.5 4.5h7M2.5 11.5h7M2.5 8h11" />
    <path d="M12 3.5v2M12 10.5v2" />
  </Icon>
)
