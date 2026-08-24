import React, { useEffect, useState } from 'react'
import { ActivityIcon, ChevronDownIcon, FileDownIcon, SearchIcon, WavesIcon } from 'lucide-react'

interface TopBarProps {
  window: string
  onWindowChange: (value: string) => void
  onExport: () => void
  exporting: boolean
  wsStatus: 'connecting' | 'connected' | 'disconnected'
  searchQuery: string
  onSearchChange: (value: string) => void
}

const WINDOWS = ['Last 24 h', 'Last 48 h', 'Last 7 d', 'Season to date']

export function TopBar({ window: timeWindow, onWindowChange, onExport, exporting, wsStatus, searchQuery, onSearchChange }: TopBarProps) {
  const [clock, setClock] = useState(() => new Date())
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-line bg-abyss-900 px-4">
      <div className="flex items-center gap-2.5">
        <WavesIcon className="h-5 w-5 text-tide" aria-hidden="true" />
        <span className="text-[15px] font-semibold tracking-tight">BlueByte AI</span>
        <span className="rounded border border-line bg-abyss-800 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-wider text-foam-muted">
          Research console
        </span>
      </div>

      <div className="relative ml-2 hidden max-w-md flex-1 items-center lg:flex">
        <SearchIcon
          className="pointer-events-none absolute left-3 h-3.5 w-3.5 text-foam-dim"
          aria-hidden="true"
        />
        <input
          type="search"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search stations, IDs, or fishing zones..."
          aria-label="Cross-domain query"
          className="h-8 w-full rounded-md border border-line bg-abyss-800 pl-8 pr-3 text-xs text-foam placeholder:text-foam-dim focus:border-tide focus:outline-none focus:ring-1 focus:ring-tide/40"
        />
      </div>

      <div className="ml-auto flex items-center gap-3">
        <div className="hidden items-center gap-2 md:flex">
          <span className="relative flex h-2 w-2">
            {wsStatus === 'connected' && (
              <span className="bb-pulse absolute inline-flex h-2 w-2 rounded-full bg-bio" />
            )}
            <span className={`relative inline-flex h-2 w-2 rounded-full ${wsStatus === 'connected' ? 'bg-bio' : wsStatus === 'connecting' ? 'bg-catch' : 'bg-risk'}`} />
          </span>
          <span className="font-mono text-2xs uppercase tracking-wider text-foam-muted">
            {wsStatus === 'connected' ? 'Feeds live' : wsStatus === 'connecting' ? 'Connecting...' : 'Offline'}
          </span>
        </div>

        <div className="relative">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-haspopup="listbox"
            className="flex h-8 items-center gap-1.5 rounded-md border border-line bg-abyss-800 px-2.5 text-xs text-foam transition-colors duration-150 hover:border-abyss-600 hover:bg-abyss-700"
          >
            <ActivityIcon className="h-3.5 w-3.5 text-foam-muted" aria-hidden="true" />
            {timeWindow}
            <ChevronDownIcon className="h-3.5 w-3.5 text-foam-dim" aria-hidden="true" />
          </button>
          {open && (
            <ul
              role="listbox"
              className="absolute right-0 z-[1200] mt-1 w-40 overflow-hidden rounded-md border border-line bg-abyss-800 py-1 shadow-2xl"
            >
              {WINDOWS.map((w) => (
                <li key={w}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={w === timeWindow}
                    onClick={() => {
                      onWindowChange(w)
                      setOpen(false)
                    }}
                    className={`flex w-full px-3 py-1.5 text-left text-xs transition-colors duration-150 hover:bg-abyss-700 ${
                      w === timeWindow ? 'text-tide' : 'text-foam-muted'
                    }`}
                  >
                    {w}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <span className="hidden font-mono text-xs tabular-nums text-foam-muted xl:inline">
          {clock.toLocaleTimeString('en-GB')} IST
        </span>

        <button
          type="button"
          onClick={onExport}
          disabled={exporting}
          className="flex h-8 items-center gap-1.5 rounded-md bg-tide px-3 text-xs font-semibold text-abyss-950 transition-colors duration-150 hover:bg-[#5ad8ec] disabled:cursor-wait disabled:opacity-70"
        >
          <FileDownIcon className="h-3.5 w-3.5" aria-hidden="true" />
          {exporting ? 'Compiling…' : 'Export report'}
        </button>
      </div>
    </header>
  )
}
