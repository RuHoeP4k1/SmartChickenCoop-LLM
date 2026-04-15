import { useState, useEffect } from 'react'
import { useTheme } from '../context/ThemeContext'
import { getSensors, getEggCalendar, getEvents } from '../api'
import logo from '../assets/chicken_logo_4x.png'
import CommandPalette from './CommandPalette'

/* ── SVG icon components (lightweight, no deps) ──────────────────── */

function IconHome({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z" />
      <path d="M9 21V12h6v9" />
    </svg>
  )
}

function IconChat({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2v10z" />
    </svg>
  )
}

function IconSensors({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  )
}

function IconChart({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path d="M7 16l4-6 4 4 5-8" />
    </svg>
  )
}

function IconBell({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 01-3.46 0" />
    </svg>
  )
}

function IconEgg({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="13" rx="7" ry="9" />
    </svg>
  )
}

function IconGear({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1.08-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1.08 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001.08 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9c.26.604.852.997 1.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1.08z" />
    </svg>
  )
}

function IconCloud({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 10a6 6 0 10-11.472 2.47A4 4 0 107 18h11a4 4 0 00.472-7.97A6.014 6.014 0 0018 10z" />
    </svg>
  )
}

function IconSun({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  )
}

function IconMoon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z" />
    </svg>
  )
}

function IconChevron({ className, direction = 'left' }) {
  return (
    <svg className={`${className} transition-transform duration-200 ${direction === 'right' ? 'rotate-180' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  )
}

function IconMenu({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12h18M3 6h18M3 18h18" />
    </svg>
  )
}

function IconX({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  )
}

function IconChicken({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="7" r="3" />
      <path d="M13 7c1-2 3-2 4-1" />
      <path d="M10 10c-4 0-6 3-6 6v2h14v-2c0-2-1-4-3-5" />
      <path d="M8 18v2M14 18v2" />
    </svg>
  )
}

function IconPackage({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 002 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
      <path d="M3.27 6.96L12 12.01l8.73-5.05M12 22.08V12" />
    </svg>
  )
}

function IconFlock({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  )
}

function IconReview({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M9 15l2 2 4-4" />
    </svg>
  )
}

/* ── Icon map ─────────────────────────────────────────────────────── */

const ICON_MAP = {
  welcome:    IconHome,
  chat:       IconChat,
  sensors:    IconSensors,
  charts:     IconChart,
  alerts:     IconBell,
  eggs:       IconEgg,
  automation: IconGear,
  weather:    IconCloud,
  chickens:   IconChicken,
  packages:   IconPackage,
  flock:      IconFlock,
  review:     IconReview,
}

/* ── Coop status hook ─────────────────────────────────────────────── */

const STATUS_FIELDS = [
  'temperature_status',
  'humidity_status',
  'h2s_level',
  'mold_risk_level',
  'feeder_status',
  'waterer_status',
]

function useCoopStatus() {
  const [status, setStatus] = useState('loading')

  useEffect(() => {
    let cancelled = false

    function compute(sensors) {
      if (!sensors) return 'normal'
      let hasCritical = false
      let hasWarning = false
      for (const key of STATUS_FIELDS) {
        const v = String(sensors[key] ?? '').toLowerCase()
        if (!v) continue
        if (v === 'critical' || v === 'empty') hasCritical = true
        if (v === 'warning' || v === 'low') hasWarning = true
      }
      if (hasCritical) return 'critical'
      if (hasWarning) return 'warning'
      return 'normal'
    }

    function poll() {
      getSensors()
        .then(data => { if (!cancelled) setStatus(compute(data?.reading)) })
        .catch(() => { /* keep last status on error */ })
    }

    poll()
    const id = setInterval(poll, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  return status
}

/* ── Footer live data hook ────────────────────────────────────────── */

function useFooterData() {
  const [temp, setTemp] = useState(null)
  const [eggsToday, setEggsToday] = useState(null)
  const [lastAlertAge, setLastAlertAge] = useState(null)

  useEffect(() => {
    function fetchTemp() {
      getSensors()
        .then(data => setTemp(data?.reading?.temperature_c ?? null))
        .catch(() => {})
    }
    fetchTemp()
    const id = setInterval(fetchTemp, 30_000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    let cancelled = false
    const now = new Date()
    const todayISO = now.toISOString().slice(0, 10)
    getEggCalendar(now.getFullYear(), now.getMonth() + 1)
      .then(data => {
        if (cancelled) return
        const entry = data?.days?.find(d => d.date === todayISO)
        setEggsToday(entry?.eggs_laid ?? null)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    getEvents(5, 'sensor_alert')
      .then(data => {
        if (cancelled) return
        const events = data?.events ?? []
        if (events.length === 0) { setLastAlertAge('none'); return }
        const latest = new Date(events[0].timestamp)
        const todayStr = new Date().toDateString()
        if (latest.toDateString() !== todayStr) { setLastAlertAge('none'); return }
        const secs = Math.floor((Date.now() - latest.getTime()) / 1000)
        if (secs < 60) setLastAlertAge('just now')
        else if (secs < 3600) setLastAlertAge(`${Math.floor(secs / 60)}m ago`)
        else setLastAlertAge(`${Math.floor(secs / 3600)}h ago`)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  return { temp, eggsToday, lastAlertAge }
}

/* ── Layout ───────────────────────────────────────────────────────── */

export default function Layout({ tabs, activeTab, onTabChange, children, currentUser, onLogout }) {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const { darkMode, toggleDarkMode } = useTheme()
  const coopStatus = useCoopStatus()
  const { temp, eggsToday, lastAlertAge } = useFooterData()

  useEffect(() => {
    function handleKeyDown(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setPaletteOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const footerParts = []
  if (temp != null) footerParts.push(`${temp.toFixed(1)}°C`)
  if (eggsToday != null) footerParts.push(`${eggsToday} egg${eggsToday !== 1 ? 's' : ''}`)
  if (lastAlertAge === 'none') footerParts.push('No alerts today')
  else if (lastAlertAge != null) footerParts.push(`Last alert ${lastAlertAge}`)
  const footerStatus = footerParts.join(' · ')

  const statusBarClass = {
    loading:  'bg-stone-200 dark:bg-stone-700',
    normal:   'bg-emerald-400',
    warning:  'bg-amber-400',
    critical: 'bg-red-400 animate-pulse',
  }[coopStatus] ?? 'bg-stone-200 dark:bg-stone-700'

  function handleNav(tabId) {
    onTabChange(tabId)
    setMobileOpen(false)
  }

  /* Shared sidebar content */
  const sidebarContent = (
    <>
      {/* Logo + branding */}
      <div className="shrink-0 px-4 py-5 flex items-center gap-3 border-b border-stone-200 dark:border-stone-700/50">
        <img src={logo} alt="Logo" className="w-9 h-9 rounded-xl object-contain shrink-0" />
        {!collapsed && (
          <div className="overflow-hidden">
            <h1 className="text-sm font-bold tracking-tight text-stone-800 dark:text-stone-100 leading-tight truncate">
              ChickenCoopComfort
            </h1>
          </div>
        )}
      </div>

      {/* Nav items */}
      <nav className="flex-1 py-3 px-2 overflow-y-auto">
        {tabs.map((tab, idx) => {
          if (tab.separator) {
            if (tab.label && !collapsed) {
              return (
                <div key={`sep-${idx}`} className="pt-4 pb-1 px-3">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500 select-none">
                    {tab.label}
                  </span>
                </div>
              )
            }
            return (
              <div
                key={`sep-${idx}`}
                className="border-t border-stone-100 dark:border-stone-700/50 mx-1 my-2"
              />
            )
          }
          const Icon = ICON_MAP[tab.id] || IconHome
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => handleNav(tab.id)}
              title={collapsed ? tab.label : undefined}
              className={`w-full flex items-center gap-3 rounded-lg text-sm font-medium transition-all duration-150 active:scale-[0.97] mb-0.5 ${
                collapsed ? 'justify-center px-2 py-2.5' : 'px-3 py-2.5'
              } ${
                isActive
                  ? 'bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-l-[3px] border-amber-500'
                  : 'text-stone-500 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700/40 hover:text-stone-800 dark:hover:text-stone-200 border-l-[3px] border-transparent'
              }`}
            >
              <Icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-amber-600 dark:text-amber-400' : ''}`} />
              {!collapsed && <span className="truncate">{tab.label}</span>}
            </button>
          )
        })}
      </nav>

      {/* Bottom controls */}
      <div className="shrink-0 px-2 py-3 border-t border-stone-200 dark:border-stone-700/50 space-y-1">
        {/* Dark mode toggle */}
        <button
          onClick={toggleDarkMode}
          title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          className={`w-full flex items-center gap-3 rounded-lg text-sm font-medium text-stone-500 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700/40 hover:text-stone-800 dark:hover:text-stone-200 transition-all duration-150 ${
            collapsed ? 'justify-center px-2 py-2.5' : 'px-3 py-2.5'
          }`}
        >
          {darkMode
            ? <IconSun className="w-5 h-5 shrink-0" />
            : <IconMoon className="w-5 h-5 shrink-0" />
          }
          {!collapsed && <span>{darkMode ? 'Light mode' : 'Dark mode'}</span>}
        </button>

        {/* Collapse toggle (desktop only) */}
        <button
          onClick={() => setCollapsed(prev => !prev)}
          className={`hidden md:flex w-full items-center gap-3 rounded-lg text-sm font-medium text-stone-400 dark:text-stone-500 hover:bg-stone-100 dark:hover:bg-stone-700/40 hover:text-stone-600 dark:hover:text-stone-300 transition-all duration-150 ${
            collapsed ? 'justify-center px-2 py-2.5' : 'px-3 py-2.5'
          }`}
        >
          <IconChevron className="w-5 h-5 shrink-0" direction={collapsed ? 'right' : 'left'} />
          {!collapsed && <span>Collapse</span>}
        </button>

        {/* Logged-in user + logout */}
        {currentUser && (
          <button
            onClick={onLogout}
            title="Sign out"
            className={`w-full flex items-center gap-3 rounded-lg text-sm font-medium text-stone-500 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-700/40 hover:text-red-500 dark:hover:text-red-400 transition-all duration-150 ${
              collapsed ? 'justify-center px-2 py-2.5' : 'px-3 py-2.5'
            }`}
          >
            <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h6a2 2 0 012 2v1" />
            </svg>
            {!collapsed && <span>{currentUser.username} · Sign out</span>}
          </button>
        )}
      </div>
    </>
  )

  return (
    <div className="h-screen flex bg-stone-50 dark:bg-stone-900 text-stone-800 dark:text-stone-100">
      <CommandPalette
        isOpen={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={onTabChange}
      />
      {/* ── Desktop sidebar ─────────────────────────────────────── */}
      <aside
        className={`hidden md:flex flex-col shrink-0 bg-white dark:bg-stone-800 border-r border-stone-200 dark:border-stone-700/50 transition-all duration-200 ${
          collapsed ? 'w-[68px]' : 'w-[220px]'
        }`}
      >
        {sidebarContent}
      </aside>

      {/* ── Mobile overlay ──────────────────────────────────────── */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40">
          {/* backdrop */}
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setMobileOpen(false)}
          />
          {/* drawer */}
          <aside className="relative w-[260px] h-full flex flex-col bg-white dark:bg-stone-800 shadow-xl">
            {sidebarContent}
          </aside>
        </div>
      )}

      {/* ── Main area ───────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile header with hamburger */}
        <header className="md:hidden shrink-0 flex items-center gap-3 px-4 py-3 bg-white dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700/50">
          <button
            onClick={() => setMobileOpen(true)}
            className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-stone-700 text-stone-600 dark:text-stone-300"
          >
            <IconMenu className="w-5 h-5" />
          </button>
          <img src={logo} alt="Logo" className="w-7 h-7 rounded-lg object-contain" />
          <span className="text-sm font-bold text-stone-800 dark:text-stone-100">ChickenCoopComfort</span>
        </header>

        {/* Page content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className={`h-[3px] w-full shrink-0 transition-colors duration-500 ${statusBarClass}`} />
          <div className="flex-1 overflow-hidden">{children}</div>
        </main>

        {/* Footer */}
        <footer className="shrink-0 border-t border-stone-200 dark:border-stone-700/50 bg-white dark:bg-stone-800 px-6 py-2.5 flex items-center justify-between">
          <p className="text-[11px] text-stone-400 dark:text-stone-500">
            ChickenCoopComfort &copy; {new Date().getFullYear()}
          </p>
          {footerStatus && (
            <p className="text-[11px] text-stone-400 dark:text-stone-500 font-mono tabular-nums">
              {footerStatus}
            </p>
          )}
        </footer>
      </div>
    </div>
  )
}
