import { useState, useEffect } from 'react'
import { getSensors, getEggCalendar, getWeather, getAutomationWindows } from '../api'

/* ── WMO weather code labels (subset from Open-Meteo) ────────────── */
const WMO = {
  0: 'Clear sky',
  1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Icy fog',
  51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
  61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow', 77: 'Snow grains',
  80: 'Rain showers', 81: 'Rain showers', 82: 'Heavy showers',
  85: 'Snow showers', 86: 'Heavy snow showers',
  95: 'Thunderstorm', 96: 'Thunderstorm + hail', 99: 'Thunderstorm + hail',
}

/* ── SVG icons ────────────────────────────────────────────────────── */
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

function IconArrow({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M12 5l7 7-7 7" />
    </svg>
  )
}

/* ── Status helpers ───────────────────────────────────────────────── */
function computeCoopStatus(sensors) {
  if (!sensors) return { level: 'normal', message: 'No sensor data received yet.' }

  const checks = [
    { key: 'temperature_status', label: 'temperature' },
    { key: 'humidity_status',    label: 'humidity' },
    { key: 'h2s_level',          label: 'H\u2082S level' },
    { key: 'mold_risk_level',    label: 'mold risk' },
    { key: 'feeder_status',      label: 'feeder level' },
    { key: 'waterer_status',     label: 'waterer level' },
  ]

  let firstWarning = null
  let firstCritical = null

  for (const { key, label } of checks) {
    const v = String(sensors[key] ?? '').toLowerCase()
    if (!v) continue
    if ((v === 'critical' || v === 'empty') && !firstCritical) firstCritical = label
    if ((v === 'warning' || v === 'low') && !firstWarning) firstWarning = label
  }

  if (firstCritical) return { level: 'critical', message: `Critical: ${firstCritical} is out of range.` }
  if (firstWarning)  return { level: 'warning',  message: `Heads up \u2014 ${firstWarning} needs attention.` }
  return { level: 'normal', message: 'Your coop is doing well.' }
}

const STATUS = {
  normal:   { dot: 'bg-emerald-500', border: 'border-l-emerald-400 dark:border-l-emerald-500', label: 'text-emerald-600 dark:text-emerald-400', badge: 'All good' },
  warning:  { dot: 'bg-amber-500',   border: 'border-l-amber-400 dark:border-l-amber-500',     label: 'text-amber-600 dark:text-amber-400',   badge: 'Warning' },
  critical: { dot: 'bg-red-500',     border: 'border-l-red-400 dark:border-l-red-500',         label: 'text-red-600 dark:text-red-400',       badge: 'Critical' },
}

function resourceBarColor(status) {
  const s = String(status ?? '').toLowerCase()
  if (s === 'critical' || s === 'empty') return 'bg-red-400'
  if (s === 'warning'  || s === 'low')   return 'bg-amber-400'
  return 'bg-emerald-400'
}

function formatSunrise(isoStr) {
  if (!isoStr) return '—'
  try {
    return new Date(isoStr).toLocaleTimeString('en-BE', { hour: '2-digit', minute: '2-digit', hour12: false })
  } catch {
    return isoStr
  }
}

function isoToday() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

/* ── Primitives ───────────────────────────────────────────────────── */
function Skeleton({ className }) {
  return <div className={`animate-pulse rounded-xl bg-stone-200 dark:bg-stone-700 ${className}`} />
}

function Card({ children, className = '', onClick, borderAccent }) {
  const accent = borderAccent ? `border-l-[3px] ${borderAccent}` : ''
  const cursor = onClick ? 'cursor-pointer hover:-translate-y-0.5 hover:shadow-md active:scale-[0.98] transition-all duration-200' : ''
  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick() } } : undefined}
      className={`bg-white dark:bg-stone-800 rounded-xl border border-stone-200 dark:border-stone-700/50 p-5 flex flex-col ${accent} ${cursor} ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  )
}

/* ── ResourceBar ──────────────────────────────────────────────────── */
function ResourceBar({ label, pct, status }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-stone-600 dark:text-stone-300">{label}</span>
        <span className="text-xs tabular-nums text-stone-500 dark:text-stone-400">
          {pct != null ? `${pct}%` : '—'}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-stone-100 dark:bg-stone-700 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${resourceBarColor(status)}`}
          style={{ width: `${pct ?? 0}%` }}
        />
      </div>
    </div>
  )
}

/* ── Main component ───────────────────────────────────────────────── */
export default function Welcome({ onNavigate }) {
  // undefined = loading, null = failed/no data, value = loaded
  const [sensors, setSensors]               = useState(undefined)
  const [eggData, setEggData]               = useState(undefined)
  const [weatherData, setWeatherData]       = useState(undefined)
  const [automationActive, setAutomationActive] = useState(undefined)
  // Captured once at mount to avoid midnight-boundary mismatch between effect and render
  const [todayISO] = useState(() => isoToday())

  useEffect(() => {
    const now    = new Date()
    const year   = now.getFullYear()
    const month  = now.getMonth() + 1
    const future = new Date(now)
    future.setDate(future.getDate() + 30)
    const end = `${future.getFullYear()}-${String(future.getMonth() + 1).padStart(2, '0')}-${String(future.getDate()).padStart(2, '0')}`

    Promise.allSettled([
      getSensors(),
      getEggCalendar(year, month),
      getWeather(),
      getAutomationWindows(todayISO, end),
    ]).then(([s, e, w, a]) => {
      setSensors(s.status === 'fulfilled' ? s.value : null)
      setEggData(e.status === 'fulfilled' ? e.value : null)
      setWeatherData(w.status === 'fulfilled' ? w.value : null)
      setAutomationActive(
        a.status === 'fulfilled'
          ? (Array.isArray(a.value?.windows) && a.value.windows.length > 0)
          : null
      )
    })
  }, [])

  const loading = sensors === undefined || eggData === undefined || weatherData === undefined || automationActive === undefined

  /* Derived values */
  const coopStatus = computeCoopStatus(sensors)
  const sc = STATUS[coopStatus.level]

  const todayEntry = eggData?.days?.find(d => d.date === todayISO)
  const eggsToday  = todayEntry?.eggs_laid ?? null
  const alertCount = sensors?.alert_count ?? 0

  const weatherTemp = weatherData?.current?.temperature_2m
  const weatherCode = weatherData?.current?.weather_code
  const weatherDesc = weatherCode != null ? (WMO[weatherCode] ?? `Code ${weatherCode}`) : null
  const weatherSunrise = weatherData?.daily?.sunrise?.[0]

  /* ── Loading skeleton ─────────────────────────────────────────── */
  if (loading) {
    return (
      <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
        <div className="max-w-4xl mx-auto grid grid-cols-3 gap-4">
          <Skeleton className="col-span-2 h-44" />
          <Skeleton className="col-span-1 h-44" />
          <Skeleton className="col-span-1 h-36" />
          <Skeleton className="col-span-2 h-36" />
          <Skeleton className="col-span-2 h-36" />
          <Skeleton className="col-span-1 h-36" />
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto grid grid-cols-3 gap-4">

        {/* ── 1. COOP STATUS ── col-span-2 ─────────────────────── */}
        <Card
          className="col-span-2"
          borderAccent={sc.border}
          onClick={() => onNavigate('sensors')}
        >
          <div className="flex items-center gap-2 mb-3">
            <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${sc.dot}`} />
            <span className={`text-xs font-semibold uppercase tracking-wide ${sc.label}`}>
              {sc.badge}
            </span>
          </div>
          <p className="text-stone-700 dark:text-stone-200 text-sm font-medium leading-snug mb-3">
            {coopStatus.message}
          </p>
          {sensors && (sensors.temperature_c != null || sensors.humidity_pct != null) && (
            <p className="text-stone-500 dark:text-stone-400 text-xs tabular-nums">
              {[
                sensors.temperature_c != null ? `${sensors.temperature_c}°C` : null,
                sensors.humidity_pct != null ? `${sensors.humidity_pct}% humidity` : null,
              ].filter(Boolean).join(' · ')}
            </p>
          )}
          <p className="text-stone-400 dark:text-stone-500 text-xs mt-auto pt-3">
            {alertCount > 0
              ? `${alertCount} active alert${alertCount !== 1 ? 's' : ''}`
              : 'No active alerts'}
          </p>
        </Card>

        {/* ── 2. EGGS TODAY ── col-span-1 ──────────────────────── */}
        <Card
          className="col-span-1 items-center justify-center text-center"
          onClick={() => onNavigate('eggs')}
        >
          <span className="text-4xl font-bold tabular-nums font-mono text-stone-800 dark:text-stone-100">
            {eggsToday !== null ? eggsToday : '\u2014'}
          </span>
          <span className="text-xs text-stone-400 dark:text-stone-500 mt-1.5">eggs today</span>
        </Card>

        {/* ── 3. RESOURCES ── col-span-1 ───────────────────────── */}
        <Card className="col-span-1 gap-4">
          <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-1">
            Resources
          </p>
          {sensors ? (
            <div className="flex flex-col gap-3">
              <ResourceBar label="Feeder"  pct={sensors.feeder_pct}  status={sensors.feeder_status} />
              <ResourceBar label="Waterer" pct={sensors.waterer_pct} status={sensors.waterer_status} />
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="space-y-1.5">
                <div className="flex justify-between">
                  <Skeleton className="h-3 w-12 rounded" />
                  <Skeleton className="h-3 w-8 rounded" />
                </div>
                <Skeleton className="h-1.5 w-full rounded-full" />
              </div>
              <div className="space-y-1.5">
                <div className="flex justify-between">
                  <Skeleton className="h-3 w-14 rounded" />
                  <Skeleton className="h-3 w-8 rounded" />
                </div>
                <Skeleton className="h-1.5 w-full rounded-full" />
              </div>
            </div>
          )}
        </Card>

        {/* ── 4. DOOR + AUTOMATION ── col-span-2 ───────────────── */}
        <Card className="col-span-2" onClick={() => onNavigate('automation')}>
          <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">
            Door &amp; Automation
          </p>
          <div className="flex items-center gap-2 flex-wrap mb-auto">
            <span className={`text-xs font-medium px-2.5 py-1 rounded-full border ${
              sensors?.door_open
                ? 'border-sky-400 text-sky-600 dark:text-sky-400 bg-sky-50 dark:bg-sky-900/20'
                : 'border-stone-300 dark:border-stone-600 text-stone-500 dark:text-stone-400 bg-stone-100 dark:bg-stone-700/50'
            }`}>
              Door: {sensors?.door_open ? 'Open' : 'Closed'}
            </span>
            <span className={`text-xs font-medium px-2.5 py-1 rounded-full border ${
              sensors?.ventilation_on
                ? 'border-emerald-400 text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/20'
                : 'border-stone-300 dark:border-stone-600 text-stone-500 dark:text-stone-400 bg-stone-100 dark:bg-stone-700/50'
            }`}>
              Ventilation: {sensors?.ventilation_on ? 'Running' : 'Off'}
            </span>
          </div>
          <p className="text-xs text-stone-400 dark:text-stone-500 mt-3">
            {automationActive === null
              ? 'Automation status unavailable'
              : automationActive
                ? 'Automation windows active'
                : 'No automation scheduled'}
          </p>
        </Card>

        {/* ── 5. WEATHER ── col-span-2 ─────────────────────────── */}
        <Card className="col-span-2">
          {weatherData && weatherTemp != null ? (
            <>
              <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">
                Weather
              </p>
              <div className="flex items-baseline gap-3">
                <span className="text-3xl font-bold tabular-nums text-stone-800 dark:text-stone-100">
                  {weatherTemp}°C
                </span>
                {weatherDesc && (
                  <span className="text-sm text-stone-500 dark:text-stone-400">
                    {weatherDesc}
                  </span>
                )}
              </div>
              {weatherSunrise && (
                <p className="text-xs text-stone-400 dark:text-stone-500 mt-2">
                  Sunrise {formatSunrise(weatherSunrise)}
                </p>
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <span className="text-xs text-stone-400 dark:text-stone-500">Weather unavailable</span>
            </div>
          )}
        </Card>

        {/* ── 6. CHAT CTA ── col-span-1 ────────────────────────── */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => onNavigate('chat')}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onNavigate('chat') } }}
          className="col-span-1 bg-amber-50 dark:bg-amber-900/10 border border-amber-100 dark:border-amber-800/30 rounded-xl p-5 flex flex-col items-center justify-center gap-3 cursor-pointer hover:-translate-y-0.5 hover:shadow-md active:scale-[0.98] transition-all duration-200"
        >
          <div className="w-12 h-12 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
            <IconChicken className="w-6 h-6 text-amber-700 dark:text-amber-400" />
          </div>
          <p className="text-sm font-medium text-stone-700 dark:text-stone-200 text-center">
            Ask ChatKippieTee
          </p>
          <IconArrow className="w-4 h-4 text-stone-400" />
        </div>

      </div>
    </div>
  )
}
