import { useState, useEffect, useCallback } from 'react'
import { getSensors, getHistory, getRiskLatest } from '../api'

/* ── Refresh icon ─────────────────────────────────────────────────── */
function RefreshIcon({ className, spinning }) {
  return (
    <svg
      className={`${className} ${spinning ? 'animate-spin' : ''}`}
      viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
    >
      <path d="M3 12a9 9 0 019-9 9.75 9.75 0 016.74 2.74L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 01-9 9 9.75 9.75 0 01-6.74-2.74L3 16" />
      <path d="M8 16H3v5" />
    </svg>
  )
}

/* ── Status tokens ────────────────────────────────────────────────── */
// Card base is always white/stone-800 — status is expressed via left border only
const STATUS = {
  normal: {
    card:   'border-stone-200 dark:border-stone-700/50 border-l-[3px] border-l-emerald-400 dark:border-l-emerald-500',
    badge:  'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-400 dark:border-emerald-700/50',
    value:  'text-stone-800 dark:text-stone-100',
  },
  warning: {
    card:   'border-stone-200 dark:border-stone-700/50 border-l-[3px] border-l-amber-400 dark:border-l-amber-500',
    badge:  'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-400 dark:border-amber-700/50',
    value:  'text-stone-800 dark:text-stone-100',
  },
  critical: {
    card:   'border-stone-200 dark:border-stone-700/50 border-l-[3px] border-l-red-400 dark:border-l-red-500',
    badge:  'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-700/50',
    value:  'text-stone-800 dark:text-stone-100',
  },
}

const RESOURCE = {
  full:  STATUS.normal,
  low:   STATUS.warning,
  empty: STATUS.critical,
}

const BINARY = {
  true: {
    card:   'border-stone-200 dark:border-stone-700/50 border-l-[3px] border-l-sky-400',
    badge:  'bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-900/20 dark:text-sky-400 dark:border-sky-700/50',
    value:  'text-stone-800 dark:text-stone-100',
  },
  false: {
    card:   'border-stone-200 dark:border-stone-700/50 border-l-[3px] border-l-stone-300 dark:border-l-stone-600',
    badge:  'bg-stone-100 text-stone-500 border-stone-200 dark:bg-stone-700/50 dark:text-stone-400 dark:border-stone-600/50',
    value:  'text-stone-500 dark:text-stone-400',
  },
}

/* ── Status → hex color (for sparkline stroke) ────────────────────── */
const STATUS_HEX = {
  normal:   '#34d399',
  warning:  '#fbbf24',
  critical: '#f87171',
  full:     '#34d399',
  low:      '#fbbf24',
  empty:    '#f87171',
}

/* ── Shared 24h history fetch (one request, multiple consumers) ───── */
let _history24hPromise = null
function getHistory24h() {
  if (!_history24hPromise) {
    _history24hPromise = getHistory('24h').catch(() => null)
  }
  return _history24hPromise
}

/* ── useSparkline ─────────────────────────────────────────────────── */
function useSparkline(metricKey) {
  const [points, setPoints] = useState(null)
  useEffect(() => {
    if (!metricKey) return
    getHistory24h()
      .then(rows => {
        if (!Array.isArray(rows)) return
        const vals = rows
          .map(r => r[metricKey])
          .filter(v => v != null && !isNaN(Number(v)))
          .map(Number)
        if (vals.length >= 2) setPoints(vals)
      })
      .catch(() => {})
  }, [metricKey])
  return points
}

/* ── Sparkline ────────────────────────────────────────────────────── */
function Sparkline({ points, color }) {
  if (!points || points.length < 2) return null
  const min = Math.min(...points)
  const max = Math.max(...points)
  const range = max - min || 1
  const W = 100
  const H = 40
  const pad = 2
  const coords = points
    .map((v, i) => {
      const x = pad + (i / (points.length - 1)) * (W - pad * 2)
      const y = pad + (1 - (v - min) / range) * (H - pad * 2)
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="w-full block"
      style={{ height: H }}
    >
      <polyline
        points={coords}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.6"
      />
    </svg>
  )
}

/* ── MetricCard ───────────────────────────────────────────────────── */
function MetricCard({ label, value, unit, status, colorMap = STATUS, metricKey }) {
  const c = colorMap[status] || STATUS.normal
  const sparkPoints = useSparkline(metricKey)
  const sparkColor = STATUS_HEX[status] || '#94a3b8'
  return (
    <div className={`rounded-lg bg-white dark:bg-stone-800 border shadow-sm p-5 pb-0 min-h-[120px] flex flex-col justify-between transition-shadow duration-150 hover:shadow-md ${c.card}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
          {label}
        </span>
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium border ${c.badge}`}>
          {status ?? '—'}
        </span>
      </div>
      <div className="flex items-baseline gap-1 mt-3">
        <span className={`text-3xl font-bold tabular-nums font-mono ${c.value}`}>
          {value ?? '—'}
        </span>
        {unit && (
          <span className="text-lg font-normal text-stone-400 dark:text-stone-500 ml-1">
            {unit}
          </span>
        )}
      </div>
      <div className="mt-3 -mx-5 overflow-hidden h-10">
        <Sparkline points={sparkPoints} color={sparkColor} />
      </div>
    </div>
  )
}

/* ── StatusCard ───────────────────────────────────────────────────── */
function StatusCard({ label, status, colorMap = STATUS, labelMap = null }) {
  const key = status != null ? status.toString() : null
  const c = (key != null ? colorMap[key] : null) || STATUS.normal
  const displayValue = labelMap && key != null && labelMap[key] != null
    ? labelMap[key]
    : (status ?? null)

  // Heat Stress Index: null/undefined → proper empty state
  if (displayValue == null) {
    return (
      <div className="rounded-lg bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700/50 border-l-[3px] border-l-stone-300 dark:border-l-stone-600 shadow-sm p-5 min-h-[120px] flex flex-col justify-between transition-shadow duration-150 hover:shadow-md">
        <span className="text-xs font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
          {label}
        </span>
        <div className="flex items-center gap-2 mt-3">
          <svg className="w-4 h-4 text-stone-300 dark:text-stone-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
          </svg>
          <span className="text-sm text-stone-400 dark:text-stone-500">No data</span>
        </div>
      </div>
    )
  }

  // Normalise: if the value looks like a known status key, badge it
  const isStatusKey = ['normal', 'warning', 'critical', 'low', 'moderate', 'high'].includes(
    String(displayValue).toLowerCase()
  )
  const badgeClass = isStatusKey ? c.badge : 'bg-stone-100 text-stone-600 border-stone-200 dark:bg-stone-700/50 dark:text-stone-300 dark:border-stone-600/50'

  return (
    <div className={`rounded-lg bg-white dark:bg-stone-800 border shadow-sm p-5 min-h-[120px] flex flex-col justify-between transition-shadow duration-150 hover:shadow-md ${c.card}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
          {label}
        </span>
        {isStatusKey && (
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium border ${badgeClass}`}>
            {String(displayValue)}
          </span>
        )}
      </div>
      <div className="mt-3">
        {isStatusKey ? (
          <span className={`text-xl font-semibold capitalize ${c.value}`}>
            {String(displayValue)}
          </span>
        ) : (
          <span className="text-xl font-semibold text-stone-700 dark:text-stone-200 capitalize">
            {String(displayValue)}
          </span>
        )}
      </div>
    </div>
  )
}

/* ── Skeleton card ────────────────────────────────────────────────── */
function SkeletonCard() {
  return (
    <div className="rounded-lg bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700/50 shadow-sm p-5 min-h-[120px]">
      <div className="skeleton h-3 w-20 mb-4 dark:opacity-20 rounded" />
      <div className="skeleton h-8 w-16 dark:opacity-20 rounded" />
    </div>
  )
}

/* ── Relative time ────────────────────────────────────────────────── */
function useRelativeTime(date) {
  const [label, setLabel] = useState('')
  useEffect(() => {
    if (!date) return
    function update() {
      const secs = Math.floor((Date.now() - date.getTime()) / 1000)
      if (secs < 5)  setLabel('just now')
      else if (secs < 60) setLabel(`${secs}s ago`)
      else setLabel(`${Math.floor(secs / 60)}m ago`)
    }
    update()
    const id = setInterval(update, 5000)
    return () => clearInterval(id)
  }, [date])
  return label
}

/* ── Section label ────────────────────────────────────────────────── */
function SectionLabel({ children, first }) {
  return (
    <p className={`text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500 mb-3 ${first ? '' : 'mt-8'}`}>
      {children}
    </p>
  )
}

/* ── Main ─────────────────────────────────────────────────────────── */
function heatScoreStatus(score) {
  if (score == null) return 'normal'
  if (score >= 67) return 'critical'
  if (score >= 34) return 'warning'
  return 'normal'
}

export default function SensorDashboard() {
  const [data, setData] = useState(undefined)
  const [riskData, setRiskData] = useState(null)
  const [error, setError] = useState(null)
  const [updatedAt, setUpdatedAt] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [showLoadingText, setShowLoadingText] = useState(false)
  const relativeTime = useRelativeTime(updatedAt)

  useEffect(() => {
    if (data !== undefined || error) {
      setShowLoadingText(false)
      return
    }
    const id = setTimeout(() => setShowLoadingText(true), 800)
    return () => clearTimeout(id)
  }, [data, error])

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true)
    try {
      const [result, risk] = await Promise.all([getSensors(), getRiskLatest().catch(() => null)])
      setData(result)
      setRiskData(risk?.snapshot ?? null)
      setUpdatedAt(new Date())
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    load()
    const id = setInterval(() => load(), 15_000)
    return () => clearInterval(id)
  }, [load])

  const r = data?.reading

  return (
    <div className="h-full overflow-y-auto bg-stone-50 dark:bg-stone-900">
      <div className="max-w-6xl mx-auto px-6 py-8">

        {/* Header row */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-stone-800 dark:text-stone-100">Live Coop Conditions</h2>
          </div>
          <div className="flex items-center gap-2">
            {updatedAt && (
              <span className="text-xs text-stone-400 dark:text-stone-500 tabular-nums font-mono">
                Updated {relativeTime}
              </span>
            )}
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              title="Refresh now"
              className="p-1.5 rounded-lg text-stone-400 hover:text-stone-600 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-stone-700/50 transition-colors disabled:opacity-40"
            >
              <RefreshIcon className="w-4 h-4" spinning={refreshing} />
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 text-red-600 dark:text-red-400 rounded-lg px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {data === undefined && !error && (
          <>
            <div className="space-y-8 animate-pulse">
              <div>
                <div className="h-3 w-16 bg-stone-200 dark:bg-stone-700 rounded mb-3" />
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[1,2,3].map(i => <SkeletonCard key={i} />)}
                </div>
              </div>
              <div>
                <div className="h-3 w-24 bg-stone-200 dark:bg-stone-700 rounded mb-3" />
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[1,2].map(i => <SkeletonCard key={i} />)}
                </div>
              </div>
            </div>
            {showLoadingText && (
              <p className="text-xs text-stone-400 dark:text-stone-500 text-center mt-4">
                Checking on the flock…
              </p>
            )}
          </>
        )}

        {/* No data */}
        {data === null && (
          <div className="flex flex-col items-center justify-center py-24 gap-3 text-center">
            <svg className="w-10 h-10 text-stone-300 dark:text-stone-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
            </svg>
            <p className="text-sm text-stone-500 dark:text-stone-400">No sensor data available yet.</p>
            <p className="text-xs text-stone-400 dark:text-stone-500">Check that the Pi is writing to the database.</p>
          </div>
        )}

        {/* Sensor cards */}
        {r && (
          <div className="animate-fade-in">

            {/* ── Climate ── */}
            <SectionLabel first>Climate</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                label="Temperature"
                value={r.temperature_c != null ? r.temperature_c.toFixed(2) : null}
                unit="°C"
                status={r.temperature_status}
                metricKey="temperature_c"
              />
              <MetricCard
                label="Humidity"
                value={r.humidity_pct != null ? r.humidity_pct.toFixed(2) : null}
                unit="%"
                status={r.humidity_status}
                metricKey="humidity_pct"
              />
              <MetricCard
                label="Thermal Humidity Index (THI)"
                value={riskData?.heat_risk_score != null ? Math.round(riskData.heat_risk_score) : null}
                unit="/ 100"
                status={heatScoreStatus(riskData?.heat_risk_score)}
              />
            </div>

            {/* ── Resource Levels ── */}
            <SectionLabel>Resource Levels</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                label="Feeder"
                value={r.feeder_pct != null ? r.feeder_pct.toFixed(0) : null}
                unit="%"
                status={r.feeder_status}
                colorMap={RESOURCE}
                metricKey="feeder_pct"
              />
              <MetricCard
                label="Waterer"
                value={r.waterer_pct != null ? r.waterer_pct.toFixed(0) : null}
                unit="%"
                status={r.waterer_status}
                colorMap={RESOURCE}
                metricKey="waterer_pct"
              />
            </div>

            {/* ── Flock ── */}
            <SectionLabel>Flock</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                label="Chickens Inside"
                value={r.chickens_inside != null ? r.chickens_inside : null}
                unit=""
                status="normal"
                metricKey="number_of_chickens"
              />
              <MetricCard
                label="Egg Count"
                value={r.egg_count != null ? r.egg_count : null}
                unit=""
                status="normal"
                metricKey="egg_count"
              />
            </div>

            {/* ── Air Quality & Risk ── */}
            <SectionLabel>Air Quality &amp; Risk</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                label="H₂S"
                value={r.h2s_ppm != null ? r.h2s_ppm.toFixed(2) : null}
                unit="ppm"
                status={r.h2s_level}
                metricKey="h2s_ppm"
              />
              <MetricCard
                label="CO₂"
                value={r.co2_ppm != null ? Math.round(r.co2_ppm) : null}
                unit="ppm"
                status={r.co2_level}
                metricKey="co2_ppm"
              />
              <MetricCard
                label="NH₃"
                value={r.nh3_ppm != null ? r.nh3_ppm.toFixed(1) : null}
                unit="ppm"
                status={r.nh3_level}
                metricKey="nh3_ppm"
              />
              <MetricCard
                label="Mold Risk Score"
                value={r.mold_risk_score != null ? r.mold_risk_score.toFixed(1) : null}
                unit="score"
                status={r.mold_risk_status}
                metricKey="mold_risk_score"
              />
            </div>

            {/* ── Coop Controls ── */}
            <SectionLabel>Coop Controls</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <StatusCard
                label="Door"
                status={r.door_open}
                colorMap={BINARY}
                labelMap={{ true: 'Open', false: 'Closed' }}
              />
              <StatusCard
                label="Ventilation"
                status={r.ventilation_on}
                colorMap={BINARY}
                labelMap={{ true: 'Running', false: 'Off' }}
              />
            </div>

            {/* AI summary */}
            {data.summary && (
              <div className="mt-8 bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700/50 rounded-lg px-5 py-4 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-stone-400 dark:text-stone-500 mb-2">AI Summary</p>
                <p className="text-sm text-stone-600 dark:text-stone-300 leading-relaxed">{data.summary}</p>
              </div>
            )}

            {r.timestamp && (
              <p className="text-xs text-stone-400 dark:text-stone-500 text-right mt-4">
                Reading from {new Date(r.timestamp).toLocaleString()}
              </p>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
