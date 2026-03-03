import { useState, useEffect } from 'react'
import { getSensors } from '../api'

// Color tokens for normal / warning / critical status
const STATUS = {
  normal: {
    card:  'border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800',
    badge: 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800',
    value: 'text-stone-800 dark:text-stone-100',
    dot:   'bg-green-500',
  },
  warning: {
    card:  'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20',
    badge: 'bg-amber-100 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border-amber-300 dark:border-amber-700',
    value: 'text-amber-700 dark:text-amber-400',
    dot:   'bg-amber-500 animate-pulse',
  },
  critical: {
    card:  'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20',
    badge: 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 border-red-200 dark:border-red-700',
    value: 'text-red-600 dark:text-red-400',
    dot:   'bg-red-500 animate-pulse',
  },
}

// For feeder / waterer (full / low / empty)
const RESOURCE = {
  full:  STATUS.normal,
  low:   STATUS.warning,
  empty: STATUS.critical,
}

// For boolean fields (door_open, ventilation_on)
const BINARY = {
  true: {
    card:  'border-sky-300 dark:border-sky-700 bg-sky-50 dark:bg-sky-900/20',
    badge: 'bg-sky-100 dark:bg-sky-900/20 text-sky-700 dark:text-sky-400 border-sky-300 dark:border-sky-700',
    value: 'text-sky-700 dark:text-sky-400',
    dot:   'bg-sky-500',
  },
  false: {
    card:  'border-stone-200 dark:border-stone-700 bg-stone-50 dark:bg-stone-700/50',
    badge: 'bg-stone-100 dark:bg-stone-700/50 text-stone-500 dark:text-stone-400 border-stone-200 dark:border-stone-600',
    value: 'text-stone-500 dark:text-stone-400',
    dot:   'bg-stone-400',
  },
}

function MetricCard({ label, value, unit, status, colorMap = STATUS }) {
  const c = colorMap[status] || STATUS.normal
  return (
    <div className={`rounded-2xl border p-5 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 ${c.card}`}>
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-medium text-stone-500 dark:text-stone-400 uppercase tracking-wider">{label}</span>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${c.badge}`}>
          {status ?? '—'}
        </span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className={`text-3xl font-bold tabular-nums ${c.value}`}>
          {value ?? '—'}
        </span>
        {unit && <span className="text-stone-400 dark:text-stone-500 text-base">{unit}</span>}
      </div>
    </div>
  )
}

function StatusCard({ label, status, colorMap = STATUS, labelMap = null }) {
  // Support boolean values as keys by converting to string
  const key = status != null ? status.toString() : null
  const c = (key != null ? colorMap[key] : null) || STATUS.normal
  const displayValue = labelMap && key != null && labelMap[key] != null
    ? labelMap[key]
    : (status ?? '—')
  return (
    <div className={`rounded-2xl border p-5 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 ${c.card}`}>
      <div className="mb-4">
        <span className="text-xs font-medium text-stone-500 dark:text-stone-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className={`w-3 h-3 rounded-full shrink-0 ${c.dot}`} />
        <span className={`text-xl font-semibold capitalize ${c.value}`}>
          {displayValue}
        </span>
      </div>
    </div>
  )
}

export default function SensorDashboard() {
  // undefined = not yet fetched, null = fetched but no data
  const [data, setData] = useState(undefined)
  const [error, setError] = useState(null)
  const [updatedAt, setUpdatedAt] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  async function load(showSpinner = false) {
    if (showSpinner) setRefreshing(true)
    try {
      const result = await getSensors()
      setData(result)
      setUpdatedAt(new Date())
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    load()
    const id = setInterval(() => load(), 15_000)
    return () => clearInterval(id)
  }, [])

  const r = data?.reading

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto">

        {/* Header row */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Live Coop Conditions</h2>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">Auto-refreshes every 15 seconds</p>
          </div>
          <div className="flex items-center gap-3">
            {updatedAt && (
              <span className="text-xs text-stone-400 dark:text-stone-500">{updatedAt.toLocaleTimeString()}</span>
            )}
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              className="text-xs border border-stone-300 dark:border-stone-600 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 hover:border-stone-400 dark:hover:border-stone-500 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
            >
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 text-red-600 dark:text-red-400 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading */}
        {data === undefined && !error && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1,2,3,4,5,6].map(i => (
                <div key={i} className="rounded-2xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-5">
                  <div className="skeleton h-3 w-20 mb-4 dark:opacity-20" />
                  <div className="skeleton h-8 w-16 dark:opacity-20" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No data */}
        {data === null && (
          <div className="text-center text-stone-400 dark:text-stone-500 py-24 text-sm">
            No sensor data available yet. Check that the Pi is writing to the database.
          </div>
        )}

        {/* Sensor cards */}
        {r && (
          <div className="space-y-6 animate-fade-in">

            {/* ── Climate ── */}
            <section>
              <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">Climate</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricCard
                  label="Temperature"
                  value={r.temperature_c != null ? r.temperature_c.toFixed(2) : null}
                  unit="°C"
                  status={r.temperature_status}
                />
                <MetricCard
                  label="Humidity"
                  value={r.humidity_pct != null ? r.humidity_pct.toFixed(2) : null}
                  unit="%"
                  status={r.humidity_status}
                />
                <StatusCard
                  label="Heat Stress Index"
                  status={r.heat_stress_index}
                />
              </div>
            </section>

            {/* ── Resource Levels ── */}
            <section>
              <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">Resource Levels</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <MetricCard
                  label="Feeder"
                  value={r.feeder_pct != null ? r.feeder_pct.toFixed(0) : null}
                  unit="%"
                  status={r.feeder_status}
                  colorMap={RESOURCE}
                />
                <MetricCard
                  label="Waterer"
                  value={r.waterer_pct != null ? r.waterer_pct.toFixed(0) : null}
                  unit="%"
                  status={r.waterer_status}
                  colorMap={RESOURCE}
                />
              </div>
            </section>

            {/* ── Flock ── */}
            <section>
              <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">Flock</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <MetricCard
                  label="Chickens Inside"
                  value={r.chickens_inside != null ? r.chickens_inside : null}
                  unit=""
                  status="normal"
                />
                <MetricCard
                  label="Egg Count"
                  value={r.egg_count != null ? r.egg_count : null}
                  unit=""
                  status="normal"
                />
              </div>
            </section>

            {/* ── Air Quality & Risk ── */}
            <section>
              <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">Air Quality & Risk</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <MetricCard
                  label="H₂S"
                  value={r.h2s_ppm != null ? r.h2s_ppm.toFixed(2) : null}
                  unit="ppm"
                  status={r.h2s_level}
                />
                <MetricCard
                  label="Mold Risk Score"
                  value={r.mold_risk_score != null ? r.mold_risk_score.toFixed(1) : null}
                  unit="score"
                  status={r.mold_risk_status}
                />
              </div>
            </section>

            {/* ── Coop Controls ── */}
            <section>
              <p className="text-xs font-semibold text-stone-400 dark:text-stone-500 uppercase tracking-widest mb-3">Coop Controls</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
            </section>

            {/* AI summary from /sensors */}
            {data.summary && (
              <div className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-2xl px-5 py-4">
                <p className="text-xs text-stone-400 dark:text-stone-500 font-medium uppercase tracking-wider mb-1">Summary</p>
                <p className="text-sm text-stone-600 dark:text-stone-300 leading-relaxed">{data.summary}</p>
              </div>
            )}

            {r.timestamp && (
              <p className="text-xs text-stone-400 dark:text-stone-500 text-right">
                Reading from {new Date(r.timestamp).toLocaleString()}
              </p>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
