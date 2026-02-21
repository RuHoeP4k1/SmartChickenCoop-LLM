import { useState, useEffect } from 'react'
import { getSensors } from '../api'
import SensorChart from './SensorChart'

// Color tokens for normal / warning / critical status
const STATUS = {
  normal: {
    card:  'border-slate-700 bg-slate-800/40',
    badge: 'bg-green-900/40 text-green-400 border-green-800/40',
    value: 'text-white',
    dot:   'bg-green-500',
  },
  warning: {
    card:  'border-amber-700/50 bg-amber-950/20',
    badge: 'bg-amber-900/40 text-amber-400 border-amber-700/40',
    value: 'text-amber-300',
    dot:   'bg-amber-500 animate-pulse',
  },
  critical: {
    card:  'border-red-700/60 bg-red-950/30',
    badge: 'bg-red-900/40 text-red-400 border-red-700/50',
    value: 'text-red-300',
    dot:   'bg-red-500 animate-pulse',
  },
}

// For feeder / waterer (full / low / empty)
const RESOURCE = {
  full:  STATUS.normal,
  low:   STATUS.warning,
  empty: STATUS.critical,
}

function MetricCard({ label, value, unit, status, colorMap = STATUS }) {
  const c = colorMap[status] || STATUS.normal
  return (
    <div className={`rounded-2xl border p-5 transition-colors ${c.card}`}>
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${c.badge}`}>
          {status ?? '—'}
        </span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className={`text-3xl font-bold tabular-nums ${c.value}`}>
          {value ?? '—'}
        </span>
        {unit && <span className="text-slate-400 text-base">{unit}</span>}
      </div>
    </div>
  )
}

function StatusCard({ label, status, colorMap = STATUS }) {
  const c = colorMap[status] || STATUS.normal
  return (
    <div className={`rounded-2xl border p-5 transition-colors ${c.card}`}>
      <div className="mb-4">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className={`w-3 h-3 rounded-full shrink-0 ${c.dot}`} />
        <span className={`text-xl font-semibold capitalize ${c.value}`}>
          {status ?? '—'}
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
    <div className="h-full overflow-y-auto px-6 py-8">
      <div className="max-w-4xl mx-auto">

        {/* Header row */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-white">Live Coop Conditions</h2>
            <p className="text-xs text-slate-500 mt-0.5">Auto-refreshes every 15 seconds</p>
          </div>
          <div className="flex items-center gap-3">
            {updatedAt && (
              <span className="text-xs text-slate-500">{updatedAt.toLocaleTimeString()}</span>
            )}
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              className="text-xs border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
            >
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-950/40 border border-red-800/50 text-red-400 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading */}
        {data === undefined && !error && (
          <div className="text-center text-slate-500 py-24 text-sm">Loading sensor data…</div>
        )}

        {/* No data */}
        {data === null && (
          <div className="text-center text-slate-600 py-24 text-sm">
            No sensor data available yet. Check that the Pi is writing to the database.
          </div>
        )}

        {/* Sensor cards */}
        {r && (
          <div className="space-y-4">
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

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <StatusCard label="Feeder"  status={r.feeder_status}  colorMap={RESOURCE} />
              <StatusCard label="Waterer" status={r.waterer_status} colorMap={RESOURCE} />
            </div>

            {/* AI summary from /sensors */}
            {data.summary && (
              <div className="bg-slate-800/40 border border-slate-700 rounded-2xl px-5 py-4">
                <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Summary</p>
                <p className="text-sm text-slate-300 leading-relaxed">{data.summary}</p>
              </div>
            )}

            {r.timestamp && (
              <p className="text-xs text-slate-600 text-right">
                Reading from {new Date(r.timestamp).toLocaleString()}
              </p>
            )}
          </div>
        )}

        {/* Historical chart — always visible, has its own loading state */}
        <SensorChart />
      </div>
    </div>
  )
}
