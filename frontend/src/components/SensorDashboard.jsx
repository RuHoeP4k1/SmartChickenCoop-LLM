import { useState, useEffect } from 'react'
import { getSensors } from '../api'
import SensorChart from './SensorChart'

// Color tokens for normal / warning / critical status
const STATUS = {
  normal: {
    card:  'border-stone-200 bg-white',
    badge: 'bg-green-100 text-green-700 border-green-200',
    value: 'text-stone-800',
    dot:   'bg-green-500',
  },
  warning: {
    card:  'border-amber-300 bg-amber-50',
    badge: 'bg-amber-100 text-amber-700 border-amber-300',
    value: 'text-amber-700',
    dot:   'bg-amber-500 animate-pulse',
  },
  critical: {
    card:  'border-red-300 bg-red-50',
    badge: 'bg-red-100 text-red-600 border-red-200',
    value: 'text-red-600',
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
        <span className="text-xs font-medium text-stone-500 uppercase tracking-wider">{label}</span>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${c.badge}`}>
          {status ?? '—'}
        </span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className={`text-3xl font-bold tabular-nums ${c.value}`}>
          {value ?? '—'}
        </span>
        {unit && <span className="text-stone-400 text-base">{unit}</span>}
      </div>
    </div>
  )
}

function StatusCard({ label, status, colorMap = STATUS }) {
  const c = colorMap[status] || STATUS.normal
  return (
    <div className={`rounded-2xl border p-5 transition-colors ${c.card}`}>
      <div className="mb-4">
        <span className="text-xs font-medium text-stone-500 uppercase tracking-wider">{label}</span>
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
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-100">
      <div className="max-w-4xl mx-auto">

        {/* Header row */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-stone-800">Live Coop Conditions</h2>
            <p className="text-xs text-stone-500 mt-0.5">Auto-refreshes every 15 seconds</p>
          </div>
          <div className="flex items-center gap-3">
            {updatedAt && (
              <span className="text-xs text-stone-400">{updatedAt.toLocaleTimeString()}</span>
            )}
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              className="text-xs border border-stone-300 text-stone-500 hover:text-stone-800 hover:border-stone-400 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
            >
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-600 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading */}
        {data === undefined && !error && (
          <div className="text-center text-stone-400 py-24 text-sm">Loading sensor data…</div>
        )}

        {/* No data */}
        {data === null && (
          <div className="text-center text-stone-400 py-24 text-sm">
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
              <div className="bg-white border border-stone-200 rounded-2xl px-5 py-4">
                <p className="text-xs text-stone-400 font-medium uppercase tracking-wider mb-1">Summary</p>
                <p className="text-sm text-stone-600 leading-relaxed">{data.summary}</p>
              </div>
            )}

            {r.timestamp && (
              <p className="text-xs text-stone-400 text-right">
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
