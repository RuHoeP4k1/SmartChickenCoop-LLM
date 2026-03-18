import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { getHistory } from '../api'

// ---------------------------------------------------------------------------
// Metric definitions — add new entries here when sensor team extends the DB
// Each entry needs: key (DB column), label, unit, color, domain (Y-axis range)
// ---------------------------------------------------------------------------
const METRICS = [
  { key: 'temperature_c',  label: 'Temperature', unit: '°C',   color: '#f97316', domain: ['auto', 'auto'] },
  { key: 'humidity_pct',   label: 'Humidity',    unit: '%',    color: '#0891b2', domain: [0, 100]         },
  { key: 'feeder_pct',     label: 'Feeder',      unit: '%',    color: '#16a34a', domain: [0, 100]         },
  { key: 'waterer_pct',    label: 'Waterer',     unit: '%',    color: '#2563eb', domain: [0, 100]         },
  { key: 'h2s_ppm',        label: 'H₂S',         unit: 'ppm',  color: '#dc2626', domain: [0, 50]          },
  { key: 'mold_risk_score',label: 'Mold Risk',   unit: '',     color: '#7c3aed', domain: [0, 100]         },
  { key: 'number_of_chickens', label: 'Chickens', unit: '',     color: '#d97706', domain: [0, 20]          },
]

const RANGES = [
  { value: '1h',  label: '1h' },
  { value: '24h', label: '24h' },
  { value: '7d',  label: '7d' },
]

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-xl px-3 py-2.5 shadow-lg text-xs">
      <p className="text-stone-400 dark:text-stone-400 mb-2">
        {new Date(label).toLocaleString([], {
          month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        })}
      </p>
      {payload.map(p => {
        const metric = METRICS.find(m => m.key === p.dataKey)
        return (
          <div key={p.dataKey} className="flex items-center gap-2 mb-0.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: p.color }} />
            <span className="text-stone-500 dark:text-stone-400">{metric?.label ?? p.dataKey}:</span>
            <span className="font-semibold" style={{ color: p.color }}>
              {Number(p.value).toFixed(2)}{metric?.unit ?? ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// X-axis tick formatter
// ---------------------------------------------------------------------------
function formatTick(ts, range) {
  const d = new Date(ts)
  if (range === '7d') return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// ---------------------------------------------------------------------------
// Main chart component
// ---------------------------------------------------------------------------
export default function SensorChart() {
  // selected: ordered array of up to 2 metric keys
  // selected[0] = left axis (primary), selected[1] = right axis (secondary)
  const [selected, setSelected] = useState(['temperature_c', 'humidity_pct'])
  const [range, setRange]       = useState('1h')
  const [data, setData]         = useState([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)

  function fetchData(silent = false, cancelled = { current: false }) {
    if (!silent) setLoading(true)
    getHistory(range)
      .then(result => {
        if (cancelled.current) return
        setData(result.readings.map(r => ({
          ...r,
          timestamp: new Date(r.timestamp).getTime(),
        })))
        setError(null)
      })
      .catch(err => { if (!cancelled.current) setError(err.message) })
      .finally(() => { if (!cancelled.current) setLoading(false) })
  }

  // Full reload when range changes
  useEffect(() => {
    const cancelled = { current: false }
    fetchData(false, cancelled)
    return () => { cancelled.current = true }
  }, [range])

  // Silent background refresh every 60 s to pick up new readings
  useEffect(() => {
    const id = setInterval(() => fetchData(true), 60_000)
    return () => clearInterval(id)
  }, [range])

  function toggleMetric(key) {
    setSelected(prev => {
      if (prev.includes(key)) {
        // deselect — but keep at least 1
        if (prev.length === 1) return prev
        return prev.filter(k => k !== key)
      }
      if (prev.length < 2) return [...prev, key]
      // already 2 selected → cycle: replace the oldest (index 0)
      return [prev[1], key]
    })
  }

  const showDual = selected.length === 2

  // X-axis domain anchored to the selected range so partial data
  // doesn't get stretched to fill the full time window
  const RANGE_MS = { '1h': 3600_000, '24h': 86400_000, '7d': 604800_000 }
  const xDomain = [Date.now() - (RANGE_MS[range] ?? RANGE_MS['1h']), Date.now()]

  return (
    <div className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-2xl p-5 mt-6 transition-all duration-200 hover:shadow-md">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div>
          <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">Historical Trends</h3>
          {!loading && (
            <p className="text-xs text-stone-400 dark:text-stone-400 mt-0.5">{data.length} readings</p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Metric picker */}
          <div className="flex gap-1 bg-stone-100 dark:bg-stone-700 rounded-lg p-1 border border-stone-200 dark:border-stone-600">
            {METRICS.map(m => {
              const isActive = selected.includes(m.key)
              return (
                <button
                  key={m.key}
                  onClick={() => toggleMetric(m.key)}
                  title={isActive ? 'Click to deselect' : selected.length === 2 ? 'Click to swap oldest' : 'Click to add'}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-white dark:bg-stone-600 text-stone-800 dark:text-stone-100 shadow-sm'
                      : 'text-stone-400 dark:text-stone-400 hover:text-stone-600 dark:hover:text-stone-200'
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0 transition-colors"
                    style={{ background: isActive ? m.color : '#d6d3d1' }}
                  />
                  {m.label}
                </button>
              )
            })}
          </div>

          {/* Range selector */}
          <div className="flex gap-1 bg-stone-100 dark:bg-stone-700 rounded-lg p-1 border border-stone-200 dark:border-stone-600">
            {RANGES.map(r => (
              <button
                key={r.value}
                onClick={() => setRange(r.value)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  range === r.value
                    ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-700'
                    : 'text-stone-400 dark:text-stone-400 hover:text-stone-700 dark:hover:text-stone-200'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart area */}
      {loading && (
        <div className="h-56 flex items-end gap-2 px-8 pb-4">
          {[40,65,30,80,55,70,45,60,35,75,50,68].map((h, i) => (
            <div key={i} className="skeleton flex-1 rounded-t-md" style={{ height: `${h}%` }} />
          ))}
        </div>
      )}
      {error && (
        <div className="flex items-center justify-center h-56 text-red-500 dark:text-red-400 text-sm">
          {error}
        </div>
      )}
      {!loading && !error && data.length === 0 && (
        <div className="flex items-center justify-center h-56 text-stone-400 dark:text-stone-500 text-sm">
          No readings in the past {range}.
        </div>
      )}

      {!loading && !error && data.length > 0 && (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart
            data={data}
            margin={{ top: 4, right: showDual ? 12 : 4, bottom: 4, left: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />

            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={xDomain}
              tickFormatter={ts => formatTick(ts, range)}
              tick={{ fill: 'var(--chart-tick)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--chart-axis)' }}
              tickLine={false}
              minTickGap={50}
            />

            {/* Left Y-axis — primary metric (selected[0]) */}
            {selected[0] && (() => {
              const m = METRICS.find(x => x.key === selected[0])
              return (
                <YAxis
                  yAxisId="left"
                  orientation="left"
                  domain={m?.domain ?? ['auto', 'auto']}
                  tickFormatter={v => `${Number(v).toFixed(0)}${m?.unit ?? ''}`}
                  tick={{ fill: m?.color ?? '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={46}
                />
              )
            })()}

            {/* Right Y-axis — secondary metric (selected[1]), only when 2 selected */}
            {showDual && (() => {
              const m = METRICS.find(x => x.key === selected[1])
              return (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={m?.domain ?? ['auto', 'auto']}
                  tickFormatter={v => `${Number(v).toFixed(0)}${m?.unit ?? ''}`}
                  tick={{ fill: m?.color ?? '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={42}
                />
              )
            })()}

            <Tooltip content={<ChartTooltip />} />

            {/* Lines — one per selected metric */}
            {selected.map((key, idx) => {
              const m = METRICS.find(x => x.key === key)
              return (
                <Line
                  key={key}
                  yAxisId={idx === 0 ? 'left' : 'right'}
                  type="linear"
                  dataKey={key}
                  stroke={m?.color ?? '#94a3b8'}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: m?.color ?? '#94a3b8', strokeWidth: 0 }}
                  connectNulls={false}
                />
              )
            })}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
