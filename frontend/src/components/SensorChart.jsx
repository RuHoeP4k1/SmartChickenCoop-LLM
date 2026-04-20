import { useState, useEffect, useId } from 'react'
import {
  AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Label,
} from 'recharts'
import { getHistory } from '../api'

// ---------------------------------------------------------------------------
// Metric definitions
// ---------------------------------------------------------------------------
const METRICS = [
  { key: 'temperature_c',      label: 'Temperature', unit: '°C',   color: '#f97316', domain: ['auto', 'auto'], decimals: 1 },
  { key: 'humidity_pct',       label: 'Humidity',    unit: '%',    color: '#0891b2', domain: [0, 100], decimals: 0 },
  { key: 'feeder_pct',         label: 'Feeder',      unit: '%',    color: '#16a34a', domain: [0, 100], decimals: 0 },
  { key: 'waterer_pct',        label: 'Waterer',     unit: '%',    color: '#2563eb', domain: [0, 100], decimals: 0 },
  { key: 'h2s_ppm',            label: 'H₂S',         unit: 'ppm',  color: '#dc2626', domain: ['auto', 'auto'], decimals: 2 },
  { key: 'co2_ppm',            label: 'CO₂',         unit: 'ppm',  color: '#b45309', domain: ['auto', 'auto'], decimals: 0 },
  { key: 'nh3_ppm',            label: 'NH₃',         unit: 'ppm',  color: '#6d28d9', domain: ['auto', 'auto'], decimals: 1 },
  { key: 'mold_risk_score',    label: 'Mold Risk',   unit: '',     color: '#7c3aed', domain: [0, 100], decimals: 1 },
]

const RANGES = [
  { value: '1h',  label: '1h'  },
  { value: '24h', label: '24h' },
  { value: '7d',  label: '7d'  },
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
              {Number(p.value).toFixed(metric?.decimals ?? 1)}{metric?.unit ?? ''}
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
// Metric pill (filter control above chart)
// ---------------------------------------------------------------------------
function MetricPill({ metric, isActive, onClick }) {
  return (
    <button
      onClick={onClick}
      title={isActive ? 'Click to deselect' : 'Click to select'}
      className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border transition-colors ${
        isActive
          ? 'bg-white dark:bg-stone-700 border-stone-300 dark:border-stone-500 text-stone-800 dark:text-stone-100 shadow-sm'
          : 'bg-transparent border-stone-200 dark:border-stone-600 text-stone-400 dark:text-stone-500 hover:bg-stone-100 dark:hover:bg-stone-700/50 hover:text-stone-600 dark:hover:text-stone-300'
      }`}
    >
      <span
        className="w-2 h-2 rounded-full shrink-0"
        style={{ background: isActive ? metric.color : '#d6d3d1' }}
      />
      {metric.label}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Main chart component
// ---------------------------------------------------------------------------
export default function SensorChart() {
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

  useEffect(() => {
    const cancelled = { current: false }
    fetchData(false, cancelled)
    return () => { cancelled.current = true }
  }, [range])

  useEffect(() => {
    const cancelled = { current: false }
    const id = setInterval(() => fetchData(true, cancelled), 60_000)
    return () => { cancelled.current = true; clearInterval(id) }
  }, [range])

  function toggleMetric(key) {
    setSelected(prev => {
      if (prev.includes(key)) {
        if (prev.length === 1) return prev
        return prev.filter(k => k !== key)
      }
      if (prev.length < 2) return [...prev, key]
      return [prev[1], key]
    })
  }

  const uid = useId()
  const leftGradId  = `leftGradient-${uid}`
  const rightGradId = `rightGradient-${uid}`

  const showDual = selected.length === 2
  const RANGE_MS = { '1h': 3600_000, '24h': 86400_000, '7d': 604800_000 }
  const xDomain = [Date.now() - (RANGE_MS[range] ?? RANGE_MS['1h']), Date.now()]

  const leftMetric  = METRICS.find(x => x.key === selected[0])
  const rightMetric = showDual ? METRICS.find(x => x.key === selected[1]) : null

  return (
    <div className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700/50 rounded-lg shadow-sm mt-6 transition-shadow duration-150 hover:shadow-md">
      {/* Card header */}
      <div className="px-5 pt-5 pb-4 border-b border-stone-100 dark:border-stone-700/50">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold tracking-tight text-stone-800 dark:text-stone-100">Historical Trends</h3>
            {!loading && (
              <p className="text-xs text-stone-400 dark:text-stone-500 mt-0.5">{data.length} readings</p>
            )}
          </div>

          {/* Time range segmented control */}
          <div className="bg-stone-100 dark:bg-stone-700 rounded-lg p-1 inline-flex gap-1 shrink-0">
            {RANGES.map(r => (
              <button
                key={r.value}
                onClick={() => setRange(r.value)}
                className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                  range === r.value
                    ? 'bg-white dark:bg-stone-600 text-amber-700 dark:text-amber-400 font-medium shadow-sm'
                    : 'text-stone-500 dark:text-stone-400 hover:text-stone-700 dark:hover:text-stone-200'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>

        {/* Metric pills */}
        <div className="flex flex-wrap gap-1.5 mt-4">
          {METRICS.map(m => (
            <MetricPill
              key={m.key}
              metric={m}
              isActive={selected.includes(m.key)}
              onClick={() => toggleMetric(m.key)}
            />
          ))}
        </div>
      </div>

      {/* Chart area */}
      <div className="px-5 pb-5 pt-4">
        {loading && (
          <div className="h-56 flex items-end gap-1.5 pb-4">
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
          <div className="flex flex-col items-center justify-center h-56 gap-2">
            <svg className="w-8 h-8 text-stone-300 dark:text-stone-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 3v18h18" /><path d="M7 16l4-6 4 4 5-8" />
            </svg>
            <p className="text-sm text-stone-400 dark:text-stone-500">No readings in the past {range}.</p>
          </div>
        )}

        {!loading && !error && data.length > 0 && (
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart
              data={data}
              margin={{ top: 8, right: showDual ? 60 : 16, bottom: 24, left: 16 }}
            >
              <defs>
                <linearGradient id={leftGradId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={leftMetric.color} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={leftMetric.color} stopOpacity={0} />
                </linearGradient>
                {showDual && (
                  <linearGradient id={rightGradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={rightMetric.color} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={rightMetric.color} stopOpacity={0} />
                  </linearGradient>
                )}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />

              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={xDomain}
                tickFormatter={ts => formatTick(ts, range)}
                tick={{ fill: 'var(--chart-tick)', fontSize: 11, fontFamily: 'Geist Mono, monospace' }}
                axisLine={{ stroke: 'var(--chart-axis)' }}
                tickLine={false}
                minTickGap={50}
              />

              {/* Left Y-axis */}
              {leftMetric && (
                <YAxis
                  yAxisId="left"
                  orientation="left"
                  domain={leftMetric.domain}
                  tickFormatter={v => `${Number(v).toFixed(leftMetric.decimals)}${leftMetric.unit}`}
                  tick={{ fill: leftMetric.color, fontSize: 11, fontFamily: 'Geist Mono, monospace' }}
                  axisLine={false}
                  tickLine={false}
                  width={54}
                >
                  <Label
                    value={`${leftMetric.label} (${leftMetric.unit})`}
                    angle={-90}
                    position="insideLeft"
                    offset={10}
                    style={{ fill: leftMetric.color, fontSize: 10, opacity: 0.7, fontFamily: 'Geist Mono, monospace' }}
                  />
                </YAxis>
              )}

              {/* Right Y-axis */}
              {showDual && rightMetric && (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={rightMetric.domain}
                  tickFormatter={v => `${Number(v).toFixed(rightMetric.decimals)}${rightMetric.unit}`}
                  tick={{ fill: rightMetric.color, fontSize: 11, fontFamily: 'Geist Mono, monospace' }}
                  axisLine={false}
                  tickLine={false}
                  width={54}
                >
                  <Label
                    value={`${rightMetric.label} (${rightMetric.unit})`}
                    angle={90}
                    position="insideRight"
                    offset={10}
                    style={{ fill: rightMetric.color, fontSize: 10, opacity: 0.7, fontFamily: 'Geist Mono, monospace' }}
                  />
                </YAxis>
              )}

              <Tooltip content={<ChartTooltip />} />

              {selected.map((key, idx) => {
                const m = METRICS.find(x => x.key === key)
                return (
                  <Area
                    key={key}
                    yAxisId={idx === 0 ? 'left' : 'right'}
                    type="linear"
                    dataKey={key}
                    stroke={m?.color ?? '#94a3b8'}
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 2, stroke: 'white', fill: m?.color ?? '#94a3b8' }}
                    fill={`url(#${idx === 0 ? leftGradId : rightGradId})`}
                    connectNulls={false}
                  />
                )
              })}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
