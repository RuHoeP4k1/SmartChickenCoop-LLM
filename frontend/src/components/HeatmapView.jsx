import { useState, useEffect } from 'react'
import { getSensors, getHeatmapLatest } from '../api'

function StatCard({ icon, label, value, sub }) {
  return (
    <div className="flex-1 rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 px-6 py-5 flex items-center gap-4">
      <div className="shrink-0 w-11 h-11 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/40 flex items-center justify-center">
        {icon}
      </div>
      <div>
        <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wide mb-0.5">{label}</p>
        <p className="text-2xl font-bold text-stone-800 dark:text-stone-100 leading-none">
          {value ?? '—'}
        </p>
        {sub && <p className="text-[11px] text-stone-400 dark:text-stone-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

function IconChicken() {
  return (
    <svg className="w-5 h-5 text-amber-600 dark:text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="7" r="3" />
      <path d="M13 7c1-2 3-2 4-1" />
      <path d="M10 10c-4 0-6 3-6 6v2h14v-2c0-2-1-4-3-5" />
      <path d="M8 18v2M14 18v2" />
    </svg>
  )
}

function IconEgg() {
  return (
    <svg className="w-5 h-5 text-amber-600 dark:text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="13" rx="7" ry="9" />
    </svg>
  )
}

export default function HeatmapView() {
  const [sensorData, setSensorData] = useState(null)
  const [heatmapUrl, setHeatmapUrl] = useState(null)
  const [heatmapTs, setHeatmapTs] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const [sensors, heatmap] = await Promise.all([
          getSensors(),
          getHeatmapLatest(),
        ])
        setSensorData(sensors?.reading ?? null)
        if (heatmap?.url) {
          setHeatmapUrl(heatmap.url)
          setHeatmapTs(heatmap.timestamp ?? null)
        }
        setError(null)
      } catch (err) {
        setError(err.message)
      }
    }
    load()
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [])

  const chickens = sensorData?.number_of_chickens
  const eggs = sensorData?.egg_count
  const ts = sensorData?.timestamp

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Flock</h2>
          <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">Live CV counts &amp; coop heatmap</p>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700/50 text-red-600 dark:text-red-400 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Stat cards */}
        <div className="flex gap-4 mb-6">
          <StatCard
            icon={<IconChicken />}
            label="Chickens"
            value={chickens}
            sub={ts ? `as of ${new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : undefined}
          />
          <StatCard
            icon={<IconEgg />}
            label="Eggs detected"
            value={eggs}
            sub="via CV pipeline"
          />
        </div>

        {/* Heatmap */}
        <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-stone-100 dark:border-stone-700">
            <p className="text-xs font-medium text-stone-500 dark:text-stone-400 uppercase tracking-wide">Distribution Heatmap</p>
            {heatmapTs && (
              <p className="text-[10px] text-stone-400 dark:text-stone-500">
                {new Date(heatmapTs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
            )}
          </div>
          <div className="p-4">
            {heatmapUrl ? (
              <img
                src={heatmapUrl}
                alt="Chicken distribution heatmap"
                className="w-full object-contain max-h-[60vh] rounded-xl"
              />
            ) : (
              <div className="flex items-center justify-center h-48 text-stone-400 dark:text-stone-500 text-sm">
                No heatmap available
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}
