import { useState, useEffect } from 'react'
import { getHeatmap } from '../api/index.js'

function formatTimestamp(ts) {
  return new Date(ts * 1000).toLocaleString()
}

export default function HeatmapView() {
  const [heatmap, setHeatmap] = useState(undefined)  // undefined = loading, null = none, object = data
  const [error, setError] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  async function fetchHeatmap() {
    setError(null)
    try {
      const data = await getHeatmap()
      setHeatmap(data)
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    fetchHeatmap()
  }, [])

  async function handleRefresh() {
    setRefreshing(true)
    await fetchHeatmap()
    setRefreshing(false)
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100">
              Chicken Distribution Heatmap
            </h2>
            <p className="text-sm text-stone-500 dark:text-stone-400 mt-1">
              Latest heatmap from the CV pipeline — updated every ~2 days
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing || heatmap === undefined}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                       bg-amber-100 text-amber-800 hover:bg-amber-200
                       dark:bg-amber-900/30 dark:text-amber-300 dark:hover:bg-amber-900/50
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>

        {/* Error state */}
        {error && (
          <div className="mb-6 p-4 rounded-2xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800
                          text-red-700 dark:text-red-300 text-sm">
            Failed to load heatmap: {error}
          </div>
        )}

        {/* Loading state */}
        {heatmap === undefined && !error && (
          <div className="rounded-2xl border border-stone-200 dark:border-stone-700
                          bg-white dark:bg-stone-800 p-8 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-stone-400 dark:text-stone-500">
              <svg className="w-8 h-8 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span className="text-sm">Loading…</span>
            </div>
          </div>
        )}

        {/* Empty state — CV pipeline hasn't sent anything yet */}
        {heatmap === null && !error && (
          <div className="rounded-2xl border-2 border-dashed border-stone-300 dark:border-stone-600
                          bg-white dark:bg-stone-800 p-12 flex flex-col items-center gap-4 text-center">
            <svg className="w-16 h-16 text-stone-300 dark:text-stone-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <div>
              <p className="text-lg font-semibold text-stone-600 dark:text-stone-300">
                No heatmap available yet
              </p>
              <p className="text-sm text-stone-400 dark:text-stone-500 mt-1 max-w-sm">
                Waiting for the CV pipeline on the Raspberry Pi to send its first heatmap.
                Images are generated every ~2 days.
              </p>
            </div>
          </div>
        )}

        {/* Heatmap image */}
        {heatmap && (
          <div className="rounded-2xl border border-stone-200 dark:border-stone-700
                          bg-white dark:bg-stone-800 overflow-hidden">

            {/* Timestamp badge */}
            <div className="flex items-center justify-between px-5 py-3
                            border-b border-stone-100 dark:border-stone-700">
              <span className="text-sm font-medium text-stone-600 dark:text-stone-300">
                Latest heatmap
              </span>
              <span className="text-xs text-stone-400 dark:text-stone-500">
                Generated: {formatTimestamp(heatmap.timestamp)}
              </span>
            </div>

            {/* Image */}
            <div className="p-4">
              <img
                src={heatmap.url}
                alt="Chicken distribution heatmap"
                className="w-full object-contain max-h-[70vh] rounded-xl"
              />
            </div>

            {/* Filename */}
            <div className="px-5 py-3 border-t border-stone-100 dark:border-stone-700">
              <span className="text-xs text-stone-400 dark:text-stone-500 font-mono">
                {heatmap.filename}
              </span>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
