import { useState, useEffect } from 'react'
import { getEvents } from '../api'

const TYPE_META = {
  sensor_alert:      { label: 'Sensor Alert', style: 'bg-amber-100 text-amber-700 border-amber-300 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-700' },
  conditions_normal: { label: 'All Clear',    style: 'bg-green-100 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-700' },
  llm_response:      { label: 'Q&A',          style: 'bg-stone-100 text-stone-600 border-stone-300 dark:bg-stone-700 dark:text-stone-300 dark:border-stone-600' },
  automation_action: { label: 'Automation',   style: 'bg-sky-100 text-sky-700 border-sky-300 dark:bg-sky-900/30 dark:text-sky-400 dark:border-sky-700' },
}

const SEV_ICON = { critical: '⚠', warning: '⚡', info: 'ℹ' }
const SEV_COLOR = { critical: 'text-red-500 dark:text-red-400', warning: 'text-amber-600 dark:text-amber-400', info: 'text-stone-400 dark:text-stone-500' }
const CARD_BORDER = { critical: 'border-red-300 dark:border-red-700', warning: 'border-amber-300 dark:border-amber-700', info: 'border-stone-200 dark:border-stone-700' }

function EventCard({ event }) {
  const [expanded, setExpanded] = useState(false)
  const meta = TYPE_META[event.event_type] || TYPE_META.llm_response
  const sources = event.sources ? (Array.isArray(event.sources) ? event.sources : JSON.parse(event.sources)) : []
  const hasBody = event.user_query || event.llm_response || event.sensor_context_filtered || sources.length > 0

  return (
    <div className={`rounded-xl border bg-white dark:bg-stone-800 overflow-hidden transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 ${CARD_BORDER[event.severity] ?? CARD_BORDER.info}`}>
      {/* Card header — always visible */}
      <div
        className={`px-4 py-3 flex items-start justify-between gap-3 ${hasBody ? 'cursor-pointer hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors' : ''}`}
        onClick={() => hasBody && setExpanded(e => !e)}
      >
        <div className="flex items-start gap-3 min-w-0">
          <span className={`mt-0.5 shrink-0 text-base ${SEV_COLOR[event.severity] ?? SEV_COLOR.info}`}>
            {SEV_ICON[event.severity] ?? 'ℹ'}
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-lg border ${meta.style}`}>
                {meta.label}
              </span>
              <span className={`text-xs font-medium ${SEV_COLOR[event.severity] ?? SEV_COLOR.info}`}>
                {event.severity}
              </span>
            </div>
            {event.user_query && (
              <p className="text-sm text-stone-700 dark:text-stone-200 truncate">{event.user_query}</p>
            )}
            {!event.user_query && event.sensor_context_filtered && (
              <p className="text-sm text-stone-500 dark:text-stone-400 truncate">{event.sensor_context_filtered}</p>
            )}
          </div>
        </div>

        <div className="shrink-0 text-right">
          <p className="text-xs text-stone-400 dark:text-stone-500">{new Date(event.timestamp).toLocaleString()}</p>
          {hasBody && (
            <p className="text-xs text-stone-300 dark:text-stone-600 mt-1">{expanded ? '▲ less' : '▼ more'}</p>
          )}
        </div>
      </div>

      {/* Expandable body */}
      {expanded && (
        <div className="px-4 pb-4 pt-3 border-t border-stone-100 dark:border-stone-700 space-y-3">
          {event.user_query && event.event_type !== 'llm_response' && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Query</p>
              <p className="text-sm text-stone-700 dark:text-stone-200 leading-relaxed">{event.user_query}</p>
            </div>
          )}
          {event.sensor_context_filtered && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Sensor Context</p>
              <p className="text-sm text-stone-600 dark:text-stone-300 leading-relaxed whitespace-pre-wrap">{event.sensor_context_filtered}</p>
            </div>
          )}
          {event.llm_response && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">AI Response</p>
              <p className="text-sm text-stone-700 dark:text-stone-200 leading-relaxed whitespace-pre-wrap">{event.llm_response}</p>
            </div>
          )}
          {sources.length > 0 && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Sources</p>
              <ul className="space-y-0.5">
                {sources.map((src, i) => (
                  <li key={i} className="text-xs text-stone-500 dark:text-stone-400 font-mono bg-stone-50 dark:bg-stone-700/50 border border-stone-200 dark:border-stone-700 rounded px-2 py-0.5 inline-block mr-1">
                    {src}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const FILTERS = [
  { value: null,               label: 'All' },
  { value: 'sensor_alert',     label: 'Alerts' },
  { value: 'automation_action',label: 'Automation' },
  { value: 'conditions_normal',label: 'All Clear' },
  { value: 'llm_response',     label: 'Q&A' },
]

export default function AlertFeed() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState(null)

  async function load(showSpinner = false) {
    if (showSpinner) setRefreshing(true)
    try {
      const result = await getEvents(30, filter)
      setEvents(result.events)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    setLoading(true)
    load()
    const id = setInterval(() => load(), 10_000)
    return () => clearInterval(id)
  }, [filter])

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Event Log</h2>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">Sensor alerts and AI responses · refreshes every 10 s</p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              className="text-xs border border-stone-300 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 hover:border-stone-400 dark:hover:border-stone-500 bg-white dark:bg-stone-800 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
            >
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
            {FILTERS.map(f => (
              <button
                key={String(f.value)}
                onClick={() => setFilter(f.value)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  filter === f.value
                    ? 'bg-amber-100 border-amber-400 text-amber-700 dark:bg-amber-900/30 dark:border-amber-600 dark:text-amber-400'
                    : 'border-stone-300 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 bg-white dark:bg-stone-800'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 rounded-xl px-4 py-3 text-sm mb-4">
            {error}
          </div>
        )}

        {loading && (
          <div className="space-y-3">
            {[1,2,3].map(i => (
              <div key={i} className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-4">
                <div className="flex items-center gap-3">
                  <div className="skeleton w-6 h-6 rounded-full shrink-0" />
                  <div className="flex-1 space-y-2">
                    <div className="skeleton h-3 w-24" />
                    <div className="skeleton h-3 w-48" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && events.length === 0 && (
          <div className="text-center text-stone-400 dark:text-stone-500 py-24 text-sm">All quiet in the coop.</div>
        )}

        <div className="space-y-3 animate-fade-in">
          {events.map(event => <EventCard key={event.id} event={event} />)}
        </div>
      </div>
    </div>
  )
}
