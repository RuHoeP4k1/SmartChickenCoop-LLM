import { useState, useEffect } from 'react'
import { getEvents } from '../api'

const TYPE_META = {
  sensor_alert:      { label: 'Sensor Alert', style: 'bg-amber-900/40 text-amber-400 border-amber-700/40' },
  conditions_normal: { label: 'All Clear',    style: 'bg-green-900/40 text-green-400 border-green-800/40' },
  llm_response:      { label: 'Q&A',          style: 'bg-slate-700/60 text-slate-300 border-slate-600/40' },
}

const SEV_ICON = { critical: '⚠', warning: '⚡', info: 'ℹ' }
const SEV_COLOR = { critical: 'text-red-400', warning: 'text-amber-400', info: 'text-slate-500' }
const CARD_BORDER = { critical: 'border-red-800/50', warning: 'border-amber-800/30', info: 'border-slate-800' }

function EventCard({ event }) {
  const [expanded, setExpanded] = useState(false)
  const meta = TYPE_META[event.event_type] || TYPE_META.llm_response
  const hasBody = event.user_query || event.llm_response || event.sensor_context_filtered

  return (
    <div className={`rounded-2xl border bg-slate-900 overflow-hidden ${CARD_BORDER[event.severity] ?? CARD_BORDER.info}`}>
      {/* Card header — always visible */}
      <div
        className={`px-4 py-3 flex items-start justify-between gap-3 ${hasBody ? 'cursor-pointer hover:bg-slate-800/40 transition-colors' : ''}`}
        onClick={() => hasBody && setExpanded(e => !e)}
      >
        <div className="flex items-start gap-3 min-w-0">
          <span className={`mt-0.5 shrink-0 text-base ${SEV_COLOR[event.severity] ?? SEV_COLOR.info}`}>
            {SEV_ICON[event.severity] ?? 'ℹ'}
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-md border ${meta.style}`}>
                {meta.label}
              </span>
              <span className={`text-xs font-medium ${SEV_COLOR[event.severity] ?? SEV_COLOR.info}`}>
                {event.severity}
              </span>
            </div>
            {event.user_query && (
              <p className="text-sm text-slate-200 truncate">{event.user_query}</p>
            )}
            {!event.user_query && event.sensor_context_filtered && (
              <p className="text-sm text-slate-400 truncate">{event.sensor_context_filtered}</p>
            )}
          </div>
        </div>

        <div className="shrink-0 text-right">
          <p className="text-xs text-slate-500">{new Date(event.timestamp).toLocaleString()}</p>
          {hasBody && (
            <p className="text-xs text-slate-600 mt-1">{expanded ? '▲ less' : '▼ more'}</p>
          )}
        </div>
      </div>

      {/* Expandable body */}
      {expanded && (
        <div className="px-4 pb-4 pt-3 border-t border-slate-800 space-y-3">
          {event.user_query && event.event_type !== 'llm_response' && (
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">Query</p>
              <p className="text-sm text-slate-300 leading-relaxed">{event.user_query}</p>
            </div>
          )}
          {event.sensor_context_filtered && (
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">Sensor Context</p>
              <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{event.sensor_context_filtered}</p>
            </div>
          )}
          {event.llm_response && (
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">AI Response</p>
              <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{event.llm_response}</p>
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
  { value: 'conditions_normal',label: 'All Clear' },
  { value: 'llm_response',     label: 'Q&A' },
]

export default function AlertFeed() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState(null)

  async function load() {
    try {
      const result = await getEvents(30, filter)
      setEvents(result.events)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    setLoading(true)
    load()
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [filter])

  return (
    <div className="h-full overflow-y-auto px-6 py-8">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-white">Event Log</h2>
            <p className="text-xs text-slate-500 mt-0.5">Sensor alerts and AI responses · refreshes every 30 s</p>
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {FILTERS.map(f => (
              <button
                key={String(f.value)}
                onClick={() => setFilter(f.value)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  filter === f.value
                    ? 'bg-green-700/30 border-green-700/50 text-green-400'
                    : 'border-slate-700 text-slate-400 hover:text-slate-200'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="bg-red-950/40 border border-red-800/50 text-red-400 rounded-xl px-4 py-3 text-sm mb-4">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-center text-slate-500 py-24 text-sm">Loading events…</div>
        )}

        {!loading && events.length === 0 && (
          <div className="text-center text-slate-600 py-24 text-sm">No events yet.</div>
        )}

        <div className="space-y-3">
          {events.map(event => <EventCard key={event.id} event={event} />)}
        </div>
      </div>
    </div>
  )
}
