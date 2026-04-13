import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { getEvents } from '../api'

// ---- card config: left-border color + badge + icon per event type/severity ----
const CARD_CONFIG = {
  sensor_alert_critical: {
    border: 'border-l-red-500',
    badge: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400',
    title: 'text-red-700 dark:text-red-400',
    icon: '🚨',
    label: 'Critical Alert',
    badgeText: 'critical',
  },
  sensor_alert_warning: {
    border: 'border-l-amber-500',
    badge: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400',
    title: 'text-amber-700 dark:text-amber-400',
    icon: '⚠️',
    label: 'Environment Alert',
    badgeText: 'warning',
  },
  sensor_alert_info: {
    border: 'border-l-amber-300',
    badge: 'bg-amber-50 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400',
    title: 'text-amber-600 dark:text-amber-400',
    icon: '📊',
    label: 'Sensor Alert',
    badgeText: 'info',
  },
  conditions_normal: {
    border: 'border-l-green-500',
    badge: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400',
    title: 'text-green-700 dark:text-green-400',
    icon: '✅',
    label: 'All Clear',
    badgeText: 'normal',
  },
  automation_action: {
    border: 'border-l-violet-500',
    badge: 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-400',
    title: 'text-violet-700 dark:text-violet-400',
    icon: '⚙️',
    label: 'Automation',
    badgeText: 'automated',
  },
  llm_response: {
    border: 'border-l-blue-500',
    badge: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400',
    title: 'text-blue-700 dark:text-blue-400',
    icon: '💬',
    label: 'AI Response',
    badgeText: 'Q&A',
  },
}

function getConfig(event) {
  if (event.event_type === 'sensor_alert') {
    return CARD_CONFIG[`sensor_alert_${event.severity}`] ?? CARD_CONFIG.sensor_alert_info
  }
  return CARD_CONFIG[event.event_type] ?? CARD_CONFIG.llm_response
}

// ---- sensor pill ----
const DOT = { green: 'bg-green-500', orange: 'bg-amber-500', red: 'bg-red-500', gray: 'bg-stone-400', blue: 'bg-blue-400' }
function Pill({ color = 'gray', children }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-stone-50 dark:bg-stone-700 text-stone-600 dark:text-stone-300 border border-stone-200 dark:border-stone-600">
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT[color] ?? DOT.gray}`} />
      {children}
    </span>
  )
}

// ---- markdown components (no prose plugin needed) ----
const MD = {
  p:      ({ children }) => <p className="mb-2 last:mb-0 text-sm leading-relaxed">{children}</p>,
  ul:     ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1 text-sm">{children}</ul>,
  ol:     ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1 text-sm">{children}</ol>,
  li:     ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-stone-800 dark:text-stone-100">{children}</strong>,
  em:     ({ children }) => <em className="italic">{children}</em>,
  code:   ({ children }) => <code className="bg-stone-100 dark:bg-stone-700 px-1 py-0.5 rounded text-xs font-mono">{children}</code>,
  h1:     ({ children }) => <h1 className="text-base font-semibold mb-1 mt-2">{children}</h1>,
  h2:     ({ children }) => <h2 className="text-sm font-semibold mb-1 mt-2">{children}</h2>,
  h3:     ({ children }) => <h3 className="text-sm font-medium mb-1 mt-1">{children}</h3>,
}

// ---- event card ----
function EventCard({ event }) {
  const [expanded, setExpanded] = useState(false)
  const config = getConfig(event)
  const sources = event.sources
    ? (Array.isArray(event.sources) ? event.sources : JSON.parse(event.sources))
    : []
  const hasBody = event.user_query || event.llm_response || event.sensor_context_filtered || sources.length > 0

  // meta pills — routing info, response time, template
  const pills = []
  if (event.routing_mode)     pills.push({ color: 'gray',  label: `routing: ${event.routing_mode}` })
  if (event.routing_decision) pills.push({ color: event.routing_decision?.toLowerCase() === 'include' ? 'green' : 'gray', label: event.routing_decision })
  if (event.prompt_template)  pills.push({ color: 'blue',  label: event.prompt_template })
  if (event.response_time_ms) pills.push({ color: 'gray',  label: `${event.response_time_ms} ms` })

  const mainText = event.user_query || (!event.llm_response && event.sensor_context_filtered)

  return (
    <div className={`rounded-xl bg-white dark:bg-stone-800 overflow-hidden shadow-sm hover:shadow-md transition-all duration-200 border border-stone-200 dark:border-stone-700 border-l-4 ${config.border}`}>

      {/* ── header (always visible) ── */}
      <div
        className={`px-5 py-4 ${hasBody ? 'cursor-pointer hover:bg-stone-50 dark:hover:bg-stone-700/30 transition-colors' : ''}`}
        onClick={() => hasBody && setExpanded(e => !e)}
      >
        {/* type row */}
        <div className="flex items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2">
            <span className="text-base leading-none">{config.icon}</span>
            <span className={`text-xs font-semibold uppercase tracking-wider ${config.title}`}>{config.label}</span>
          </div>
          <span className={`text-xs font-semibold px-3 py-0.5 rounded-full shrink-0 ${config.badge}`}>
            {config.badgeText}
          </span>
        </div>

        {/* main text */}
        {mainText && (
          <p className="text-sm text-stone-700 dark:text-stone-200 leading-relaxed line-clamp-2">
            {mainText}
          </p>
        )}

        {/* pills */}
        {pills.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-3">
            {pills.map((p, i) => <Pill key={i} color={p.color}>{p.label}</Pill>)}
          </div>
        )}

        {/* timestamp + expand indicator */}
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-stone-400 dark:text-stone-500">
            {new Date(event.timestamp).toLocaleString()}
          </span>
          {hasBody && (
            <span className="text-xs text-stone-400 dark:text-stone-500">{expanded ? '▲ less' : '▼ more'}</span>
          )}
        </div>
      </div>

      {/* ── expanded body ── */}
      {expanded && (
        <div className="px-5 pb-5 pt-4 border-t border-stone-100 dark:border-stone-700 space-y-4">
          {event.user_query && event.event_type !== 'llm_response' && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1.5">Query</p>
              <p className="text-sm text-stone-700 dark:text-stone-200 leading-relaxed">{event.user_query}</p>
            </div>
          )}
          {event.sensor_context_filtered && event.event_type !== 'llm_response' && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1.5">Sensor Context</p>
              <pre className="text-xs text-stone-600 dark:text-stone-300 whitespace-pre-wrap leading-relaxed bg-stone-50 dark:bg-stone-700/50 rounded-lg p-3 overflow-x-auto">
                {event.sensor_context_filtered}
              </pre>
            </div>
          )}
          {event.llm_response && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1.5">AI Response</p>
              <div className="text-stone-700 dark:text-stone-200">
                <ReactMarkdown components={MD}>{event.llm_response}</ReactMarkdown>
              </div>
            </div>
          )}
          {sources.length > 0 && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1.5">Sources</p>
              <div className="flex flex-wrap gap-1.5">
                {sources.map((src, i) => (
                  <span key={i} className="text-xs text-stone-500 dark:text-stone-400 font-mono bg-stone-50 dark:bg-stone-700/50 border border-stone-200 dark:border-stone-700 rounded-lg px-2 py-0.5">
                    {src}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---- filter bar ----
const FILTERS = [
  { value: null,                label: 'All' },
  { value: 'sensor_alert',      label: 'Alerts' },
  { value: 'automation_action', label: 'Automation' },
  { value: 'conditions_normal', label: 'All Clear' },
  { value: 'llm_response',      label: 'Q&A' },
]

export default function AlertFeed() {
  const [events, setEvents]       = useState([])
  const [loading, setLoading]     = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError]         = useState(null)
  const [filter, setFilter]       = useState(null)

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

        {/* header */}
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <div>
            <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Event Log</h2>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">Sensor alerts and AI responses · refreshes every 10 s</p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              className="text-xs border border-stone-300 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 hover:border-stone-400 bg-white dark:bg-stone-800 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
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
            {[1, 2, 3].map(i => (
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
