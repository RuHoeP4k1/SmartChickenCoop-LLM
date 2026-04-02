import { useState, useEffect } from 'react'
import { getReviewEvents, submitReview, exportReviews } from '../api'

const SEV_COLOR = { critical: 'text-red-500 dark:text-red-400', warning: 'text-amber-600 dark:text-amber-400', info: 'text-stone-400 dark:text-stone-500' }
const CARD_BORDER = { critical: 'border-red-300 dark:border-red-700', warning: 'border-amber-300 dark:border-amber-700', info: 'border-stone-200 dark:border-stone-700' }

const FILTERS = [
  { value: null,    label: 'All' },
  { value: 'false', label: 'Unreviewed' },
  { value: 'true',  label: 'Reviewed' },
]

function ReviewCard({ event, onSaved }) {
  const [expanded, setExpanded] = useState(false)
  const [isGood, setIsGood] = useState(event.is_good)
  const [routingCorrect, setRoutingCorrect] = useState(event.routing_correct)
  const [notes, setNotes] = useState(event.review_notes || '')
  const [saving, setSaving] = useState(false)

  const hasReview = event.review_id != null
  const sources = event.sources ? (Array.isArray(event.sources) ? event.sources : JSON.parse(event.sources)) : []
  const hasSensors = event.routing_decision === 'include'

  async function save(good) {
    setSaving(true)
    try {
      await submitReview({
        event_id: event.id,
        is_good: good,
        routing_correct: routingCorrect,
        notes,
      })
      setIsGood(good)
      onSaved()
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className={`rounded-2xl border bg-white dark:bg-stone-800 overflow-hidden transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 ${CARD_BORDER[event.severity] ?? CARD_BORDER.info}`}>
      {/* Header */}
      <div
        className="px-4 py-3 flex items-start justify-between gap-3 cursor-pointer hover:bg-stone-50 dark:hover:bg-stone-700/50 transition-colors"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-start gap-3 min-w-0">
          <span className={`mt-0.5 shrink-0 text-base ${SEV_COLOR[event.severity] ?? SEV_COLOR.info}`}>
            {event.severity === 'critical' ? '!' : 'i'}
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              {/* Review status badge */}
              {isGood === true && <span className="text-xs font-semibold px-2 py-0.5 rounded-md border bg-green-100 text-green-700 border-green-300 dark:bg-green-900/30 dark:text-green-400 dark:border-green-700">Good</span>}
              {isGood === false && <span className="text-xs font-semibold px-2 py-0.5 rounded-md border bg-red-100 text-red-700 border-red-300 dark:bg-red-900/30 dark:text-red-400 dark:border-red-700">Bad</span>}
              {isGood == null && <span className="text-xs font-semibold px-2 py-0.5 rounded-md border bg-stone-100 text-stone-500 border-stone-300 dark:bg-stone-700 dark:text-stone-400 dark:border-stone-600">Unrated</span>}
              {/* Routing badge */}
              {hasSensors && <span className="text-xs px-2 py-0.5 rounded-md border bg-blue-50 text-blue-600 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-700">Sensors</span>}
              {event.prompt_template && <span className="text-xs px-2 py-0.5 rounded-md border bg-stone-50 text-stone-500 border-stone-200 dark:bg-stone-700 dark:text-stone-400 dark:border-stone-600">{event.prompt_template}</span>}
              {event.response_time_ms != null && <span className="text-xs text-stone-400 dark:text-stone-500">{event.response_time_ms}ms</span>}
            </div>
            <p className="text-sm text-stone-700 dark:text-stone-200 truncate">{event.user_query}</p>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <p className="text-xs text-stone-400 dark:text-stone-500">{new Date(event.timestamp).toLocaleString()}</p>
          <p className="text-xs text-stone-300 dark:text-stone-600 mt-1">{expanded ? 'less' : 'more'}</p>
        </div>
      </div>

      {/* Expanded body */}
      {expanded && (
        <div className="px-4 pb-4 pt-3 border-t border-stone-100 dark:border-stone-700 space-y-3">
          {/* Query */}
          <div>
            <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Query</p>
            <p className="text-sm text-stone-700 dark:text-stone-200 leading-relaxed">{event.user_query}</p>
          </div>

          {/* Sensor context */}
          {event.sensor_context_filtered && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Sensor Context</p>
              <p className="text-sm text-stone-600 dark:text-stone-300 leading-relaxed whitespace-pre-wrap">{event.sensor_context_filtered}</p>
            </div>
          )}

          {/* AI Response */}
          {event.llm_response && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">AI Response</p>
              <p className="text-sm text-stone-700 dark:text-stone-200 leading-relaxed whitespace-pre-wrap">{event.llm_response}</p>
            </div>
          )}

          {/* Sources */}
          {sources.length > 0 && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Sources</p>
              <div className="flex flex-wrap gap-1">
                {sources.map((src, i) => (
                  <span key={i} className="text-xs text-stone-500 dark:text-stone-400 font-mono bg-stone-50 dark:bg-stone-700/50 border border-stone-200 dark:border-stone-700 rounded px-2 py-0.5">{src}</span>
                ))}
              </div>
            </div>
          )}

          {/* Routing metadata */}
          {event.routing_mode && (
            <div>
              <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Routing</p>
              <p className="text-xs text-stone-500 dark:text-stone-400">
                Mode: <span className="font-mono">{event.routing_mode}</span> | Decision: <span className="font-mono">{event.routing_decision}</span>
              </p>
            </div>
          )}

          {/* Rating form */}
          <div className="border-t border-stone-100 dark:border-stone-700 pt-3 space-y-3">
            <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium">Rate this response</p>

            <div className="flex items-center gap-2">
              <button
                onClick={() => save(true)}
                disabled={saving}
                className={`text-sm px-4 py-1.5 rounded-lg border transition-colors ${
                  isGood === true
                    ? 'bg-green-100 border-green-400 text-green-700 dark:bg-green-900/40 dark:border-green-600 dark:text-green-400'
                    : 'border-stone-300 dark:border-stone-600 text-stone-500 dark:text-stone-400 hover:bg-green-50 dark:hover:bg-green-900/20'
                }`}
              >
                Good
              </button>
              <button
                onClick={() => save(false)}
                disabled={saving}
                className={`text-sm px-4 py-1.5 rounded-lg border transition-colors ${
                  isGood === false
                    ? 'bg-red-100 border-red-400 text-red-700 dark:bg-red-900/40 dark:border-red-600 dark:text-red-400'
                    : 'border-stone-300 dark:border-stone-600 text-stone-500 dark:text-stone-400 hover:bg-red-50 dark:hover:bg-red-900/20'
                }`}
              >
                Bad
              </button>

              {hasSensors && (
                <div className="flex items-center gap-1 ml-4">
                  <span className="text-xs text-stone-400 dark:text-stone-500">Routing correct?</span>
                  <button
                    onClick={() => setRoutingCorrect(routingCorrect === true ? null : true)}
                    className={`text-xs px-2 py-1 rounded border transition-colors ${
                      routingCorrect === true
                        ? 'bg-green-100 border-green-400 text-green-700 dark:bg-green-900/40 dark:border-green-600 dark:text-green-400'
                        : 'border-stone-300 dark:border-stone-600 text-stone-400'
                    }`}
                  >Y</button>
                  <button
                    onClick={() => setRoutingCorrect(routingCorrect === false ? null : false)}
                    className={`text-xs px-2 py-1 rounded border transition-colors ${
                      routingCorrect === false
                        ? 'bg-red-100 border-red-400 text-red-700 dark:bg-red-900/40 dark:border-red-600 dark:text-red-400'
                        : 'border-stone-300 dark:border-stone-600 text-stone-400'
                    }`}
                  >N</button>
                </div>
              )}
            </div>

            <textarea
              value={notes}
              onChange={e => setNotes(e.target.value)}
              placeholder="Optional notes..."
              rows={2}
              className="w-full text-sm border border-stone-300 dark:border-stone-600 rounded-lg px-3 py-2 bg-white dark:bg-stone-900 text-stone-700 dark:text-stone-200 placeholder:text-stone-400 dark:placeholder:text-stone-600 resize-none"
            />

            {notes !== (event.review_notes || '') && isGood != null && (
              <button
                onClick={() => save(isGood)}
                disabled={saving}
                className="text-xs border border-stone-300 dark:border-stone-600 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 bg-white dark:bg-stone-800 rounded-lg px-3 py-1.5 transition-colors"
              >
                Save notes
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function downloadCSV(data) {
  if (!data.length) return
  const keys = Object.keys(data[0])
  const csv = [
    keys.join(','),
    ...data.map(row => keys.map(k => {
      const v = row[k]
      if (v == null) return ''
      const s = typeof v === 'object' ? JSON.stringify(v) : String(v)
      return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replace(/"/g, '""')}"` : s
    }).join(','))
  ].join('\n')

  const blob = new Blob([csv], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `response_reviews_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

export default function ResponseReview() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState(null)

  async function load(showSpinner = false) {
    if (showSpinner) setRefreshing(true)
    try {
      const result = await getReviewEvents(50, filter)
      setEvents(result.events)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  async function handleExport() {
    try {
      const result = await exportReviews()
      downloadCSV(result.data)
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    setLoading(true)
    load()
  }, [filter])

  const reviewed = events.filter(e => e.review_id != null).length
  const total = events.length

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Response Review</h2>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">
              Rate live Q&A responses for paper evaluation · {reviewed}/{total} reviewed
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={handleExport}
              className="text-xs border border-stone-300 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 hover:border-stone-400 dark:hover:border-stone-500 bg-white dark:bg-stone-800 rounded-lg px-3 py-1.5 transition-colors"
            >
              Export CSV
            </button>
            <button
              onClick={() => load(true)}
              disabled={refreshing}
              className="text-xs border border-stone-300 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-100 hover:border-stone-400 dark:hover:border-stone-500 bg-white dark:bg-stone-800 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
            >
              {refreshing ? 'Loading...' : 'Refresh'}
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
              <div key={i} className="rounded-2xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-4">
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
          <div className="text-center text-stone-400 dark:text-stone-500 py-24 text-sm">
            No responses to review yet. Ask questions via Chat to generate data.
          </div>
        )}

        <div className="space-y-3 animate-fade-in">
          {events.map(event => <ReviewCard key={event.id} event={event} onSaved={() => load()} />)}
        </div>
      </div>
    </div>
  )
}
