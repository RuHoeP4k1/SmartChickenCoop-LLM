import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { askQuestion } from '../api'

/* ── tiny send icon ──────────────────────────────────────────────── */
function SendIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 2L11 13" />
      <path d="M22 2L15 22l-4-9-9-4z" />
    </svg>
  )
}

/* ── message bubbles ─────────────────────────────────────────────── */

function UserMessage({ content }) {
  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-xl bg-amber-600 dark:bg-amber-700 text-white rounded-2xl rounded-br-sm px-4 py-3 text-sm leading-relaxed shadow-sm">
        {content}
      </div>
    </div>
  )
}

function AssistantMessage({ msg }) {
  const [openSource, setOpenSource] = useState(null)

  const uniqueSources = msg.sources ? [...new Set(msg.sources)] : []

  function toggleSource(src) {
    setOpenSource(prev => (prev === src ? null : src))
  }

  // All chunks for the currently-open source
  const visibleChunks = openSource && msg.chunks
    ? msg.chunks.filter(c => c.source === openSource)
    : []

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-2xl w-full">
        {/* Message bubble */}
        <div className={`rounded-2xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed border shadow-sm ${
          msg.has_critical
            ? 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700/50'
            : 'bg-white dark:bg-stone-800 border-stone-200 dark:border-stone-700'
        }`}>
          {msg.has_critical && (
            <div className="flex items-center gap-1.5 text-red-600 dark:text-red-400 text-xs font-semibold uppercase tracking-wider mb-2 pb-2 border-b border-red-200 dark:border-red-700/50">
              <span>⚠</span>
              <span>Critical alert active</span>
            </div>
          )}
          <ReactMarkdown
            className="text-stone-800 dark:text-stone-200 text-sm leading-relaxed"
            components={{
              p:      ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="font-semibold text-stone-900 dark:text-stone-100">{children}</strong>,
              em:     ({ children }) => <em className="italic text-stone-600 dark:text-stone-400">{children}</em>,
              ul:     ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
              ol:     ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
              li:     ({ children }) => <li className="text-stone-700 dark:text-stone-300">{children}</li>,
              h1:     ({ children }) => <h1 className="text-base font-bold text-stone-900 dark:text-stone-100 mb-1">{children}</h1>,
              h2:     ({ children }) => <h2 className="text-sm font-bold text-stone-900 dark:text-stone-100 mb-1">{children}</h2>,
              h3:     ({ children }) => <h3 className="text-sm font-semibold text-stone-700 dark:text-stone-300 mb-1">{children}</h3>,
              code:   ({ children }) => <code className="bg-stone-100 dark:bg-stone-700 text-amber-700 dark:text-amber-300 rounded px-1 text-xs font-mono border border-stone-200 dark:border-stone-600">{children}</code>,
              pre:    ({ children }) => <pre className="bg-stone-100 dark:bg-stone-700 border border-stone-200 dark:border-stone-600 rounded-lg p-3 text-xs font-mono overflow-x-auto my-2">{children}</pre>,
            }}
          >
            {msg.content}
          </ReactMarkdown>
        </div>

        {/* Source chips — clickable, toggle chunk drawer */}
        {uniqueSources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5 px-1">
            {uniqueSources.map(src => (
              <button
                key={src}
                onClick={() => toggleSource(src)}
                className={`text-xs border rounded-full px-2.5 py-0.5 transition-colors ${
                  openSource === src
                    ? 'bg-amber-200 dark:bg-amber-700/50 text-amber-800 dark:text-amber-200 border-amber-400 dark:border-amber-500'
                    : 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-700/40 hover:bg-amber-100 dark:hover:bg-amber-800/30'
                }`}
              >
                {src} {openSource === src ? '▲' : '▼'}
              </button>
            ))}
          </div>
        )}

        {/* Chunk drawer */}
        {visibleChunks.length > 0 && (
          <div className="mt-2 px-1 space-y-2">
            {visibleChunks.map((chunk, i) => (
              <div
                key={i}
                className="bg-stone-50 dark:bg-stone-800/60 border border-stone-200 dark:border-stone-700 rounded-xl px-3 py-2.5"
              >
                <p className="text-xs font-semibold text-amber-700 dark:text-amber-400 mb-1">
                  {chunk.source} — chunk {i + 1}
                </p>
                <ReactMarkdown
                  className="text-xs text-stone-600 dark:text-stone-300 leading-relaxed"
                  components={{
                    p:      ({ children }) => <p className="mb-1.5 last:mb-0">{children}</p>,
                    strong: ({ children }) => <strong className="font-semibold text-stone-800 dark:text-stone-200">{children}</strong>,
                    em:     ({ children }) => <em className="italic">{children}</em>,
                    ul:     ({ children }) => <ul className="list-disc list-inside mb-1.5 space-y-0.5">{children}</ul>,
                    ol:     ({ children }) => <ol className="list-decimal list-inside mb-1.5 space-y-0.5">{children}</ol>,
                    li:     ({ children }) => <li>{children}</li>,
                    code:   ({ children }) => <code className="bg-stone-100 dark:bg-stone-700 text-amber-700 dark:text-amber-300 rounded px-1 font-mono border border-stone-200 dark:border-stone-600">{children}</code>,
                    pre:    ({ children }) => <pre className="bg-stone-100 dark:bg-stone-700 border border-stone-200 dark:border-stone-600 rounded-lg p-2 font-mono overflow-x-auto my-1.5">{children}</pre>,
                  }}
                >
                  {chunk.content}
                </ReactMarkdown>
              </div>
            ))}
          </div>
        )}

        {/* Sensor summary line */}
        {msg.sensor_context && (
          <p className="text-xs text-stone-400 dark:text-stone-500 mt-1.5 px-1 leading-relaxed">
            {msg.sensor_context.replace('Current coop conditions:\n', '')}
          </p>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex justify-start mb-4">
      <div className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
        {[0, 150, 300].map(delay => (
          <span
            key={delay}
            className="w-2 h-2 bg-amber-400 rounded-full animate-bounce"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  )
}

/* ── main panel ──────────────────────────────────────────────────── */

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Welcome to ChickenCoopComfort! I'm ChatKippieTee, your personal chickencoop assistant — always here to give insights about chickenkeeping and tell you what's going on with your flock. Ask me anything about health, coop conditions, feed, behaviour, or general chicken-keeping advice.",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [useSensors, setUseSensors] = useState(true)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function submit(e) {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    const history = messages.slice(1).map(m => ({ role: m.role, content: m.content }))

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      const result = await askQuestion(query, history, useSensors)
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: result.answer,
          sources: result.sources,
          chunks: result.chunks,
          sensor_context: result.sensor_context,
          has_critical: result.has_critical,
        },
      ])
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Could not reach the backend: ${err.message}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-stone-50 dark:bg-stone-900">
      {/* Message thread */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto">
          {messages.map((msg, i) =>
            msg.role === 'user'
              ? <UserMessage key={i} content={msg.content} />
              : <AssistantMessage key={i} msg={msg} />
          )}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-stone-200 dark:border-stone-700/50 bg-white dark:bg-stone-800 px-4 py-4">
        <div className="max-w-3xl mx-auto flex items-center gap-2 mb-2">
          <button
            type="button"
            onClick={() => setUseSensors(prev => !prev)}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border transition-colors ${
              useSensors
                ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-600'
                : 'bg-stone-100 dark:bg-stone-700 text-stone-500 dark:text-stone-400 border-stone-300 dark:border-stone-600'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${useSensors ? 'bg-amber-500' : 'bg-stone-400'}`} />
            Sensors {useSensors ? 'on' : 'off'}
          </button>
        </div>
        <form onSubmit={submit} className="max-w-3xl mx-auto relative">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about your flock…"
            disabled={loading}
            className="w-full bg-stone-50 dark:bg-stone-700/50 border border-stone-300 dark:border-stone-600 rounded-full px-5 py-3 pr-14 text-sm text-stone-800 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 disabled:opacity-50 transition"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 bg-amber-600 hover:bg-amber-500 active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed text-white p-2.5 rounded-full transition-all duration-150"
          >
            <SendIcon className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  )
}
