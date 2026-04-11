import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { askQuestion } from '../api'

/* ── Icons ───────────────────────────────────────────────────────── */
function SendIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  )
}

function ChickenAvatarIcon() {
  return (
    <div className="w-8 h-8 rounded-full bg-amber-100 dark:bg-amber-900/40 border border-amber-200 dark:border-amber-700/50 flex items-center justify-center shrink-0 text-base select-none">
      🐔
    </div>
  )
}

/* ── Suggested prompt chips (shown only in empty state) ──────────── */
const SUGGESTED_PROMPTS = [
  'How are my chickens doing?',
  'Is the coop temperature okay?',
  'Any alerts I should know about?',
]

/* ── Message bubbles ─────────────────────────────────────────────── */
function UserMessage({ content }) {
  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-xl bg-amber-600 dark:bg-amber-700 text-white rounded-xl rounded-br-sm px-4 py-3 text-sm leading-relaxed shadow-sm">
        {content}
      </div>
    </div>
  )
}

function AssistantMessage({ msg, isWelcome }) {
  const [openSource, setOpenSource] = useState(null)

  const uniqueSources = msg.sources ? [...new Set(msg.sources)] : []

  function toggleSource(src) {
    setOpenSource(prev => (prev === src ? null : src))
  }

  const visibleChunks = openSource && msg.chunks
    ? msg.chunks.filter(c => c.source === openSource)
    : []

  return (
    <div className={`flex items-start gap-3 mb-4 ${isWelcome ? 'justify-center' : 'justify-start'}`}>
      {!isWelcome && <ChickenAvatarIcon />}
      <div className="max-w-2xl w-full">
        {/* Bubble */}
        <div className={`rounded-xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed border shadow-sm ${
          msg.has_critical
            ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700/50'
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

        {/* Source chips */}
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
              <div key={i} className="bg-stone-50 dark:bg-stone-800/60 border border-stone-200 dark:border-stone-700 rounded-xl px-3 py-2.5">
                <p className="text-xs font-semibold text-amber-700 dark:text-amber-400 mb-1">
                  {chunk.source} — chunk {i + 1}
                </p>
                <ReactMarkdown
                  className="text-xs text-stone-600 dark:text-stone-300 leading-relaxed"
                  components={{
                    p:    ({ children }) => <p className="mb-1.5 last:mb-0">{children}</p>,
                    code: ({ children }) => <code className="bg-stone-100 dark:bg-stone-700 text-amber-700 dark:text-amber-300 rounded px-1 font-mono border border-stone-200 dark:border-stone-600">{children}</code>,
                  }}
                >
                  {chunk.content}
                </ReactMarkdown>
              </div>
            ))}
          </div>
        )}

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
    <div className="flex items-start gap-3 mb-4">
      <ChickenAvatarIcon />
      <div
        aria-label="ChatKippieTee is thinking…"
        className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-xl rounded-bl-sm px-4 py-3 flex items-center gap-2 shadow-sm"
      >
        {[0, 150, 300].map(delay => (
          <span
            key={delay}
            className="w-2 h-2 bg-amber-400 rounded-full animate-bounce"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
        <span className="text-xs text-stone-400 dark:text-stone-500 ml-1">ChatKippieTee is thinking…</span>
      </div>
    </div>
  )
}

/* ── Empty state (only welcome message) ──────────────────────────── */
function EmptyState({ onSuggest }) {
  return (
    <div className="flex flex-col items-center" style={{ paddingTop: '10%' }}>
      {/* Avatar + welcome bubble */}
      <div className="flex items-start gap-3 w-full max-w-xl">
        <div className="w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/40 border border-amber-200 dark:border-amber-700/50 flex items-center justify-center shrink-0 text-xl select-none mt-0.5">
          🐔
        </div>
        <div className="bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed shadow-sm text-stone-800 dark:text-stone-200">
          Welcome to <span className="font-semibold">ChickenCoopComfort</span>! I'm ChatKippieTee, your personal chickencoop assistant — always here to give insights about chickenkeeping and tell you what's going on with your flock. Ask me anything about health, coop conditions, feed, behaviour, or general chicken-keeping advice.
        </div>
      </div>

      {/* Suggested prompts */}
      <div className="flex flex-wrap justify-center gap-2 mt-6 max-w-lg">
        {SUGGESTED_PROMPTS.map(prompt => (
          <button
            key={prompt}
            onClick={() => onSuggest(prompt)}
            className="border border-stone-200 dark:border-stone-600 rounded-full px-4 py-2 text-sm text-stone-600 dark:text-stone-300 hover:bg-amber-50 dark:hover:bg-amber-900/20 hover:border-amber-200 dark:hover:border-amber-700/50 hover:text-amber-700 dark:hover:text-amber-400 transition-colors cursor-pointer bg-white dark:bg-stone-800 shadow-sm"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ── Status toggle pill ──────────────────────────────────────────── */
function StatusPill({ active, color, label, onClick }) {
  const colors = {
    amber: {
      on:  'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-700/50',
      dot: 'bg-amber-500',
      off: 'bg-stone-100 dark:bg-stone-700/50 text-stone-400 dark:text-stone-500 border-stone-200 dark:border-stone-600/50',
    },
    emerald: {
      on:  'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-700/50',
      dot: 'bg-emerald-500',
      off: 'bg-stone-100 dark:bg-stone-700/50 text-stone-400 dark:text-stone-500 border-stone-200 dark:border-stone-600/50',
    },
  }
  const c = colors[color]
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border transition-colors ${active ? c.on : c.off}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${active ? c.dot : 'bg-stone-300 dark:bg-stone-500'}`} />
      {label}
    </button>
  )
}

/* ── Main panel ──────────────────────────────────────────────────── */
export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '__welcome__', // sentinel — renders EmptyState instead
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [useSensors, setUseSensors] = useState(true)
  const [useTaskContext, setUseTaskContext] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  const isEmptyState = messages.length === 1 && messages[0].content === '__welcome__'

  useEffect(() => {
    if (!isEmptyState) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, loading, isEmptyState])

  function handleSuggest(prompt) {
    setInput(prompt)
    inputRef.current?.focus()
  }

  async function submit(e) {
    e?.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    // Replace welcome sentinel with a real conversation
    const prevMessages = isEmptyState ? [] : messages.slice(1)
    const history = prevMessages.map(m => ({ role: m.role, content: m.content }))

    setInput('')
    setMessages(prev => {
      const base = isEmptyState ? [] : prev
      return [...base, { role: 'user', content: query }]
    })
    setLoading(true)

    try {
      const result = await askQuestion(query, history, useSensors, true, useTaskContext)
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

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const hasInput = input.trim().length > 0

  return (
    <div className="h-full flex flex-col bg-stone-50 dark:bg-stone-900">
      {/* Status bar */}
      <div className="shrink-0 flex items-center gap-2 px-4 py-2 border-b border-stone-200 dark:border-stone-700/50 bg-white dark:bg-stone-800">
        <span className="text-xs text-stone-400 dark:text-stone-500 font-medium mr-1">Context:</span>
        <StatusPill
          active={useSensors}
          color="amber"
          label={useSensors ? 'Live sensors' : 'Sensors off'}
          onClick={() => setUseSensors(p => !p)}
        />
        <StatusPill
          active={useTaskContext}
          color="emerald"
          label={useTaskContext ? 'Task history' : 'Task history off'}
          onClick={() => setUseTaskContext(p => !p)}
          title="Include recent chores & automation windows in the next message"
        />
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto">
          {isEmptyState ? (
            <EmptyState onSuggest={handleSuggest} />
          ) : (
            messages.map((msg, i) =>
              msg.role === 'user'
                ? <UserMessage key={i} content={msg.content} />
                : <AssistantMessage key={i} msg={msg} />
            )
          )}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="shrink-0 bg-white dark:bg-stone-800 border-t border-stone-200 dark:border-stone-700/50 px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className={`flex items-center gap-2 bg-stone-50 dark:bg-stone-700/50 border rounded-xl shadow-sm p-1.5 transition-colors ${
            hasInput
              ? 'border-amber-300 dark:border-amber-600/50'
              : 'border-stone-200 dark:border-stone-600'
          }`}>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your flock…"
              disabled={loading}
              className="flex-1 bg-transparent border-0 focus:outline-none focus:ring-0 text-sm text-stone-800 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 disabled:opacity-50 px-2"
            />
            <button
              type="button"
              onClick={submit}
              disabled={loading || !hasInput}
              className={`p-2 rounded-lg transition-all duration-150 active:scale-95 shrink-0 ${
                hasInput && !loading
                  ? 'bg-amber-600 hover:bg-amber-500 text-white'
                  : 'bg-stone-200 dark:bg-stone-600 text-stone-400 dark:text-stone-500 cursor-not-allowed'
              }`}
            >
              <SendIcon className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
