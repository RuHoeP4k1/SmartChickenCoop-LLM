import { useState, useRef, useEffect } from 'react'
import { askQuestion } from '../api'

function UserMessage({ content }) {
  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-xl bg-green-700 text-white rounded-2xl rounded-br-sm px-4 py-3 text-sm leading-relaxed">
        {content}
      </div>
    </div>
  )
}

function AssistantMessage({ msg }) {
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-2xl w-full">
        {/* Message bubble */}
        <div className={`rounded-2xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed ${
          msg.has_critical
            ? 'bg-red-950/50 border border-red-700/50'
            : 'bg-slate-800'
        }`}>
          {msg.has_critical && (
            <div className="flex items-center gap-1.5 text-red-400 text-xs font-semibold uppercase tracking-wider mb-2 pb-2 border-b border-red-800/40">
              <span>⚠</span>
              <span>Critical alert active</span>
            </div>
          )}
          <p className="text-slate-100 whitespace-pre-wrap">{msg.content}</p>
        </div>

        {/* Source chips */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5 px-1">
            {[...new Set(msg.sources)].map(src => (
              <span
                key={src}
                className="text-xs bg-slate-800 text-slate-500 border border-slate-700 rounded-md px-2 py-0.5"
              >
                {src}
              </span>
            ))}
          </div>
        )}

        {/* Sensor summary line */}
        {msg.sensor_context && (
          <p className="text-xs text-slate-500 mt-1.5 px-1 leading-relaxed">
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
      <div className="bg-slate-800 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
        {[0, 150, 300].map(delay => (
          <span
            key={delay}
            className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  )
}

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi, I'm ChickenGuard. Ask me anything about your flock — health, coop conditions, feed, behaviour, or general chicken-keeping advice.",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function submit(e) {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      const result = await askQuestion(query)
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: result.answer,
          sources: result.sources,
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
    <div className="h-full flex flex-col">
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
      <div className="shrink-0 border-t border-slate-800 bg-slate-900 px-4 py-4">
        <form onSubmit={submit} className="max-w-3xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about your flock…"
            disabled={loading}
            className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-green-600 focus:ring-1 focus:ring-green-600/20 disabled:opacity-50 transition"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-5 py-3 rounded-xl text-sm font-medium transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
