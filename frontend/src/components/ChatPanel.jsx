import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { askQuestion } from '../api'

function UserMessage({ content }) {
  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-xl bg-amber-700 text-white rounded-2xl rounded-br-sm px-4 py-3 text-sm leading-relaxed">
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
        <div className={`rounded-2xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed border ${
          msg.has_critical
            ? 'bg-red-50 border-red-300'
            : 'bg-white border-stone-200'
        }`}>
          {msg.has_critical && (
            <div className="flex items-center gap-1.5 text-red-600 text-xs font-semibold uppercase tracking-wider mb-2 pb-2 border-b border-red-200">
              <span>⚠</span>
              <span>Critical alert active</span>
            </div>
          )}
          <ReactMarkdown
            className="text-stone-800 text-sm leading-relaxed"
            components={{
              p:      ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="font-semibold text-stone-900">{children}</strong>,
              em:     ({ children }) => <em className="italic text-stone-600">{children}</em>,
              ul:     ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
              ol:     ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
              li:     ({ children }) => <li className="text-stone-700">{children}</li>,
              h1:     ({ children }) => <h1 className="text-base font-bold text-stone-900 mb-1">{children}</h1>,
              h2:     ({ children }) => <h2 className="text-sm font-bold text-stone-900 mb-1">{children}</h2>,
              h3:     ({ children }) => <h3 className="text-sm font-semibold text-stone-700 mb-1">{children}</h3>,
              code:   ({ children }) => <code className="bg-stone-100 text-amber-700 rounded px-1 text-xs font-mono border border-stone-200">{children}</code>,
            }}
          >
            {msg.content}
          </ReactMarkdown>
        </div>

        {/* Source chips */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5 px-1">
            {[...new Set(msg.sources)].map(src => (
              <span
                key={src}
                className="text-xs bg-stone-200 text-stone-500 border border-stone-300 rounded-md px-2 py-0.5"
              >
                {src}
              </span>
            ))}
          </div>
        )}

        {/* Sensor summary line */}
        {msg.sensor_context && (
          <p className="text-xs text-stone-400 mt-1.5 px-1 leading-relaxed">
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
      <div className="bg-white border border-stone-200 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
        {[0, 150, 300].map(delay => (
          <span
            key={delay}
            className="w-2 h-2 bg-stone-400 rounded-full animate-bounce"
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
      content: "Welcome to ChickenCoopComfort! I'm ChatKippieTee, your personal chickencoop assistant — always here to give insights about chickenkeeping and tell you what's going on with your flock. Ask me anything about health, coop conditions, feed, behaviour, or general chicken-keeping advice.",
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

    // Build history from current messages before appending the new user turn.
    // Skip index 0 (static welcome message), send only role + content.
    const history = messages.slice(1).map(m => ({ role: m.role, content: m.content }))

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      const result = await askQuestion(query, history)
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
    <div className="h-full flex flex-col bg-stone-100">
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
      <div className="shrink-0 border-t border-stone-200 bg-white px-4 py-4">
        <form onSubmit={submit} className="max-w-3xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about your flock…"
            disabled={loading}
            className="flex-1 bg-stone-50 border border-stone-300 rounded-xl px-4 py-3 text-sm text-stone-800 placeholder-stone-400 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50 transition"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-amber-600 hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-5 py-3 rounded-xl text-sm font-medium transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
