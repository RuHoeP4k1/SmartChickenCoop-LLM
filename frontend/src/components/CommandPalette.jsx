import { useState, useEffect, useRef } from 'react'

const QUICK_QUESTIONS = [
  'How are my chickens doing?',
  'Is the coop temperature okay?',
  'Any alerts I should know about?',
]

export default function CommandPalette({ isOpen, onClose, tabs, activeTab, onTabChange }) {
  const [query, setQuery] = useState('')
  const dialogRef = useRef(null)
  const inputRef = useRef(null)
  const onCloseRef = useRef(onClose)
  const closingFromCode = useRef(false)
  useEffect(() => { onCloseRef.current = onClose })

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    let focusId
    if (isOpen) {
      if (!dialog.open) dialog.showModal()
      setQuery('')
      // Defer focus so the dialog is visible first
      focusId = setTimeout(() => inputRef.current?.focus(), 0)
    } else if (dialog.open) {
      closingFromCode.current = true
      dialog.close()
    }
    return () => clearTimeout(focusId)
  }, [isOpen])

  // Only invoke onClose when the dialog closes due to user action (Escape),
  // not when we close it programmatically via closingFromCode.
  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    function handleClose() {
      if (closingFromCode.current) { closingFromCode.current = false; return }
      onCloseRef.current()
    }
    dialog.addEventListener('close', handleClose)
    return () => dialog.removeEventListener('close', handleClose)
  }, [])

  function handleBackdropClick(e) {
    // Click landed on the <dialog> itself (the backdrop area), not inside the panel
    if (e.target === dialogRef.current) onClose()
  }

  const navTabs = tabs.filter(t => !t.separator && t.id && t.label)
  const filtered = query.trim()
    ? navTabs.filter(t => t.label.toLowerCase().includes(query.toLowerCase()))
    : navTabs

  function navigate(tabId) {
    onTabChange(tabId)
    onClose()
  }

  return (
    <dialog
      ref={dialogRef}
      onClick={handleBackdropClick}
      aria-label="Command palette"
      style={{ margin: 0, padding: 0, width: '100vw', height: '100vh', maxWidth: 'none', maxHeight: 'none', border: 'none', background: 'transparent' }}
      className="backdrop:bg-black/40"
    >
      <div className="flex justify-center" style={{ marginTop: '15vh' }}>
        {/* Panel — stop click propagation so backdrop click doesn't fire */}
        <div
          className="bg-white dark:bg-stone-800 rounded-xl shadow-2xl max-w-md w-full mx-4 overflow-hidden"
          onClick={e => e.stopPropagation()}
        >
          {/* Search input */}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search tabs…"
            className="w-full border-b border-stone-200 dark:border-stone-700 px-4 py-3 text-sm bg-transparent text-stone-800 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none"
          />

          <div className="py-1 max-h-80 overflow-y-auto">
            {/* Navigate section */}
            {filtered.length > 0 && (
              <>
                <p className="px-4 py-1 text-[10px] uppercase tracking-wider text-stone-400 dark:text-stone-500">
                  Navigate
                </p>
                {filtered.map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => navigate(tab.id)}
                    className={`w-full text-left px-4 py-2.5 hover:bg-stone-50 dark:hover:bg-stone-700/50 rounded-lg text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'text-amber-600 dark:text-amber-400 font-medium'
                        : 'text-stone-700 dark:text-stone-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </>
            )}

            {/* Quick questions — only shown when not filtering */}
            {!query.trim() && (
              <>
                <p className="px-4 py-1 text-[10px] uppercase tracking-wider text-stone-400 dark:text-stone-500 mt-1">
                  Quick questions
                </p>
                {QUICK_QUESTIONS.map(q => (
                  <button
                    key={q}
                    onClick={() => navigate('chat')}
                    className="w-full text-left px-4 py-2.5 hover:bg-stone-50 dark:hover:bg-stone-700/50 rounded-lg text-sm text-stone-700 dark:text-stone-200 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
    </dialog>
  )
}
