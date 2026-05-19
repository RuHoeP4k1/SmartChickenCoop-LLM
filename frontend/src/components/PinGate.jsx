import { useState } from 'react'

const PIN = '1234'
const SESSION_KEY = 'automation_unlocked'

export default function PinGate({ children }) {
  const [unlocked, setUnlocked] = useState(
    () => sessionStorage.getItem(SESSION_KEY) === '1'
  )
  const [input, setInput] = useState('')
  const [shake, setShake] = useState(false)

  if (unlocked) return children

  function handleDigit(d) {
    const next = input + d
    if (next.length < PIN.length) {
      setInput(next)
      return
    }
    if (next === PIN) {
      sessionStorage.setItem(SESSION_KEY, '1')
      setUnlocked(true)
    } else {
      setShake(true)
      setInput('')
      setTimeout(() => setShake(false), 500)
    }
  }

  function handleDelete() {
    setInput(prev => prev.slice(0, -1))
  }

  const dots = Array.from({ length: PIN.length }, (_, i) => i < input.length)

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-4">
      <div className="text-center">
        <div className="text-2xl mb-1">🔒</div>
        <h2 className="text-lg font-semibold text-stone-700 dark:text-stone-200">Automation Panel</h2>
        <p className="text-sm text-stone-400 dark:text-stone-500 mt-1">Enter PIN to continue</p>
      </div>

      {/* dots */}
      <div className={`flex gap-4 ${shake ? 'animate-shake' : ''}`}>
        {dots.map((filled, i) => (
          <div
            key={i}
            className={`w-4 h-4 rounded-full border-2 transition-colors ${
              filled
                ? 'bg-amber-500 border-amber-500'
                : 'bg-transparent border-stone-300 dark:border-stone-600'
            }`}
          />
        ))}
      </div>

      {/* numpad */}
      <div className="grid grid-cols-3 gap-3">
        {[1,2,3,4,5,6,7,8,9].map(d => (
          <button
            key={d}
            onClick={() => handleDigit(String(d))}
            className="w-16 h-16 rounded-xl text-xl font-medium bg-stone-100 dark:bg-stone-700 text-stone-800 dark:text-stone-100 hover:bg-stone-200 dark:hover:bg-stone-600 active:scale-95 transition-all"
          >
            {d}
          </button>
        ))}
        <div />
        <button
          onClick={() => handleDigit('0')}
          className="w-16 h-16 rounded-xl text-xl font-medium bg-stone-100 dark:bg-stone-700 text-stone-800 dark:text-stone-100 hover:bg-stone-200 dark:hover:bg-stone-600 active:scale-95 transition-all"
        >
          0
        </button>
        <button
          onClick={handleDelete}
          className="w-16 h-16 rounded-xl text-xl bg-stone-100 dark:bg-stone-700 text-stone-500 dark:text-stone-400 hover:bg-stone-200 dark:hover:bg-stone-600 active:scale-95 transition-all flex items-center justify-center"
        >
          ⌫
        </button>
      </div>
    </div>
  )
}
