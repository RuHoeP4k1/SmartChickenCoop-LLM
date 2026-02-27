import { useState, useMemo, useRef, useEffect } from 'react'

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const MONTH_NAMES = [
  'January','February','March','April','May','June',
  'July','August','September','October','November','December',
]

function dateKey(year, month, day) {
  return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
}

function cellColor(count) {
  if (!count)    return 'bg-stone-50 border-stone-200'
  if (count <= 2) return 'bg-green-50 border-green-200'
  if (count <= 4) return 'bg-green-100 border-green-300'
  return 'bg-amber-100 border-amber-300'
}

function InlineInput({ value, onSave }) {
  const ref = useRef(null)
  const [val, setVal] = useState(value || '')

  useEffect(() => { ref.current?.focus(); ref.current?.select() }, [])

  function commit() {
    const n = parseInt(val, 10)
    onSave(isNaN(n) || n < 0 ? 0 : n)
  }

  return (
    <input
      ref={ref}
      type="number"
      inputMode="numeric"
      min="0"
      max="99"
      value={val}
      onChange={e => setVal(e.target.value)}
      onBlur={commit}
      onKeyDown={e => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') onSave(value || 0) }}
      className="w-12 text-center bg-white border border-amber-400 rounded-lg px-1 py-0.5 text-sm font-semibold text-stone-800 focus:outline-none focus:ring-2 focus:ring-amber-400/50"
    />
  )
}

export default function EggCalendar() {
  const [month, setMonth] = useState(() => {
    const now = new Date()
    return { year: now.getFullYear(), month: now.getMonth() }
  })
  const [eggData, setEggData] = useState(() => {
    try { return JSON.parse(localStorage.getItem('egg_calendar') || '{}') }
    catch { return {} }
  })
  const [editing, setEditing] = useState(null)

  const today = new Date()
  const todayKey = dateKey(today.getFullYear(), today.getMonth(), today.getDate())

  const daysInMonth = new Date(month.year, month.month + 1, 0).getDate()
  const firstDay = new Date(month.year, month.month, 1).getDay()

  const monthlyTotal = useMemo(() => {
    const prefix = `${month.year}-${String(month.month + 1).padStart(2, '0')}`
    return Object.entries(eggData)
      .filter(([k]) => k.startsWith(prefix))
      .reduce((sum, [, v]) => sum + v, 0)
  }, [eggData, month])

  function saveEgg(key, count) {
    const updated = { ...eggData }
    if (count > 0) updated[key] = count
    else delete updated[key]
    setEggData(updated)
    localStorage.setItem('egg_calendar', JSON.stringify(updated))
    setEditing(null)
  }

  function prevMonth() {
    setMonth(m => m.month === 0
      ? { year: m.year - 1, month: 11 }
      : { year: m.year, month: m.month - 1 })
    setEditing(null)
  }
  function nextMonth() {
    setMonth(m => m.month === 11
      ? { year: m.year + 1, month: 0 }
      : { year: m.year, month: m.month + 1 })
    setEditing(null)
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-100">
      <div className="max-w-2xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-stone-800">Egg Calendar</h2>
            <p className="text-xs text-stone-500 mt-0.5">Tap a day to log how many eggs were collected</p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-amber-600 tabular-nums">{monthlyTotal}</p>
            <p className="text-xs text-stone-400">eggs this month</p>
          </div>
        </div>

        {/* Calendar card */}
        <div className="bg-white border border-stone-200 rounded-2xl p-5">

          {/* Month navigation */}
          <div className="flex items-center justify-between mb-4">
            <button
              onClick={prevMonth}
              className="text-stone-400 hover:text-stone-800 transition-colors px-2 py-1 rounded-lg hover:bg-stone-100 active:scale-95"
            >
              ← Prev
            </button>
            <h3 className="text-sm font-semibold text-stone-800">
              {MONTH_NAMES[month.month]} {month.year}
            </h3>
            <button
              onClick={nextMonth}
              className="text-stone-400 hover:text-stone-800 transition-colors px-2 py-1 rounded-lg hover:bg-stone-100 active:scale-95"
            >
              Next →
            </button>
          </div>

          {/* Day-of-week headers */}
          <div className="grid grid-cols-7 gap-1 mb-1">
            {DAYS.map(d => (
              <div key={d} className="text-center text-xs font-medium text-stone-400 py-1">{d}</div>
            ))}
          </div>

          {/* Calendar grid */}
          <div className="grid grid-cols-7 gap-1">
            {/* Leading empty cells */}
            {Array.from({ length: firstDay }).map((_, i) => (
              <div key={`empty-${i}`} />
            ))}

            {/* Day cells */}
            {Array.from({ length: daysInMonth }).map((_, i) => {
              const day = i + 1
              const key = dateKey(month.year, month.month, day)
              const count = eggData[key] || 0
              const isToday = key === todayKey
              const isEditing = editing === key

              return (
                <div
                  key={key}
                  onClick={() => !isEditing && setEditing(key)}
                  className={`aspect-square rounded-xl border p-1.5 flex flex-col items-center justify-center cursor-pointer transition-all duration-150 hover:shadow-sm ${cellColor(count)} ${
                    isToday ? 'ring-2 ring-amber-400' : ''
                  }`}
                >
                  <span className="text-xs text-stone-400 leading-none mb-0.5">{day}</span>
                  {isEditing ? (
                    <InlineInput value={count} onSave={v => saveEgg(key, v)} />
                  ) : (
                    count > 0 && (
                      <span className="text-sm font-bold text-stone-700 tabular-nums">{count}</span>
                    )
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-4 mt-4">
          {[
            { label: 'None', cls: 'bg-stone-50 border-stone-200' },
            { label: '1–2', cls: 'bg-green-50 border-green-200' },
            { label: '3–4', cls: 'bg-green-100 border-green-300' },
            { label: '5+',  cls: 'bg-amber-100 border-amber-300' },
          ].map(l => (
            <div key={l.label} className="flex items-center gap-1.5">
              <span className={`w-3 h-3 rounded border ${l.cls}`} />
              <span className="text-xs text-stone-400">{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
