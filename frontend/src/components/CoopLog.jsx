import { useEffect, useMemo, useState } from 'react'
import {
  getEggCalendar,
  setEggEntry,
  triggerReconcile,
  getChoreDefinitions,
  addChoreLog,
  deleteChoreLog,
  getAutomationWindows,
  createAutomationWindow,
  deleteAutomationWindow,
} from '../api'

const AUTOMATION_TASKS = ['ventilation', 'door', 'feeder']
const TASK_COLORS = {
  ventilation: 'border-sky-400',
  door: 'border-violet-400',
  feeder: 'border-amber-400',
}
const TASK_DOT = {
  ventilation: 'bg-sky-400',
  door: 'bg-violet-400',
  feeder: 'bg-amber-400',
}
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function isoDate(d) {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

// ─── Date Range Picker ───────────────────────────────────────────────────────
function DateRangePicker({ startDate, endDate, onRangeChange }) {
  const [cursor, setCursor] = useState(() => {
    const d = startDate ? new Date(startDate + 'T00:00:00') : new Date()
    return { year: d.getFullYear(), month: d.getMonth() + 1 }
  })
  const [hover, setHover] = useState(null)
  const [picking, setPicking] = useState('start')

  const gridDays = useMemo(() => {
    const first = new Date(cursor.year, cursor.month - 1, 1)
    const dow = first.getDay()
    const days = []
    for (let i = 0; i < 42; i++) {
      days.push(new Date(cursor.year, cursor.month - 1, 1 - dow + i))
    }
    return days
  }, [cursor])

  const monthLabel = new Date(cursor.year, cursor.month - 1, 1)
    .toLocaleString(undefined, { month: 'short', year: 'numeric' })

  function prevM() {
    setCursor(c => c.month === 1 ? { year: c.year - 1, month: 12 } : { year: c.year, month: c.month - 1 })
  }
  function nextM() {
    setCursor(c => c.month === 12 ? { year: c.year + 1, month: 1 } : { year: c.year, month: c.month + 1 })
  }

  function handleClick(iso) {
    if (picking === 'start') {
      onRangeChange(iso, iso)
      setPicking('end')
    } else {
      const [s, e] = iso < startDate ? [iso, startDate] : [startDate, iso]
      onRangeChange(s, e)
      setPicking('start')
    }
  }

  function isInRange(iso) {
    return startDate && endDate && iso > startDate && iso < endDate
  }
  function isHoverRange(iso) {
    if (!startDate || picking !== 'end' || !hover) return false
    const lo = hover < startDate ? hover : startDate
    const hi = hover < startDate ? startDate : hover
    return iso > lo && iso < hi
  }

  return (
    <div className="bg-stone-50 dark:bg-stone-700/50 rounded-xl border border-stone-200 dark:border-stone-600 p-2.5 text-xs">
      {/* month nav */}
      <div className="flex items-center justify-between mb-2">
        <button onClick={prevM} className="w-5 text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 text-sm">‹</button>
        <span className="font-medium text-stone-600 dark:text-stone-300 text-[11px]">{monthLabel}</span>
        <button onClick={nextM} className="w-5 text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 text-sm">›</button>
      </div>
      {/* day headers */}
      <div className="grid grid-cols-7 text-center text-stone-400 dark:text-stone-500 mb-0.5 text-[10px]">
        {['S','M','T','W','T','F','S'].map((d, i) => <div key={i}>{d}</div>)}
      </div>
      {/* days */}
      <div className="grid grid-cols-7 gap-px">
        {gridDays.map((d, i) => {
          const iso = isoDate(d)
          const inMonth = d.getMonth() + 1 === cursor.month
          const isStart = iso === startDate
          const isEnd   = iso === endDate && endDate !== startDate
          const inR  = isInRange(iso)
          const inH  = isHoverRange(iso)

          let cls = 'h-6 w-full flex items-center justify-center cursor-pointer text-[11px] transition-colors '
          if (!inMonth) cls += 'opacity-25 '
          if (isStart)  cls += 'bg-amber-500 text-white font-bold rounded-full '
          else if (isEnd) cls += 'bg-amber-500 text-white font-bold rounded-full '
          else if (inR) cls += 'bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 '
          else if (inH) cls += 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 '
          else          cls += 'hover:bg-stone-200 dark:hover:bg-stone-600 text-stone-700 dark:text-stone-300 rounded '

          return (
            <div
              key={i}
              className={cls}
              onClick={() => inMonth && handleClick(iso)}
              onMouseEnter={() => inMonth && setHover(iso)}
              onMouseLeave={() => setHover(null)}
            >
              {d.getDate()}
            </div>
          )
        })}
      </div>
      {/* display + hint */}
      <div className="mt-2 flex items-center justify-between text-[10px]">
        <span className="text-stone-500 dark:text-stone-400">
          {startDate || '—'} → {endDate || '—'}
        </span>
        <span className={`font-medium ${picking === 'start' ? 'text-amber-600' : 'text-sky-500'}`}>
          {picking === 'start' ? 'pick start' : 'pick end'}
        </span>
      </div>
    </div>
  )
}

// ─── Analog Clock ────────────────────────────────────────────────────────────
const OUTER_HOURS = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
const INNER_HOURS = [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
const QUARTER_MIN = [0, 15, 30, 45]
const CX = 100, CY = 100, OUTER_R = 74, INNER_R = 50

function clockPos(idx, n, r) {
  const angle = (idx / n) * 2 * Math.PI - Math.PI / 2
  return { x: CX + Math.cos(angle) * r, y: CY + Math.sin(angle) * r }
}

function AnalogClock({ initialHour, onPick, onClose }) {
  const [phase, setPhase]     = useState(initialHour != null ? 'minute' : 'hour')
  const [hour, setHour]       = useState(initialHour ?? null)
  const [hovH, setHovH]       = useState(null)
  const [hovM, setHovM]       = useState(null)

  function pickHour(h) { setHour(h); setPhase('minute') }
  function pickMinute(m) {
    onPick(`${String(hour).padStart(2, '0')}:${String(m).padStart(2, '0')}`)
  }

  // clock hand to selected hour
  const handLine = (() => {
    if (hour == null || phase !== 'hour') return null
    const isOuter = OUTER_HOURS.includes(hour)
    const arr = isOuter ? OUTER_HOURS : INNER_HOURS
    const idx = arr.indexOf(hour)
    const r = isOuter ? OUTER_R - 14 : INNER_R - 12
    const { x, y } = clockPos(idx, 12, r)
    return <line x1={CX} y1={CY} x2={x} y2={y} stroke="#f59e0b" strokeWidth={2} strokeLinecap="round" />
  })()

  return (
    <div className="bg-white dark:bg-stone-800 rounded-2xl shadow-2xl border border-stone-200 dark:border-stone-700 p-3 w-[220px] select-none">
      {/* header */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-semibold text-stone-500 uppercase tracking-wider">
          {phase === 'hour'
            ? 'Select hour'
            : `${String(hour).padStart(2, '0')} : select min`}
        </span>
        <button onClick={onClose} className="text-stone-300 hover:text-stone-600 dark:hover:text-stone-200 text-sm leading-none">✕</button>
      </div>

      <svg width="200" height="200" className="cursor-pointer">
        {/* dial */}
        <circle cx={CX} cy={CY} r={92} fill="#fafaf9" />

        {phase === 'hour' && (
          <>
            {OUTER_HOURS.map((h, i) => {
              const { x, y } = clockPos(i, 12, OUTER_R)
              const sel = hour === h
              const hov = hovH === h
              return (
                <g key={h} onClick={() => pickHour(h)} onMouseEnter={() => setHovH(h)} onMouseLeave={() => setHovH(null)}>
                  <circle cx={x} cy={y} r={15} fill={sel ? '#f59e0b' : hov ? '#fef3c7' : 'transparent'} />
                  <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
                    fontSize={13} fontWeight={sel ? '700' : '400'} fill={sel ? '#fff' : '#44403c'}
                    style={{ pointerEvents: 'none' }}
                  >{h}</text>
                </g>
              )
            })}
            {INNER_HOURS.map((h, i) => {
              const { x, y } = clockPos(i, 12, INNER_R)
              const sel = hour === h
              const hov = hovH === h
              return (
                <g key={`i${h}`} onClick={() => pickHour(h)} onMouseEnter={() => setHovH(h)} onMouseLeave={() => setHovH(null)}>
                  <circle cx={x} cy={y} r={13} fill={sel ? '#f59e0b' : hov ? '#fef3c7' : 'transparent'} />
                  <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
                    fontSize={11} fontWeight={sel ? '700' : '400'} fill={sel ? '#fff' : '#78716c'}
                    style={{ pointerEvents: 'none' }}
                  >{String(h).padStart(2, '0')}</text>
                </g>
              )
            })}
            {handLine}
          </>
        )}

        {phase === 'minute' && (
          <>
            {/* hour in center */}
            <text x={CX} y={CY - 8} textAnchor="middle" dominantBaseline="central"
              fontSize={26} fontWeight="700" fill="#f59e0b">{String(hour).padStart(2, '0')}</text>
            <text x={CX} y={CY + 16} textAnchor="middle" dominantBaseline="central"
              fontSize={10} fill="#a8a29e">tap a minute</text>
            {/* quarter circles */}
            {QUARTER_MIN.map((m, i) => {
              const { x, y } = clockPos(i, 4, OUTER_R)
              const hov = hovM === m
              return (
                <g key={m} onClick={() => pickMinute(m)} onMouseEnter={() => setHovM(m)} onMouseLeave={() => setHovM(null)}>
                  <circle cx={x} cy={y} r={20} fill={hov ? '#fef3c7' : 'transparent'} />
                  <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
                    fontSize={14} fontWeight="600" fill="#44403c"
                    style={{ pointerEvents: 'none' }}
                  >:{String(m).padStart(2, '0')}</text>
                </g>
              )
            })}
          </>
        )}

        <circle cx={CX} cy={CY} r={4} fill="#f59e0b" />
      </svg>

      {phase === 'minute' && (
        <button onClick={() => setPhase('hour')} className="w-full text-[11px] text-stone-400 hover:text-stone-600 dark:hover:text-stone-200 text-center mt-1">
          ← back to hour
        </button>
      )}
    </div>
  )
}

// ─── Time Picker Field ───────────────────────────────────────────────────────
function TimePickerField({ label, value, onChange }) {
  const [open, setOpen] = useState(false)
  const initialHour = value ? parseInt(value.split(':')[0]) : null

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-stone-500 dark:text-stone-400 w-7 shrink-0">{label}</span>
        <button
          onClick={() => setOpen(o => !o)}
          className={`flex-1 px-3 py-1.5 rounded-lg border text-xs text-left flex items-center justify-between transition-colors ${
            value
              ? 'bg-white dark:bg-stone-700 border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-200 font-medium'
              : 'bg-stone-50 dark:bg-stone-700/50 border-dashed border-stone-300 dark:border-stone-600 text-stone-400'
          }`}
        >
          <span>{value ?? '--:--'}</span>
          {value && (
            <span
              className="text-stone-300 hover:text-red-400 ml-1"
              onClick={e => { e.stopPropagation(); onChange(null) }}
            >✕</span>
          )}
        </button>
      </div>
      {open && (
        <div className="absolute left-9 top-full mt-1 z-50">
          <AnalogClock
            initialHour={initialHour}
            onPick={t => { onChange(t); setOpen(false) }}
            onClose={() => setOpen(false)}
          />
        </div>
      )}
    </div>
  )
}

// ─── Main CoopLog ────────────────────────────────────────────────────────────
export default function CoopLog() {
  const [cursor, setCursor] = useState(() => {
    const n = new Date()
    return { year: n.getFullYear(), month: n.getMonth() + 1 }
  })
  const [data, setData]       = useState(null)
  const [defs, setDefs]       = useState([])
  const [selected, setSelected] = useState(null)
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const [cal, d] = await Promise.all([
        getEggCalendar(cursor.year, cursor.month),
        defs.length ? Promise.resolve({ definitions: defs }) : getChoreDefinitions(),
      ])
      setData(cal)
      if (!defs.length) setDefs(d.definitions || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { refresh() }, [cursor.year, cursor.month])

  const dayMap = useMemo(() => {
    const m = {}
    if (data?.days) for (const d of data.days) m[d.date] = d
    return m
  }, [data])

  const selectedDay = selected ? dayMap[selected] : null
  const selectedChores = useMemo(() => {
    if (!selected || !data?.chores) return []
    return data.chores.filter(c => {
      const ed = typeof c.entry_date === 'string' ? c.entry_date : c.entry_date?.slice?.(0, 10)
      return ed === selected
    })
  }, [selected, data])
  const selectedWindows = useMemo(() => {
    if (!selected || !data?.automation_windows) return []
    return data.automation_windows.filter(w => w.start_date <= selected && w.end_date >= selected)
  }, [selected, data])

  const gridDays = useMemo(() => {
    const first = new Date(cursor.year, cursor.month - 1, 1)
    const startDow = first.getDay()
    const cells = []
    for (let i = 0; i < 42; i++) {
      cells.push(new Date(cursor.year, cursor.month - 1, 1 - startDow + i))
    }
    return cells
  }, [cursor])

  function prevMonth() {
    setCursor(c => c.month === 1 ? { year: c.year - 1, month: 12 } : { year: c.year, month: c.month - 1 })
  }
  function nextMonth() {
    setCursor(c => c.month === 12 ? { year: c.year + 1, month: 1 } : { year: c.year, month: c.month + 1 })
  }

  async function handleSetEggs(val) {
    if (!selected) return
    try { await setEggEntry(selected, val); await refresh() } catch (e) { setError(e.message) }
  }
  async function handleReconcile() {
    if (!selected) return
    try { await triggerReconcile(selected); await refresh() } catch (e) { setError(e.message) }
  }
  async function handleAddChore(defId, items, notes) {
    if (!selected) return
    try { await addChoreLog({ entry_date: selected, definition_id: defId, checked_items: items, notes }); await refresh() }
    catch (e) { setError(e.message) }
  }
  async function handleDeleteChore(id) {
    try { await deleteChoreLog(id); await refresh() } catch (e) { setError(e.message) }
  }
  async function handleAddWindow(w) {
    try { await createAutomationWindow(w); await refresh() } catch (e) { setError(e.message) }
  }
  async function handleDeleteWindow(id) {
    try { await deleteAutomationWindow(id); await refresh() } catch (e) { setError(e.message) }
  }

  const monthLabel = new Date(cursor.year, cursor.month - 1, 1)
    .toLocaleString(undefined, { month: 'long', year: 'numeric' })

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-6xl mx-auto animate-fade-in">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold text-stone-800 dark:text-stone-100">Coop Log</h1>
          <div className="flex items-center gap-2">
            <button onClick={prevMonth} className="px-3 py-1 rounded border border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-200">‹</button>
            <span className="min-w-[10rem] text-center font-medium text-stone-700 dark:text-stone-200">{monthLabel}</span>
            <button onClick={nextMonth} className="px-3 py-1 rounded border border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-200">›</button>
          </div>
        </div>

        {error && <div className="mb-4 p-3 rounded bg-red-100 text-red-800 text-sm">{error}</div>}
        {loading && <div className="mb-4 text-sm text-stone-500">Loading…</div>}

        <div className="grid grid-cols-[2fr_1fr] gap-6">
          {/* Calendar */}
          <div className="bg-white dark:bg-stone-800 rounded-lg shadow p-4">
            <div className="grid grid-cols-7 gap-1 text-xs text-stone-500 mb-1">
              {WEEKDAYS.map(w => <div key={w} className="text-center font-medium">{w}</div>)}
            </div>
            <div className="grid grid-cols-7 gap-1">
              {gridDays.map((d, i) => {
                const iso = isoDate(d)
                const inMonth = d.getMonth() + 1 === cursor.month
                const info = dayMap[iso]
                const hasAuto = info?.automation_tasks?.length > 0
                const primary = hasAuto ? info.automation_tasks[0] : null
                const borderCls = primary ? TASK_COLORS[primary] : 'border-transparent'
                const isSel = iso === selected
                return (
                  <button
                    key={i}
                    onClick={() => setSelected(iso)}
                    title={info
                      ? `Eggs: ${info.eggs_laid}${info.chore_count ? ` · ${info.chore_count} chore(s)` : ''}${hasAuto ? ` · ${info.automation_tasks.join(', ')}` : ''}`
                      : iso}
                    className={`aspect-square rounded border-2 ${borderCls} ${isSel ? 'ring-2 ring-amber-400' : ''} ${inMonth ? 'bg-stone-50 dark:bg-stone-700' : 'bg-stone-100/40 dark:bg-stone-800 opacity-50'} flex flex-col items-center justify-start p-1 text-xs hover:bg-amber-50 dark:hover:bg-stone-600 transition`}
                  >
                    <span className="text-stone-600 dark:text-stone-300 self-end">{d.getDate()}</span>
                    {info && info.eggs_laid > 0 && (
                      <span className="mt-1 font-semibold text-amber-600 dark:text-amber-400">🥚 {info.eggs_laid}</span>
                    )}
                    {info && info.chore_count > 0 && (
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-0.5" />
                    )}
                  </button>
                )
              })}
            </div>
            {/* Legend */}
            <div className="flex flex-wrap gap-3 mt-3 text-xs text-stone-500">
              {AUTOMATION_TASKS.map(t => (
                <div key={t} className="flex items-center gap-1">
                  <span className={`inline-block w-3 h-3 rounded border-2 ${TASK_COLORS[t]}`} />
                  {t}
                </div>
              ))}
              <div className="flex items-center gap-1">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" /> chore logged
              </div>
            </div>
          </div>

          {/* Side panel */}
          <DayPanel
            selected={selected}
            selectedDay={selectedDay}
            selectedChores={selectedChores}
            selectedWindows={selectedWindows}
            defs={defs}
            onSetEggs={handleSetEggs}
            onReconcile={handleReconcile}
            onAddChore={handleAddChore}
            onDeleteChore={handleDeleteChore}
            onAddWindow={handleAddWindow}
            onDeleteWindow={handleDeleteWindow}
          />
        </div>
      </div>
    </div>
  )
}

// ─── Day Panel ───────────────────────────────────────────────────────────────
function DayPanel({ selected, selectedDay, selectedChores, selectedWindows, defs, onSetEggs, onReconcile, onAddChore, onDeleteChore, onAddWindow, onDeleteWindow }) {
  const [eggInput, setEggInput]     = useState('')
  const [choreDefId, setChoreDefId] = useState('')
  const [checkedItems, setCheckedItems] = useState([])
  const [choreNotes, setChoreNotes] = useState('')
  // automation
  const [winTasks, setWinTasks]     = useState([])
  const [winStart, setWinStart]     = useState(selected || '')
  const [winEnd, setWinEnd]         = useState(selected || '')
  const [winStartTime, setWinStartTime] = useState(null)
  const [winEndTime, setWinEndTime]     = useState(null)

  useEffect(() => {
    setEggInput(selectedDay?.eggs_laid ?? '')
    setChoreDefId('')
    setCheckedItems([])
    setChoreNotes('')
    setWinTasks([])
    setWinStart(selected || '')
    setWinEnd(selected || '')
    setWinStartTime(null)
    setWinEndTime(null)
  }, [selected, selectedDay?.eggs_laid])

  if (!selected) {
    return (
      <div className="bg-white dark:bg-stone-800 rounded-lg shadow p-4 text-sm text-stone-500">
        Select a day to view or edit its log.
      </div>
    )
  }

  const currentDef = defs.find(d => d.id === Number(choreDefId))

  async function handleSubmitWindows() {
    if (!winTasks.length) return
    for (const task of winTasks) {
      await onAddWindow({
        task,
        start_date: winStart || selected,
        end_date:   winEnd || winStart || selected,
        start_time: winStartTime || null,
        end_time:   winEndTime || null,
        days_of_week: [0, 1, 2, 3, 4, 5, 6],
      })
    }
    setWinTasks([])
    setWinStartTime(null)
    setWinEndTime(null)
  }

  function toggleTask(task) {
    setWinTasks(prev =>
      prev.includes(task)
        ? prev.filter(t => t !== task)
        : prev.length < 3 ? [...prev, task] : prev
    )
  }

  return (
    <div className="bg-white dark:bg-stone-800 rounded-lg shadow p-4 space-y-5 text-sm overflow-visible">
      {/* date header */}
      <div>
        <div className="font-semibold text-stone-800 dark:text-stone-100">{selected}</div>
        {selectedDay && (
          <div className="text-xs text-stone-500 mt-1 space-x-1">
            {selectedDay.eggs_source && <span>eggs: {selectedDay.eggs_source}</span>}
            {selectedDay.automation_hours > 0 && <span>· {selectedDay.automation_hours}h automation</span>}
            {selectedDay.feeder_pct_consumed != null && <span>· feed {selectedDay.feeder_pct_consumed}%</span>}
          </div>
        )}
      </div>

      {/* ── Eggs ── */}
      <section>
        <div className="font-medium mb-2 text-stone-700 dark:text-stone-200">Eggs</div>
        <div className="flex gap-2">
          <input
            type="number" min="0" max="100"
            value={eggInput}
            onChange={e => setEggInput(e.target.value)}
            className="w-20 px-2 py-1 border border-stone-300 dark:border-stone-600 rounded bg-white dark:bg-stone-700"
          />
          <button onClick={() => onSetEggs(Number(eggInput) || 0)}
            className="px-3 py-1 bg-amber-500 text-white rounded hover:bg-amber-600">
            Save
          </button>
          <button onClick={onReconcile}
            className="px-3 py-1 border border-stone-300 dark:border-stone-600 rounded hover:bg-stone-100 dark:hover:bg-stone-700"
            title="Recompute from camera samples">
            Reconcile
          </button>
        </div>
      </section>

      {/* ── Chores ── */}
      <section>
        <div className="font-medium mb-2 text-stone-700 dark:text-stone-200">Chores</div>
        {selectedChores.length > 0 && (
          <ul className="mb-2 space-y-1">
            {selectedChores.map(c => (
              <li key={c.id} className="flex items-center justify-between gap-2 text-xs bg-stone-50 dark:bg-stone-700 px-2 py-1 rounded">
                <span>{c.label}{c.checked_items?.length ? ` — ${c.checked_items.join(', ')}` : ''}</span>
                <button onClick={() => onDeleteChore(c.id)} className="text-red-500 hover:text-red-600">×</button>
              </li>
            ))}
          </ul>
        )}
        <select
          value={choreDefId}
          onChange={e => { setChoreDefId(e.target.value); setCheckedItems([]) }}
          className="w-full px-2 py-1 border border-stone-300 dark:border-stone-600 rounded bg-white dark:bg-stone-700 mb-2"
        >
          <option value="">— add chore —</option>
          {defs.map(d => <option key={d.id} value={d.id}>{d.label}</option>)}
        </select>
        {currentDef?.checklist_items?.length > 0 && (
          <div className="space-y-1 mb-2">
            {currentDef.checklist_items.map(item => (
              <label key={item} className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={checkedItems.includes(item)}
                  onChange={e => setCheckedItems(prev => e.target.checked ? [...prev, item] : prev.filter(x => x !== item))}
                />
                {item}
              </label>
            ))}
          </div>
        )}
        {choreDefId && (
          <>
            <input
              type="text" placeholder="notes (optional)"
              value={choreNotes}
              onChange={e => setChoreNotes(e.target.value)}
              className="w-full px-2 py-1 border border-stone-300 dark:border-stone-600 rounded bg-white dark:bg-stone-700 mb-2 text-xs"
            />
            <button
              onClick={() => { onAddChore(Number(choreDefId), checkedItems, choreNotes); setChoreDefId(''); setCheckedItems([]); setChoreNotes('') }}
              className="px-3 py-1 bg-emerald-600 text-white rounded hover:bg-emerald-700 text-xs"
            >
              Log chore
            </button>
          </>
        )}
      </section>

      {/* ── Automation ── */}
      <section className="space-y-3">
        <div className="font-medium text-stone-700 dark:text-stone-200">Automation</div>

        {/* existing windows */}
        {selectedWindows.length > 0 && (
          <ul className="space-y-1">
            {selectedWindows.map(w => (
              <li key={w.id} className="flex items-center justify-between gap-2 text-xs bg-stone-50 dark:bg-stone-700 px-2 py-1.5 rounded">
                <span className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${TASK_DOT[w.task] ?? 'bg-stone-400'}`} />
                  {w.task} · {w.start_date}{w.start_date !== w.end_date ? ` → ${w.end_date}` : ''}
                  {w.start_time && w.end_time ? ` · ${w.start_time}–${w.end_time}` : ''}
                </span>
                <button onClick={() => onDeleteWindow(w.id)} className="text-red-400 hover:text-red-600 shrink-0">×</button>
              </li>
            ))}
          </ul>
        )}

        {/* task checkboxes */}
        <div>
          <div className="text-[11px] text-stone-400 dark:text-stone-500 mb-1.5 uppercase tracking-wider">Tasks (1–3)</div>
          <div className="space-y-2">
            {AUTOMATION_TASKS.map(task => (
              <label key={task} className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={winTasks.includes(task)}
                  onChange={() => toggleTask(task)}
                  disabled={!winTasks.includes(task) && winTasks.length >= 3}
                  className="rounded accent-sky-500"
                />
                <span className={`w-2 h-2 rounded-full ${TASK_DOT[task]}`} />
                <span className="text-sm capitalize text-stone-700 dark:text-stone-200">{task}</span>
              </label>
            ))}
          </div>
        </div>

        {/* date range picker */}
        <div>
          <div className="text-[11px] text-stone-400 dark:text-stone-500 mb-1.5 uppercase tracking-wider">Period</div>
          <DateRangePicker
            startDate={winStart}
            endDate={winEnd}
            onRangeChange={(s, e) => { setWinStart(s); setWinEnd(e) }}
          />
        </div>

        {/* time pickers */}
        <div>
          <div className="text-[11px] text-stone-400 dark:text-stone-500 mb-1.5 uppercase tracking-wider">Time window</div>
          <div className="space-y-2">
            <TimePickerField label="From" value={winStartTime} onChange={setWinStartTime} />
            <TimePickerField label="To"   value={winEndTime}   onChange={setWinEndTime} />
          </div>
        </div>

        <button
          disabled={winTasks.length === 0}
          onClick={handleSubmitWindows}
          className="w-full px-3 py-2 bg-sky-600 text-white rounded-lg hover:bg-sky-700 text-xs font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Add window{winTasks.length > 1 ? ` (${winTasks.length} tasks)` : ''}
        </button>
      </section>
    </div>
  )
}
