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
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function isoDate(d) {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

export default function CoopLog() {
  const [cursor, setCursor] = useState(() => {
    const n = new Date()
    return { year: n.getFullYear(), month: n.getMonth() + 1 }
  })
  const [data, setData] = useState(null)
  const [defs, setDefs] = useState([])
  const [selected, setSelected] = useState(null) // iso date
  const [error, setError] = useState(null)
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
    return data.automation_windows.filter(w => {
      return w.start_date <= selected && w.end_date >= selected
    })
  }, [selected, data])

  // Build 6-week grid (Sun=0)
  const gridDays = useMemo(() => {
    const first = new Date(cursor.year, cursor.month - 1, 1)
    const startDow = first.getDay() // 0=Sun
    const cells = []
    for (let i = 0; i < 42; i++) {
      const d = new Date(cursor.year, cursor.month - 1, 1 - startDow + i)
      cells.push(d)
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
    try {
      await setEggEntry(selected, val)
      await refresh()
    } catch (e) { setError(e.message) }
  }

  async function handleReconcile() {
    if (!selected) return
    try {
      await triggerReconcile(selected)
      await refresh()
    } catch (e) { setError(e.message) }
  }

  async function handleAddChore(definitionId, checkedItems, notes) {
    if (!selected) return
    try {
      await addChoreLog({
        entry_date: selected,
        definition_id: definitionId,
        checked_items: checkedItems,
        notes,
      })
      await refresh()
    } catch (e) { setError(e.message) }
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
                    title={info ? `Eggs: ${info.eggs_laid}${info.chore_count ? ` · ${info.chore_count} chore(s)` : ''}${hasAuto ? ` · ${info.automation_tasks.join(', ')}` : ''}${info.feeder_pct_consumed != null ? ` · feed ${info.feeder_pct_consumed}%` : ''}${info.waterer_pct_consumed != null ? ` · water ${info.waterer_pct_consumed}%` : ''}` : iso}
                    className={`aspect-square rounded border-2 ${borderCls} ${isSel ? 'ring-2 ring-amber-400' : ''} ${inMonth ? 'bg-stone-50 dark:bg-stone-700' : 'bg-stone-100/40 dark:bg-stone-800 opacity-50'} flex flex-col items-center justify-start p-1 text-xs hover:bg-amber-50 dark:hover:bg-stone-600 transition`}
                  >
                    <span className="text-stone-600 dark:text-stone-300 self-end">{d.getDate()}</span>
                    {info && info.eggs_laid > 0 && (
                      <span className="mt-1 font-semibold text-amber-600 dark:text-amber-400">
                        🥚 {info.eggs_laid}
                      </span>
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

function DayPanel({ selected, selectedDay, selectedChores, selectedWindows, defs, onSetEggs, onReconcile, onAddChore, onDeleteChore, onAddWindow, onDeleteWindow }) {
  const [eggInput, setEggInput] = useState('')
  const [choreDefId, setChoreDefId] = useState('')
  const [checkedItems, setCheckedItems] = useState([])
  const [choreNotes, setChoreNotes] = useState('')
  const [winTask, setWinTask] = useState('ventilation')
  const [winEnd, setWinEnd] = useState('')
  const [winStartTime, setWinStartTime] = useState('')
  const [winEndTime, setWinEndTime] = useState('')

  useEffect(() => {
    setEggInput(selectedDay?.eggs_laid ?? '')
    setChoreDefId('')
    setCheckedItems([])
    setChoreNotes('')
    setWinEnd(selected || '')
  }, [selected, selectedDay?.eggs_laid])

  if (!selected) {
    return (
      <div className="bg-white dark:bg-stone-800 rounded-lg shadow p-4 text-sm text-stone-500">
        Select a day to view or edit its log.
      </div>
    )
  }

  const currentDef = defs.find(d => d.id === Number(choreDefId))

  return (
    <div className="bg-white dark:bg-stone-800 rounded-lg shadow p-4 space-y-4 text-sm">
      <div>
        <div className="font-semibold text-stone-800 dark:text-stone-100">{selected}</div>
        {selectedDay && (
          <div className="text-xs text-stone-500 mt-1">
            {selectedDay.eggs_source && <span>eggs: {selectedDay.eggs_source}</span>}
            {selectedDay.automation_hours > 0 && <span> · {selectedDay.automation_hours}h automation</span>}
            {selectedDay.feeder_pct_consumed != null && <span> · feed {selectedDay.feeder_pct_consumed}%</span>}
            {selectedDay.waterer_pct_consumed != null && <span> · water {selectedDay.waterer_pct_consumed}%</span>}
          </div>
        )}
      </div>

      {/* Eggs */}
      <section>
        <div className="font-medium mb-1 text-stone-700 dark:text-stone-200">Eggs</div>
        <div className="flex gap-2">
          <input
            type="number" min="0" max="100"
            value={eggInput}
            onChange={e => setEggInput(e.target.value)}
            className="w-20 px-2 py-1 border border-stone-300 dark:border-stone-600 rounded bg-white dark:bg-stone-700"
          />
          <button
            onClick={() => onSetEggs(Number(eggInput) || 0)}
            className="px-3 py-1 bg-amber-500 text-white rounded hover:bg-amber-600"
          >Save</button>
          <button
            onClick={onReconcile}
            className="px-3 py-1 border border-stone-300 dark:border-stone-600 rounded hover:bg-stone-100 dark:hover:bg-stone-700"
            title="Recompute from camera samples (won't overwrite manual edits)"
          >Reconcile</button>
        </div>
      </section>

      {/* Chores */}
      <section>
        <div className="font-medium mb-1 text-stone-700 dark:text-stone-200">Chores</div>
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
              type="text"
              placeholder="notes (optional)"
              value={choreNotes}
              onChange={e => setChoreNotes(e.target.value)}
              className="w-full px-2 py-1 border border-stone-300 dark:border-stone-600 rounded bg-white dark:bg-stone-700 mb-2 text-xs"
            />
            <button
              onClick={() => { onAddChore(Number(choreDefId), checkedItems, choreNotes); setChoreDefId(''); setCheckedItems([]); setChoreNotes('') }}
              className="px-3 py-1 bg-emerald-600 text-white rounded hover:bg-emerald-700 text-xs"
            >Log chore</button>
          </>
        )}
      </section>

      {/* Automation */}
      <section>
        <div className="font-medium mb-1 text-stone-700 dark:text-stone-200">Automation</div>
        {selectedWindows.length > 0 && (
          <ul className="mb-2 space-y-1">
            {selectedWindows.map(w => (
              <li key={w.id} className="flex items-center justify-between gap-2 text-xs bg-stone-50 dark:bg-stone-700 px-2 py-1 rounded">
                <span>
                  {w.task} · {w.start_date} → {w.end_date}
                  {w.start_time && w.end_time ? ` · ${w.start_time}–${w.end_time}` : ''}
                </span>
                <button onClick={() => onDeleteWindow(w.id)} className="text-red-500 hover:text-red-600">×</button>
              </li>
            ))}
          </ul>
        )}
        <div className="space-y-2">
          <select
            value={winTask}
            onChange={e => setWinTask(e.target.value)}
            className="w-full px-2 py-1 border border-stone-300 dark:border-stone-600 rounded bg-white dark:bg-stone-700 text-xs"
          >
            {AUTOMATION_TASKS.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <div className="flex gap-2 text-xs">
            <label className="flex-1">start <input type="date" value={selected} disabled className="w-full px-2 py-1 border rounded bg-stone-100 dark:bg-stone-900 border-stone-300 dark:border-stone-600" /></label>
            <label className="flex-1">end <input type="date" value={winEnd} onChange={e => setWinEnd(e.target.value)} className="w-full px-2 py-1 border rounded bg-white dark:bg-stone-700 border-stone-300 dark:border-stone-600" /></label>
          </div>
          <div className="flex gap-2 text-xs">
            <label className="flex-1">from <input type="time" value={winStartTime} onChange={e => setWinStartTime(e.target.value)} className="w-full px-2 py-1 border rounded bg-white dark:bg-stone-700 border-stone-300 dark:border-stone-600" /></label>
            <label className="flex-1">to <input type="time" value={winEndTime} onChange={e => setWinEndTime(e.target.value)} className="w-full px-2 py-1 border rounded bg-white dark:bg-stone-700 border-stone-300 dark:border-stone-600" /></label>
          </div>
          <button
            onClick={() => onAddWindow({
              task: winTask,
              start_date: selected,
              end_date: winEnd || selected,
              start_time: winStartTime || null,
              end_time: winEndTime || null,
              days_of_week: [0,1,2,3,4,5,6],
            })}
            className="px-3 py-1 bg-sky-600 text-white rounded hover:bg-sky-700 text-xs"
          >Add window</button>
        </div>
      </section>
    </div>
  )
}
