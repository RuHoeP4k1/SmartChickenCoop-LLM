import { useState, useEffect } from 'react'
import { getSensors, getRiskLatest, getDeviceControl, setDeviceControl } from '../api'

function Toggle({ enabled, onChange, disabled }) {
  return (
    <button
      onClick={() => !disabled && onChange(!enabled)}
      role="switch"
      aria-checked={enabled}
      disabled={disabled}
      className={`relative inline-flex h-7 w-12 items-center rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-500/50 dark:focus:ring-offset-stone-800 disabled:opacity-40 disabled:cursor-not-allowed ${
        enabled
          ? 'bg-gradient-to-r from-green-500 to-green-400 shadow-inner'
          : 'bg-stone-300 dark:bg-stone-600'
      }`}
    >
      <span className={`inline-block h-5 w-5 rounded-full shadow-md transition-all duration-300 ${
        enabled ? 'translate-x-6 bg-white' : 'translate-x-0.5 bg-white dark:bg-stone-300'
      }`} />
    </button>
  )
}

function StatusDot({ active }) {
  return (
    <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${
      active ? 'bg-green-500 animate-pulse' : 'bg-stone-300 dark:bg-stone-600'
    }`} />
  )
}

function SensorBadge({ label, value, status }) {
  const colors = {
    normal:   'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-700/50',
    warning:  'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border-amber-300 dark:border-amber-700/50',
    critical: 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 border-red-200 dark:border-red-700/50',
  }
  return (
    <div className={`inline-flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-lg border ${colors[status] || colors.normal}`}>
      <span className="text-stone-500 dark:text-stone-400">{label}:</span>
      <span className="font-semibold">{value ?? '—'}</span>
    </div>
  )
}

function ActionBtn({ onClick, disabled, variant = 'default', children }) {
  const base = 'text-xs font-medium px-3 py-1.5 rounded-lg transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed'
  const variants = {
    default:  'bg-stone-100 dark:bg-stone-700 text-stone-700 dark:text-stone-200 hover:bg-stone-200 dark:hover:bg-stone-600',
    primary:  'bg-amber-500 text-white hover:bg-amber-600',
    danger:   'bg-red-500 text-white hover:bg-red-600',
    success:  'bg-green-500 text-white hover:bg-green-600',
  }
  return (
    <button onClick={onClick} disabled={disabled} className={`${base} ${variants[variant]}`}>
      {children}
    </button>
  )
}

// Keys that go through the draft/save flow (settings vs immediate commands)
const SETTINGS_KEYS = ['fan_auto', 'door_auto', 'feeder_auto', 'chickens_owned']

export default function AutomationPanel() {
  const [device, setDevice] = useState(null)
  const [draft, setDraft] = useState(null)   // null = no pending changes
  const [sensorData, setSensorData] = useState(undefined)
  const [riskData, setRiskData] = useState(null)
  const [error, setError] = useState(null)
  const [saving, setSaving] = useState(false)
  const [fanSlider, setFanSlider] = useState(50)

  useEffect(() => {
    async function load() {
      try {
        const [dc, sensors, risk] = await Promise.all([
          getDeviceControl(),
          getSensors(),
          getRiskLatest(),
        ])
        setDevice(dc?.state ?? null)
        // don't overwrite draft on poll — user may have unsaved changes
        setSensorData(sensors)
        setRiskData(risk?.snapshot ?? null)
        setError(null)
      } catch (err) {
        setError(err.message)
      }
    }
    load()
    const id = setInterval(load, 10_000)
    return () => clearInterval(id)
  }, [])

  // Queue a settings change into the draft (no DB write yet)
  function updateDraft(fields) {
    setDraft(prev => ({
      ...(prev ?? Object.fromEntries(SETTINGS_KEYS.map(k => [k, device[k]]))),
      ...fields,
    }))
  }

  // Read display value — draft takes priority for settings keys
  function val(key) {
    return draft !== null && key in draft ? draft[key] : device?.[key]
  }

  // Save all pending draft fields to DB at once
  async function saveDraft() {
    setSaving(true)
    try {
      const result = await setDeviceControl(draft)
      setDevice(result.state)
      setDraft(null)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  // Immediate command (door open/close, feeder dispense, fan override)
  async function command(fields) {
    setSaving(true)
    try {
      const result = await setDeviceControl(fields)
      setDevice(result.state)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  const isDirty = draft !== null
  const r = sensorData?.reading
  const d = device

  const fanOverrideActive = d?.fan_override_pct != null
  const fanDisplaySpeed = fanOverrideActive
    ? d.fan_override_pct
    : (d?.fan_speed_pct ?? null)

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Automation Controls</h2>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700/50 text-red-600 dark:text-red-400 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Unsaved changes banner */}
        {isDirty && (
          <div className="flex items-center justify-between bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 rounded-xl px-4 py-3 mb-6">
            <span className="text-sm text-amber-700 dark:text-amber-400 font-medium">Unsaved changes</span>
            <div className="flex gap-2">
              <ActionBtn onClick={() => setDraft(null)} disabled={saving} variant="default">
                Discard
              </ActionBtn>
              <ActionBtn onClick={saveDraft} disabled={saving} variant="primary">
                {saving ? 'Saving…' : 'Save changes'}
              </ActionBtn>
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {d === null && !error && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6">
                <div className="skeleton h-4 w-40 mb-6" />
                <div className="skeleton h-6 w-24 mb-4" />
                <div className="skeleton h-16 w-full" />
              </div>
            ))}
          </div>
        )}

        {d && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

            {/* Fan / Ventilation card */}
            <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🌀</span>
                  <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">Ventilation</h3>
                </div>
                <StatusDot active={r?.ventilation_on} />
              </div>

              {/* Speed display */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                fanDisplaySpeed != null
                  ? 'bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-700/50'
                  : 'bg-stone-50 dark:bg-stone-700/40 border border-stone-200 dark:border-stone-600'
              }`}>
                <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">
                  {fanOverrideActive ? 'Override speed' : 'Auto speed'}
                </p>
                <p className="text-2xl font-bold text-stone-800 dark:text-stone-100">
                  {fanDisplaySpeed != null ? `${Math.round(fanDisplaySpeed)}%` : '—'}
                </p>
                {d.fan_status_pct != null && (
                  <p className="text-xs text-stone-400 dark:text-stone-500 mt-0.5">
                    Actual: {Math.round(d.fan_status_pct)}%
                  </p>
                )}
              </div>

              {/* Auto toggle — queues into draft */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600 dark:text-stone-300">Auto mode</span>
                <Toggle
                  enabled={val('fan_auto') ?? true}
                  onChange={v => updateDraft({ fan_auto: v })}
                  disabled={saving}
                />
              </div>

              {/* Risk badges */}
              {r && (
                <div className="flex flex-wrap gap-1.5 mb-4">
                  <SensorBadge label="Heat" value={r.heat_risk_level?.split(' - ')[0] ?? '—'} status={r.heat_risk_level?.toLowerCase().startsWith('high') ? 'critical' : r.heat_risk_level?.toLowerCase().startsWith('med') ? 'warning' : 'normal'} />
                  <SensorBadge label="CO₂" value={r.co2_level ?? '—'} status={r.co2_level ?? 'normal'} />
                </div>
              )}

              {/* Manual override — immediate command */}
              <div className="border-t border-stone-100 dark:border-stone-700 pt-4 mt-2">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-3">
                  Manual override
                </p>
                {fanOverrideActive && (
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">
                      Override active: {Math.round(d.fan_override_pct)}%
                    </span>
                    <ActionBtn onClick={() => command({ clear_fan_override: true })} disabled={saving} variant="danger">
                      Clear
                    </ActionBtn>
                  </div>
                )}
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={fanSlider}
                    onChange={e => setFanSlider(Number(e.target.value))}
                    className="flex-1 accent-amber-500"
                  />
                  <span className="text-xs font-mono text-stone-600 dark:text-stone-300 w-8 text-right">{fanSlider}%</span>
                  <ActionBtn
                    onClick={() => command({ fan_override_pct: fanSlider })}
                    disabled={saving}
                    variant="primary"
                  >
                    Set
                  </ActionBtn>
                </div>
              </div>

              {/* Fan rate from risk snapshot */}
              {riskData?.fan_rate_m3h != null && (
                <div className="mt-3 pt-3 border-t border-stone-100 dark:border-stone-700">
                  <span className="text-sm font-bold text-stone-700 dark:text-stone-200">
                    {Math.round(riskData.fan_rate_m3h)} m³/h
                  </span>
                  {riskData.decision_reason && (
                    <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5 leading-relaxed">
                      {riskData.decision_reason}
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Door card */}
            <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚪</span>
                  <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">Door</h3>
                </div>
                <StatusDot active={r?.door_open} />
              </div>

              {/* Status */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                r?.door_open
                  ? 'bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-700/50'
                  : 'bg-stone-50 dark:bg-stone-700/40 border border-stone-200 dark:border-stone-600'
              }`}>
                <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Status</p>
                <p className={`text-2xl font-bold ${r?.door_open ? 'text-sky-700 dark:text-sky-400' : 'text-stone-600 dark:text-stone-300'}`}>
                  {d.door_status ?? (r?.door_open ? 'Open' : 'Closed')}
                </p>
                {d.door_target && (
                  <p className="text-xs text-stone-400 dark:text-stone-500 mt-0.5">
                    Target: {d.door_target}
                  </p>
                )}
              </div>

              {/* Auto toggle — queues into draft */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600 dark:text-stone-300">Auto mode</span>
                <Toggle
                  enabled={val('door_auto') ?? true}
                  onChange={v => updateDraft({ door_auto: v })}
                  disabled={saving}
                />
              </div>

              {/* Sensor context */}
              {r && (
                <div className="flex flex-wrap gap-1.5 mb-4">
                  <SensorBadge label="Inside" value={r.chickens_inside ?? '—'} status="normal" />
                  <SensorBadge label="Owned" value={val('chickens_owned')} status="normal" />
                </div>
              )}

              {/* Manual commands — immediate */}
              <div className="border-t border-stone-100 dark:border-stone-700 pt-4 mt-2">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-3">
                  Manual command
                </p>
                <div className="flex gap-2">
                  <ActionBtn
                    onClick={() => command({ door_target: 'open' })}
                    disabled={saving}
                    variant="success"
                  >
                    Open
                  </ActionBtn>
                  <ActionBtn
                    onClick={() => command({ door_target: 'closed' })}
                    disabled={saving}
                    variant="danger"
                  >
                    Close
                  </ActionBtn>
                </div>
              </div>

              {/* Flock size — queues into draft */}
              <div className="border-t border-stone-100 dark:border-stone-700 pt-4 mt-4">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-2">
                  Flock size
                </p>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={val('chickens_owned') ?? 10}
                    onChange={e => {
                      const v = parseInt(e.target.value, 10)
                      if (!isNaN(v) && v > 0) updateDraft({ chickens_owned: v })
                    }}
                    className="w-20 text-sm px-2 py-1 rounded-lg border border-stone-200 dark:border-stone-600 bg-stone-50 dark:bg-stone-700 text-stone-800 dark:text-stone-100 focus:outline-none focus:ring-2 focus:ring-amber-500/50"
                  />
                  <span className="text-xs text-stone-400 dark:text-stone-500">chickens owned</span>
                </div>
              </div>
            </div>

            {/* Feed lid card */}
            <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🪣</span>
                  <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">Feed Lid</h3>
                </div>
                <StatusDot active={d.feeder_status === 'closed' || d.feeder_target === 'closed'} />
              </div>

              {/* Status */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                (d.feeder_status ?? d.feeder_target) === 'open'
                  ? 'bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-700/50'
                  : 'bg-stone-50 dark:bg-stone-700/40 border border-stone-200 dark:border-stone-600'
              }`}>
                <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Status</p>
                <p className={`text-2xl font-bold capitalize ${
                  (d.feeder_status ?? d.feeder_target) === 'open'
                    ? 'text-sky-700 dark:text-sky-400'
                    : 'text-stone-600 dark:text-stone-300'
                }`}>
                  {d.feeder_status ?? d.feeder_target ?? '—'}
                </p>
                {d.feeder_target && d.feeder_status && d.feeder_target !== d.feeder_status && (
                  <p className="text-xs text-stone-400 dark:text-stone-500 mt-0.5">
                    Target: {d.feeder_target}
                  </p>
                )}
              </div>

              {/* Auto toggle — queues into draft */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600 dark:text-stone-300">Auto mode</span>
                <Toggle
                  enabled={val('feeder_auto') ?? false}
                  onChange={v => updateDraft({ feeder_auto: v })}
                  disabled={saving}
                />
              </div>

              {/* Auto rule description */}
              <div className="bg-stone-50 dark:bg-stone-700/40 rounded-xl p-3 border border-stone-200 dark:border-stone-600 mb-4">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-1.5">Rule</p>
                <p className="text-xs text-stone-600 dark:text-stone-300 leading-relaxed">
                  Closes at nightfall to block rodents. Opens again at dawn with the coop door.
                </p>
              </div>

              {/* Manual commands — immediate */}
              <div className="border-t border-stone-100 dark:border-stone-700 pt-4 mt-2">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-3">
                  Manual command
                </p>
                <div className="flex gap-2">
                  <ActionBtn
                    onClick={() => command({ feeder_target: 'open' })}
                    disabled={saving || (d.feeder_status ?? d.feeder_target) === 'open'}
                    variant="success"
                  >
                    Open
                  </ActionBtn>
                  <ActionBtn
                    onClick={() => command({ feeder_target: 'closed' })}
                    disabled={saving || (d.feeder_status ?? d.feeder_target) === 'closed'}
                    variant="danger"
                  >
                    Close
                  </ActionBtn>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* Last updated indicator */}
        {d && (
          <p className="mt-4 text-xs text-stone-400 dark:text-stone-500 text-right">
            Device state synced from DB ·{' '}
            {d.updated_at ? new Date(d.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'}
          </p>
        )}
      </div>
    </div>
  )
}
