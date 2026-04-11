import { useState, useEffect } from 'react'
import { getSensors, getRiskLatest } from '../api'

function Toggle({ enabled, onChange }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      role="switch"
      aria-checked={enabled}
      className={`relative inline-flex h-7 w-12 items-center rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-500/50 dark:focus:ring-offset-stone-800 ${
        enabled
          ? 'bg-gradient-to-r from-green-500 to-green-400 shadow-inner'
          : 'bg-stone-300 dark:bg-stone-600'
      }`}
    >
      <span
        className={`inline-block h-5 w-5 rounded-full shadow-md transition-all duration-300 ${
          enabled
            ? 'translate-x-6 bg-white'
            : 'translate-x-0.5 bg-white dark:bg-stone-300'
        }`}
      />
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

export default function AutomationPanel() {
  const [sensorData, setSensorData] = useState(undefined)
  const [riskData, setRiskData] = useState(null)
  const [error, setError] = useState(null)
  const [toggles, setToggles] = useState(() => {
    try { return JSON.parse(localStorage.getItem('automation_toggles') || '{}') }
    catch { return {} }
  })

  useEffect(() => {
    async function load() {
      try {
        const [sensors, risk] = await Promise.all([getSensors(), getRiskLatest()])
        setSensorData(sensors)
        setRiskData(risk?.snapshot ?? null)
        setError(null)
      } catch (err) { setError(err.message) }
    }
    load()
    const id = setInterval(load, 15_000)
    return () => clearInterval(id)
  }, [])

  function setToggle(key, value) {
    const updated = { ...toggles, [key]: value }
    setToggles(updated)
    localStorage.setItem('automation_toggles', JSON.stringify(updated))
  }

  const r = sensorData?.reading

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Automation Controls</h2>
          <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">Monitor and manage coop automation systems</p>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700/50 text-red-600 dark:text-red-400 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {sensorData === undefined && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[1, 2].map(i => (
              <div key={i} className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6">
                <div className="skeleton h-4 w-40 mb-6" />
                <div className="skeleton h-6 w-24 mb-4" />
                <div className="skeleton h-16 w-full" />
              </div>
            ))}
          </div>
        )}

        {/* Automation cards */}
        {r && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Door Automation */}
            <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚪</span>
                  <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">Door Automation</h3>
                </div>
                <StatusDot active={r.door_open} />
              </div>

              {/* Current status */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                r.door_open
                  ? 'bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-700/50'
                  : 'bg-stone-50 dark:bg-stone-700/40 border border-stone-200 dark:border-stone-600'
              }`}>
                <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Current Status</p>
                <p className={`text-lg font-bold ${r.door_open ? 'text-sky-700 dark:text-sky-400' : 'text-stone-600 dark:text-stone-300'}`}>
                  Door is {r.door_open ? 'Open' : 'Closed'}
                </p>
              </div>

              {/* Enable toggle */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600 dark:text-stone-300">Enable automation</span>
                <Toggle
                  enabled={toggles.door_auto ?? false}
                  onChange={v => setToggle('door_auto', v)}
                />
              </div>

              {/* Logic rule */}
              <div className="bg-stone-50 dark:bg-stone-700/40 rounded-xl p-3 border border-stone-200 dark:border-stone-600">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-1.5">Rule</p>
                <p className="text-xs text-stone-600 dark:text-stone-300 leading-relaxed">
                  Door closes when all chickens are inside and it's after sunset.
                  Opens again after sunrise if conditions are safe.
                </p>
              </div>

              {/* Related sensors */}
              <div className="mt-4 flex flex-wrap gap-2">
                <SensorBadge label="Chickens inside" value={r.chickens_inside} status="normal" />
              </div>
            </div>

            {/* Ventilation Automation */}
            <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🌀</span>
                  <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100">Ventilation Automation</h3>
                </div>
                <StatusDot active={r.ventilation_on} />
              </div>

              {/* Current status */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                r.ventilation_on
                  ? 'bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-700/50'
                  : 'bg-stone-50 dark:bg-stone-700/40 border border-stone-200 dark:border-stone-600'
              }`}>
                <p className="text-xs text-stone-400 dark:text-stone-500 uppercase tracking-wider font-medium mb-1">Current Status</p>
                <p className={`text-lg font-bold ${r.ventilation_on ? 'text-sky-700 dark:text-sky-400' : 'text-stone-600 dark:text-stone-300'}`}>
                  Fan is {r.ventilation_on ? 'Running' : 'Off'}
                </p>
              </div>

              {/* Enable toggle */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600 dark:text-stone-300">Enable automation</span>
                <Toggle
                  enabled={toggles.vent_auto ?? false}
                  onChange={v => setToggle('vent_auto', v)}
                />
              </div>

              {/* Logic rule */}
              <div className="bg-stone-50 dark:bg-stone-700/40 rounded-xl p-3 border border-stone-200 dark:border-stone-600">
                <p className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-1.5">Rule</p>
                <p className="text-xs text-stone-600 dark:text-stone-300 leading-relaxed">
                  Fan activates when mold risk or heat stress potential reaches warning or critical levels.
                  Turns off when conditions return to normal.
                </p>
              </div>

              {/* Related sensors */}
              <div className="mt-4 flex flex-wrap gap-2">
                <SensorBadge label="Mold risk" value={r.mold_risk_status} status={r.mold_risk_status} />
                <SensorBadge label="Heat stress" value={r.heat_stress_index} status={r.heat_stress_index} />
              </div>

              {/* Fan rate from risk snapshot */}
              {riskData?.fan_rate_m3h != null && (
                <div className="mt-4 pt-4 border-t border-stone-100 dark:border-stone-700">
                  <span className="text-xl font-bold text-stone-800 dark:text-stone-100">
                    {Math.round(riskData.fan_rate_m3h)} m³/h
                  </span>
                  {riskData.decision_reason && (
                    <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5 leading-relaxed">
                      {riskData.decision_reason}
                    </p>
                  )}
                  {riskData.created_at && (
                    <p className="text-[10px] text-stone-400 dark:text-stone-500 mt-1">
                      {new Date(riskData.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Info note */}
        {r && (
          <div className="mt-6 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 rounded-xl px-4 py-3 text-xs text-amber-700 dark:text-amber-400">
            <strong>Note:</strong> Automation toggles are visual only for now. The hardware integration is being developed separately.
            Sensor status updates live every 15 seconds.
          </div>
        )}
      </div>
    </div>
  )
}
