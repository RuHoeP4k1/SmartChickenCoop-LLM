import { useState, useEffect } from 'react'
import { getSensors } from '../api'

function Toggle({ enabled, onChange }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
        enabled ? 'bg-green-500' : 'bg-stone-300'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200 ${
          enabled ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  )
}

function StatusDot({ active }) {
  return (
    <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${
      active ? 'bg-green-500 animate-pulse' : 'bg-stone-300'
    }`} />
  )
}

function SensorBadge({ label, value, status }) {
  const colors = {
    normal:   'bg-green-50 text-green-700 border-green-200',
    warning:  'bg-amber-50 text-amber-700 border-amber-300',
    critical: 'bg-red-50 text-red-600 border-red-200',
  }
  return (
    <div className={`inline-flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-lg border ${colors[status] || colors.normal}`}>
      <span className="text-stone-500">{label}:</span>
      <span className="font-semibold">{value ?? '—'}</span>
    </div>
  )
}

export default function AutomationPanel() {
  const [sensorData, setSensorData] = useState(undefined)
  const [error, setError] = useState(null)
  const [toggles, setToggles] = useState(() => {
    try { return JSON.parse(localStorage.getItem('automation_toggles') || '{}') }
    catch { return {} }
  })

  useEffect(() => {
    async function load() {
      try {
        const result = await getSensors()
        setSensorData(result)
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
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-100">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-base font-semibold text-stone-800">Automation Controls</h2>
          <p className="text-xs text-stone-500 mt-0.5">Monitor and manage coop automation systems</p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-600 rounded-xl px-4 py-3 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {sensorData === undefined && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[1, 2].map(i => (
              <div key={i} className="rounded-2xl border border-stone-200 bg-white p-6">
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
            <div className="rounded-2xl border border-stone-200 bg-white p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚪</span>
                  <h3 className="text-sm font-semibold text-stone-800">Door Automation</h3>
                </div>
                <StatusDot active={r.door_open} />
              </div>

              {/* Current status */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                r.door_open ? 'bg-sky-50 border border-sky-200' : 'bg-stone-50 border border-stone-200'
              }`}>
                <p className="text-xs text-stone-400 uppercase tracking-wider font-medium mb-1">Current Status</p>
                <p className={`text-lg font-bold ${r.door_open ? 'text-sky-700' : 'text-stone-600'}`}>
                  Door is {r.door_open ? 'Open' : 'Closed'}
                </p>
              </div>

              {/* Enable toggle */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600">Enable automation</span>
                <Toggle
                  enabled={toggles.door_auto ?? false}
                  onChange={v => setToggle('door_auto', v)}
                />
              </div>

              {/* Logic rule */}
              <div className="bg-stone-50 rounded-xl p-3 border border-stone-200">
                <p className="text-xs font-medium text-stone-400 uppercase tracking-wider mb-1.5">Rule</p>
                <p className="text-xs text-stone-600 leading-relaxed">
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
            <div className="rounded-2xl border border-stone-200 bg-white p-6 transition-all duration-200 hover:shadow-md">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🌀</span>
                  <h3 className="text-sm font-semibold text-stone-800">Ventilation Automation</h3>
                </div>
                <StatusDot active={r.ventilation_on} />
              </div>

              {/* Current status */}
              <div className={`rounded-xl px-4 py-3 mb-4 ${
                r.ventilation_on ? 'bg-sky-50 border border-sky-200' : 'bg-stone-50 border border-stone-200'
              }`}>
                <p className="text-xs text-stone-400 uppercase tracking-wider font-medium mb-1">Current Status</p>
                <p className={`text-lg font-bold ${r.ventilation_on ? 'text-sky-700' : 'text-stone-600'}`}>
                  Fan is {r.ventilation_on ? 'Running' : 'Off'}
                </p>
              </div>

              {/* Enable toggle */}
              <div className="flex items-center justify-between mb-4 px-1">
                <span className="text-sm text-stone-600">Enable automation</span>
                <Toggle
                  enabled={toggles.vent_auto ?? false}
                  onChange={v => setToggle('vent_auto', v)}
                />
              </div>

              {/* Logic rule */}
              <div className="bg-stone-50 rounded-xl p-3 border border-stone-200">
                <p className="text-xs font-medium text-stone-400 uppercase tracking-wider mb-1.5">Rule</p>
                <p className="text-xs text-stone-600 leading-relaxed">
                  Fan activates when mold risk or heat stress potential reaches warning or critical levels.
                  Turns off when conditions return to normal.
                </p>
              </div>

              {/* Related sensors */}
              <div className="mt-4 flex flex-wrap gap-2">
                <SensorBadge label="Mold risk" value={r.mold_risk_status} status={r.mold_risk_status} />
                <SensorBadge label="Heat stress" value={r.heat_stress_index} status={r.heat_stress_index} />
              </div>
            </div>
          </div>
        )}

        {/* Info note */}
        {r && (
          <div className="mt-6 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-xs text-amber-700">
            <strong>Note:</strong> Automation toggles are visual only for now. The hardware integration is being developed separately.
            Sensor status updates live every 15 seconds.
          </div>
        )}
      </div>
    </div>
  )
}
