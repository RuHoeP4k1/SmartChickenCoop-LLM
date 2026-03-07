import { useState, useEffect } from 'react'
import { getWeather } from '../api'

// WMO weather interpretation codes → human-readable labels
const WMO = {
  0: 'Clear sky',
  1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Icy fog',
  51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
  61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow',
  77: 'Snow grains',
  80: 'Rain showers', 81: 'Rain showers', 82: 'Heavy showers',
  85: 'Snow showers', 86: 'Heavy snow showers',
  95: 'Thunderstorm', 96: 'Thunderstorm + hail', 99: 'Thunderstorm + hail',
}

function wmoLabel(code) {
  return WMO[code] ?? `Code ${code}`
}

// Short weekday from ISO date string e.g. "2025-03-07"
function shortDay(dateStr, index) {
  if (index === 0) return 'Today'
  const d = new Date(dateStr + 'T12:00:00')
  return d.toLocaleDateString('en-BE', { weekday: 'short' })
}

// Format ISO datetime to HH:MM
function formatTime(isoStr) {
  if (!isoStr) return '—'
  const d = new Date(isoStr)
  return d.toLocaleTimeString('en-BE', { hour: '2-digit', minute: '2-digit', hour12: false })
}

function SimpleCard({ label, value, unit, sub }) {
  return (
    <div className="rounded-2xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-5 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5">
      <div className="text-xs font-medium text-stone-500 dark:text-stone-400 uppercase tracking-wider mb-3">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-bold tabular-nums text-stone-800 dark:text-stone-100">{value}</span>
        {unit && <span className="text-base text-stone-400 dark:text-stone-500">{unit}</span>}
      </div>
      {sub && <div className="mt-1 text-xs text-stone-400 dark:text-stone-500">{sub}</div>}
    </div>
  )
}

function SunCard({ sunrise, sunset, daylightSec }) {
  const hours = daylightSec ? (daylightSec / 3600).toFixed(1) : '—'
  return (
    <div className="rounded-2xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 p-5">
      <div className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-4">Daylight · Leuven</div>
      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-xs text-stone-400 dark:text-stone-500 mb-1">Sunrise</div>
          <div className="text-xl font-bold text-stone-800 dark:text-stone-100">{formatTime(sunrise)}</div>
        </div>
        <div>
          <div className="text-xs text-stone-400 dark:text-stone-500 mb-1">Daylight</div>
          <div className="text-xl font-bold text-amber-600 dark:text-amber-400">{hours}h</div>
        </div>
        <div>
          <div className="text-xs text-stone-400 dark:text-stone-500 mb-1">Sunset</div>
          <div className="text-xl font-bold text-stone-800 dark:text-stone-100">{formatTime(sunset)}</div>
        </div>
      </div>
    </div>
  )
}

function ForecastRow({ dates, maxTemps, minTemps, precipSums, codes }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider mb-3">3-Day Forecast</h3>
      <div className="grid grid-cols-3 gap-3">
        {dates.map((date, i) => (
          <div key={date} className="rounded-2xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-4 text-center">
            <div className="text-xs font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider mb-2">
              {shortDay(date, i)}
            </div>
            <div className="text-xs text-stone-500 dark:text-stone-400 mb-3 leading-snug min-h-[2.5rem] flex items-center justify-center">
              {wmoLabel(codes[i])}
            </div>
            <div className="flex items-baseline justify-center gap-1.5 mb-1">
              <span className="text-lg font-bold text-stone-800 dark:text-stone-100">{maxTemps[i] != null ? Math.round(maxTemps[i]) : '—'}°</span>
              <span className="text-sm text-stone-400 dark:text-stone-500">/ {minTemps[i] != null ? Math.round(minTemps[i]) : '—'}°</span>
            </div>
            {precipSums[i] > 0 && (
              <div className="text-xs text-sky-500 dark:text-sky-400">{precipSums[i].toFixed(1)} mm</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Weather() {
  const [data, setData] = useState(undefined)   // undefined = loading
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  async function load() {
    try {
      const result = await getWeather()
      setData(result)
      setError(null)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    load()
    const id = setInterval(load, 600_000)  // refresh every 10 minutes
    return () => clearInterval(id)
  }, [])

  const c = data?.current ?? {}
  const d = data?.daily ?? {}

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-stone-800 dark:text-stone-100">Outdoor Conditions</h1>
            <p className="text-sm text-stone-500 dark:text-stone-400 mt-0.5">Leuven, Belgium · Open-Meteo</p>
          </div>
          <div className="flex items-center gap-3">
            {lastUpdated && (
              <span className="text-xs text-stone-400 dark:text-stone-500">
                Updated {lastUpdated.toLocaleTimeString('en-BE', { hour: '2-digit', minute: '2-digit', hour12: false })}
              </span>
            )}
            <button
              onClick={load}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Loading skeleton */}
        {data === undefined && !error && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 animate-pulse">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-28 rounded-2xl bg-stone-200 dark:bg-stone-700" />
            ))}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-2xl border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 p-5 text-red-600 dark:text-red-400 text-sm">
            Could not load weather data: {error}
          </div>
        )}

        {/* Data */}
        {data && (
          <div className="space-y-6">

            {/* Current conditions */}
            <div>
              <h3 className="text-sm font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider mb-3">Current Conditions</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <SimpleCard
                  label="Temperature"
                  value={c.temperature_2m != null ? c.temperature_2m.toFixed(1) : '—'}
                  unit="°C"
                  sub={wmoLabel(c.weather_code)}
                />
                <SimpleCard
                  label="Feels Like"
                  value={c.apparent_temperature != null ? c.apparent_temperature.toFixed(1) : '—'}
                  unit="°C"
                />
                <SimpleCard
                  label="Humidity"
                  value={c.relative_humidity_2m != null ? c.relative_humidity_2m : '—'}
                  unit="%"
                />
                <SimpleCard
                  label="Wind Speed"
                  value={c.wind_speed_10m != null ? c.wind_speed_10m.toFixed(1) : '—'}
                  unit="km/h"
                />
                <SimpleCard
                  label="Precipitation"
                  value={c.precipitation != null ? c.precipitation.toFixed(1) : '—'}
                  unit="mm"
                  sub="last hour"
                />
              </div>
            </div>

            {/* Sunrise / Sunset */}
            {d.sunrise && (
              <SunCard
                sunrise={d.sunrise[0]}
                sunset={d.sunset[0]}
                daylightSec={d.daylight_duration?.[0]}
              />
            )}

            {/* 3-day forecast */}
            {d.time && (
              <ForecastRow
                dates={d.time}
                maxTemps={d.temperature_2m_max}
                minTemps={d.temperature_2m_min}
                precipSums={d.precipitation_sum}
                codes={d.weather_code}
              />
            )}

          </div>
        )}
      </div>
    </div>
  )
}
