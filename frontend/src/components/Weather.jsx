import { useState, useEffect } from 'react'
import { getWeather, setLocation } from '../api'

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

function wmoLabel(code) { return WMO[code] ?? `Code ${code}` }

function shortDay(dateStr, index) {
  if (index === 0) return 'Today'
  return new Date(dateStr + 'T12:00:00').toLocaleDateString('en-BE', { weekday: 'short' })
}

function formatTime(isoStr) {
  if (!isoStr) return '—'
  return new Date(isoStr).toLocaleTimeString('en-BE', { hour: '2-digit', minute: '2-digit', hour12: false })
}

function SimpleCard({ label, value, unit, sub }) {
  return (
    <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-5 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5">
      <div className="text-xs font-medium text-stone-500 dark:text-stone-400 uppercase tracking-wider mb-3">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-bold tabular-nums text-stone-800 dark:text-stone-100">{value}</span>
        {unit && <span className="text-base text-stone-400 dark:text-stone-500">{unit}</span>}
      </div>
      {sub && <div className="mt-1 text-xs text-stone-400 dark:text-stone-500">{sub}</div>}
    </div>
  )
}

function SunCard({ sunrise, sunset, daylightSec, locationName }) {
  const hours = daylightSec ? (daylightSec / 3600).toFixed(1) : '—'
  return (
    <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 p-5">
      <div className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-4">
        Daylight{locationName ? ` · ${locationName}` : ''}
      </div>
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
          <div key={date} className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-4 text-center">
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

function LocationPicker({ onSaved }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [saving, setSaving] = useState(false)
  const [searchError, setSearchError] = useState(null)

  async function search(e) {
    e.preventDefault()
    if (!query.trim()) return
    setSearching(true)
    setSearchError(null)
    setResults([])
    try {
      const res = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=5&language=en&format=json`
      )
      const data = await res.json()
      setResults(data.results ?? [])
      if (!data.results?.length) setSearchError('No locations found.')
    } catch {
      setSearchError('Search failed. Check your connection.')
    } finally {
      setSearching(false)
    }
  }

  async function pick(result) {
    setSaving(true)
    try {
      const name = [result.name, result.admin1, result.country].filter(Boolean).join(', ')
      await setLocation(result.latitude, result.longitude, name)
      onSaved(name)
    } catch {
      setSearchError('Could not save location.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-800 p-6 mb-6">
      <h3 className="text-sm font-semibold text-stone-700 dark:text-stone-200 mb-1">Set your coop location</h3>
      <p className="text-xs text-stone-500 dark:text-stone-400 mb-4">
        Search for your city or village to get local weather for your coop.
      </p>
      <form onSubmit={search} className="flex gap-2 mb-3">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g. Ghent, Brussels, London…"
          className="flex-1 px-3 py-2 rounded-lg border border-stone-300 dark:border-stone-600
                     bg-white dark:bg-stone-700 text-stone-900 dark:text-stone-100 text-sm
                     focus:outline-none focus:ring-2 focus:ring-amber-400"
        />
        <button
          type="submit"
          disabled={searching}
          className="px-4 py-2 rounded-lg bg-amber-400 hover:bg-amber-500 text-stone-900 font-medium text-sm
                     disabled:opacity-50 transition-colors"
        >
          {searching ? '…' : 'Search'}
        </button>
      </form>
      {searchError && <p className="text-xs text-red-500 dark:text-red-400 mb-2">{searchError}</p>}
      {results.length > 0 && (
        <ul className="space-y-1">
          {results.map((r, i) => (
            <li key={i}>
              <button
                onClick={() => pick(r)}
                disabled={saving}
                className="w-full text-left px-3 py-2 rounded-lg text-sm
                           hover:bg-amber-50 dark:hover:bg-amber-900/20
                           text-stone-700 dark:text-stone-300 transition-colors
                           disabled:opacity-50"
              >
                <span className="font-medium">{r.name}</span>
                {r.admin1 && <span className="text-stone-400 dark:text-stone-500"> · {r.admin1}</span>}
                {r.country && <span className="text-stone-400 dark:text-stone-500"> · {r.country}</span>}
                <span className="text-xs text-stone-400 dark:text-stone-500 ml-2">
                  {r.latitude.toFixed(2)}, {r.longitude.toFixed(2)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function Weather() {
  const [data, setData] = useState(undefined)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [showPicker, setShowPicker] = useState(false)

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
    const id = setInterval(load, 600_000)
    return () => clearInterval(id)
  }, [])

  function handleLocationSaved(name) {
    setShowPicker(false)
    load()
  }

  const c = data?.current ?? {}
  const d = data?.daily ?? {}
  const locationName = data?.location_name ?? null

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-stone-800 dark:text-stone-100">Outdoor Conditions</h1>
            <p className="text-sm text-stone-500 dark:text-stone-400 mt-0.5">
              {locationName ? `${locationName} · Open-Meteo` : 'Open-Meteo'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {lastUpdated && (
              <span className="text-xs text-stone-400 dark:text-stone-500">
                Updated {lastUpdated.toLocaleTimeString('en-BE', { hour: '2-digit', minute: '2-digit', hour12: false })}
              </span>
            )}
            <button
              onClick={() => setShowPicker(v => !v)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors"
            >
              {locationName ? 'Change location' : 'Set location'}
            </button>
            <button
              onClick={load}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Location picker */}
        {(showPicker || (!locationName && data !== undefined && !error)) && (
          <LocationPicker onSaved={handleLocationSaved} />
        )}

        {/* Loading skeleton */}
        {data === undefined && !error && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 animate-pulse">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-28 rounded-xl bg-stone-200 dark:bg-stone-700" />
            ))}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 p-5 text-red-600 dark:text-red-400 text-sm">
            Could not load weather data: {error}
          </div>
        )}

        {/* Data */}
        {data && (
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider mb-3">Current Conditions</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <SimpleCard label="Temperature" value={c.temperature_2m != null ? c.temperature_2m.toFixed(1) : '—'} unit="°C" sub={wmoLabel(c.weather_code)} />
                <SimpleCard label="Feels Like" value={c.apparent_temperature != null ? c.apparent_temperature.toFixed(1) : '—'} unit="°C" />
                <SimpleCard label="Humidity" value={c.relative_humidity_2m != null ? c.relative_humidity_2m : '—'} unit="%" />
                <SimpleCard label="Wind Speed" value={c.wind_speed_10m != null ? c.wind_speed_10m.toFixed(1) : '—'} unit="km/h" />
                <SimpleCard label="Precipitation" value={c.precipitation != null ? c.precipitation.toFixed(1) : '—'} unit="mm" sub="last hour" />
              </div>
            </div>

            {d.sunrise && (
              <SunCard
                sunrise={d.sunrise[0]}
                sunset={d.sunset[0]}
                daylightSec={d.daylight_duration?.[0]}
                locationName={locationName}
              />
            )}

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
