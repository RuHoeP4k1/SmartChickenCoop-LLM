export async function askQuestion(query, useSensors = true, useHybrid = true) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, use_sensors: useSensors, use_hybrid: useHybrid }),
  })
  if (!res.ok) throw new Error(`Request failed (${res.status})`)
  return res.json()
}

export async function getSensors() {
  const res = await fetch('/sensors')
  if (res.status === 404) return null  // no sensor data written yet
  if (!res.ok) throw new Error(`Sensors request failed (${res.status})`)
  return res.json()
}

export async function getHistory(range = '1h') {
  const res = await fetch(`/sensors/history?range=${range}`)
  if (!res.ok) throw new Error(`History request failed (${res.status})`)
  return res.json()
}

export async function getEvents(limit = 30, eventType = null) {
  const params = new URLSearchParams({ limit })
  if (eventType) params.append('event_type', eventType)
  const res = await fetch(`/events?${params}`)
  if (!res.ok) throw new Error(`Events request failed (${res.status})`)
  return res.json()
}
