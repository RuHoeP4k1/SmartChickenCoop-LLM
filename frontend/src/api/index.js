const apiKey = import.meta.env.VITE_API_KEY
const authHeaders = apiKey ? { Authorization: `Bearer ${apiKey}` } : {}

export async function askQuestion(query, history = [], useSensors = true, useHybrid = true) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify({ query, history, use_sensors: useSensors, use_hybrid: useHybrid }),
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

export async function getWeather() {
  const res = await fetch('/weather')
  if (!res.ok) throw new Error(`Weather request failed (${res.status})`)
  return res.json()
}

export async function getReviewEvents(limit = 50, reviewed = null) {
  const params = new URLSearchParams({ limit })
  if (reviewed !== null) params.append('reviewed', reviewed)
  const res = await fetch(`/reviews?${params}`)
  if (!res.ok) throw new Error(`Reviews request failed (${res.status})`)
  return res.json()
}

export async function submitReview(review) {
  const res = await fetch('/reviews', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify(review),
  })
  if (!res.ok) throw new Error(`Review submit failed (${res.status})`)
  return res.json()
}

export async function exportReviews() {
  const res = await fetch('/reviews/export')
  if (!res.ok) throw new Error(`Export failed (${res.status})`)
  return res.json()
}
