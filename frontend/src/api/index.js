function getAuthHeaders() {
  const token = localStorage.getItem('access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function apiFetch(url, options = {}) {
  const res = await fetch(url, options)
  if (res.status === 401) {
    localStorage.removeItem('access_token')
    localStorage.removeItem('username')
    window.location.reload()
  }
  return res
}

export async function loginUser(username, password) {
  const res = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (res.status === 401) throw new Error('Invalid username or password')
  if (!res.ok) throw new Error(`Login failed (${res.status})`)
  return res.json()
}

export async function registerUser(username, password) {
  const res = await fetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (res.status === 409) throw new Error('Username already taken')
  if (!res.ok) throw new Error(`Register failed (${res.status})`)
  return res.json()
}

export async function getMe() {
  const token = localStorage.getItem('access_token')
  if (!token) return null
  const res = await fetch('/auth/me', { headers: { Authorization: `Bearer ${token}` } })
  if (res.status === 401) return null
  if (!res.ok) return null
  return res.json()
}

export function logout() {
  localStorage.removeItem('access_token')
  localStorage.removeItem('username')
}

export async function askQuestion(query, history = [], useSensors = true, useHybrid = true, includeTaskContext = false) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({
      query, history,
      use_sensors: useSensors,
      use_hybrid: useHybrid,
      include_task_context: includeTaskContext,
    }),
  })
  if (!res.ok) throw new Error(`Request failed (${res.status})`)
  return res.json()
}

// ---------- Coop Log (eggs, chores, automation) ----------

export async function getEggCalendar(year, month) {
  const res = await apiFetch(`/eggs/calendar?year=${year}&month=${month}`, { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Egg calendar failed (${res.status})`)
  return res.json()
}

export async function setEggEntry(isoDate, eggsLaid) {
  const res = await apiFetch(`/eggs/calendar/${isoDate}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ eggs_laid: eggsLaid }),
  })
  if (!res.ok) throw new Error(`Egg entry failed (${res.status})`)
  return res.json()
}

export async function triggerReconcile(isoDate) {
  const res = await apiFetch(`/eggs/reconcile?entry_date=${isoDate}`, {
    method: 'POST', headers: { ...getAuthHeaders() },
  })
  if (!res.ok) throw new Error(`Reconcile failed (${res.status})`)
  return res.json()
}

export async function getChoreDefinitions() {
  const res = await apiFetch('/chores/definitions', { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Chore defs failed (${res.status})`)
  return res.json()
}

export async function addChoreLog(entry) {
  const res = await apiFetch('/chores/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify(entry),
  })
  if (!res.ok) throw new Error(`Chore log failed (${res.status})`)
  return res.json()
}

export async function deleteChoreLog(id) {
  const res = await apiFetch(`/chores/log/${id}`, { method: 'DELETE', headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Chore delete failed (${res.status})`)
  return res.json()
}

export async function getAutomationWindows(start, end) {
  const res = await apiFetch(`/automation/windows?start=${start}&end=${end}`, { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Automation fetch failed (${res.status})`)
  return res.json()
}

export async function createAutomationWindow(window) {
  const res = await apiFetch('/automation/windows', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify(window),
  })
  if (!res.ok) throw new Error(`Automation create failed (${res.status})`)
  return res.json()
}

export async function deleteAutomationWindow(id) {
  const res = await apiFetch(`/automation/windows/${id}`, { method: 'DELETE', headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Automation delete failed (${res.status})`)
  return res.json()
}

export async function getSensors() {
  const res = await apiFetch('/sensors', { headers: { ...getAuthHeaders() } })
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`Sensors request failed (${res.status})`)
  return res.json()
}

export async function getHistory(range = '1h') {
  const res = await apiFetch(`/sensors/history?range=${range}`, { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`History request failed (${res.status})`)
  return res.json()
}

export async function getEvents(limit = 30, eventType = null) {
  const params = new URLSearchParams({ limit })
  if (eventType) params.append('event_type', eventType)
  const res = await apiFetch(`/events?${params}`, { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Events request failed (${res.status})`)
  return res.json()
}

export async function getWeather() {
  const res = await apiFetch('/weather')
  if (!res.ok) throw new Error(`Weather request failed (${res.status})`)
  return res.json()
}

export async function getReviewEvents(limit = 50, reviewed = null) {
  const params = new URLSearchParams({ limit })
  if (reviewed !== null) params.append('reviewed', reviewed)
  const res = await apiFetch(`/reviews?${params}`, { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Reviews request failed (${res.status})`)
  return res.json()
}

export async function submitReview(review) {
  const res = await apiFetch('/reviews', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify(review),
  })
  if (!res.ok) throw new Error(`Review submit failed (${res.status})`)
  return res.json()
}

export async function exportReviews() {
  const res = await apiFetch('/reviews/export', { headers: { ...getAuthHeaders() } })
  if (!res.ok) throw new Error(`Export failed (${res.status})`)
  return res.json()
}

export async function evaluateAutomation(ventEnabled, doorEnabled) {
  const res = await apiFetch('/automation/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ vent_enabled: ventEnabled, door_enabled: doorEnabled }),
  })
  if (!res.ok) throw new Error(`Automation evaluate failed (${res.status})`)
  return res.json()
}

export async function getRiskLatest() {
  const res = await apiFetch('/risk/latest', { headers: { ...getAuthHeaders() } })
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`Risk fetch failed (${res.status})`)
  return res.json()
}
