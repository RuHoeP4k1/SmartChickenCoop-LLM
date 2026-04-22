const apiKey = import.meta.env.VITE_API_KEY
const authHeaders = apiKey ? { Authorization: `Bearer ${apiKey}` } : {}

export async function askQuestion(query, history = [], useSensors = true, useHybrid = true, includeTaskContext = false) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
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

export async function askQuestionStream(
  query,
  history = [],
  useSensors = true,
  useHybrid = true,
  includeTaskContext = false,
  { onChunk, onDone, onError } = {},
) {
  let res
  try {
    res = await fetch('/ask-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders },
      body: JSON.stringify({
        query, history,
        use_sensors: useSensors,
        use_hybrid: useHybrid,
        include_task_context: includeTaskContext,
      }),
    })
  } catch (err) {
    onError?.(err)
    return
  }

  if (!res.ok) {
    onError?.(new Error(`Request failed (${res.status})`))
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split('\n\n')
      buffer = frames.pop() // keep incomplete trailing frame

      for (const frame of frames) {
        if (!frame.startsWith('data: ')) continue
        try {
          const data = JSON.parse(frame.slice(6))
          if (data.type === 'chunk') onChunk?.(data.delta)
          else if (data.type === 'done') onDone?.(data)
          else if (data.type === 'error') onError?.(new Error(data.message))
        } catch {
          // ignore malformed frames
        }
      }
    }
  } catch (err) {
    onError?.(err)
  }
}

// ---------- Coop Log (eggs, chores, automation) ----------

export async function getEggCalendar(year, month) {
  const res = await fetch(`/eggs/calendar?year=${year}&month=${month}`)
  if (!res.ok) throw new Error(`Egg calendar failed (${res.status})`)
  return res.json()
}

export async function setEggEntry(isoDate, eggsLaid) {
  const res = await fetch(`/eggs/calendar/${isoDate}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify({ eggs_laid: eggsLaid }),
  })
  if (!res.ok) throw new Error(`Egg entry failed (${res.status})`)
  return res.json()
}

export async function triggerReconcile(isoDate) {
  const res = await fetch(`/eggs/reconcile?entry_date=${isoDate}`, {
    method: 'POST', headers: { ...authHeaders },
  })
  if (!res.ok) throw new Error(`Reconcile failed (${res.status})`)
  return res.json()
}

export async function getChoreDefinitions() {
  const res = await fetch('/chores/definitions')
  if (!res.ok) throw new Error(`Chore defs failed (${res.status})`)
  return res.json()
}

export async function addChoreLog(entry) {
  const res = await fetch('/chores/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify(entry),
  })
  if (!res.ok) throw new Error(`Chore log failed (${res.status})`)
  return res.json()
}

export async function deleteChoreLog(id) {
  const res = await fetch(`/chores/log/${id}`, { method: 'DELETE', headers: { ...authHeaders } })
  if (!res.ok) throw new Error(`Chore delete failed (${res.status})`)
  return res.json()
}

export async function getAutomationWindows(start, end) {
  const res = await fetch(`/automation/windows?start=${start}&end=${end}`)
  if (!res.ok) throw new Error(`Automation fetch failed (${res.status})`)
  return res.json()
}

export async function createAutomationWindow(window) {
  const res = await fetch('/automation/windows', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify(window),
  })
  if (!res.ok) throw new Error(`Automation create failed (${res.status})`)
  return res.json()
}

export async function deleteAutomationWindow(id) {
  const res = await fetch(`/automation/windows/${id}`, { method: 'DELETE', headers: { ...authHeaders } })
  if (!res.ok) throw new Error(`Automation delete failed (${res.status})`)
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

export async function evaluateAutomation(ventEnabled, doorEnabled) {
  const res = await fetch('/automation/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify({ vent_enabled: ventEnabled, door_enabled: doorEnabled }),
  })
  if (!res.ok) throw new Error(`Automation evaluate failed (${res.status})`)
  return res.json()
}

export async function getRiskLatest() {
  const res = await fetch('/risk/latest')
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`Risk fetch failed (${res.status})`)
  return res.json()
}

export async function getDeviceControl() {
  const res = await fetch('/device/control')
  if (!res.ok) throw new Error(`Device control fetch failed (${res.status})`)
  return res.json()
}

export async function setDeviceControl(patch) {
  const res = await fetch('/device/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify(patch),
  })
  if (!res.ok) throw new Error(`Device control set failed (${res.status})`)
  return res.json()
}
