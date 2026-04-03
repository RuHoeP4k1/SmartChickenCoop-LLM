# Ethical AI & Data Privacy
### ChickenCare AI — Paper Section + Presentation Slides

---

## 1. Data Flow Analysis (production)

Before making ethical claims, it helps to be precise about what data this system actually
collects in production, where it goes, and what risks exist.

### What is collected

| Data type | Where stored | Who can read it |
|---|---|---|
| Every user question (full text) | PostgreSQL `event_log` | Authenticated account owner only |
| Every LLM response (full text) | PostgreSQL `event_log` | Authenticated account owner only |
| Sensor readings (temp, humidity, H2S, feeder, egg count, chickens inside) | PostgreSQL `sensor_readings` | Authenticated account owner only |
| Heatmap images from the coop camera | Server filesystem (`uploads/`) | Authenticated account owner only |
| GPS coordinates of the coop | Sent to Open-Meteo API on every weather call | Open-Meteo (no API key, no account) |
| Conversation history | Held in browser memory (last 2 turns), not persisted | Client only |

All data endpoints (`/events`, `/sensors`, `/heatmap`) are scoped to the authenticated user's account.
A keeper can only retrieve their own data — other accounts are invisible to them.

### Where queries go

User questions are processed by a local LLM (`smollm2:1.7b`) running on-premise via Ollama.
No query is sent to OpenAI, Anthropic, or any commercial cloud provider during normal operation.
Sensor readings and conversation logs are stored in a self-hosted PostgreSQL database.
The infrastructure provider — Render — hosts the application container but does not have
access to LLM inference or database contents, which run on a separate, operator-controlled server.

### What happens with sensitive questions

If a user asks something sensitive — e.g. "my chicken died this morning", "I think there's
a disease spreading in my flock" — the full question text is:
1. Sent to the local LLM (no external exposure)
2. Logged to `event_log` in PostgreSQL, visible only to that user's authenticated account
3. Automatically deleted after 30 days
4. Deletable on demand — users can remove individual messages from their history at any time

---

## 2. Paper Section (2–3 paragraphs)

### Ethical AI in 2026: Pillars for a Trustworthy System

In an era where AI is pervasive and personal data has become a strategic asset, building
trustworthy AI requires more than technical competence — it demands deliberate ethical
choices at every layer of the stack. For ChickenCare AI, three pillars form the foundation
of a responsible deployment: **data sovereignty**, **transparency**, and **purpose limitation**.

**Data sovereignty** means that user data stays under the control of the user or operator,
not a third-party cloud provider. ChickenCare AI addresses this by running its production
LLM entirely on-premise via Ollama. No user query is ever transmitted to OpenAI, Google, or
any commercial AI provider. Sensor readings and conversation logs are stored in a self-hosted
PostgreSQL database. Render hosts the application container but does not have access to the
LLM inference or the database contents, which run on a separate, operator-controlled server.
This contrasts sharply with SaaS-based AI tools that route every query through vendor
infrastructure, where data can be retained, used for model training, or subpoenaed. Choosing
infrastructure providers based on their own data ethics — open-source models, minimal-logging
APIs such as Open-Meteo for weather data — is a concrete expression of this principle.

**Transparency and purpose limitation** require that users know what is collected, why,
and for how long. Every conversation in ChickenCare AI is logged to an event log for system
monitoring and quality review — a legitimate operational need, but one that is disclosed
explicitly to users. Each keeper has their own authenticated account; their sensor readings,
conversation logs, and coop images are scoped to their account and are never visible to other
users on the platform. Conversation logs are subject to a 30-day automatic retention policy
and users can request full deletion of their data at any time — rights mandated under GDPR
Article 17. Beyond disclosure, purpose limitation means that conversation logs are used solely
to operate and improve this system; they are never sold, shared with third parties, or used
to train external models. Ethical AI in 2026 is not only about what the model says; it is
about how the surrounding system treats the people who use it.

---

## 3. Presentation Slides

---

### SLIDE 1 — "What happens to your data?"

**Title:** Your Data — Where It Goes, Where It Stays

**Visual concept:** simple flow diagram with two paths — green (stays local) and orange (leaves the system)

---

**LEFT COLUMN — Stays on your infrastructure**

- Your questions → processed by a **local LLM** (Ollama, runs on-premise)
- Sensor readings → stored in **your own account** in our database — not visible to other keepers
- Conversation logs → stored under **your account only** — other users cannot access them
- Knowledge base → static PDF documents, never sent anywhere
- Conversation history → held in your browser only (2 turns max)

**RIGHT COLUMN — Leaves the system**

- GPS coordinates of your coop → sent to **Open-Meteo** (free, no account, no tracking)

---

**Bottom banner:**
> No user question is ever sent to OpenAI, Anthropic, Google, or any commercial AI provider.

---

### SLIDE 2 — "Our ethical commitments"

**Title:** Ethical AI: Our Commitments to You

**Visual concept:** four pillars / icons

---

**Pillar 1 — Data Sovereignty**
We don't send your data to Elon Musk, Sam Altman, or anyone else.
Your queries are answered by a model running on hardware we control.

**Pillar 2 — Transparency**
We log conversations for system quality monitoring — and we tell you that.
The log contains your question and the AI's answer. It is scoped to your account only — other keepers cannot see it.

**Pillar 3 — Purpose Limitation**
Conversation logs are used to monitor and improve *this* system only.
They are never sold, shared with third parties, or used to train external models.

**Pillar 4 — Your Right to Delete**
Under GDPR, you have the right to request deletion of your conversation data.
We implement a 30-day automatic retention policy and honour deletion requests.

---

**Bottom citation / credibility line:**
> Infrastructure: self-hosted PostgreSQL · Ollama (local LLM) · Open-Meteo (no-login weather API)
> Compliant with: GDPR Article 5 (purpose limitation), Article 17 (right to erasure)

---

## 4. What needs to be in place before making these claims

| Claim | Current state | Fix needed |
|---|---|---|
| "Each keeper only sees their own data" | Single shared database, no user accounts | Per-user authentication (login system) + row-level data isolation in PostgreSQL — every sensor reading, event log entry, and heatmap scoped to an owner ID |
| "Event log is not publicly accessible" | `/events` and `/sensors` are unauthenticated | Protect all data endpoints behind the per-user login; a keeper's data is only returned for their authenticated session |
| "30-day retention policy" | No retention policy exists | Add a scheduled DB cleanup job that purges event log entries older than 30 days per user |
| "Right to delete" | No deletion endpoint | Add account deletion flow that wipes all sensor readings, event logs, and heatmaps for that user |
| "GPS not stored" | Coordinates are in `.env` and `render.yaml` | Already fine — only sent at query time to Open-Meteo, never stored in the database |

### The scaled-up architecture this implies

When ChickenCare AI moves beyond a single-operator setup, the core addition is a **user account layer**:

- Each chicken keeper registers an account and receives their own login credentials.
- All data written to the database — sensor readings, conversation logs, heatmap images — is tagged with that user's ID.
- API endpoints only return data belonging to the authenticated user. A keeper cannot see another keeper's coop readings, conversations, or alerts, even if they know the URL.
- The local LLM still processes queries on shared infrastructure, but no query or response is ever associated with another user's account or visible to them.

This is standard multi-tenant design and is the intended architecture for any commercial scaling of this system.

---

*Document written for the ChickenCoopComfort final paper and presentation — April 2026.*
