# Implementation Plan: Per-User Login System + Row-Level Data Isolation

> Based on the fixes required in `ETHICS_DATA_PRIVACY_OVERVIEW.md` — Fix #1.
> **Current state:** single shared database, no user accounts.
> **Goal:** every sensor reading, event log entry, chore, egg entry, automation window, and heatmap is scoped to an `owner_id`. A keeper can only retrieve their own data.

---

## What needs to be in place

| Claim | Current state | Fix |
|---|---|---|
| "Each keeper only sees their own data" | Single shared DB, no user accounts | Per-user auth + row-level isolation on every table |
| "Event log is not publicly accessible" | `/events`, `/sensors` are unauthenticated | Protect all data endpoints behind JWT auth |

---

## Step 1 — Database: `users` table + `owner_id` columns

**File:** `backend/db_utils.py`

Add two new SQL constants and wire them into `setup_database()`.

### 1a. `CREATE_USERS_SQL`

```sql
CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    username   TEXT NOT NULL UNIQUE,
    hashed_pw  TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
```

### 1b. `MIGRATE_ADD_OWNER_ID_SQL`

Add `owner_id INTEGER REFERENCES users(id)` (nullable, `IF NOT EXISTS`) to all 8 user-scoped tables:

```sql
ALTER TABLE sensor_readings_colson   ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE cv_counts_colson         ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE risk_snapshots           ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE event_log                ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE response_reviews         ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE egg_calendar_entries     ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE chore_log                ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
ALTER TABLE automation_windows       ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);

CREATE INDEX IF NOT EXISTS idx_sensor_owner     ON sensor_readings_colson(owner_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_event_owner      ON event_log(owner_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_egg_owner        ON egg_calendar_entries(owner_id, entry_date);
CREATE INDEX IF NOT EXISTS idx_chore_owner      ON chore_log(owner_id, entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_automation_owner ON automation_windows(owner_id, start_date, end_date);
```

Old rows get `owner_id = NULL` — intentional. Legacy data is treated as "unowned" and is never returned to authenticated users.

### 1c. `egg_calendar_entries` PK change

The current PK is `(entry_date)` alone. Two users can now log eggs on the same date, so the PK must become composite:

```sql
ALTER TABLE egg_calendar_entries DROP CONSTRAINT egg_calendar_entries_pkey;
ALTER TABLE egg_calendar_entries ADD PRIMARY KEY (owner_id, entry_date);
```

> **Gotcha:** `DROP CONSTRAINT` will fail if existing rows have `owner_id IS NULL`. For a clean dev DB, truncate the table first. Also update `upsert_egg_entry()`'s `ON CONFLICT (entry_date)` clause to `ON CONFLICT (owner_id, entry_date)`.

### 1d. New helper functions in `db_utils.py`

```python
def create_user(username: str, hashed_pw: str) -> int:
    # INSERT INTO users ... RETURNING id

def get_user_by_username(username: str) -> Optional[Dict]:
    # returns {id, username, hashed_pw} or None
```

### 1e. Update all user-scoped DB functions

Every function that reads or writes user data must accept `owner_id: int` and filter/tag accordingly.

| Function | Change |
|---|---|
| `get_latest_sensor_reading()` | add `owner_id` param, `WHERE s.owner_id = %s` |
| `get_recent_readings()` | add `owner_id` param |
| `get_sensor_history()` | add `owner_id` param |
| `insert_sensor_reading()` | add `owner_id` to INSERT |
| `insert_cv_count()` | add `owner_id` to INSERT |
| `get_latest_cv_count()` | add `owner_id` param |
| `insert_event()` | add `owner_id` to INSERT (default `None` for system events) |
| `get_recent_events()` | add `owner_id` param |
| `upsert_egg_entry()` | add `owner_id` + update `ON CONFLICT` target |
| `get_egg_entries_for_month()` | add `owner_id` param |
| `get_cv_egg_counts_for_date()` | add `owner_id` param |
| `insert_chore_log()` | add `owner_id` |
| `delete_chore_log()` | add `AND owner_id = %s` guard |
| `get_chore_log_in_range()` | add `AND cl.owner_id = %s` |
| `get_feeder_waterer_samples_for_date()` | add `owner_id` param |
| `insert_automation_window()` | add `owner_id` |
| `delete_automation_window()` | add `AND owner_id = %s` guard |
| `get_automation_windows_in_range()` | add `AND owner_id = %s` |
| `get_events_for_review()` | add `AND e.owner_id = %s` |
| `export_reviews()` | add `AND e.owner_id = %s` |
| `insert_risk_snapshot()` | add `owner_id` |
| `get_latest_risk_snapshot()` | add `owner_id` param |

> `get_chore_definitions()` is **global** (shared across users) — no change needed.

> **Delete endpoint gotcha:** always add `AND owner_id = %s` to DELETE queries so user A cannot delete user B's rows by guessing integer IDs. If `rowcount == 0`, return 404 — the row either doesn't exist or belongs to someone else.

---

## Step 2 — New file: `backend/auth.py`

All JWT logic lives here. Nothing else imports it except `app.py`.

### 2a. Add to `requirements.txt`

```
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

> `bcrypt==5.0.0` is already present — `passlib[bcrypt]` will use it. No version conflict.

### 2b. Contents of `backend/auth.py`

```python
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os

SECRET_KEY = os.environ["JWT_SECRET"]   # hard error if missing — intentional fail-fast
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24h default

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str: ...
def verify_password(plain: str, hashed: str) -> bool: ...

def create_access_token(data: dict) -> str:
    # payload: {"sub": str(user_id), "username": username, "exp": ...}

def decode_token(token: str) -> dict:
    # raises JWTError on invalid/expired token
    # returns the payload dict
```

> **JWT spec:** use `"sub": str(user_id)` (string). Cast back to `int` when extracting `owner_id` in the FastAPI dependency.

---

## Step 3 — `app.py`: auth endpoints + JWT dependency

**File:** `app.py`

### 3a. New Pydantic models

```python
class RegisterRequest(BaseModel):
    username: Annotated[str, StringConstraints(min_length=3, max_length=50)]
    password: Annotated[str, StringConstraints(min_length=8, max_length=100)]

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    user_id: int
```

### 3b. New auth endpoints

| Endpoint | Method | Request | Response | Errors |
|---|---|---|---|---|
| `/auth/register` | POST | `{username, password}` | `{id, username}` | 409 if username taken |
| `/auth/login` | POST | `{username, password}` | `{access_token, token_type, username, user_id}` | 401 if wrong credentials |
| `/auth/me` | GET | — (token in header) | `{user_id, username}` | 401 if invalid/expired |

### 3c. Replace `verify_key` with JWT dependency

Remove the existing `_API_KEY` / `HTTPBearer` / `verify_key` block. Replace with:

```python
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from backend.auth import decode_token

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(_oauth2_scheme)) -> dict:
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user_id": int(payload["sub"]), "username": payload["username"]}
```

### 3d. Thread `owner_id` through all protected endpoints

Replace `_=Depends(verify_key)` with `current_user: dict = Depends(get_current_user)`, then extract `owner_id = current_user["user_id"]` and pass it to every DB call.

**Protected endpoints (require auth):**

- `POST /ask` — pass `owner_id` to `insert_event()`
- `GET /events` — pass `owner_id` to `get_recent_events()`
- `GET /sensors` — pass `owner_id` to `get_latest_sensor_reading()`
- `GET /sensors/history` — pass `owner_id`
- `GET /risk/latest` — pass `owner_id` to `get_latest_risk_snapshot()`
- `GET /eggs/calendar` — pass `owner_id`
- `POST /eggs/calendar/{entry_date}` — pass `owner_id`
- `POST /eggs/reconcile` — pass `owner_id`
- `POST /chores/log` — pass `owner_id`
- `DELETE /chores/log/{log_id}` — pass `owner_id` for ownership check
- `GET /automation/windows` — pass `owner_id`
- `POST /automation/windows` — pass `owner_id`
- `DELETE /automation/windows/{win_id}` — pass `owner_id`
- `POST /automation/evaluate` — pass `owner_id` to `get_latest_sensor_reading()`
- `GET /reviews` — pass `owner_id` to `get_events_for_review()`
- `GET /reviews/export` — pass `owner_id` to `export_reviews()`
- `POST /heatmap/upload` — pass `owner_id`, save to `uploads/heatmaps/{owner_id}/`
- `GET /heatmap/latest` — scope to `uploads/heatmaps/{owner_id}/`

**Unprotected endpoints (stay open):**

- `GET /health`
- `GET /api/info`
- `GET /weather`
- `GET /chores/definitions`
- `POST /setup-db`

### 3e. Heatmap isolation

Store files per user in `uploads/heatmaps/{owner_id}/` instead of a shared flat directory. Serve via an authenticated API route (not raw `StaticFiles`) so file URLs are access-controlled.

### 3f. Scheduler gotcha

`scheduler.py` calls `insert_event()` as a background system process with no user context. Keep `owner_id` defaulting to `None` in `insert_event()` — these are "system events". The alerts feed should show the current user's LLM events plus all unowned system events.

---

## Step 4 — Pi sensor writer: `scripts/pi_sensor_writer.py`

Since `owner_id` is nullable, existing Pi writes continue to work (they produce `owner_id = NULL` rows, which no authenticated user will see). To properly tag data with an owner:

1. Add to the Pi's `.env`:
   ```
   SENSOR_OWNER_ID=<user id of the Pi's owner>
   ```

2. Read it in `pi_sensor_writer.py`:
   ```python
   _SENSOR_OWNER_ID = int(os.getenv("SENSOR_OWNER_ID", "0")) or None
   ```
   Pass `owner_id=_SENSOR_OWNER_ID` when calling `insert_sensor_reading()`.

> **Recommended:** create a dedicated user (e.g., `pi_colson`) during setup, retrieve its ID, and write that ID into the Pi `.env`. This ties sensor data to a real user account, matching the ethics requirement.

---

## Step 5 — Frontend: `frontend/src/api/index.js`

Replace static `VITE_API_KEY` header with a dynamic function reading from `localStorage`:

```js
// Remove:
const apiKey = import.meta.env.VITE_API_KEY
const authHeaders = apiKey ? { Authorization: `Bearer ${apiKey}` } : {}

// Replace with:
function getAuthHeaders() {
  const token = localStorage.getItem('access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}
```

Replace every `...authHeaders` spread with `...getAuthHeaders()`.

Add a global `apiFetch()` wrapper to intercept 401 responses:

```js
async function apiFetch(url, options = {}) {
  const res = await fetch(url, options)
  if (res.status === 401) {
    localStorage.removeItem('access_token')
    window.location.reload()
  }
  return res
}
```

Replace all `fetch(...)` calls in this file with `apiFetch(...)`.

Add new exported functions:

```js
export async function loginUser(username, password) { ... }
  // POST /auth/login → { access_token, token_type, username, user_id }
  // Throws on 401

export async function registerUser(username, password) { ... }
  // POST /auth/register → { id, username }
  // Throws 'Username already taken' on 409

export async function getMe() { ... }
  // GET /auth/me → { user_id, username } or null on 401

export function logout() {
  localStorage.removeItem('access_token')
  localStorage.removeItem('username')
}
```

---

## Step 6 — New file: `frontend/src/components/LoginPage.jsx`

A simple two-mode form (Login / Register toggle).

**Behaviour:**
1. On submit: calls `loginUser()` or (`registerUser()` then `loginUser()`)
2. On success: stores `access_token` + `username` in `localStorage`, calls `onLogin(username)` prop
3. On error: shows inline error message

**Visual style:** centered card using `bg-white dark:bg-stone-800 rounded-2xl shadow-lg p-8` — matching existing component language. Include the chicken logo.

---

## Step 7 — Frontend: `frontend/src/App.jsx`

Add auth state. On mount, call `getMe()` to validate the stored token. Gate the whole app behind the result:

```jsx
const [currentUser, setCurrentUser] = useState(null)
const [authChecked, setAuthChecked] = useState(false)

useEffect(() => {
  getMe().then(user => {
    setCurrentUser(user)   // null if token missing/expired
    setAuthChecked(true)
  })
}, [])

if (!authChecked) return null          // brief loading pause
if (!currentUser) return <LoginPage onLogin={handleLogin} />
// else: render normal tab layout
```

Pass `currentUser` and `onLogout` down to `Layout` so it can show the username and a logout button in the sidebar (next to the dark mode toggle in `Layout.jsx`).

> **Token persistence:** `localStorage` survives page refresh, so users stay logged in for 24h (configurable via `JWT_EXPIRE_MINUTES`). The `getMe()` check on mount validates the token server-side before trusting it.

---

## Step 8 — Environment variables

Add to `.env`:

```
JWT_SECRET=<run: python -c "import secrets; print(secrets.token_hex(32))">
JWT_EXPIRE_MINUTES=1440
```

The app will **crash at startup** if `JWT_SECRET` is missing — this is intentional.

Add to the Pi's `.env`:

```
SENSOR_OWNER_ID=<user id of the Pi's owner>
```

Remove `VITE_API_KEY` from the frontend `.env` once the JWT migration is complete.

---

## Sequencing and dependencies

```
Step 1  DB schema changes
  └─ Step 2  backend/auth.py (new)
       └─ Step 3  app.py (auth endpoints + JWT dependency)
            ├─ Step 4  pi_sensor_writer.py  ← parallel, independent after Step 1
            └─ Step 5  api/index.js
                 ├─ Step 6  LoginPage.jsx (new)
                 └─ Step 7  App.jsx
```

Run `POST /setup-db` (or restart the server) immediately after Step 1 to apply schema changes before any code changes go live.

---

## Gotchas summary

| # | Gotcha | Fix |
|---|---|---|
| 1 | `egg_calendar_entries` PK must change from `(entry_date)` to `(owner_id, entry_date)` | Truncate table first on dev DB; update `ON CONFLICT` clause in `upsert_egg_entry()` |
| 2 | `scheduler.py` calls `insert_event()` without `owner_id` | Keep `owner_id` defaulting to `None` in `insert_event()` — system events are unowned |
| 3 | Delete endpoints must check ownership: `WHERE id = %s AND owner_id = %s` | Without this, user A can silently delete user B's rows by guessing IDs |
| 4 | `passlib[bcrypt]` must be added explicitly — `bcrypt` alone is not enough | Add `passlib[bcrypt]==1.7.4` to `requirements.txt` |
| 5 | JWT token expires mid-session (default 24h) → all calls return 401 | Global `apiFetch()` wrapper clears localStorage and reloads on 401 |
| 6 | Heatmap `StaticFiles` mount is not access-controlled | Serve heatmaps through an authenticated API route, not raw static files |
| 7 | `JWT_SECRET` missing → app crash at startup | This is intentional. Document it in `.env.example` |

---

*Plan written April 2026 — for ChickenCoopComfort final paper implementation.*
