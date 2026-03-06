# Mold-risk prevention model

## Goal
Prevent mold by keeping the coop **inside a safe band**, instead of reacting only when mold is already likely.
Mold needs **time** in damp conditions to start growing, so we explicitly model *accumulated exposure*.

## Key idea: State machine + Exposure memory
We run a small state machine with 3 states:

- **SAFE**: conditions are dry enough; no action or minimal maintenance pulse.
- **WATCH**: early dampness risk; **prevent** persistence with low-effort actions.
- **DANGER**: condensation/mold conditions likely; act urgently (automation first).

### Why not only thresholds?
A single spike of humidity shouldn't cause big alarms, but **hours of moderately high humidity** can still be dangerous.
So we add an **exposure** value (0–100) that represents *how long dampness has persisted*.

## Exposure model (the “memory”)
Each sample (e.g., every 10 minutes) we update exposure:

- In **DANGER zone**: exposure rises **fast** (wetness accumulates quickly)
- In **WATCH zone**: exposure rises **slower**
- In **SAFE zone**: exposure decays **slowly** (drying/reset takes time)

This mirrors reality: the coop doesn’t instantly “reset” after one good reading.

## Zones and thresholds (what we detect)
We use:
- **Humidity (%RH)** (e.g., WATCH ≥ 80%, DANGER ≥ 90%)
- **Condensation margin**: `margin = temp - dew_point`
  - WATCH when margin is small (≈ ≤ 3°C)
  - DANGER when margin is very small (≈ ≤ 1.5°C)

Night-time can slightly increase risk (surfaces cool → condensation easier).

## State transitions: stability features
To avoid flapping (fan on/off repeatedly):
- **Hysteresis**: to leave WATCH/DANGER, conditions must improve *more than just barely*.
- **Minimum time in state**: require e.g. 20 minutes before switching states.

## “Client effort minimizer” actions
We prefer automation and minimal manual effort:

- **WATCH**: short ventilation pulse (fan ON for X min, then OFF), + 1-minute checklist
- **DANGER**: stronger pulse schedule + “one-tap anti-mold mode” + focused checklist
- **SAFE**: usually nothing; optional maintenance pulse if exposure is still elevated

## What we log and what we send to RAG
### Log (every sample)
- state, exposure, humidity, dew point, condensation margin, fan state

### Emit events
- `MOLD_MANAGER_STATUS` every sample (dashboard/debugging)
- `MOLD_ALERT` only on state changes (notifications)
- `AUTOMATION_COMMAND_REQUEST` when fan must switch ON/OFF

### What RAG should say (strict, grounded)
RAG should only use:
- current state (SAFE/WATCH/DANGER)
- key facts (humidity, margin, exposure, urgency)
- the recommended actions list (automation + short checklist)
No “invented” advice or commands.

## Why this is prevention-first
We act early in WATCH, and we keep acting until exposure truly comes down.
That prevents dampness from persisting long enough for mold to start.

# Mold-prevention pipeline (overzicht)

## 1) Sensor inputs (per sample / window)
**Doel:** 10-minuten samenvatting van `temp_c`, `humidity_rh`, plus context `day_night`.  
**Uitbreidingen / efficiënter:**
- Timestamp + `sample_minutes` uit data afleiden (niet hardcoden).
- DQ flags: missing/outlier detection, en “fallback-to-last-good”.
- Meerdere sensoren: gebruik `temp_inside_mean` + `temp_inside_max` en kies “worst-case” hotspot voor condensatie.

**Irregularities:**
- `day_night` is een string zonder validatie (“day/night” verwacht).

---

## 2) Physics feature: Dew point + condensation margin
**Code:** `dew_point_celsius()` → `dp`, dan `margin_c = temp_c - dp`  
**Doel:** schimmel/condensatie is vooral gevaarlijk als **dew point dicht bij temperatuur ligt** (kleine margin). 
**Uitbreidingen / efficiënter:**
- Precompute constants / avoid log for extreme low RH (micro-opt).
- Voeg “surface temp” schatting toe (dak/hoek kan kouder zijn dan lucht).
- Voeg “outside temp” toe: grotere kans op koude oppervlakken → lagere margin.
---
## 3) Zone detection (WATCH/DANGER zones)
**Code:** `in_watch_zone`, `in_danger_zone` op basis van `margin_c`, `humidity_rh`, en `night_margin_bonus`  
**Doel:** snel bepalen of het klimaat **nu** al “mold-friendly” is.  
**Uitbreidingen / efficiënter:**
- Maak de zone rules één functie (minder duplicatie, testbaar).
- Pas night-factor aan op basis van echte metingen (temp drop rate).
- Combineer RH + margin in één score (vb. “condensation risk index”).

**eventuele aanpassingen:**
- `night_margin_bonus` is hardcoded (0.2°C). Ok voor MVP, maar later tunen.

---

## 4) Exposure model (memory over tijd)
**Code:** exposure ↑ snel in danger, ↑ langzaam in watch, ↓ langzaam in safe  
**Doel:** “mold risk” hangt af van **duur**: een korte piek ≠ urenlang vochtig.  
**Uitbreidingen / efficiënter:**
- Exposure update met integers (x10) om floats te vermijden op Pi.
- Exposure decay sneller als fan aan staat (ventilatie versnelt drogen).
- Voeg “bedding wetness” feature toe (als je dat ooit meet).

**Irregularities:**
- Exposure update gebruikt altijd `params.sample_minutes`; als sampling ooit verschilt, klopt exposure niet.

---

## 5) State machine (SAFE / WATCH / DANGER)
**Code:** target state op basis van zones + exposure thresholds, met hysterese + min time  
**Doel:** stabiele waarschuwingen en automations zonder flapping.  
**Uitbreidingen / efficiënter:**
- Extra state “RECOVERY” (optioneel) als je een fase wil tussen WATCH→SAFE.
- Log state transitions expliciet (handig voor evaluatie).
- Maak exit/hysteresis thresholds “symmetric” en documenteer tuning.

**Irregularities:**
- `min_time_met` gebruikt `minutes_in_state` dat **reset naar 0 op state change** (goed),
  maar dat beïnvloedt ook je fan duty-cycles (zie hieronder).

---

## 6) Fan scheduling (ON/OFF pulses + anti-flapping)
**Code:** in WATCH en DANGER pulsschema op basis van `minutes_in_state % cycle`, plus `min_minutes_between_fan_changes`  
**Doel:** automatisch drogen met minimale user effort, zonder relais kapot te togglen.  
**Uitbreidingen / efficiënter:**
- Gebruik een aparte `fan_phase_minutes` teller ipv `minutes_in_state` (robuster).
- Voeg “manual override” check toe (user kan fan locken).
- Voeg “max continuous on time” toe voor veiligheid.

**Irregularities (belangrijk):**
- Fan schedule gebruikt `memory.minutes_in_state` **na reset** bij state change.
  Daardoor start elke WATCH/DANGER cyclus opnieuw op hetzelfde punt (meestal ok),
  maar kan onbedoeld “fan ON direct” geven bij elke transition.
- In SAFE: je zet `desired_fan_on=False` bijna altijd, maar exposure kan nog hoog zijn:
  je hebt een pulse voorzien (goed), maar zonder aparte pulse-timer kan dat “1 window ON” worden en daarna OFF.

---

## 7) Messaging / RAG payload + events
**Code:** `why_facts`, `client_actions`, `automation_instruction`, `events`  
**Doel:** RAG kan uitleggen *waarom* en *wat te doen*; events voor logging/notifications/automation.  
**Uitbreidingen / efficiënter:**
- Maak payload super compact (state + minutes + 2–3 numerieke causes) op Pi.
- Gebruik cause-codes i.p.v. f-strings (scheelt RAM/CPU).
- Emit `MOLD_ALERT` alleen op state changes (nu al zo), en log status als JSONL.

**Irregularities (kritisch):**
- Je snippet eindigt met `]` i.p.v. `}` in de return → syntaxis error.
- `events: List[Dict[str, Any]]` gebruikt `List/Dict/Any` imports; check dat die bovenaan staan.
- `automation_instruction` zegt `"fan": "ON/OFF"` maar je logt ook `fan_is_on` (duplicatie).
- Je maakt f-strings in `why_facts` elke window → duur op Raspberry Pi (kan compacter).

---

# Quick wins (1 uur werk, veel winst)
1) **Fix return bracket** (`}` i.p.v. `]`) en check imports (`math`, typing, Enum, dataclass).
2) **Sort/validate inputs**: `day_night` only in {"day","night"}; clamp RH 0..100; handle None.
3) **Replace why_facts/client_actions** door compacte codes + ints (Pi-friendly).
4) **Separate fan schedule timer** van `minutes_in_state` (stabiliteit + testbaarheid).