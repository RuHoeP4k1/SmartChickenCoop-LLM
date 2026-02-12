# 🐔 Smart Chicken Coop - Full Team Coordination
## Second Semester Kickoff | January 2026

---

## Where We Stand

**First Semester: ✅ Proof of Concept Complete**
- Working RAG pipeline (LangChain + Chroma + Qwen)
- Demo sensor data integration (JSON format)
- Basic Gradio chatbot interface
- All running locally on laptop with test data

**Second Semester Goal: 🎯 Operational Prototype**
- Real sensors → Real data → Real advice
- Polished interface that looks professional
- Automated features that actually work
- Something we can demo with confidence

---

## What's Coming: The Vision

### The User Experience We're Building

```
┌─────────────────────────────────────────────────────────────┐
│                    SMART COOP DASHBOARD                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🌡️ Temperature: 24.3°C  ✓     💧 Humidity: 58%  ✓         │
│  🐔 Chickens Inside: 4/4  ✓    🚪 Door: CLOSED              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 🤖 AI Assistant                                       │   │
│  │                                                       │   │
│  │ "Good evening! All 4 chickens are safely inside.     │   │
│  │  Door closed automatically at 18:32 (sunset +30min). │   │
│  │  Conditions look great for tonight."                 │   │
│  │                                                       │   │
│  │ [Ask a question...]                        [Send]    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  📊 [Dashboard]  ⚙️ [Automation]  🔔 [Alerts]  💬 [Chat]    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**This is what we're building. This is what we demo.**

---

## Features We Need Consensus On

### 🎚️ Tier System: Modular Features

We need to agree on what goes where. Proposal:

| Tier | Name | Features | Target User |
|------|------|----------|-------------|
| **1** | Basic | Monitoring, manual controls, basic AI Q&A | "I just want to see what's happening" |
| **2** | Automation | Smart door, temp-based fan, alerts, rule builder | "I want it to handle routine stuff" |
| **3** | Advanced | CV counting, behavior analysis, health trends, preditor detection | "I want insights about my chickens" |


**❓ Questions for the team:**
- Does this breakdown make sense?
- What's realistic for our prototype? (I suggest Tier 1 + 2 fully working, Tier 3-4 as demos)
- Any features missing or in the wrong tier?

---

## Sensor Integration: What We Need

### Data Format Agreement

**Current:** Demo JSON files loaded manually

**Target:** Real-time data every 15 minutes from Raspberry Pi?

```json
{
  "timestamp": "2026-01-15T14:30:00",
  "temperature_c": 23.5,
  "humidity_pct": 62,
  "co2_ppm": 850,
  "ammonia_ppm": 12,
  "air_velocity_ms": 0.8,
  "heat_stress_index": "normal",
  "feeder_status": "OK",
  "waterer_status": "Low",
  "door_state": "closed",
  "fan_state": "off",
  "light_state": "on"
}
```

**🔧 For Sensors Team:**
- Can we finalize this JSON schema together?
- What's the realistic polling interval? (15 min? 5 min? On-change?)
- How do we handle sensor failures / missing data?

**🔧 For CV Team:**
- What data will you provide? Chicken count? Locations? Behavior labels?
- Same JSON format or separate endpoint?

---

## Actuator Controls: What Can We Control?

### Manual Controls (Always Available)

| Actuator | Control Options | Notes |
|----------|-----------------|-------|
| **Door** | Open / Close | Safety: don't close if chickens outside |
| **Fan** | Off / Low / High | Or: Off / On with speed % |
| **Lights** | Off / On / Dim % | Schedule support needed |
| **Heater** | Off / On | Winter only, threshold-based |

### Control Flow

```
User App  ───►  FastAPI Server  ───►  Raspberry Pi  ───►  Actuator
                     │
                     ▼
              Log action + 
              Update state
```

**🔧 For Sensors/Hardware Team:**
- Which actuators do we actually have working?
- Response time expectations? Latency issues?

---

## Automation: Rule Engine Design

### The Core Concept

Users create rules. System executes them automatically.
Make them self explanatory. 

```
RULE: "Evening Lockdown"
────────────────────────
WHEN:  time > sunset + 30 minutes
AND:   all_chickens_inside = true
THEN:  close_door()
       send_notification("🚪 Coop secured for the night")
```

### Rule Building Blocks

**Conditions (IF):**
| Type | Examples |
|------|----------|
| Time | `time = 07:00`, `time > sunset`, `time < sunrise + 1h` |
| Sensor | `temperature > 28`, `humidity > 75`, `co2 > 2000` |
| State | `door = open`, `fan = off`, `all_chickens_inside` |
| Combo | `temp > 30 AND humidity > 70` |

**Actions (THEN):**
| Type | Examples |
|------|----------|
| Device | `open_door()`, `fan_on(high)`, `lights_off()` |
| Alert | `send_notification(msg)`, `send_email(msg)` |
| Log | `log_event(type, details)` |

### Pre-Built Rule Templates

Users shouldn't have to build from scratch. We provide templates:

1. **🌅 Morning Routine** - Open door at sunrise
2. **🌙 Evening Lockdown** - Close door after sunset when all inside
3. **🌡️ Heat Management** - Fan on when temp > threshold
4. **💧 Humidity Control** - Fan on when humidity > threshold
5. **⚠️ Critical Alert** - Notify on dangerous conditions
6. **🏖️ Vacation Mode** - Enhanced monitoring, all automations active

**❓ Questions for the team:**
- What other rule templates should we have?
- In which depts do we wanna work these out? (highlight potentials, work out 1?)
---

## Predator Detection & Response

### The Problem
Survey said 55% of keepers worry about predators (foxes, martens).

### Our Solution: Layered Approach

**Layer 1: Motion Detection (Basic)**
```
PIR sensor triggers → Immediate alert to owner
                   → Auto-lock door if chickens inside
                   → Log event with timestamp
```

**Layer 2: CV Classification (Advanced)**
```
Camera detects movement → YOLO model classifies
                       → Fox/marten detected?
                       → YES: Alert + Lock + Optional deterrent
                       → NO: Log as "other movement"
```

**Layer 3: Deterrent System (Optional Hardware)**
```
Predator confirmed → Bright lights ON
                  → Sound alarm (optional)
                  → Sprinkler (optional)
                  → Requires user confirmation first? (avoid false positives)
```

### Implementation Priority

| Phase | What | When |
|-------|------|------|
| **MVP** | Motion alert + auto-lock | This semester |
| **V2** | CV predator classification | If CV team has bandwidth |
| **Future** | Smart deterrent system | Hardware dependent |

**🔧 For CV Team:**
- Is predator detection feasible with our setup?
- What dataset would we need? (fox, marten, cat images)
- Can we run this on Raspberry Pi or need cloud?

---

## The Interface: Making It Look Professional

### What We Need

**Not Gradio.** Something that looks like a real product.

**Options:**
1. **React/Vue web app** - Most professional, more work
2. **Streamlit** - Easier than React, better than Gradio
3. **Custom HTML/CSS/JS** - Full control, moderate effort
4. **AI-assisted build** - Use Claude/similar to generate interface

### Key Screens

1. **Dashboard** - At-a-glance status, main screen
2. **Live Data** - Charts, graphs, historical trends
3. **Controls** - Manual device control panel
4. **Automation** - Rule management, enable/disable
5. **Alerts** - Notification history, settings
6. **AI Chat** - The RAG assistant interface
7. **Settings** - Configuration, preferences

### Design Principles

- **Mobile-first** - Chicken keepers check on their phone
- **Clear status indicators** - Green/Yellow/Red at a glance
- **Minimal clicks** - Quick actions accessible immediately
- **Professional but friendly** - Not clinical, not childish

---

## What AI Workgroup Is Doing

### Current Sprint Tasks

| Task | Status | Notes |
|------|--------|-------|
| Expand knowledge base | 🔄 In Progress | Adding more welfare docs |
| Improve RAG retrieval | 📋 Planned | MMR tuning, better chunking |
| System prompt refinement | 📋 Planned | Structured output format |
| FastAPI backend setup | 📋 Planned | Endpoints for all integrations |
| Interface prototyping | 📋 Planned | Deciding on tech stack |
| RAG vs non-RAG evaluation | 📋 Planned | 20-30 test questions |

### What We Need From Other Teams

**From Sensors:**
- Finalized JSON data format
- Test data for development (real readings from KABAN?)
- Actuator command interface spec

**From CV:**
- Output format for chicken count/location
- Integration timeline
- Predator detection feasibility assessment

---

## Integration Points: Who Talks to Who

```
                    ┌─────────────────┐
                    │   USER (App)    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  FastAPI Server │◄───── AI Workgroup builds this
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ RAG + LLM     │   │ Database      │   │ Rule Engine   │
│ (AI Advice)   │   │ (History)     │   │ (Automation)  │
└───────────────┘   └───────────────┘   └───────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Raspberry Pi   │◄───── Sensors team builds this
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Sensors       │   │ Camera (CV)   │   │ Actuators     │
│ (Temp, Humid) │   │ (Counting)    │   │ (Door, Fan)   │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## What We're Showing at Final Demo

### The Story We Tell

> "Meet our smart chicken coop. It monitors your flock 24/7, handles routine tasks automatically, and gives you expert advice whenever you need it."

### Live Demo Sequence

1. **Show dashboard** - Real data, real status
2. **Trigger an alert** - "Watch what happens when temperature spikes"
3. **Show automation** - "The door closed automatically when all chickens were inside"
4. **Ask the AI** - "Why aren't my chickens laying eggs?" → Get actionable advice
5. **Show predator response** - "Motion detected → Alert sent → Door locked"

### Wow Factors

- **It actually works** - Real sensors, real data, real responses
- **It looks professional** - Not a student project look
- **It's modular** - Show how features can be added/removed
- **It's smart** - AI gives genuinely useful advice, not generic info

---

## Action Items: This Week

### Everyone
- [ ] Review this document
- [ ] Comment on feature tier breakdown
- [ ] Flag any concerns or blockers

### AI Workgroup
- [ ] Finalize FastAPI endpoint spec
- [ ] Start interface mockups
- [ ] Expand knowledge base

### Sensors Team
- [ ] Confirm JSON data format
- [ ] Provide test data from KABAN
- [ ] Document actuator command interface

### CV Team
- [ ] Confirm output format for chicken data
- [ ] Assess predator detection feasibility
- [ ] Integration timeline estimate

### CFD Team
- [ ] Define how CFD insights reach the AI system
- [ ] Format for CFD-based recommendations

---

## Next Meeting

**When:** [TBD - this week]
**Focus:** Lock in data formats + feature priorities
**Goal:** Everyone leaves knowing exactly what to build

---

## Questions?

Let's discuss and get aligned. The more we agree now, the smoother the integration later.

**Remember:** We're building something that could actually help chicken keepers. Let's make it good. 🐔

