"""
ventilation_controller.py  –  v4
=================================
Direct-inversion ventilation control for poultry housing.
No dataclasses — all functions take plain arguments.
No heater available: only a variable-speed fan.

Priority order (hard hierarchy)
--------------------------------
  1. Gas safety    : CO2 floor always respected; H2S hard override
  2. Heat stress   : T_in > T_max  → increase fan if outside is cooler
  3. Moisture      : RH_in > RH_max → increase fan if outside is drier
  4. Cold stress   : T_in < T_min  → reduce fan, but never below gas floor

Core idea — ratio inversion (no model needed after first cycle)
---------------------------------------------------------------
Instead of correcting a ventilation rate, we invert the mass balance
directly from the sensor reading:

  VR_needed = VR_prev * (C_measured - C_ambient) / (C_target - C_ambient)

If CO2 is 10% above target, we need 10% more airflow. No bird production
model is involved — systematic model errors cannot accumulate.

On cold start (first cycle, no VR_prev) a model-based estimate seeds the
first cycle only. After that sensors take over completely.

Units
-----
  Ventilation : m3/h
  Temperature : C
  RH          : fraction 0–1
  CO2, H2S    : ppm
  Moisture    : g/s per bird
  Heat        : W per bird
"""

import math


# Physics helpers


def absolute_humidity(T, RH):
    """
    Absolute humidity [kg water / kg dry air].
    Buck equation — same as your original code.
    """
    Psat = 0.61121 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
    return 0.622 * (Psat * RH / (101.325 - Psat * RH))


def air_density(T):
    """Dry air density [kg/m3]."""
    return 353.0 / (T + 273.15)


def latent_heat_of_vaporisation(T):
    """Latent heat [J/g] interpolated from lookup table."""
    temps  = [0,2,4,10,14,18,20,25,30,34,40,44,50,54,60,70,80,90,96]
    latent = [2500.9,2496.4,2491.2,2477.2,2467.7,2458.3,2453.5,2441.7,
              2429.8,2420.3,2406.4,2396.4,2381.9,2372.3,2357.7,2333.7,
              2308.0,2282.5,2266.9]
    if T <= temps[0]:  return latent[0]
    if T >= temps[-1]: return latent[-1]
    for i in range(len(temps) - 1):
        if temps[i] <= T <= temps[i + 1]:
            return latent[i] + (latent[i+1]-latent[i])*(T-temps[i])/(temps[i+1]-temps[i])


def bird_heat_production(W, T):
    """
    Heat and moisture production for one bird.
    Your original formulas.

    Parameters
    ----------
    W : body weight [kg]
    T : indoor temperature [C]

    Returns
    -------
    total_heat   [W]
    sensible     [W]
    latent       [W]
    moisture     [g/s]
    """
    total    = 10.62 * (W ** 0.75)
    sensible = (0.61 * (1000 + 20*(20 - T) - 0.228*T**2)) * (total / 1000)
    latent   = total - sensible
    moisture = latent / latent_heat_of_vaporisation(T)
    return total, sensible, latent, moisture


def co2_production_per_bird(W, T, RQ=0.9):
    """
    CO2 production [L/day per bird].
    Your original formula.
    """
    total_heat = 10.62 * (W ** 0.75)
    HP_kcal    = total_heat * 0.8598452279
    return (HP_kcal * RQ) / (3.815 + 1.232 * RQ)



# Sensor validation
SENSOR_RANGES = {
    "T_in":   (-15.0,  50.0),
    "CO2_in": (300.0, 5000.0),
    "RH_in":  (0.05,  1.0),
    "T_amb":  (-20.0, 45.0),
    "RH_amb": (0.05,  1.0),
    "H2S_in": (0.0,   100.0),
}

def validate_sensors(T_in, CO2_in, RH_in, T_amb, RH_amb, H2S_in,
                     ranges=None):
    """
    Check each sensor channel against plausibility bounds.

    Returns
    -------
    faults : list of str
        Names of channels outside their valid range.
    valid  : dict {channel: bool}
    """
    r = ranges or SENSOR_RANGES
    readings = {
        "T_in":   T_in,
        "CO2_in": CO2_in,
        "RH_in":  RH_in,
        "T_amb":  T_amb,
        "RH_amb": RH_amb,
        "H2S_in": H2S_in,
    }
    valid  = {ch: r[ch][0] <= v <= r[ch][1] for ch, v in readings.items()}
    faults = [ch for ch, ok in valid.items() if not ok]
    return faults, valid


# Direct inversion functions


def co2_seed_rate(n_birds, q_co2_Lday, CO2_target, CO2_ambient):
    """
    Model-based CO2 ventilation estimate [m3/h] for cold start only.
    After the first cycle use co2_inversion_rate() instead.
    """
    q_m3h = (n_birds * q_co2_Lday) / (24 * 1000)
    delta  = (CO2_target - CO2_ambient) * 1e-6
    if delta <= 0:
        return float('inf')
    return q_m3h / delta


def co2_inversion_rate(prev_rate, CO2_measured, CO2_target, CO2_ambient):
    """
    Required ventilation rate [m3/h] derived purely from the CO2 sensor.

    Derivation
    ----------
    At steady state:   VR * (C_in - C_amb) = Q_birds
    Previous cycle:    Q_birds = VR_prev * (CO2_prev - C_amb)   [measured]
    New rate needed:   VR_new  = Q_birds  / (CO2_target - C_amb)
                               = VR_prev * (CO2_measured - C_amb)
                                         / (CO2_target   - C_amb)

    No bird model needed. If birds produce more CO2 than assumed,
    the sensor reads higher and the rate scales up automatically.
    """
    numerator   = CO2_measured - CO2_ambient
    denominator = CO2_target   - CO2_ambient
    if denominator <= 0:
        return float('inf')
    if numerator <= 0:
        # CO2 is at or below ambient — minimum ventilation is enough
        return 0.0
    return prev_rate * (numerator / denominator)


def moisture_inversion_rate(prev_rate, AH_in, AH_target_max, AH_outdoor):
    """
    Required ventilation rate [m3/h] derived purely from the RH sensor.

    Same ratio logic as CO2:
      VR_new = VR_prev * (AH_in - AH_outdoor) / (AH_target_max - AH_outdoor)

    Returns (rate, impossible) where impossible=True if outdoor air is
    already more humid than the indoor target (nothing ventilation can do).
    """
    numerator   = AH_in         - AH_outdoor
    denominator = AH_target_max - AH_outdoor
    if denominator <= 0:
        return 0.0, True    # outdoor too moist — ventilation can't help
    if numerator <= 0:
        return 0.0, False   # indoor already drier than target — no action needed
    return prev_rate * (numerator / denominator), False


def temperature_inversion_rate(n_birds, q_sensible, T_target, T_amb):
    """
    Required ventilation rate [m3/h] to hold indoor temperature at T_target.

    Heat balance:  VR * rho * cp * (T_target - T_amb) = Q_sensible
    => VR = Q_sensible / (rho * cp * (T_target - T_amb))

    Returns 0 if T_amb >= T_target (ventilation would make things worse).
    Note: this function still uses the bird model because temperature
    depends on sensible heat production which cannot be directly inverted
    from a single temperature sensor without knowing the previous rate
    and the house thermal mass. For a 10-min cycle the steady-state
    approximation is acceptable.
    """
    delta_T = T_target - T_amb
    if delta_T <= 0:
        return 0.0
    rho = air_density(T_amb)
    cp  = 1005.0   # J/(kg*K)
    Q   = n_birds * q_sensible
    return (Q / (rho * cp * delta_T)) * 3600   # m3/s -> m3/h


# Main control function


def compute_ventilation_rate(
    # ── Sensor inputs ──────────────────────────────────────────────────
    T_in, CO2_in, RH_in, T_amb, RH_amb, H2S_in,
    # ── Setpoints ──────────────────────────────────────────────────────
    T_min, T_max, RH_max, CO2_target, CO2_ambient,
    H2S_warning, H2S_emergency,
    # ── Flock parameters (model — used only for cold start / temp) ─────
    n_birds, q_sensible, q_co2_Lday, m_water_per_bird,
    # ── Controller state (pass in, get back in returned dict) ──────────
    prev_rate,          # m3/h — commanded rate from previous cycle
    prev_valid,         # dict — last known good sensor values
    initialised,        # bool — False on very first call
    # ── Hardware limits ────────────────────────────────────────────────
    vent_min=20.0,      # m3/h
    vent_max=150,   # m3/h
    max_slew=3000.0,    # m3/h per cycle
):
    """
    Compute the ventilation rate for one control cycle.

    Parameters
    ----------
    (see module docstring for units)

    Returns
    -------
    rate          : float  — commanded ventilation rate [m3/h]
    state         : dict   — updated controller state to pass into next call
    diagnostics   : dict   — what drove this decision (for logging/display)

    Example
    -------
    # First call
    rate, state, diag = compute_ventilation_rate(
        T_in=20.0, CO2_in=1800, RH_in=0.62, T_amb=12.0, RH_amb=0.55, H2S_in=0.0,
        T_min=16, T_max=22, RH_max=0.70, CO2_target=2000, CO2_ambient=400,
        H2S_warning=1.0, H2S_emergency=5.0,
        n_birds=500, q_sensible=13.0, q_co2_Lday=3.8, m_water_per_bird=0.0046,
        prev_rate=0.0, prev_valid={}, initialised=False,
    )
    actuator.set(rate)

    # Every subsequent call — pass state back in
    rate, state, diag = compute_ventilation_rate(..., **state)
    """

    notes  = []
    faults = []

    # ── 1. Validate sensors, freeze faults ────────────────────────────
    faults, valid = validate_sensors(T_in, CO2_in, RH_in, T_amb, RH_amb, H2S_in)

    # Replace faulty readings with last known good values
    readings = {"T_in": T_in, "CO2_in": CO2_in, "RH_in": RH_in,
                "T_amb": T_amb, "RH_amb": RH_amb, "H2S_in": H2S_in}
    for ch in faults:
        if ch in prev_valid:
            readings[ch] = prev_valid[ch]
            notes.append(f"{ch} faulty — using last known value {prev_valid[ch]:.2f}")
        else:
            notes.append(f"{ch} faulty — no previous value, using raw reading")

    # Update last-good store
    new_prev_valid = dict(prev_valid)
    for ch, ok in valid.items():
        if ok:
            new_prev_valid[ch] = readings[ch]

    # Unpack (possibly substituted) readings
    T_in   = readings["T_in"]
    CO2_in = readings["CO2_in"]
    RH_in  = readings["RH_in"]
    T_amb  = readings["T_amb"]
    RH_amb = readings["RH_amb"]
    H2S_in = readings["H2S_in"]

    # ── 2. Cold start — seed prev_rate from model ──────────────────────
    cold_start = False
    if not initialised:
        prev_rate  = co2_seed_rate(n_birds, q_co2_Lday, CO2_target, CO2_ambient)
        prev_rate  = max(vent_min, min(prev_rate, vent_max))
        cold_start = True
        notes.append(f"Cold start: seeded from model at {prev_rate:.0f} m3/h")

    # ── 3. Precompute humidity values ─────────────────────────────────
    AH_in      = absolute_humidity(T_in,  RH_in)
    AH_out     = absolute_humidity(T_amb, RH_amb)
    AH_tgt_max = absolute_humidity(T_in,  RH_max)   # AH ceiling at current T_in

    # ── PRIORITY 1: Gas — CO2 and H2S ────────────────────────────────

    # H2S emergency: drop everything, slam fan to max
    if H2S_in >= H2S_emergency:
        rate = vent_max
        state = dict(prev_rate=rate, prev_valid=new_prev_valid, initialised=True)
        diag  = dict(limiting="H2S emergency", h2s_alert="emergency",
                     vr_co2=rate, vr_moisture=0, vr_temp=0,
                     rh_impossible=False, cold_start=cold_start,
                     sensor_faults=faults, notes=notes+["H2S EMERGENCY — fan at maximum"])
        return rate, state, diag

    h2s_alert = "none"
    if H2S_in >= H2S_warning:
        h2s_alert = "warning"
        notes.append(f"H2S warning ({H2S_in:.1f} ppm) — ventilation boosted")

    # CO2 ratio inversion
    vr_co2 = co2_inversion_rate(prev_rate, CO2_in, CO2_target, CO2_ambient)
    vr_co2 = max(vr_co2, vent_min)

    # H2S warning: proportional additive boost on top of CO2 rate
    if h2s_alert == "warning":
        boost  = vent_max * 0.15 * (H2S_in - H2S_warning) / max(H2S_warning, 1e-9)
        vr_co2 = min(vr_co2 + boost, vent_max)

    # ── PRIORITY 2: Heat stress ────────────────────────────────────────
    vr_temp = 0.0
    if T_in > T_max:
        if T_amb < T_in:
            vr_temp = temperature_inversion_rate(n_birds, q_sensible, T_max, T_amb)
            notes.append(f"Heat stress: T_in={T_in:.1f}C — ventilating to cool")
        else:
            notes.append("Heat stress: outside warmer than inside — ventilation cannot cool")

    # ── PRIORITY 3: Moisture ───────────────────────────────────────────
    vr_moisture  = 0.0
    rh_impossible = False
    if RH_in > RH_max:
        vr_moisture, rh_impossible = moisture_inversion_rate(
            prev_rate, AH_in, AH_tgt_max, AH_out)
        if rh_impossible:
            notes.append("RH target unachievable: outdoor air too moist")
            vr_moisture = 0.0
        else:
            vr_moisture = max(vr_moisture, vent_min)
            notes.append(f"Excess moisture: RH_in={RH_in:.2f} — ventilating to dry")

    # ── Combine: highest need wins (gas is always the floor) ───────────
    candidates = {"CO2/H2S": vr_co2, "Heat stress": vr_temp, "Moisture": vr_moisture}
    limiting   = max(candidates, key=candidates.__getitem__)
    target     = candidates[limiting]
    target     = max(target, vr_co2)   # gas floor always enforced

    # ── PRIORITY 4: Cold stress — reduce fan but respect gas floor ─────
    if T_in < T_min:
        target   = vr_co2          # reduce to gas minimum only
        limiting = "CO2/H2S (cold-stress floor)"
        notes.append(f"Cold stress: T_in={T_in:.1f}C < T_min={T_min:.1f}C — fan at gas-safety floor")

    # ── Slew rate limit ────────────────────────────────────────────────
    delta  = max(-max_slew, min(target - prev_rate, max_slew))
    rate   = max(vent_min, min(prev_rate + delta, vent_max))

    # ── Return rate + updated state + diagnostics ──────────────────────
    state = dict(
        prev_rate   = rate,
        prev_valid  = new_prev_valid,
        initialised = True,
    )
    diag = dict(
        limiting      = limiting,
        h2s_alert     = h2s_alert,
        vr_co2        = vr_co2,
        vr_moisture   = vr_moisture,
        vr_temp       = vr_temp,
        rh_impossible = rh_impossible,
        cold_start    = cold_start,
        sensor_faults = faults,
        notes         = notes,
    )
    return rate, state, diag



# Demo

if __name__ == "__main__":

    # Per-bird production from your original formulas
    W, T_ref = 3.0, 21.0
    _, q_sens, _, m_water = bird_heat_production(W, T_ref)
    q_co2 = co2_production_per_bird(W, T_ref)

    print(f"Per-bird: sensible={q_sens:.2f} W  moisture={m_water:.4f} g/s  CO2={q_co2:.2f} L/day\n")

    # Shared flock / setpoint args — unpack into every call
    flock = dict(n_birds=500, q_sensible=q_sens, q_co2_Lday=q_co2, m_water_per_bird=m_water)
    sp    = dict(T_min=16, T_max=22, RH_max=0.70, CO2_target=2000, CO2_ambient=400,
                 H2S_warning=1.0, H2S_emergency=5.0)
    hw    = dict(vent_min=50, vent_max=36000, max_slew=3000)

    # Initial controller state
    init_state = dict(prev_rate=0.0, prev_valid={}, initialised=False)

    scenarios = [
        ("Normal — CO2 slightly high",
         dict(T_in=20.0, CO2_in=2200, RH_in=0.62, T_amb=12.0, RH_amb=0.55, H2S_in=0.0)),
        ("Heat stress — outside cooler",
         dict(T_in=24.5, CO2_in=1800, RH_in=0.60, T_amb=20.0, RH_amb=0.50, H2S_in=0.0)),
        ("Heat stress — outside warmer",
         dict(T_in=23.0, CO2_in=1700, RH_in=0.58, T_amb=26.0, RH_amb=0.45, H2S_in=0.0)),
        ("Excess moisture",
         dict(T_in=20.0, CO2_in=1600, RH_in=0.78, T_amb=14.0, RH_amb=0.88, H2S_in=0.0)),
        ("Cold stress",
         dict(T_in=14.0, CO2_in=1900, RH_in=0.65, T_amb=4.0,  RH_amb=0.60, H2S_in=0.0)),
        ("H2S warning",
         dict(T_in=20.0, CO2_in=1700, RH_in=0.60, T_amb=15.0, RH_amb=0.55, H2S_in=2.0)),
        ("H2S emergency",
         dict(T_in=20.0, CO2_in=1700, RH_in=0.60, T_amb=15.0, RH_amb=0.55, H2S_in=6.5)),
        ("Faulty CO2 sensor",
         dict(T_in=20.0, CO2_in=9999, RH_in=0.62, T_amb=12.0, RH_amb=0.55, H2S_in=0.0)),
    ]

    print(f"{'Scenario':<35} {'VR m3/h':>8}  {'Limiting':<30}  {'H2S':>10}")
    print("-" * 90)
    for name, sensors in scenarios:
        rate, _, diag = compute_ventilation_rate(
            **sensors, **sp, **flock, **hw, **init_state)
        print(f"{name:<35} {rate:>8.0f}  {diag['limiting']:<30}  {diag['h2s_alert']:>10}")
        for note in diag["notes"]:
            print(f"  {'':35} {note}")
        if diag["sensor_faults"]:
            print(f"  {'':35} Faults: {diag['sensor_faults']}")

    # Multi-cycle: CO2 drifting upward
    print("\n=== Multi-cycle: CO2 rising — ratio inversion tracks it without a PI loop ===")
    print(f"{'Cycle':<6} {'CO2_in':>8} {'VR m3/h':>9}  {'Limiting'}")
    print("-" * 45)
    state = dict(prev_rate=0.0, prev_valid={}, initialised=False)
    for i in range(1, 10):
        co2 = 1500 + i * 120
        rate, state, diag = compute_ventilation_rate(
            T_in=20.0, CO2_in=co2, RH_in=0.60, T_amb=12.0, RH_amb=0.55, H2S_in=0.0,
            **sp, **flock, **hw, **state)
        print(f"{i:<6} {co2:>8.0f} {rate:>9.0f}  {diag['limiting']}")