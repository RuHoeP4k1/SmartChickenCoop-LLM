"""
predictive_ventilation_controller.py
====================================
Risk-driven predictive ventilation control for poultry housing.

Architecture
------------
1. Safety Layer
   - sensor validation with fallback to previous valid values
   - cold-start CO2 seed
   - CO2 minimum ventilation floor
   - H2S warning boost
   - H2S emergency override
2. Risk-Driven Proactive Layer
   - converts heat-risk indicators into a normalized heat demand
3. Physical Feasibility Layer
   - only requests cooling when outdoor air can actually cool
   - keeps moisture control secondary and only when outdoor air is drier
4. Final Actuator Layer
   - combines targets, applies slew-rate limiting and hardware bounds

The controller is intentionally simple: no MPC, no machine learning, no
optimization solver. Every control term is traceable to a small formula.
"""

import math


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def absolute_humidity(T, RH):
    """
    Absolute humidity [kg water / kg dry air].
    Buck equation.
    """
    psat = 0.61121 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
    return 0.622 * (psat * RH / (101.325 - psat * RH))


def air_density(T):
    """Dry air density [kg/m3]."""
    return 353.0 / (T + 273.15)


def latent_heat_of_vaporisation(T):
    """Latent heat [J/g] interpolated from a lookup table."""
    temps = [0, 2, 4, 10, 14, 18, 20, 25, 30, 34, 40, 44, 50, 54, 60, 70, 80, 90, 96]
    latent = [
        2500.9, 2496.4, 2491.2, 2477.2, 2467.7, 2458.3, 2453.5, 2441.7, 2429.8,
        2420.3, 2406.4, 2396.4, 2381.9, 2372.3, 2357.7, 2333.7, 2308.0, 2282.5,
        2266.9,
    ]
    if T <= temps[0]:
        return latent[0]
    if T >= temps[-1]:
        return latent[-1]

    for i in range(len(temps) - 1):
        if temps[i] <= T <= temps[i + 1]:
            span = temps[i + 1] - temps[i]
            return latent[i] + (latent[i + 1] - latent[i]) * (T - temps[i]) / span

    return latent[-1]


def bird_heat_production(W, T):
    """
    Heat and moisture production for one bird.

    Returns
    -------
    total_heat   [W]
    sensible     [W]
    latent       [W]
    moisture     [g/s]
    """
    total = 10.62 * (W ** 0.75)
    sensible = (0.61 * (1000 + 20 * (20 - T) - 0.228 * T ** 2)) * (total / 1000)
    latent = total - sensible
    moisture = latent / latent_heat_of_vaporisation(T)
    return total, sensible, latent, moisture


def co2_production_per_bird(W, T, RQ=0.9):
    """CO2 production [L/day per bird]."""
    total_heat = 10.62 * (W ** 0.75)
    hp_kcal = total_heat * 0.8598452279
    return (hp_kcal * RQ) / (3.815 + 1.232 * RQ)


def co2_seed_rate(n_birds, q_co2_Lday, CO2_target, CO2_ambient):
    """
    Model-based CO2 ventilation estimate [m3/h] for cold start only.
    """
    q_m3h = (n_birds * q_co2_Lday) / (24 * 1000)
    delta = (CO2_target - CO2_ambient) * 1e-6
    if delta <= 0:
        return float("inf")
    return q_m3h / delta


def co2_inversion_rate(prev_rate, CO2_measured, CO2_target, CO2_ambient):
    """
    Required ventilation rate [m3/h] derived purely from the CO2 sensor.
    """
    numerator = CO2_measured - CO2_ambient
    denominator = CO2_target - CO2_ambient
    if denominator <= 0:
        return float("inf")
    if numerator <= 0:
        return 0.0
    return prev_rate * (numerator / denominator)


def moisture_inversion_rate(prev_rate, AH_in, AH_target_max, AH_outdoor):
    """
    Required ventilation rate [m3/h] derived purely from the RH sensor.

    Returns (rate, impossible) where impossible=True if outdoor air is
    already more humid than the indoor target.
    """
    numerator = AH_in - AH_outdoor
    denominator = AH_target_max - AH_outdoor
    if denominator <= 0:
        return 0.0, True
    if numerator <= 0:
        return 0.0, False
    return prev_rate * (numerator / denominator), False


def temperature_inversion_rate(n_birds, q_sensible, T_target, T_amb):
    """
    Required ventilation rate [m3/h] to hold indoor temperature at T_target.
    """
    delta_T = T_target - T_amb
    if delta_T <= 0:
        return 0.0
    rho = air_density(T_amb)
    cp = 1005.0
    Q = n_birds * q_sensible
    return (Q / (rho * cp * delta_T)) * 3600.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def round_if_number(value, digits=3):
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


# ---------------------------------------------------------------------------
# Sensor validation and safety helpers
# ---------------------------------------------------------------------------


SENSOR_RANGES = {
    "T_in": (-15.0, 50.0),
    "CO2_in": (300.0, 5000.0),
    "RH_in": (0.05, 1.0),
    "T_amb": (-20.0, 45.0),
    "RH_amb": (0.05, 1.0),
    "H2S_in": (0.0, 100.0),
}


def validate_sensors(T_in, CO2_in, RH_in, T_amb, RH_amb, H2S_in, ranges=None):
    """
    Check each sensor channel against plausibility bounds.

    Returns
    -------
    faults : list[str]
    valid  : dict[str, bool]
    """
    r = ranges or SENSOR_RANGES
    readings = {
        "T_in": T_in,
        "CO2_in": CO2_in,
        "RH_in": RH_in,
        "T_amb": T_amb,
        "RH_amb": RH_amb,
        "H2S_in": H2S_in,
    }
    valid = {channel: r[channel][0] <= value <= r[channel][1] for channel, value in readings.items()}
    faults = [channel for channel, ok in valid.items() if not ok]
    return faults, valid


def apply_sensor_fallback(readings, prev_valid, valid, faults):
    """
    Replace faulty channels with the last known valid value when available.
    """
    notes = []
    substituted = dict(readings)
    new_prev_valid = dict(prev_valid)

    for channel in faults:
        if channel in prev_valid:
            substituted[channel] = prev_valid[channel]
            notes.append(f"{channel} faulty - using last known value {prev_valid[channel]:.2f}")
        else:
            notes.append(f"{channel} faulty - no previous value available, keeping raw reading")

    for channel, ok in valid.items():
        if ok:
            new_prev_valid[channel] = substituted[channel]

    return substituted, new_prev_valid, notes


def compute_gas_safety_floor(
    *,
    prev_rate,
    CO2_in,
    CO2_target,
    CO2_ambient,
    H2S_in,
    H2S_warning,
    H2S_emergency,
    vent_min,
    vent_max,
):
    """
    Safety-first gas layer with the same behavior as the legacy controller.
    """
    notes = []

    if H2S_in >= H2S_emergency:
        notes.append("H2S EMERGENCY - fan at maximum")
        return {
            "emergency": True,
            "h2s_alert": "emergency",
            "vr_co2": vent_max,
            "notes": notes,
        }

    h2s_alert = "none"
    vr_co2 = co2_inversion_rate(prev_rate, CO2_in, CO2_target, CO2_ambient)
    vr_co2 = max(vr_co2, vent_min)

    if H2S_in >= H2S_warning:
        h2s_alert = "warning"
        boost = vent_max * 0.15 * (H2S_in - H2S_warning) / max(H2S_warning, 1e-9)
        vr_co2 = min(vr_co2 + boost, vent_max)
        notes.append(f"H2S warning ({H2S_in:.1f} ppm) - ventilation boosted")

    return {
        "emergency": False,
        "h2s_alert": h2s_alert,
        "vr_co2": vr_co2,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Risk-driven predictive helpers
# ---------------------------------------------------------------------------


def normalize_heat_risk_score(heat_risk_score):
    """
    Accept either 0..1 or 0..100 risk scores and normalize to 0..1.
    """
    if heat_risk_score is None:
        return 0.0

    score = float(heat_risk_score)
    if score > 1.0:
        score = score / 100.0
    return clamp(score, 0.0, 1.0)


def compute_heat_demand(
    heat_risk_score,
    heat_thi_slope_per_hour,
    heat_high_thi_streak_minutes,
    heat_data_coverage_last_hour,
):
    """
    Convert heat-risk indicators into:
      heat_demand     in [0, 1]
      heat_confidence in [0, 1]

    Transparent weighting:
      - risk score dominates the decision
      - a positive THI slope adds early-warning demand
      - a longer high-THI streak adds persistence demand
      - low data coverage reduces confidence, not the raw demand signal
    """
    notes = []

    risk_component = normalize_heat_risk_score(heat_risk_score)
    slope_value = max(0.0, float(heat_thi_slope_per_hour or 0.0))
    streak_value = max(0.0, float(heat_high_thi_streak_minutes or 0.0))
    coverage = clamp(float(heat_data_coverage_last_hour or 0.0), 0.0, 1.0)

    slope_bonus = 0.15 * clamp(slope_value / 3.0, 0.0, 1.0)
    streak_bonus = 0.15 * clamp(streak_value / 60.0, 0.0, 1.0)
    heat_demand = clamp(0.70 * risk_component + slope_bonus + streak_bonus, 0.0, 1.0)
    heat_confidence = coverage

    notes.append(f"Heat risk score normalized to {risk_component:.2f}")
    if slope_bonus > 0.0:
        notes.append(f"Positive THI slope adds early-warning bonus ({slope_value:.2f} THI/h)")
    if streak_bonus > 0.0:
        notes.append(f"High-THI persistence adds bonus ({streak_value:.0f} min streak)")
    if coverage < 0.75:
        notes.append(f"Reduced confidence due to limited last-hour coverage ({coverage:.2f})")
    if heat_demand < 0.15:
        heat_demand = 0.0
        notes.append("Heat demand below activation threshold")

    return heat_demand, heat_confidence, notes


def compute_cooling_potential(T_in, T_amb):
    """
    Cooling potential in [0, 1].

    0 when the outdoor air is not cooler than the indoor air.
    Otherwise it grows linearly with the indoor-outdoor temperature gap and
    saturates when the gap reaches 8 C.
    """
    delta_t = T_in - T_amb
    if delta_t <= 0:
        return 0.0
    return clamp(delta_t / 8.0, 0.0, 1.0)


def compute_base_cooling_rate(T_in, T_amb, T_max, n_birds, q_sensible):
    """
    Base physically feasible cooling rate from sensible heat balance.

    Below T_max the controller uses the current indoor temperature as the
    stabilizing target. Above T_max it tries to pull toward T_max, but never
    below what the outdoor air can physically support.
    """
    if T_amb >= T_in:
        return 0.0, None

    target_temperature = min(T_in, T_max)
    target_temperature = max(target_temperature, T_amb + 0.5)
    base_rate = temperature_inversion_rate(n_birds, q_sensible, target_temperature, T_amb)
    return base_rate, target_temperature


def compute_predictive_heat_target(
    *,
    T_in,
    T_amb,
    T_min,
    T_max,
    n_birds,
    q_sensible,
    heat_risk_score,
    heat_thi_slope_per_hour,
    heat_high_thi_streak_minutes,
    heat_data_coverage_last_hour,
):
    """
    Risk-driven proactive cooling request.
    """
    notes = []
    heat_demand, heat_confidence, demand_notes = compute_heat_demand(
        heat_risk_score=heat_risk_score,
        heat_thi_slope_per_hour=heat_thi_slope_per_hour,
        heat_high_thi_streak_minutes=heat_high_thi_streak_minutes,
        heat_data_coverage_last_hour=heat_data_coverage_last_hour,
    )
    notes.extend(demand_notes)

    if T_in <= T_min:
        notes.append(f"Indoor temperature is at or below T_min ({T_min:.1f}C) - proactive cooling disabled")
        return 0.0, heat_demand, heat_confidence, 0.0, notes

    cooling_potential = compute_cooling_potential(T_in, T_amb)
    if cooling_potential <= 0.0:
        notes.append("Outdoor air is not cooler than indoor air - ventilation cannot provide sensible cooling")
        return 0.0, heat_demand, heat_confidence, cooling_potential, notes

    base_heat_rate, base_target_temperature = compute_base_cooling_rate(
        T_in=T_in,
        T_amb=T_amb,
        T_max=T_max,
        n_birds=n_birds,
        q_sensible=q_sensible,
    )
    if base_heat_rate <= 0.0:
        notes.append("Base cooling rate is zero under current outdoor conditions")
        return 0.0, heat_demand, heat_confidence, cooling_potential, notes

    effective_signal = heat_demand * heat_confidence * cooling_potential
    vr_temp = base_heat_rate * effective_signal

    if vr_temp > 0.0:
        notes.append(
            f"Predictive cooling active: base={base_heat_rate:.0f} m3/h, "
            f"signal={effective_signal:.2f}, target={base_target_temperature:.1f}C"
        )

    return vr_temp, heat_demand, heat_confidence, cooling_potential, notes


def compute_moisture_target(*, T_in, RH_in, T_amb, RH_amb, RH_max, prev_rate, vent_min):
    """
    Secondary moisture control.
    Ventilation is only used when outdoor air is drier than indoor air.
    """
    notes = []
    vr_moisture = 0.0
    rh_impossible = False

    if RH_in <= RH_max:
        return vr_moisture, rh_impossible, notes

    ah_in = absolute_humidity(T_in, RH_in)
    ah_out = absolute_humidity(T_amb, RH_amb)
    ah_target_max = absolute_humidity(T_in, RH_max)

    if ah_out >= ah_in:
        notes.append("Outdoor air is not drier than indoor air - moisture ventilation skipped")
        return vr_moisture, rh_impossible, notes

    vr_moisture, rh_impossible = moisture_inversion_rate(prev_rate, ah_in, ah_target_max, ah_out)
    if rh_impossible:
        notes.append("RH target unachievable - outdoor air is too moist relative to the indoor RH ceiling")
        return 0.0, rh_impossible, notes

    vr_moisture = max(vr_moisture, vent_min)
    notes.append(f"Moisture control active: RH_in={RH_in:.2f} above RH_max={RH_max:.2f}")
    return vr_moisture, rh_impossible, notes


# ---------------------------------------------------------------------------
# Final actuator layer
# ---------------------------------------------------------------------------


def apply_actuator_constraints(target, prev_rate, vent_min, vent_max, max_slew):
    """
    Apply hardware constraints and slew-rate limiting.
    """
    notes = []
    bounded_target = clamp(target, vent_min, vent_max)
    if bounded_target != target:
        if bounded_target == vent_max:
            notes.append("Target clipped to hardware maximum")
        elif bounded_target == vent_min:
            notes.append("Target clipped to hardware minimum")

    delta = clamp(bounded_target - prev_rate, -max_slew, max_slew)
    rate = clamp(prev_rate + delta, vent_min, vent_max)
    if delta != bounded_target - prev_rate:
        notes.append("Slew-rate limit active")

    return rate, notes


# ---------------------------------------------------------------------------
# Main control function
# ---------------------------------------------------------------------------


def compute_ventilation_rate(
    # Sensor inputs
    T_in,
    CO2_in,
    RH_in,
    T_amb,
    RH_amb,
    H2S_in,
    # Environmental setpoints
    T_min,
    T_max,
    RH_max,
    CO2_target,
    CO2_ambient,
    H2S_warning,
    H2S_emergency,
    # Flock / physical parameters
    n_birds,
    q_sensible,
    q_co2_Lday,
    m_water_per_bird,
    # Heat-risk inputs
    heat_risk_score,
    heat_thi_slope_per_hour,
    heat_high_thi_streak_minutes,
    heat_data_coverage_last_hour,
    heat_risk_level=None,
    heat_thi_mean=None,
    heat_thi_max=None,
    # Controller state
    prev_rate=0.0,
    prev_valid=None,
    initialised=False,
    # Hardware limits
    vent_min=20.0,
    vent_max=150.0,
    max_slew=3000.0,
):
    """
    Compute the ventilation rate for one control cycle.

    Returns
    -------
    rate        : float
    state       : dict
    diagnostics : dict

    Notes
    -----
    `m_water_per_bird` is accepted for compatibility with the legacy call
    signature, but the current moisture logic still uses sensor inversion.
    """
    del m_water_per_bird

    notes = []
    prev_valid = prev_valid or {}

    # 1. Safety layer: validate sensors and freeze faults.
    raw_readings = {
        "T_in": T_in,
        "CO2_in": CO2_in,
        "RH_in": RH_in,
        "T_amb": T_amb,
        "RH_amb": RH_amb,
        "H2S_in": H2S_in,
    }
    sensor_faults, valid = validate_sensors(T_in, CO2_in, RH_in, T_amb, RH_amb, H2S_in)
    readings, new_prev_valid, fallback_notes = apply_sensor_fallback(
        raw_readings,
        prev_valid,
        valid,
        sensor_faults,
    )
    notes.extend(fallback_notes)

    T_in = readings["T_in"]
    CO2_in = readings["CO2_in"]
    RH_in = readings["RH_in"]
    T_amb = readings["T_amb"]
    RH_amb = readings["RH_amb"]
    H2S_in = readings["H2S_in"]

    cold_start = False
    if not initialised:
        prev_rate = co2_seed_rate(n_birds, q_co2_Lday, CO2_target, CO2_ambient)
        prev_rate = clamp(prev_rate, vent_min, vent_max)
        cold_start = True
        notes.append(f"Cold start: seeded from CO2 model at {prev_rate:.0f} m3/h")

    gas_layer = compute_gas_safety_floor(
        prev_rate=prev_rate,
        CO2_in=CO2_in,
        CO2_target=CO2_target,
        CO2_ambient=CO2_ambient,
        H2S_in=H2S_in,
        H2S_warning=H2S_warning,
        H2S_emergency=H2S_emergency,
        vent_min=vent_min,
        vent_max=vent_max,
    )
    notes.extend(gas_layer["notes"])

    if gas_layer["emergency"]:
        rate = vent_max
        state = {
            "prev_rate": rate,
            "prev_valid": new_prev_valid,
            "initialised": True,
        }
        diagnostics = {
            "limiting": "H2S emergency",
            "h2s_alert": gas_layer["h2s_alert"],
            "vr_co2": round_if_number(gas_layer["vr_co2"], 1),
            "vr_temp": 0.0,
            "vr_moisture": 0.0,
            "heat_demand": 0.0,
            "heat_confidence": 0.0,
            "cooling_potential": 0.0,
            "sensor_faults": sensor_faults,
            "cold_start": cold_start,
            "heat_risk_level": heat_risk_level,
            "heat_thi_mean": heat_thi_mean,
            "heat_thi_max": heat_thi_max,
            "notes": notes,
        }
        return rate, state, diagnostics

    vr_co2 = gas_layer["vr_co2"]

    # 2. Risk-driven proactive layer: translate risk to demand.
    vr_temp, heat_demand, heat_confidence, cooling_potential, heat_notes = compute_predictive_heat_target(
        T_in=T_in,
        T_amb=T_amb,
        T_min=T_min,
        T_max=T_max,
        n_birds=n_birds,
        q_sensible=q_sensible,
        heat_risk_score=heat_risk_score,
        heat_thi_slope_per_hour=heat_thi_slope_per_hour,
        heat_high_thi_streak_minutes=heat_high_thi_streak_minutes,
        heat_data_coverage_last_hour=heat_data_coverage_last_hour,
    )
    notes.extend(heat_notes)

    # 3. Physical feasibility layer: simple secondary moisture control.
    vr_moisture, rh_impossible, moisture_notes = compute_moisture_target(
        T_in=T_in,
        RH_in=RH_in,
        T_amb=T_amb,
        RH_amb=RH_amb,
        RH_max=RH_max,
        prev_rate=prev_rate,
        vent_min=vent_min,
    )
    notes.extend(moisture_notes)

    # 4. Final actuator layer: combine requests then enforce actuator limits.
    candidates = {
        "CO2/H2S safety floor": vr_co2,
        "Predictive heat": vr_temp,
        "Moisture": vr_moisture,
    }
    limiting = max(candidates, key=candidates.get)
    target = max(vr_co2, vr_temp, vr_moisture)
    rate, actuator_notes = apply_actuator_constraints(
        target=target,
        prev_rate=prev_rate,
        vent_min=vent_min,
        vent_max=vent_max,
        max_slew=max_slew,
    )
    notes.extend(actuator_notes)

    state = {
        "prev_rate": rate,
        "prev_valid": new_prev_valid,
        "initialised": True,
    }
    diagnostics = {
        "limiting": limiting,
        "h2s_alert": gas_layer["h2s_alert"],
        "vr_co2": round_if_number(vr_co2, 1),
        "vr_temp": round_if_number(vr_temp, 1),
        "vr_moisture": round_if_number(vr_moisture, 1),
        "heat_demand": round_if_number(heat_demand, 3),
        "heat_confidence": round_if_number(heat_confidence, 3),
        "cooling_potential": round_if_number(cooling_potential, 3),
        "rh_impossible": rh_impossible,
        "cold_start": cold_start,
        "sensor_faults": sensor_faults,
        "heat_risk_level": heat_risk_level,
        "heat_thi_mean": heat_thi_mean,
        "heat_thi_max": heat_thi_max,
        "notes": notes,
    }
    return rate, state, diagnostics


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    W, T_ref = 3.0, 21.0
    _, q_sens, _, m_water = bird_heat_production(W, T_ref)
    q_co2 = co2_production_per_bird(W, T_ref)

    flock = {
        "n_birds": 500,
        "q_sensible": q_sens,
        "q_co2_Lday": q_co2,
        "m_water_per_bird": m_water,
    }
    setpoints = {
        "T_min": 16.0,
        "T_max": 22.0,
        "RH_max": 0.70,
        "CO2_target": 2000.0,
        "CO2_ambient": 400.0,
        "H2S_warning": 1.0,
        "H2S_emergency": 5.0,
    }
    hardware = {
        "vent_min": 50.0,
        "vent_max": 36000.0,
        "max_slew": 3000.0,
    }
    state = {
        "prev_rate": 0.0,
        "prev_valid": {},
        "initialised": False,
    }

    scenarios = [
        (
            "Normal gas floor",
            {
                "T_in": 20.0,
                "CO2_in": 2200.0,
                "RH_in": 0.62,
                "T_amb": 12.0,
                "RH_amb": 0.55,
                "H2S_in": 0.0,
                "heat_risk_score": 15.0,
                "heat_thi_slope_per_hour": 0.2,
                "heat_high_thi_streak_minutes": 0,
                "heat_data_coverage_last_hour": 1.0,
            },
        ),
        (
            "Predictive cooling below threshold",
            {
                "T_in": 21.3,
                "CO2_in": 1700.0,
                "RH_in": 0.60,
                "T_amb": 16.0,
                "RH_amb": 0.50,
                "H2S_in": 0.0,
                "heat_risk_score": 72.0,
                "heat_thi_slope_per_hour": 2.4,
                "heat_high_thi_streak_minutes": 40,
                "heat_data_coverage_last_hour": 1.0,
            },
        ),
        (
            "Hot but no cooling potential",
            {
                "T_in": 24.0,
                "CO2_in": 1800.0,
                "RH_in": 0.58,
                "T_amb": 27.0,
                "RH_amb": 0.40,
                "H2S_in": 0.0,
                "heat_risk_score": 88.0,
                "heat_thi_slope_per_hour": 1.8,
                "heat_high_thi_streak_minutes": 50,
                "heat_data_coverage_last_hour": 1.0,
            },
        ),
        (
            "H2S emergency",
            {
                "T_in": 20.0,
                "CO2_in": 1700.0,
                "RH_in": 0.60,
                "T_amb": 15.0,
                "RH_amb": 0.55,
                "H2S_in": 6.5,
                "heat_risk_score": 20.0,
                "heat_thi_slope_per_hour": 0.0,
                "heat_high_thi_streak_minutes": 0,
                "heat_data_coverage_last_hour": 1.0,
            },
        ),
    ]

    print(f"{'Scenario':<34} {'VR m3/h':>10}  {'Limiting':<24} {'HeatDem':>8} {'CoolPot':>8}")
    print("-" * 95)
    for name, scenario in scenarios:
        rate, state, diag = compute_ventilation_rate(
            **scenario,
            **setpoints,
            **flock,
            **hardware,
            **state,
        )
        print(
            f"{name:<34} {rate:>10.0f}  {diag['limiting']:<24} "
            f"{diag['heat_demand']:>8} {diag['cooling_potential']:>8}"
        )
        for note in diag["notes"]:
            print(f"  {note}")
