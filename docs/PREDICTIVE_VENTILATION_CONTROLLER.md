# Predictive Ventilation Controller

This document explains the logic of [`predictive_ventilation_controller.py`](../scripts/predictive_ventilation_controller.py) in a simple and structured way.

## Goal

The controller is designed to ventilate a poultry house in a way that is:

- safe
- proactive
- physically realistic
- easy to understand

It does not use Model Predictive Control, machine learning, or optimization solvers.
Instead, it uses a layered rule-based design.

## Main Idea

The old controller mainly reacted after indoor thresholds were already exceeded.

The new controller still keeps all important safety protections, but it also uses the heat-risk indicators from the risk model to act earlier when heat stress is building up.

The control logic is split into four layers:

1. Safety Layer
2. Risk-Driven Proactive Layer
3. Physical Feasibility Layer
4. Final Actuator Layer

## Inputs

The controller uses normal sensor inputs:

- indoor temperature `T_in`
- indoor CO2 `CO2_in`
- indoor relative humidity `RH_in`
- outdoor temperature `T_amb`
- outdoor relative humidity `RH_amb`
- indoor H2S `H2S_in`

It also uses heat-risk inputs from `risk_calculation.py`:

- `heat_risk_score`
- `heat_thi_slope_per_hour`
- `heat_high_thi_streak_minutes`
- `heat_data_coverage_last_hour`

Optional extra context can also be passed:

- `heat_risk_level`
- `heat_thi_mean`
- `heat_thi_max`

## Output

The main function returns three things:

1. `rate`
   The commanded ventilation rate in `m3/h`
2. `state`
   The updated controller state for the next control cycle
3. `diagnostics`
   A readable summary of what limited the decision and why

## Variable List

### Sensor inputs

- `T_in`: Indoor air temperature in degrees Celsius.
- `CO2_in`: Indoor CO2 concentration in ppm.
- `RH_in`: Indoor relative humidity as a fraction from `0` to `1`.
- `T_amb`: Outdoor air temperature in degrees Celsius.
- `RH_amb`: Outdoor relative humidity as a fraction from `0` to `1`.
- `H2S_in`: Indoor hydrogen sulfide concentration in ppm.

### Setpoints and safety thresholds

- `T_min`: Lower indoor temperature limit below which extra cooling should be avoided.
- `T_max`: Upper indoor temperature target used by the heat-balance cooling calculation.
- `RH_max`: Maximum desired indoor relative humidity as a fraction from `0` to `1`.
- `CO2_target`: Target maximum indoor CO2 concentration in ppm.
- `CO2_ambient`: Expected outdoor or baseline CO2 concentration in ppm.
- `H2S_warning`: H2S level in ppm where the controller starts adding a safety boost.
- `H2S_emergency`: H2S level in ppm where the fan is forced to maximum.

### Flock and physical parameters

- `n_birds`: Number of birds currently producing heat and CO2 inside the house.
- `q_sensible`: Sensible heat production per bird in watts.
- `q_co2_Lday`: CO2 production per bird in liters per day.
- `m_water_per_bird`: Moisture production per bird in grams per second, kept for interface compatibility.

### Heat-risk inputs

- `heat_risk_score`: Heat-risk severity from the risk model, accepted as either `0..1` or `0..100`.
- `heat_thi_slope_per_hour`: THI trend in THI points per hour, where positive values mean rising heat stress.
- `heat_high_thi_streak_minutes`: Consecutive minutes spent above the high-THI threshold.
- `heat_data_coverage_last_hour`: Fraction from `0` to `1` describing how complete the recent risk data is.
- `heat_risk_level`: Optional text label describing the qualitative heat-risk category.
- `heat_thi_mean`: Optional average THI value from the recent risk window.
- `heat_thi_max`: Optional maximum THI value from the recent risk window.

### Controller state

- `prev_rate`: Ventilation command from the previous control cycle in `m3/h`.
- `prev_valid`: Dictionary of the last valid sensor values used for fault fallback.
- `initialised`: Boolean flag showing whether the controller has already completed its first cycle.

### Hardware constraints

- `vent_min`: Minimum allowed fan airflow in `m3/h`.
- `vent_max`: Maximum allowed fan airflow in `m3/h`.
- `max_slew`: Maximum allowed airflow change per control cycle in `m3/h`.

### Main internal variables

- `vr_co2`: Gas-safety ventilation floor required by CO2 and H2S logic.
- `vr_temp`: Predictive cooling airflow request from the heat-risk layer.
- `vr_moisture`: Secondary drying airflow request from the moisture layer.
- `heat_demand`: Normalized heat-control request from `0` to `1`.
- `heat_confidence`: Confidence in the heat-demand signal from `0` to `1`.
- `cooling_potential`: Cooling usefulness of outdoor air from `0` to `1`.
- `target`: Combined airflow request before actuator constraints.
- `rate`: Final airflow command after all limits are applied.

### Diagnostics

- `limiting`: Text label naming which layer or target dominated the final decision.
- `sensor_faults`: List of sensor channels that failed validation in the current cycle.
- `rh_impossible`: Boolean flag showing that ventilation cannot solve humidity under current outdoor conditions.
- `cold_start`: Boolean flag showing that the controller seeded itself from the startup CO2 model.
- `notes`: Human-readable explanation of the main decisions and safeguards applied.

## Layer 1: Safety Layer

This layer always has priority.

### 1. Sensor validation

Each sensor is checked against plausible physical limits.

If a sensor is invalid:

- the controller tries to reuse the last valid reading
- if no old value exists, it keeps the raw value and records a note

This prevents one faulty reading from causing unstable behavior.

### 2. Cold start

On the very first cycle, the controller has no previous ventilation rate.

So it creates an initial estimate from the CO2 balance:

- more birds means more CO2 production
- higher allowed indoor CO2 means less ventilation required

This is only used to seed the controller at startup.

### 3. CO2 minimum ventilation floor

The controller computes the minimum required ventilation from CO2 using inversion logic:

`VR_new = VR_prev * (CO2_measured - CO2_ambient) / (CO2_target - CO2_ambient)`

This gas-based airflow is the minimum floor that must always be respected.

### 4. H2S safety overrides

Two H2S responses are kept:

- warning: add a ventilation boost on top of the CO2 floor
- emergency: force the fan directly to maximum

If H2S reaches emergency level, all other logic is ignored.

## Layer 2: Risk-Driven Proactive Layer

This layer converts heat-risk indicators into a simple control signal.

### Heat demand

The helper function `compute_heat_demand()` calculates:

- `heat_demand` in `[0, 1]`
- `heat_confidence` in `[0, 1]`

The design is intentionally simple:

- `heat_risk_score` is the main driver
- a positive THI slope adds an early-warning bonus
- a longer high-THI streak adds a persistence bonus
- low data coverage reduces confidence

In simple terms:

- high risk means the controller should start asking for cooling
- a rising THI means act earlier
- a long hot streak means the situation is persistent, not temporary
- poor data coverage means trust the signal less

There is also a small activation threshold so weak signals do not cause unnecessary fan movement.

## Layer 3: Physical Feasibility Layer

This layer checks whether ventilation can actually help.

### Cooling potential

Ventilation can only cool the house if the outdoor air is cooler than the indoor air.

The helper function `compute_cooling_potential(T_in, T_amb)` does this:

- if `T_amb >= T_in`, cooling potential is `0`
- if `T_amb < T_in`, cooling potential increases with the temperature difference

This avoids fake cooling requests when hot outdoor air would not help.

### Base cooling rate

The script uses the physical heat-balance function:

- `temperature_inversion_rate()`

This gives a base ventilation rate that is physically meaningful.

The predictive heat target is then:

`vr_temp = base_heat_rate * heat_demand * heat_confidence * cooling_potential`

This means:

- no heat demand -> no heat ventilation request
- no cooling potential -> no cooling request
- low confidence -> weaker request

### Moisture control

Moisture control is still available, but it is secondary.

The script:

- computes indoor and outdoor absolute humidity
- only requests extra drying airflow if outdoor air is actually drier

If outside air is not drier, the controller does not pretend ventilation can solve the moisture problem.

## Layer 4: Final Actuator Layer

The controller combines the three airflow requests:

- gas safety floor
- predictive heat target
- moisture target

The final target is:

`final_target = max(vr_co2, vr_temp, vr_moisture)`

This means:

- gas safety is always protected
- heat can demand more airflow when justified
- moisture can also demand more airflow when drying is feasible

After that, the controller applies:

- hardware minimum fan limit
- hardware maximum fan limit
- slew-rate limiting

Slew-rate limiting prevents the commanded fan speed from changing too abruptly between cycles.

## Diagnostics

The controller returns readable diagnostics for logging and debugging.

Important fields include:

- `limiting`
- `vr_co2`
- `vr_temp`
- `vr_moisture`
- `heat_demand`
- `heat_confidence`
- `cooling_potential`
- `sensor_faults`
- `notes`

This makes it easier to answer questions like:

- Was the fan driven by gas safety or by heat risk?
- Did the controller want cooling but outdoor air was too warm?
- Was a sensor faulty?
- Was the command clipped by hardware or slew limits?

## Why This Design Is Useful

This controller is a good middle ground:

- simpler than MPC
- more proactive than pure threshold control
- still based on physical reasoning
- easy to inspect and explain

It should be easier to trust in practice because each decision can be traced back to a short list of formulas and conditions.

## Example Decision Flow

A typical cycle works like this:

1. Validate sensors and replace faulty values with the last valid reading if possible.
2. Compute the gas safety floor from CO2 and H2S.
3. Convert heat-risk inputs into `heat_demand`.
4. Check whether outdoor air can actually cool the building.
5. Compute a predictive heat ventilation target.
6. Compute a secondary moisture target if outdoor air is drier.
7. Take the maximum of gas, heat, and moisture airflow.
8. Apply slew-rate and hardware limits.
9. Return the final command and diagnostics.

## Future Extensions

Two logical next steps are:

### 1. Weather forecast integration

Possible use:

- increase heat demand if hot outdoor conditions are expected soon
- reduce confidence in cooling if the outdoor temperature is expected to rise quickly

### 2. Mold or VTT model integration

Possible use:

- convert mold-growth risk into a separate moisture-risk ventilation request
- combine that moisture-risk request with the existing moisture control layer

This can be added without changing the overall layered structure.
