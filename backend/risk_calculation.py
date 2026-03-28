from __future__ import annotations
import math
from typing import Any, Dict, List
from enum import Enum
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import dataclass
from math import exp, log
from typing import Iterable, List, Optional

#=============================================================================
# SUPABASE DATA FETCHING AND MAPPING
# =============================================================================

import os
from supabase import create_client, Client
from typing import List, Dict, Any


def get_supabase_client() -> Client:
    """Create a Supabase client from environment variables."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY."
        )

    return create_client(supabase_url, supabase_key)


def fetch_recent_environment_readings(
    table_name: str = "sensor_readings",
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """
    Fetch recent valid environment rows from Supabase.

    Rows are returned newest first and only include the fields needed to
    compute a THI series.
    """
    client = get_supabase_client()

    try:
        response = (
            client.table(table_name)
            .select("timestamp,temperature_c,humidity_pct")
            .not_.is_("temperature_c", "null")
            .not_.is_("humidity_pct", "null")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read from Supabase table '{table_name}': {exc}"
        ) from exc

    return list(response.data or [])


def fetch_latest_environment_reading(
    table_name: str = "sensor_readings",
) -> Dict[str, Any]:
    """Fetch the latest valid raw environment row from Supabase."""
    rows = fetch_recent_environment_readings(table_name=table_name, limit=1)
    if not rows:
        raise ValueError(f"No valid row found in Supabase table '{table_name}'.")

    return rows[0]


def build_environment_inputs_from_supabase(
    reading: Optional[Dict[str, Any]] = None,
    table_name: str = "sensor_readings",
) -> Dict[str, float]:
    """Map the latest raw Supabase row to minimal environment inputs."""
    if reading is None:
        reading = fetch_latest_environment_reading(table_name=table_name)

    try:
        return {
            "temperature_c": float(reading["temperature_c"]),
            "humidity_pct": float(reading["humidity_pct"]),
        }
    except KeyError as exc:
        raise ValueError(f"Missing required field in Supabase row: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid temperature_c or humidity_pct value in Supabase row.") from exc
    
#=============================================================================
# THI_SERIES CALCULATION
# concept: 12 rijen worden opgehaald, dat is 2 uur aan data bij 10-minuten intervallen. 
# Daarmee kan een THI-serie worden opgebouwd die de afgelopen 2 uur laat zien hoe de THI zich heeft ontwikkeld. 
# Op basis van die serie kan worden berekend hoe lang de THI al boven een bepaalde drempelwaarde is gebleven, wat een belangrijke factor is voor het risico op hittestress bij kippen.
#=============================================================================


def _calculate_thi(temp_c: float, humidity_pct: float) -> float:
    """Compute THI from dry-bulb temperature and relative humidity."""
    twb = wet_bulb_temperature_c(temp_c, humidity_pct)
    return 0.85 * temp_c + 0.15 * twb


def _calculate_thi_streak_bonus( 
    thi: float,
    thi_streak_minutes: int,
    streak_thi_threshold: float,
    streak_threshold_minutes: int,
    streak_max_bonus: float,
    streak_base_bonus_at_threshold: float,
    streak_growth_rate: float,
) -> float:
    """
    Compute a bounded exponential-like bonus for sustained THI exposure.

    The bonus starts at `streak_base_bonus_at_threshold` once the minimum streak
    duration is reached and then approaches `streak_max_bonus`.
    """
    if thi < streak_thi_threshold:
        return 0.0
    if thi_streak_minutes < streak_threshold_minutes:
        return 0.0
    if streak_max_bonus <= 0:
        return 0.0

    base_bonus = max(0.0, min(streak_base_bonus_at_threshold, streak_max_bonus))
    extra_minutes = max(0, thi_streak_minutes - streak_threshold_minutes)
    remaining_bonus = max(0.0, streak_max_bonus - base_bonus)
    growth_factor = 1.0 - math.exp(-streak_growth_rate * extra_minutes)
    bonus = base_bonus + remaining_bonus * growth_factor
    return min(bonus, streak_max_bonus)


def build_thi_series_from_readings(
    readings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a THI series from Supabase readings.

    The function assumes the readings are already ordered newest first by the
    Supabase query.
    """
    thi_series: List[Dict[str, Any]] = []
    for row in readings:
        timestamp = row.get("timestamp")
        temperature_c = row.get("temperature_c")
        humidity_pct = row.get("humidity_pct")

        if timestamp is None or temperature_c is None or humidity_pct is None:
            continue

        try:
            temperature_value = float(temperature_c)
            humidity_value = float(humidity_pct)
        except (TypeError, ValueError):
            continue

        thi_series.append(
            {
                "timestamp": timestamp,
                "temperature_c": temperature_value,
                "humidity_pct": humidity_value,
                "thi": _calculate_thi(temperature_value, humidity_value),
            }
        )
    return thi_series


def calculate_thi_streak_minutes(
    thi_series: List[Dict[str, Any]],
    thi_threshold: float,
    interval_minutes: int = 10,
) -> int:
    """
    Calculate how long THI has continuously stayed above a threshold.

    The input series is expected to be ordered newest first.
    """
    consecutive_rows = 0

    for point in thi_series:
        thi_value = point.get("thi")
        if thi_value is None or float(thi_value) < thi_threshold:
            break
        consecutive_rows += 1

    return consecutive_rows * interval_minutes


def _build_thi_series(readings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper for THI series construction."""
    return build_thi_series_from_readings(readings)

#=============================================================================
# helper functions
#=============================================================================
def wet_bulb_temperature_c(t_db_c: float, rh_percent: float) -> float:
    """
    Approximate wet-bulb temperature (°C) from dry-bulb temperature (°C)
    and relative humidity (%) using Stull (2011) approximation.

    Valid for typical ambient conditions (roughly 0–50°C, 5–99% RH).
    Good enough for control/early warning use cases.
    """
    rh = max(1.0, min(rh_percent, 99.0))  # keep in a safe range
    t = t_db_c

    twb = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return twb

#=============================================================================
# HEAT RISK CALCULATION FUNCTION
#=============================================================================

def compute_heat_risk(
    temperature_c: float,
    humidity_pct: float,
    thi_streak_minutes: int = 0,
    streak_thi_threshold: float = 25.0, #vanaf hier zijn alle variabelen voor de streak
    streak_threshold_minutes: int = 30, # vanaf hier begint de bonus te lopen
    streak_max_bonus: float = 0.20, # maximale bonus die kan worden bereikt door een lange streak
    streak_base_bonus_at_threshold: float = 0.05, # bonus die al wordt toegekend zodra de streak drempel is bereikt (bijv. 30 minuten boven 25°C)
    streak_growth_rate: float = 0.06, # bepaalt hoe snel de bonus groeit met extra minuten boven de drempel. Hogere waarde = snellere groei richting max bonus.
) -> Dict[str, Any]:
    if not 0 <= humidity_pct <= 100:
        raise ValueError("humidity_pct must be between 0 and 100")
    if thi_streak_minutes < 0:
        raise ValueError("thi_streak_minutes must be non-negative")

    score = 0.0
    contributing_factors: List[str] = []

    thi = _calculate_thi(temperature_c, humidity_pct)

# De score wordt bepaald door de huidige THI-waarde, met hogere scores voor hogere THI.
    if thi < 18:
        score = 0.0
        contributing_factors.append("THI within optimal range")

    elif thi < 20:
        score = 0.15
        contributing_factors.append("THI slightly elevated")

    elif thi < 22:
        score = 0.35
        contributing_factors.append("THI approaching critical threshold")

    elif thi < 24:
        score = 0.6
        contributing_factors.append("THI above critical threshold (performance decline starts)")

    elif thi < 26:
        score = 0.8
        contributing_factors.append("THI high (clear heat stress zone)")

    elif thi < 28:
        score = 0.9
        contributing_factors.append("THI very high (strong performance impact)")

    else:
        score = 1.0
        contributing_factors.append("THI extreme (severe heat stress)")

        streak_bonus = _calculate_thi_streak_bonus(
            thi=thi,
            thi_streak_minutes=thi_streak_minutes,
            streak_thi_threshold=streak_thi_threshold,
            streak_threshold_minutes=streak_threshold_minutes,
            streak_max_bonus=streak_max_bonus,
            streak_base_bonus_at_threshold=streak_base_bonus_at_threshold,
            streak_growth_rate=streak_growth_rate,
        )

    if streak_bonus > 0:
        score += streak_bonus
        contributing_factors.append(
            "Sustained THI exposure: THI >= "
            f"{streak_thi_threshold:.1f} for {thi_streak_minutes} minutes "
            f"(+{streak_bonus:.2f} score)"
        )

    score = max(0.0, min(score, 1.0))

    if score < 0.5:
        level = "LOW - Monitor"
    elif score < 0.75:
        level = "MEDIUM - Elevated risk - Prepare to act"
    else:
        level = "HIGH - Take action now"

    return {
        "event_type": "HEAT_RISK",
        "risk_score": round(score * 100, 1),
        "risk_level": level,
        "contributing_factors": contributing_factors,
        "thi": round(thi, 2),
        "temperature_c": round(temperature_c, 2),
        "humidity_pct": round(humidity_pct, 2),
    }

#=============================================================================
# VTT MOLD GROWTH MODEL IMPLEMENTATION
#=============================================================================


class SensitivityLevel(str, Enum):
    VERY_SENSITIVE = "very_sensitive"
    SENSITIVE = "sensitive"
    MEDIUM_RESISTANT = "medium_resistant"
    RESISTANT = "resistant"

class DeclineMethod(str, Enum):
    WOOD = "WOOD"
    NON_WOOD = "NON_WOOD"


@dataclass(frozen=True)
class SensitivityParams:
    k11: float
    k12: float
    A: float
    B: float
    C: float
    rh_above_20: float

# Original sensitivity table from the paper
# Source paper table: very sensitive, sensitive, medium resistant, resistant
ORIGINAL_SENSITIVITIES = {
    SensitivityLevel.VERY_SENSITIVE: SensitivityParams(
        k11=1.0, k12=2.0, A=1.0, B=7.0, C=2.0, rh_above_20=70.0
    ),
    SensitivityLevel.SENSITIVE: SensitivityParams(
        k11=0.578, k12=0.386, A=0.3, B=6.0, C=1.0, rh_above_20=70.0
    ),
    SensitivityLevel.MEDIUM_RESISTANT: SensitivityParams(
        k11=0.072, k12=0.097, A=0.0, B=5.0, C=1.5, rh_above_20=70.0
    ),
    SensitivityLevel.RESISTANT: SensitivityParams(
        k11=0.033, k12=0.014, A=0.0, B=3.0, C=1.0, rh_above_20=70.0
    ),
}

# equations
@dataclass(frozen=True)
class VTTOriginalParams:
    """
    Originele VTT-parameters.

    Defaults:
    - sensitivity: VERY_SENSITIVE
    - wood_type_w: 0 = pine, 1 = spruce
    - surface_quality_sq: 0 = sawn, 1 = kiln dried
    - p_t, p_rh, p_c: originele waarden uit het model
    - decline_method: WOOD (originele toepassing is hout)
    - c_decline: 1.0 als standaard intensiteit
    - sample_minutes: numerieke timestep in minuten voor discrete integratie
    """
    sensitivity: SensitivityLevel = SensitivityLevel.VERY_SENSITIVE
    wood_type_w: int = 0 # 0 = pine, 1 = spruce
    surface_quality_sq: int = 0 # 0 = sawn, 1 = kiln dried
    p_t: float = 0.45
    p_rh: float = 9.0
    p_c: float = 58.0
    decline_method: DeclineMethod = DeclineMethod.WOOD
    c_decline: float = 1.0
    # VTT definieert dM/dt per 24 uur; sample_minutes bepaalt alleen de integratiestap.
    sample_minutes: int = 10
    m_min: float = 0.0
    m_max_hard: float = 6.0 # harde bovengrens voor M, ook al kan het model theoretisch hoger gaan. Dit helpt numerische stabiliteit.

    def __post_init__(self) -> None:
        if self.sample_minutes <= 0:
            raise ValueError("sample_minutes must be > 0")

@dataclass
class VTTState:
    m: float = 0.0
    consecutive_unfavourable_minutes: int = 0

@dataclass
class VTTStepResult:
    m: float
    mold_risk_level: str
    dmdt_per_24h: float #slope van M per 24 uur
    rhcrit: float
    mmax: float
    favourable_for_growth: bool


def get_sensitivity_params(level: SensitivityLevel) -> SensitivityParams:
    return ORIGINAL_SENSITIVITIES[level]


def rh_crit_original(temp_c: float, rh_above_20: float) -> float:
    """
    RHcrit volgens het originele model, maar met de gecorrigeerde formule uit
    het corrigendum van de paper.

    Corrigendum:
    RHcrit = -0.00267*T^3 + 0.160*T^2 - 3.13*T + 100   when T <= 20
             RH>20                                     when T >= 20
    """
    if temp_c <= 20.0:
        return -0.00267 * (temp_c ** 3) + 0.160 * (temp_c ** 2) - 3.13 * temp_c + 100.0
    return rh_above_20


def compute_k1(m: float, sens: SensitivityParams) -> float:
    return sens.k11 if m < 1.0 else sens.k12


def compute_mmax(rh: float, rhcrit: float, sens: SensitivityParams) -> float: #mmax is de verzadigingswaarde van M bij gegeven RH
    denom = rhcrit - 100.0
    if abs(denom) < 1e-12:
        x = 0.0
    else:
        x = (rhcrit - rh) / denom # genormaliseerde afstand van RH tot RHcrit, geschaald naar [0, 1] waarbij 0 bij RH=100% is en 1 bij RH=RHcrit

    mmax = sens.A + sens.B * x - sens.C * (x ** 2)
    return max(0.0, mmax)


def compute_k2(m: float, mmax: float) -> float:
    return max(1.0 - exp(2.3 * (m - mmax)), 0.0)


def growth_rate_per_24h(
    *,
    temp_c: float,
    rh: float,
    m: float,
    params: VTTOriginalParams,
    sens: SensitivityParams,
    rhcrit: float,
) -> float:
    """
    Originele VTT groeivergelijking:
    dM/dt = k1*k2 / (7 * exp(-pT ln(T) - pRH ln(RH) + 0.14W - 0.33SQ + pC))

    dM/dt is uitgedrukt per 24 uur.
    """
    k1 = compute_k1(m, sens)
    mmax = compute_mmax(rh, rhcrit, sens)
    k2 = compute_k2(m, mmax)

    # Numerieke bescherming - log(0) is niet gedefinieerd, dus we zorgen dat T en RH binnen een redelijke range blijven voor de logaritme.
    t_for_log = max(temp_c, 0.1)
    rh_for_log = min(max(rh, 0.1), 100)
    
    exponent_term = (
        -params.p_t * log(t_for_log)
        -params.p_rh * log(rh_for_log)
        + 0.14 * params.wood_type_w
        - 0.33 * params.surface_quality_sq
        + params.p_c
    )

    denominator = 7.0 * exp(exponent_term)
    return (k1 * k2) / denominator


def decline_rate_per_24h(state: VTTState, params: VTTOriginalParams) -> float:
    """
    Originele decline-regel.
    """
    hours_below = state.consecutive_unfavourable_minutes / 60.0

    if params.decline_method == DeclineMethod.NON_WOOD:
        base_decline = -0.01
    else:
        if hours_below <= 6.0:
            base_decline = -0.01
        elif hours_below <= 24.0:
            base_decline = 0.0
        else:
            base_decline = -0.008

    return params.c_decline * base_decline


def vtt_original_step(
    *,
    temp_c: float,
    rh: float,
    state: VTTState,
    params: VTTOriginalParams,
) -> VTTStepResult:
    """
    Eén numerieke stap van het originele VTT-model.
    De toestand is M(t) = state.m.
    Let op: in VTT is dM/dt uitgedrukt per 24 uur, niet per timestep.
    Daarom wordt de increment geschaald met sample_minutes / (24 * 60).
    """
    rh = min(max(rh, 0.0), 100.0)
    sens = get_sensitivity_params(params.sensitivity)
    rhcrit = rh_crit_original(temp_c, sens.rh_above_20)

    favourable = rh >= rhcrit

    if favourable:
        state.consecutive_unfavourable_minutes = 0
        dmdt_24h = growth_rate_per_24h(
            temp_c=temp_c,
            rh=rh,
            m=state.m,
            params=params,
            sens=sens,
            rhcrit=rhcrit,
        )
    else:
        state.consecutive_unfavourable_minutes += params.sample_minutes
        # Physical guard: once M hits zero, it cannot decline below zero.
        # Keep M flat at 0 until favourable conditions create growth again.
        if state.m <= 0.0:
            dmdt_24h = 0.0
        else:
            dmdt_24h = decline_rate_per_24h(state, params)

    delta_m = dmdt_24h * (params.sample_minutes / (24.0 * 60.0))
    state.m += delta_m
    # Numerieke stabiliteit: houd M altijd binnen de modelgrenzen [0, 6].
    state.m = max(0.0, min(state.m, 6.0))

    mmax = compute_mmax(rh, rhcrit, sens)

    return VTTStepResult(
        m=state.m,
        mold_risk_level=classify_mold_risk_level(state.m),
        dmdt_per_24h=dmdt_24h,
        rhcrit=rhcrit,
        mmax=mmax,
        favourable_for_growth=favourable,
    )


def run_vtt_original_series(
    *,
    temperatures_c: Iterable[float],
    rhs_percent: Iterable[float],
    params: VTTOriginalParams,
    initial_m: float = 0.0,
) -> List[VTTStepResult]:
    state = VTTState(m=initial_m, consecutive_unfavourable_minutes=0)
    results: List[VTTStepResult] = []

    for temp_c, rh in zip(temperatures_c, rhs_percent):
        results.append(
            vtt_original_step(
                temp_c=temp_c,
                rh=rh,
                state=state,
                params=params,
            )
        )

    return results


def mean_m(results: List[VTTStepResult]) -> float:
    if not results:
        return 0.0
    return sum(r.m for r in results) / len(results)


def classify_mold_risk_level(m: float) -> str:
    """Map the VTT mold index M to a qualitative mold risk level."""
    if m < 1.0:
        return "low"
    if m < 2.5:
        return "medium"
    if m < 4.5:
        return "high"
    return "severe"


def summarize_mold_risk(results: List[VTTStepResult]) -> Dict[str, Any]:
    """Return the mean mold index and its qualitative risk level."""
    m = mean_m(results)
    return {
        "m": round(m, 4),
        "mold_risk_level": classify_mold_risk_level(m),
    }

