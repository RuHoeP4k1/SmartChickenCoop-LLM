"""
consumption.py — Feed / water consumption calculation from sensor deltas.

The feeder_pct and waterer_pct columns in sensor_readings_colson are a
percentage of a full container. Across a day the percentage goes down as
birds eat/drink and sometimes jumps back up when the user refills.

Consumption for a day = sum of all negative deltas (i.e. eaten/drunk), with
refills ignored. A value above 100 % is valid — it just means the container
was refilled and drained more than once.

    consumed = sum(max(0, prev_pct - curr_pct) for consecutive samples)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

from backend.db_utils import get_feeder_waterer_samples_for_date

log = logging.getLogger("chickencareai.consumption")


def _sum_negative_deltas(samples: list, key: str) -> float:
    total = 0.0
    prev: Optional[float] = None
    for row in samples:
        val = row.get(key)
        if val is None:
            continue
        curr = float(val)
        if prev is not None and curr < prev:
            total += (prev - curr)
        prev = curr
    return round(total, 1)


def consumption_for_date(target_date: date) -> Dict[str, Optional[float]]:
    """
    Return {feeder_pct_consumed, waterer_pct_consumed} for the given date.
    Values may exceed 100 when the container was refilled mid-day.
    Returns None for a metric if there are no samples.
    """
    samples = get_feeder_waterer_samples_for_date(target_date)
    if not samples:
        return {"feeder_pct_consumed": None, "waterer_pct_consumed": None}

    return {
        "feeder_pct_consumed": _sum_negative_deltas(samples, "feeder_pct"),
        "waterer_pct_consumed": _sum_negative_deltas(samples, "waterer_pct"),
    }
