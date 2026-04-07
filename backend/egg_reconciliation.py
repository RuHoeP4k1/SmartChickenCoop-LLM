"""
egg_reconciliation.py — Daily egg count reconciliation.

Logic
─────
The CV pipeline (cv_counts_colson) samples how many eggs are visible in the
nest at each tick. Across a day the count can go up (hen lays) or down
(user collects). To compute "how many eggs were actually laid today" we
walk the day's samples in order and sum only the POSITIVE deltas:

    laid_today = sum(max(0, count[i] - count[i-1]) for i in 1..n)

This matches the user's "logical way": an increase = a new egg laid,
a decrease = eggs were collected.

Manual edits (source='manual' in egg_calendar_entries) are always preserved —
the automatic job will not overwrite a human-entered value.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from backend.db_utils import (
    get_cv_egg_counts_for_date,
    upsert_egg_entry,
)

log = logging.getLogger("chickencareai.egg_reconciliation")


def reconcile_day(target_date: Optional[date] = None) -> int:
    """
    Reconcile the egg calendar entry for a given date (default: yesterday).
    Returns the computed eggs_laid count, or 0 if no CV samples exist.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    samples = get_cv_egg_counts_for_date(target_date)
    if not samples:
        log.info("No CV samples for %s — skipping reconciliation", target_date)
        return 0

    # Skip None values at the head — a None baseline followed by a real
    # reading would otherwise generate a spurious positive delta equal to
    # the whole count. Likewise mid-series None rows are skipped without
    # advancing prev, so the next real sample is compared against the
    # last known good value.
    eggs_laid = 0
    prev: Optional[int] = None
    for row in samples:
        curr = row.get("egg_count")
        if curr is None:
            continue
        if prev is not None and curr > prev:
            eggs_laid += (curr - prev)
        prev = curr

    # upsert_egg_entry preserves manual edits automatically
    upsert_egg_entry(target_date, eggs_laid, source="auto")
    log.info("Reconciled %s: %d eggs laid (from %d samples)", target_date, eggs_laid, len(samples))
    return eggs_laid


def reconcile_yesterday() -> int:
    """Convenience wrapper — scheduler calls this at 23:55 each day."""
    return reconcile_day(date.today() - timedelta(days=1))
