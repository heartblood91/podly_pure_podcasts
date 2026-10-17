"""Fixed-time feed refresh settings (no app imports, safe from cycles)."""

import os
from typing import Any, Dict, Optional


def refresh_cron_kwargs() -> Optional[Dict[str, Any]]:
    """APScheduler cron trigger kwargs, or None to keep the interval refresh.

    Enabled by PODLY_REFRESH_CRON_HOURS (e.g. "9,12,18").
    """
    hours = os.environ.get("PODLY_REFRESH_CRON_HOURS", "").strip()
    if not hours:
        return None
    return {
        "trigger": "cron",
        "hour": hours,
        "minute": os.environ.get("PODLY_REFRESH_CRON_MINUTE", "0").strip() or "0",
        "timezone": os.environ.get("PODLY_REFRESH_TIMEZONE", "Europe/Paris").strip()
        or "Europe/Paris",
    }
