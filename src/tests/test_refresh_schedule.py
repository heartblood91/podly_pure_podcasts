from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger

from app import background
from app.refresh_schedule import refresh_cron_kwargs

PARIS = ZoneInfo("Europe/Paris")


@pytest.fixture
def cron_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODLY_REFRESH_CRON_HOURS", "9,12,18")
    monkeypatch.delenv("PODLY_REFRESH_CRON_MINUTE", raising=False)
    monkeypatch.delenv("PODLY_REFRESH_TIMEZONE", raising=False)


def test_interval_refresh_without_cron(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODLY_REFRESH_CRON_HOURS", raising=False)
    fake = MagicMock()
    monkeypatch.setattr(background, "scheduler", fake)
    background.add_background_job(10)
    kwargs = fake.add_job.call_args.kwargs
    assert kwargs["trigger"] == "interval"
    assert kwargs["minutes"] == 10


def test_cron_refresh_job(monkeypatch: pytest.MonkeyPatch, cron_env: None) -> None:
    fake = MagicMock()
    monkeypatch.setattr(background, "scheduler", fake)
    background.add_background_job(10)
    kwargs = fake.add_job.call_args.kwargs
    assert kwargs["trigger"] == "cron"
    assert kwargs["hour"] == "9,12,18"
    assert kwargs["minute"] == "0"
    assert kwargs["timezone"] == "Europe/Paris"
    assert kwargs["coalesce"] is True
    assert kwargs["max_instances"] == 1


def _fire_times(start: datetime, count: int) -> list[datetime]:
    kwargs = dict(refresh_cron_kwargs() or {})
    kwargs.pop("trigger")
    trigger = CronTrigger(**kwargs)
    times: list[datetime] = []
    previous = None
    now = start
    for _ in range(count):
        fire = trigger.get_next_fire_time(previous, now)
        assert fire is not None
        times.append(fire)
        previous, now = fire, fire
    return times


def test_cron_fires_at_paris_wall_clock(cron_env: None) -> None:
    times = _fire_times(datetime(2026, 10, 24, 10, 0, tzinfo=PARIS), 6)
    assert [(t.day, t.hour, t.minute) for t in times] == [
        (24, 12, 0),
        (24, 18, 0),
        (25, 9, 0),  # DST ends on 2026-10-25, still 09:00 local
        (25, 12, 0),
        (25, 18, 0),
        (26, 9, 0),
    ]
    assert all(str(t.tzinfo) == "Europe/Paris" for t in times)
