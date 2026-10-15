"""
When a schedule runs next. Times are the computer's local time, stored as ``YYYY-MM-DDTHH:MM:SS``.

Triggers:
    interval   every ``interval_minutes`` after the last attempt finished (or the schedule was created)
    daily      every day at ``at_time`` ("HH:MM")
    weekly     every week on ``weekday`` (0 = Monday) at ``at_time``
    new_data   whenever the data source changes (checked on every scheduler pass; no fixed time)
"""

import re
from datetime import datetime, timedelta

TRIGGERS = ("interval", "daily", "weekly", "new_data")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MIN_INTERVAL_MINUTES = 5
_FORMAT = "%Y-%m-%dT%H:%M:%S"


def to_text(moment):
    return moment.strftime(_FORMAT) if moment else None


def from_text(text):
    if not text:
        return None
    try:
        return datetime.strptime(str(text)[:19], _FORMAT)
    except ValueError:
        return None


def parse_time(text):
    """``"HH:MM"`` → (hour, minute)."""
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(text or ""))
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError("Use a time like 02:00 or 18:30 (24-hour clock).")
    return int(match.group(1)), int(match.group(2))


def parse_weekday(value):
    """0-6, or a day name such as "monday" / "mon"."""
    if isinstance(value, int) or str(value).strip().isdigit():
        day = int(value)
        if 0 <= day <= 6:
            return day
    else:
        text = str(value or "").strip().lower()
        for index, name in enumerate(WEEKDAYS):
            if len(text) >= 2 and name.lower().startswith(text):  # "mo", "tue", "thursday", ...
                return index
    raise ValueError("Choose a day of the week (Monday to Sunday).")


def parse_every(text):
    """``"30m"``, ``"6h"``, ``"1d"`` or a number of minutes → minutes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([mhd]?)\s*", str(text or "").lower())
    if not match:
        raise ValueError("Use an interval like 30m, 6h or 1d.")
    minutes = float(match.group(1)) * {"": 1, "m": 1, "h": 60, "d": 1440}[match.group(2)]
    return int(round(minutes))


def validate_trigger(trigger, interval_minutes=None, at_time=None, weekday=None):
    """Check and normalise trigger settings; returns the values to store."""
    if trigger not in TRIGGERS:
        raise ValueError(f"Choose when to retrain: one of {', '.join(TRIGGERS)}.")
    values = {"trigger": trigger, "interval_minutes": None, "at_time": None, "weekday": None}
    if trigger == "interval":
        try:
            minutes = int(interval_minutes)
        except (TypeError, ValueError):
            raise ValueError("Give the interval in minutes.") from None
        if minutes < MIN_INTERVAL_MINUTES:
            raise ValueError(f"The interval must be at least {MIN_INTERVAL_MINUTES} minutes.")
        values["interval_minutes"] = minutes
    elif trigger in ("daily", "weekly"):
        hour, minute = parse_time(at_time)
        values["at_time"] = f"{hour:02d}:{minute:02d}"
        if trigger == "weekly":
            values["weekday"] = parse_weekday(weekday)
    return values


def next_run(schedule, after):
    """The first moment after ``after`` at which the schedule is due; None for ``new_data``."""
    trigger = schedule.get("trigger")
    if trigger == "interval":
        return after + timedelta(minutes=int(schedule.get("interval_minutes") or MIN_INTERVAL_MINUTES))
    if trigger in ("daily", "weekly"):
        hour, minute = parse_time(schedule.get("at_time"))
        candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if trigger == "daily":
            return candidate if candidate > after else candidate + timedelta(days=1)
        candidate += timedelta(days=(int(schedule.get("weekday") or 0) - after.weekday()) % 7)
        return candidate if candidate > after else candidate + timedelta(days=7)
    return None


def describe(schedule):
    """A short sentence for lists: "Every 6 hours", "Daily at 02:00", ..."""
    trigger = schedule.get("trigger")
    if trigger == "interval":
        minutes = int(schedule.get("interval_minutes") or 0)
        if minutes % 1440 == 0:
            amount, unit = minutes // 1440, "day"
        elif minutes % 60 == 0:
            amount, unit = minutes // 60, "hour"
        else:
            amount, unit = minutes, "minute"
        return f"Every {unit}" if amount == 1 else f"Every {amount} {unit}s"
    if trigger == "daily":
        return f"Daily at {schedule.get('at_time')}"
    if trigger == "weekly":
        return f"Every {WEEKDAYS[int(schedule.get('weekday') or 0)]} at {schedule.get('at_time')}"
    return "When new data arrives"
