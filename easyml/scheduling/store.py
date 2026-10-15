"""Reading and writing schedules and their events (``database/schedules.db``, ``schedule_events.db``)."""

from datetime import datetime, timedelta

from easyml.scheduling.timing import from_text, to_text
from easyml.utils.database_operations import (
    claim_record,
    delete_record,
    enter_record_to_table,
    ensure_table,
    fetch_rows,
    update_record,
)

SCHEDULES = "schedules"
EVENTS = "schedule_events"
STALE_HOURS = 6  # a claim older than this belongs to a scheduler that died; it may be taken over


def now_text(now=None):
    return to_text(now or datetime.now())


def list_schedules():
    return fetch_rows(SCHEDULES)


def get_schedule(name_or_id):
    """A schedule by id or name. Raises LookupError."""
    text = str(name_or_id).strip()
    rows = fetch_rows(SCHEDULES, "id = ?", (int(text),)) if text.isdigit() else []
    rows = rows or fetch_rows(SCHEDULES, "name = ?", (text,))
    if not rows:
        raise LookupError(f"No schedule {name_or_id!r}. List them with `easy-ml schedule list`.")
    return rows[0]


def name_taken(name, except_id=None):
    return any(row["id"] != except_id for row in fetch_rows(SCHEDULES, "lower(name) = lower(?)", (name,)))


def insert_schedule(**values):
    ensure_table(SCHEDULES)
    stamp = now_text()
    return enter_record_to_table(SCHEDULES, created_at=stamp, updated_at=stamp, **values)


def update_schedule(schedule_id, touch=True, **values):
    ensure_table(SCHEDULES)
    if touch:
        values["updated_at"] = now_text()
    update_record(SCHEDULES, schedule_id, **values)


def delete_schedule(schedule_id):
    ensure_table(SCHEDULES)
    ensure_table(EVENTS)
    delete_record(SCHEDULES, schedule_id)
    for event in fetch_rows(EVENTS, "schedule_id = ?", (schedule_id,)):
        delete_record(EVENTS, event["id"])


def claim(schedule_id, runner, now=None):
    """Mark the schedule as running for ``runner``; False when another scheduler is already on it."""
    ensure_table(SCHEDULES)
    now = now or datetime.now()
    stale = to_text(now - timedelta(hours=STALE_HOURS))
    return claim_record(SCHEDULES, schedule_id, "running_since IS NULL OR running_since < ?", (stale,),
                        running_since=to_text(now), runner=runner)


def release(schedule_id, **values):
    update_schedule(schedule_id, touch=False, running_since=None, runner=None, job_id=None, **values)


def is_running(row, now=None):
    started = from_text(row.get("running_since"))
    return bool(started) and started >= (now or datetime.now()) - timedelta(hours=STALE_HOURS)


def add_event(**values):
    ensure_table(EVENTS)
    return enter_record_to_table(EVENTS, **values)


def list_events(schedule_id, limit=50):
    return fetch_rows(EVENTS, "schedule_id = ?", (int(schedule_id),), order="id DESC", limit=limit)


def last_events():
    """The latest event of every schedule: {schedule_id: event}."""
    latest = {}
    for event in fetch_rows(EVENTS, "id IN (SELECT MAX(id) FROM schedule_events GROUP BY schedule_id)"):
        latest[event["schedule_id"]] = event
    return latest


def run_badges():
    """For Run History: which runs are a schedule's current model, and what happened to scheduled runs."""
    champions = {}
    for row in fetch_rows(SCHEDULES):
        if row.get("champion_run_id"):
            champions.setdefault(int(row["champion_run_id"]), []).append(row["name"])
    decisions = {}
    for event in fetch_rows(EVENTS, "challenger_run_id IS NOT NULL"):
        decisions[int(event["challenger_run_id"])] = event["decision"]
    return champions, decisions
