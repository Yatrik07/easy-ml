"""
Running schedules: find due ones, retrain, compare with the model in use, promote or keep.

One attempt (``prepare`` then ``execute``):

1. Claim the schedule in the database, so two schedulers (the studio and a worker) never run it twice.
2. Find the data. Time triggers skip unchanged data ("no new data"); "Run now" always trains.
3. Copy the data into the uploads folder and plan a retrain of the current model (the champion), which
   applies the data-prep recipe and checks the columns.
4. Train in this thread. The new run is recorded with ``origin = "schedule"``.
5. Compare it with the champion on the new run's test rows; promote it when it is at least
   ``min_improvement`` better, otherwise keep the champion (the new run stays in Run History).
6. Record an event, work out the next run time and release the claim, also when something failed.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

from easyml.scheduling import store
from easyml.scheduling.sources import SETTLE_SECONDS, find_data, snapshot
from easyml.scheduling.timing import from_text, next_run, to_text
from easyml.utils.logger import write_log

NO_NEW_DATA = "no new data"


class ScheduleBusy(RuntimeError):
    """The schedule is already running (in this or another process)."""


@dataclass
class PreparedRun:
    schedule: dict
    trigger: str
    data_name: str
    signature: str
    plan: object
    started_at: datetime
    runner: str
    cleanup: list = field(default_factory=list)


def _recipe_argument(value):
    """The schedule's ``recipe`` column → ``plan_retrain(recipe=...)``."""
    text = str(value or "").strip()
    if not text:
        return None          # the run's own recipe, if it has one
    if text.lower() == "none":
        return False         # use the data as it is
    return text


def is_due(row, now, settle_seconds=SETTLE_SECONDS):
    """``(due, data)``: whether the schedule should run now and the data it would use."""
    if not row.get("enabled") or store.is_running(row, now):
        return False, None
    if row.get("trigger") != "new_data":
        due_at = from_text(row.get("next_run_at"))
        if due_at is None or now < due_at:
            return False, None
    try:
        data = find_data(row.get("source_path"), row.get("task"), row.get("file_pattern"),
                         now=now.timestamp(), settle_seconds=settle_seconds)
    except OSError:
        data = None
    if row.get("trigger") == "new_data":
        return bool(data) and data.signature != row.get("last_signature"), data
    return True, data


def _finish(row, now, **values):
    next_at = next_run(row, now)
    store.release(row["id"], last_run_at=to_text(now), next_run_at=to_text(next_at), **values)


def prepare(schedule_id, trigger="manual", runner="manual", now=None, settle_seconds=0, data=None):
    """
    Claim the schedule and get everything ready to train. Returns a PreparedRun, or None when there is no
    new data for a timed run (the schedule is then simply moved to its next time).

    Raises ScheduleBusy, LookupError (unknown schedule / deleted run), FileNotFoundError or ValueError
    (unusable data). The claim is released again on any error.
    """
    from easyml.config.paths import UPLOADS_DIR
    from easyml.retraining import plan_retrain

    now = now or datetime.now()
    row = store.get_schedule(schedule_id)
    if not store.claim(row["id"], runner, now):
        raise ScheduleBusy(f"Schedule '{row['name']}' is already running.")
    cleanup = []
    try:
        if data is None:
            data = find_data(row.get("source_path"), row.get("task"), row.get("file_pattern"),
                             now=now.timestamp(), settle_seconds=settle_seconds)
        if data is None:
            raise FileNotFoundError("No data file found yet (or it is still being written).")
        if trigger != "manual" and data.signature == row.get("last_signature"):
            _finish(row, now, last_status=NO_NEW_DATA,
                    last_message=f"Checked {data.name}: unchanged since the last run, so no model was trained.")
            return None
        path = snapshot(data, UPLOADS_DIR)
        if path != data.path:
            cleanup.append(path)
        champion = int(row.get("champion_run_id") or row["base_run_id"])
        name = f"{row['name']}-{now.strftime('%Y%m%d-%H%M')}"
        plan = plan_retrain(champion, path, name=name, original_dataset_name=data.name,
                            recipe=_recipe_argument(row.get("recipe")))
        cleaned = plan.kwargs.get("dataset_path")
        if cleaned and cleaned != path:  # the data-prep recipe wrote a cleaned copy
            cleanup.append(cleaned)
        return PreparedRun(schedule=row, trigger=trigger, data_name=data.name, signature=data.signature, plan=plan,
                           started_at=now, runner=runner, cleanup=cleanup)
    except Exception as exc:
        from easyml.utils.database_operations import RunNotFoundError

        _remove(cleanup)
        message = str(exc)
        if isinstance(exc, RunNotFoundError):
            message = (f"Run #{row.get('champion_run_id') or row['base_run_id']} no longer exists. "
                       "Delete this schedule and create a new one from another run.")
        event = {"schedule_id": row["id"], "trigger": trigger, "data_file": data.name if data else None,
                 "started_at": to_text(now), "finished_at": to_text(datetime.now()), "decision": "failed",
                 "message": message, "champion_before": row.get("champion_run_id") or row["base_run_id"],
                 "champion_after": row.get("champion_run_id") or row["base_run_id"]}
        store.add_event(**event)
        values = {"last_status": "failed", "last_message": message}
        if data is not None and trigger == "new data":
            values["last_signature"] = data.signature  # retry when the data changes again, not on every pass
        _finish(row, now, **values)
        if isinstance(exc, RunNotFoundError):
            raise LookupError(message) from exc
        raise


def _remove(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def execute(prepared, notify=None):
    """
    Train, compare and record the result. Returns the event (a dict). Training errors (and a stop
    request, ``InterruptedError``) are recorded as a failed event and then raised again.
    """
    from easyml.retraining import execute_plan
    from easyml.scheduling.compare import compare, is_better
    from easyml.utils.database_operations import update_record

    row, plan = prepared.schedule, prepared.plan
    champion = int(plan.parent_run_id)
    event = {"schedule_id": row["id"], "trigger": prepared.trigger, "data_file": prepared.data_name,
             "started_at": to_text(prepared.started_at), "champion_before": champion, "champion_after": champion}
    try:
        result = execute_plan(plan)
        update_record("runs", result.run_id, origin="schedule")
        outcome = compare(champion, result.run_id)
        promoted = is_better(outcome, row.get("min_improvement") or 0.0)
        event.update(challenger_run_id=result.run_id, metric=outcome["metric"],
                     champion_score=outcome["champion_score"], challenger_score=outcome["challenger_score"],
                     compared_on=outcome["compared_on"], decision="promoted" if promoted else "kept champion",
                     champion_after=result.run_id if promoted else champion,
                     message=_message(promoted, champion, result.run_id, outcome))
    except BaseException as exc:
        stopped = isinstance(exc, (InterruptedError, KeyboardInterrupt))
        event.update(decision="failed", message="Stopped before it finished." if stopped else f"Training failed: {exc}")
        _record(prepared, event, notify)
        raise
    finally:
        _remove(prepared.cleanup)
    _record(prepared, event, notify)
    return event


def _message(promoted, champion, challenger, outcome):
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    scores = (f"{outcome['metric']} {fmt(outcome['challenger_score'])} vs {fmt(outcome['champion_score'])} "
              f"on {outcome['compared_on']}")
    text = (f"Run #{challenger} replaces run #{champion} ({scores})." if promoted
            else f"Run #{champion} stays in use; new run #{challenger} was not better ({scores}).")
    return f"{text} {outcome['note']}".strip()


def _record(prepared, event, notify):
    finished = datetime.now()
    event["finished_at"] = to_text(finished)
    event["id"] = store.add_event(**{k: v for k, v in event.items() if k != "id"})
    row = prepared.schedule
    values = {"last_status": event["decision"], "last_message": event.get("message")}
    if event["decision"] != "failed":
        values["champion_run_id"] = event["champion_after"]
    if event["decision"] != "failed" or prepared.trigger == "new data":
        values["last_signature"] = prepared.signature  # a failed "new data" attempt waits for newer data
    _finish(row, finished, **values)
    write_log(f"Schedule '{row['name']}': {event['decision']}. {event.get('message') or ''}", "info",
              os.path.basename(__file__))
    if notify:
        try:
            notify({**event, "schedule_name": row["name"]})
        except Exception:  # a notification must never break the scheduler
            pass


def run_schedule(schedule_id, trigger="manual", runner="manual", now=None, settle_seconds=0, notify=None, data=None):
    """Prepare and execute one attempt in this thread (inside a job, so it can be stopped). Returns the event or None."""
    from easyml.utils.cancellation import register_job, reset_current_job, set_current_job

    prepared = prepare(schedule_id, trigger=trigger, runner=runner, now=now, settle_seconds=settle_seconds, data=data)
    if prepared is None:
        return None
    job = register_job(f"Schedule: {prepared.schedule['name']}")
    store.update_schedule(prepared.schedule["id"], touch=False, job_id=job.job_id)
    token = set_current_job(job)
    try:
        return execute(prepared, notify=notify)
    finally:
        job.finished = True
        reset_current_job(token)


def run_due(now=None, runner="worker", settle_seconds=SETTLE_SECONDS, notify=None, log=None):
    """
    One scheduler pass: run every due schedule, one after another. Returns the events of this pass.
    Errors of one schedule are recorded on it and never stop the others.
    """
    events = []
    for row in store.list_schedules():
        moment = now or datetime.now()
        due, data = is_due(row, moment, settle_seconds)
        if not due:
            continue
        trigger = "new data" if row.get("trigger") == "new_data" else "time"
        if log:
            log(f"Schedule '{row['name']}': {trigger}, retraining run #{row.get('champion_run_id') or row['base_run_id']}...")
        try:
            event = run_schedule(row["id"], trigger=trigger, runner=runner, now=moment, settle_seconds=settle_seconds,
                                 notify=notify, data=data)
        except ScheduleBusy:
            continue
        except Exception as exc:  # already recorded on the schedule
            if log:
                log(f"Schedule '{row['name']}' failed: {exc}")
            continue
        if event:
            events.append(event)
            if log:
                log(f"Schedule '{row['name']}': {event['decision']}. {event.get('message') or ''}")
        elif log:
            log(f"Schedule '{row['name']}': no new data.")
    return events
