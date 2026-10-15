"""
Scheduled retraining from Python.

    from easyml import schedules

    schedules.create(5, "C:/exports/customers/", name="churn-nightly", daily="02:00")
    schedules.create(5, "C:/exports/customers.csv", name="churn-on-change", on_new_data=True,
                     recipe="customers", min_improvement=0.01)
    schedules.list()                      # every schedule with its current model and last result
    event = schedules.run_now("churn-nightly")
    print(event["decision"], event["message"])   # "promoted" / "kept champion" / "failed"
    schedules.events("churn-nightly")

Schedules run while the studio is open, with ``easy-ml schedule worker`` (or :func:`worker`), or once per
call of ``easy-ml schedule run-due`` / :func:`run_due` (for cron or Windows Task Scheduler).
"""


def _trigger(every=None, daily=None, weekly=None, on_new_data=False):
    from easyml.scheduling.timing import parse_every

    chosen = [bool(every), bool(daily), bool(weekly), bool(on_new_data)]
    if sum(chosen) != 1:
        raise ValueError("Choose exactly one of every=, daily=, weekly= or on_new_data=True.")
    if every:
        return {"trigger": "interval", "interval_minutes": parse_every(every)}
    if daily:
        return {"trigger": "daily", "at_time": daily}
    if weekly:
        day, at = weekly.split() if isinstance(weekly, str) else weekly
        return {"trigger": "weekly", "weekday": day, "at_time": at}
    return {"trigger": "new_data"}


def create(run_id, source, name, every=None, daily=None, weekly=None, on_new_data=False, pattern=None, recipe=None,
           min_improvement=0.0, enabled=True):
    """
    Retrain run ``run_id`` automatically and keep whichever model is better.

    Args:
        run_id: The run to keep fresh.
        source: A data file (retrained on when it changes), a folder (its newest file matching ``pattern``),
            or for image runs a folder of class folders.
        name: The schedule's name.
        every / daily / weekly / on_new_data: When to retrain (exactly one): ``every="6h"`` (or "30m", "1d"),
            ``daily="02:00"``, ``weekly="monday 02:00"``, or ``on_new_data=True`` (as soon as the data changes).
        pattern: File names to use in a folder, e.g. ``"orders_*.csv"`` (default: CSV and Excel files).
        recipe: Data-prep recipe to clean new data with (name or id), ``"none"`` to use it as it is;
            by default the recipe the run was trained with, if any.
        min_improvement: Use a new model only if it scores at least this much better (0.01 = 1 point).
        enabled: Create it switched on.

    Returns:
        dict: The schedule (``id``, ``when``, ``next_run_at``, ``champion``, ...).
    """
    from easyml.scheduling import service

    data = {"run_id": run_id, "source_path": str(source), "name": name, "file_pattern": pattern,
            "recipe": "" if recipe is None else str(recipe), "min_improvement": min_improvement, "enabled": enabled}
    data.update(_trigger(every, daily, weekly, on_new_data))
    return service.create(data)


def list_schedules():
    """Every schedule with its current model (``champion``), next run and last result."""
    from easyml.scheduling import service

    return service.list_summaries()


def get(schedule):
    """One schedule by name or id (raises LookupError)."""
    from easyml.scheduling import service, store

    return service.summary(store.get_schedule(schedule))


def run_now(schedule):
    """Retrain now on the source's current data (blocks until done). Returns the event dict."""
    from easyml.scheduling.engine import run_schedule

    return run_schedule(schedule, trigger="manual", runner="python")


def enable(schedule):
    from easyml.scheduling import service

    return service.set_enabled(schedule, True)


def disable(schedule):
    from easyml.scheduling import service

    return service.set_enabled(schedule, False)


def delete(schedule):
    """Delete a schedule. The runs it trained stay in Run History."""
    from easyml.scheduling import service

    service.delete(schedule)


def events(schedule, limit=20):
    """What the schedule did, newest first: ``decision`` is "promoted", "kept champion" or "failed"."""
    from easyml.scheduling import service, store

    row = store.get_schedule(schedule)
    return [service.event_summary(e) for e in store.list_events(row["id"], limit=limit)]


def run_due():
    """Run the schedules that are due now, once. Returns their events."""
    import os
    import socket

    from easyml.scheduling.engine import run_due as _run_due

    return _run_due(runner=f"python:{socket.gethostname()}:{os.getpid()}")


def worker(poll_seconds=30):
    """Run schedules until stopped (Ctrl+C), like ``easy-ml schedule worker``."""
    from easyml.scheduling.host import worker as _worker

    return _worker(poll_seconds=poll_seconds)


list = list_schedules  # noqa: A001  (schedules.list() reads naturally; nothing below needs the builtin)

__all__ = ["create", "list", "list_schedules", "get", "run_now", "enable", "disable", "delete", "events", "run_due",
           "worker"]
