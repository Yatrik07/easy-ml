"""Creating, changing and describing schedules: shared by the studio API, the CLI and ``easyml.schedules``."""

import os
from datetime import datetime

from easyml.scheduling import store
from easyml.scheduling.sources import DEFAULT_PATTERN, find_data, matching_files
from easyml.scheduling.timing import describe, from_text, next_run, to_text, validate_trigger

_FIELDS = ("name", "run_id", "source_path", "file_pattern", "trigger", "interval_minutes", "at_time", "weekday",
           "recipe", "min_improvement", "enabled")


def _run_spec(run_id):
    from easyml.retraining import run_spec

    spec = run_spec(int(run_id))
    if not spec["retrain"].get("supported"):
        raise ValueError(f"Run #{run_id} cannot be retrained here: {spec['retrain'].get('reason')}")
    return spec


def _recipe_value(value, task):
    from easyml.retraining import IMAGE_TASK

    text = str(value if value is not None else "").strip()
    if text.lower() in ("", "auto", "default"):
        return ""
    if text.lower() == "none":
        return "none"
    if task == IMAGE_TASK:
        raise ValueError("Data-prep recipes apply to tables and text, not image folders.")
    from easyml.dataprep.recipe import get_recipe_row

    return str(get_recipe_row(int(text) if text.isdigit() else text)["id"])  # LookupError when unknown


def _validated(data, current=None):
    """Merge ``data`` over ``current`` and check everything; returns the columns to store."""
    merged = dict(current or {})
    merged.update({k: v for k, v in data.items() if k in _FIELDS and v is not None})
    if current and "run_id" in data and int(data["run_id"]) != int(current["base_run_id"]):
        raise ValueError("A schedule's run cannot be changed; create a new schedule instead.")
    run_id = merged.get("run_id") or merged.get("base_run_id")
    if not run_id:
        raise ValueError("Choose the run to retrain.")
    spec = _run_spec(current.get("champion_run_id") or current["base_run_id"] if current else run_id)

    name = str(merged.get("name") or "").strip()
    if not name:
        raise ValueError("Give the schedule a name.")
    if len(name) > 80:
        raise ValueError("Schedule names can have at most 80 characters.")
    if store.name_taken(name, except_id=current["id"] if current else None):
        raise ValueError(f"A schedule named {name!r} already exists.")

    source = str(merged.get("source_path") or "").strip().strip('"')
    if not source:
        raise ValueError("Choose the file or folder the new data comes from.")
    source = os.path.abspath(os.path.expanduser(source))
    if not os.path.exists(source):
        raise FileNotFoundError(f"{source} does not exist.")
    image = spec["retrain"]["input"] == "image_folder"
    if image and not os.path.isdir(source):
        raise ValueError("Image runs retrain on a folder with one sub-folder per class.")
    pattern = str(merged.get("file_pattern") or "").strip() or DEFAULT_PATTERN

    try:
        min_improvement = float(merged.get("min_improvement") or 0.0)
    except (TypeError, ValueError):
        raise ValueError("The minimum improvement must be a number.") from None
    if not 0.0 <= min_improvement <= 1.0:
        raise ValueError("The minimum improvement must be between 0 and 1 (0.01 = one point of accuracy).")

    values = validate_trigger(merged.get("trigger") or "", merged.get("interval_minutes"), merged.get("at_time"),
                              merged.get("weekday"))
    values.update(name=name, task=spec["task"], source_path=source,
                  file_pattern=None if image or os.path.isfile(source) else pattern,
                  recipe=_recipe_value(merged.get("recipe"), spec["task"]), min_improvement=min_improvement,
                  enabled=1 if merged.get("enabled", True) not in (False, 0, "0", "false", "False") else 0)
    return values


def create(data, now=None):
    """Create a schedule from a dict (see ``_FIELDS``); returns its summary."""
    now = now or datetime.now()
    values = _validated(data)
    run_id = int(data["run_id"])
    try:  # the data that exists now is not "new": only later changes trigger a retrain
        current = find_data(values["source_path"], values["task"], values["file_pattern"], settle_seconds=0)
    except OSError:
        current = None
    values.update(base_run_id=run_id, champion_run_id=run_id, last_signature=current.signature if current else None,
                  next_run_at=to_text(next_run(values, now)))
    return summary(store.get_schedule(store.insert_schedule(**values)))


def update(name_or_id, data, now=None):
    row = store.get_schedule(name_or_id)
    values = _validated(data, current=row)
    timing_changed = any(values[k] != row.get(k) for k in ("trigger", "interval_minutes", "at_time", "weekday"))
    if timing_changed or (values["enabled"] and not row.get("enabled")):
        values["next_run_at"] = to_text(next_run(values, now or datetime.now()))
    store.update_schedule(row["id"], **values)
    return summary(store.get_schedule(row["id"]))


def set_enabled(name_or_id, enabled, now=None):
    row = store.get_schedule(name_or_id)
    values = {"enabled": 1 if enabled else 0}
    if enabled and not row.get("enabled"):
        values["next_run_at"] = to_text(next_run(row, now or datetime.now()))
    store.update_schedule(row["id"], **values)
    return summary(store.get_schedule(row["id"]))


def delete(name_or_id):
    row = store.get_schedule(name_or_id)
    if store.is_running(row):
        raise ValueError(f"Schedule '{row['name']}' is running; stop it first.")
    store.delete_schedule(row["id"])
    return row


def _run_brief(run_id):
    from easyml.utils.database_operations import RunNotFoundError, get_record_by_id
    from easyml.utils.run_summary import summarize_run

    try:
        run = summarize_run(get_record_by_id(int(run_id)))
    except (RunNotFoundError, TypeError, ValueError):
        return {"id": run_id, "exists": False}
    return {"id": run["id"], "exists": True, "name": run["run_name"], "best_model": run["best_model_label"],
            "score": run["best_score"], "metric": run["metric"]}


def summary(row, last_event=None, now=None):
    """A schedule as the UI and CLI show it (paths are the user's own, so they are included)."""
    from easyml.utils.run_summary import task_label

    return {
        "id": row["id"],
        "name": row["name"],
        "task": row.get("task"),
        "task_label": task_label(row.get("task")),
        "base_run": _run_brief(row["base_run_id"]),
        "champion": _run_brief(row.get("champion_run_id") or row["base_run_id"]),
        "trigger": row.get("trigger"),
        "interval_minutes": row.get("interval_minutes"),
        "at_time": row.get("at_time"),
        "weekday": row.get("weekday"),
        "when": describe(row),
        "source_path": row.get("source_path"),
        "file_pattern": row.get("file_pattern"),
        "recipe": row.get("recipe") or "",
        "recipe_label": _recipe_label(row.get("recipe")),
        "min_improvement": row.get("min_improvement") or 0.0,
        "enabled": bool(row.get("enabled")),
        "next_run_at": row.get("next_run_at") if row.get("enabled") and row.get("trigger") != "new_data" else None,
        "last_run_at": row.get("last_run_at"),
        "last_status": row.get("last_status"),
        "last_message": row.get("last_message"),
        "running": store.is_running(row, now),
        "job_id": row.get("job_id") if store.is_running(row, now) else None,
        "last_event": event_summary(last_event) if last_event else None,
    }


def _recipe_label(value):
    text = str(value or "")
    if not text:
        return "The run's own recipe (if any)"
    if text == "none":
        return "None (use the data as it is)"
    try:
        from easyml.dataprep.recipe import get_recipe_row

        return get_recipe_row(int(text))["name"]
    except (LookupError, ValueError):
        return f"Recipe #{text} (deleted)"


def event_summary(event):
    return {k: event.get(k) for k in ("id", "schedule_id", "trigger", "data_file", "started_at", "finished_at",
                                      "challenger_run_id", "champion_before", "champion_after", "metric",
                                      "champion_score", "challenger_score", "compared_on", "decision", "message")}


def list_summaries():
    latest = store.last_events()
    return [summary(row, latest.get(row["id"])) for row in store.list_schedules()]


def check_source(run_id, source_path, file_pattern=None):
    """
    What a schedule would see in ``source_path``: the file it would use and whether it has the run's columns.
    Returns ``{"ok", "kind", "message", "files", "chosen", "missing_columns"}``.
    """
    spec = _run_spec(run_id)
    source = os.path.abspath(os.path.expanduser(str(source_path or "").strip().strip('"')))
    result = {"ok": False, "kind": None, "message": "", "files": [], "chosen": None, "missing_columns": [],
              "source_path": source}
    if not source_path or not os.path.exists(source):
        result["message"] = "This file or folder does not exist."
        return result
    image = spec["retrain"]["input"] == "image_folder"
    if image:
        result["kind"] = "image_folder"
        if not os.path.isdir(source):
            result["message"] = "Choose a folder with one sub-folder per class."
            return result
        classes = sorted(d.name for d in os.scandir(source) if d.is_dir())
        result.update(ok=len(classes) >= 2, files=classes,
                      message=f"{len(classes)} class folder(s)." if len(classes) >= 2 else "Needs at least two class folders.")
        return result
    if os.path.isdir(source):
        result["kind"] = "folder"
        files = matching_files(source, file_pattern)
        result["files"] = [name for name, _, _ in files[:20]]
        if not files:
            result["message"] = "No matching files yet. The schedule will use the newest one that appears."
            result["ok"] = True
            return result
        chosen = os.path.join(source, files[0][0])
    else:
        result["kind"] = "file"
        chosen = source
    result["chosen"] = os.path.basename(chosen)
    try:
        from easyml.dataprep.engine import read_table

        columns = [str(c) for c in read_table(chosen, nrows=5).columns]
    except Exception as exc:
        result["message"] = f"Could not read {os.path.basename(chosen)}: {exc}"
        return result
    missing = [c for c in spec["retrain"].get("required_columns") or [] if c not in columns]
    result["missing_columns"] = missing
    result["ok"] = True
    if missing:
        result["message"] = (f"{result['chosen']} lacks {', '.join(missing)}. That is fine if a data-prep recipe "
                             "creates them; otherwise retraining will fail.")
    else:
        result["message"] = f"{result['chosen']} has every column the run needs."
    return result


def next_due(rows=None):
    """The earliest next run time of the enabled timed schedules (for status lines)."""
    times = [from_text(r.get("next_run_at")) for r in (rows or store.list_schedules())
             if r.get("enabled") and r.get("trigger") != "new_data"]
    times = [t for t in times if t]
    return min(times) if times else None
