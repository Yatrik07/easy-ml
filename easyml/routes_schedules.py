"""
Scheduled retraining endpoints: list, create, change, delete, run now, and each schedule's history.

A schedule's data source is a path the user types (a file or folder they keep updating), so unlike
uploads it may be outside the data folder; it is copied into the uploads folder before each use.
"""

from flask import Blueprint, current_app, jsonify, request

from easyml.route_helpers import error_response, start_training_job

schedules_bp = Blueprint("easyml_schedules", __name__)


def _body():
    return request.get_json(silent=True) or {}


def _error(exc):
    from easyml.scheduling.engine import ScheduleBusy

    if isinstance(exc, ScheduleBusy):
        status = 409
    elif isinstance(exc, LookupError) and not isinstance(exc, KeyError):
        status = 404
    elif isinstance(exc, FileNotFoundError):  # a data source problem, not a missing endpoint
        status = 400
    else:
        return error_response(exc)
    return jsonify({"status": "error", "status_code": status, "error_message": str(exc), "message": str(exc)}), status


@schedules_bp.route("/schedules", methods=["GET", "POST"])
def schedules():
    from easyml.scheduling import service
    from easyml.scheduling.host import POLL_SECONDS, studio_scheduler_active

    try:
        if request.method == "POST":
            return jsonify(service.create(_body())), 201
        return jsonify({"schedules": service.list_summaries(),
                        "scheduler": {"studio": studio_scheduler_active(), "poll_seconds": POLL_SECONDS}})
    except Exception as exc:
        return _error(exc)


@schedules_bp.route("/schedules/check-source", methods=["POST"])
def check_source():
    from easyml.scheduling import service

    body = _body()
    try:
        return jsonify(service.check_source(body.get("run_id"), body.get("source_path"), body.get("file_pattern")))
    except Exception as exc:
        return _error(exc)


@schedules_bp.route("/schedules/<int:schedule_id>", methods=["GET", "PUT", "DELETE"])
def schedule(schedule_id):
    from easyml.scheduling import service, store

    try:
        if request.method == "PUT":
            return jsonify(service.update(schedule_id, _body()))
        if request.method == "DELETE":
            service.delete(schedule_id)
            return jsonify({"deleted": schedule_id})
        row = store.get_schedule(schedule_id)
        events = [service.event_summary(e) for e in store.list_events(schedule_id, limit=100)]
        return jsonify({**service.summary(row, events[0] if events else None), "events": events})
    except Exception as exc:
        return _error(exc)


@schedules_bp.route("/schedules/<int:schedule_id>/toggle", methods=["POST"])
def toggle(schedule_id):
    from easyml.scheduling import service, store

    body = _body()
    try:
        enabled = body.get("enabled")
        if enabled is None:
            enabled = not store.get_schedule(schedule_id).get("enabled")
        return jsonify(service.set_enabled(schedule_id, bool(enabled)))
    except Exception as exc:
        return _error(exc)


@schedules_bp.route("/schedules/<int:schedule_id>/run", methods=["POST"])
def run_now(schedule_id):
    """Retrain now on the source's current data, as a job the studio can follow (``202`` with ``job_id``)."""
    from easyml.scheduling import engine, store

    socketio = current_app.extensions.get("easyml_socketio")

    def notify(event):
        if socketio is not None:
            socketio.emit("schedule_event", event)

    try:
        prepared = engine.prepare(schedule_id, trigger="manual", runner="studio")
    except Exception as exc:
        return _error(exc)
    job = start_training_job(lambda: engine.execute(prepared, notify=notify), f"Schedule: {prepared.schedule['name']}")
    store.update_schedule(prepared.schedule["id"], touch=False, job_id=job.job_id)
    return jsonify({"status": "started", "job_id": job.job_id, "task": prepared.plan.task,
                    "parent_run_id": prepared.plan.parent_run_id, "name": prepared.plan.name,
                    "warnings": prepared.plan.warnings, "message": "Retraining started"}), 202


@schedules_bp.route("/schedules/<int:schedule_id>/events", methods=["GET"])
def events(schedule_id):
    from easyml.scheduling import service, store

    try:
        store.get_schedule(schedule_id)
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
        return jsonify({"events": [service.event_summary(e) for e in store.list_events(schedule_id, limit=limit)]})
    except Exception as exc:
        return _error(exc)
