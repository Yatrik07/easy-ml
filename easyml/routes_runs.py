"""
Endpoints that act on a saved run: its settings, retraining it on new data, and (later in this
module) exporting/importing it and its result charts.
"""

import os
import tempfile

from flask import Blueprint, jsonify, request, send_file

from easyml.retraining import IMAGE_TASK, DatasetMismatchError, plan_retrain, run_spec
from easyml.run_archive import TRUST_WARNING, export_run, import_archive, max_import_bytes
from easyml.route_helpers import (
    clean_display_filename,
    create_run_folder,
    error_response,
    remove_orphan_run_folder,
    resolve_data_file,
    start_training_job,
)
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.logger import write_log
from easyml.utils.utils import sanitize_run_name

runs_bp = Blueprint("easyml_runs", __name__)


@runs_bp.route("/runs/<int:run_id>/config", methods=["GET"])
def run_config(run_id):
    try:
        return jsonify(run_spec(run_id))
    except Exception as exc:
        return error_response(exc)


@runs_bp.route("/runs/<int:run_id>/report", methods=["GET"])
def run_report(run_id):
    """The run as one self-contained HTML file (download)."""
    from flask import Response

    from easyml.reporting.report import build_report

    try:
        html, name = build_report(run_id)
    except Exception as exc:
        return error_response(exc)
    disposition = "inline" if request.args.get("view") in ("1", "true") else "attachment"
    return Response(html, mimetype="text/html", headers={"Content-Disposition": f'{disposition}; filename="{name}"'})


@runs_bp.route("/runs/<int:run_id>/meta", methods=["PUT"])
def run_meta(run_id):
    """Change a run's name, notes and/or tags: ``{run_name?, notes?, tags?}``."""
    from easyml.utils.run_meta import update_run_meta

    data = request.get_json(silent=True) or {}
    try:
        return jsonify(update_run_meta(run_id, run_name=data.get("run_name"), notes=data.get("notes"),
                                       tags=data.get("tags")))
    except Exception as exc:
        return error_response(exc)


@runs_bp.route("/runs/<int:run_id>/sample", methods=["GET"])
def run_sample(run_id):
    """A random real row of the run's data (preferably its test set) to try the model with."""
    from easyml.core.artifacts import sample_row

    try:
        record = get_record_by_id(run_id)
        if record["task"] == IMAGE_TASK:
            raise ValueError("Image runs do not keep a copy of their images; choose an image to try.")
        return jsonify(sample_row(record["folder_path"], record["task"]))
    except Exception as exc:
        return error_response(exc)


@runs_bp.route("/runs/<int:run_id>/data-check", methods=["POST"])
def run_data_check(run_id):
    """
    Does new data look like the run's training data? Multipart ``file`` (CSV/Excel) or JSON ``{"rows": [...]}``.
    Returns the report of :func:`easyml.monitoring.datacheck.check_frame`.
    """
    import pandas as pd

    from easyml.monitoring.datacheck import check_frame, reference_profile
    from easyml.route_helpers import _read_uploaded_table

    try:
        record = get_record_by_id(run_id)
        if record["task"] == IMAGE_TASK:
            raise ValueError("The data check works on tables and text, not images.")
        if "file" in request.files:
            frame = _read_uploaded_table(request.files["file"])
        else:
            rows = (request.get_json(silent=True) or {}).get("rows")
            if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
                raise ValueError('Send a CSV/Excel "file", or JSON {"rows": [{...}, ...]}.')
            frame = pd.DataFrame(rows)
        return jsonify(check_frame(reference_profile(record["folder_path"]), frame))
    except Exception as exc:
        return error_response(exc)


@runs_bp.route("/runs/<int:run_id>/retrain", methods=["POST"])
def retrain_run(run_id):
    write_log(f"new incoming request to retrain run {run_id}", "info", os.path.basename(__file__))
    data = request.get_json(silent=True) or {}
    try:
        task = get_record_by_id(run_id)["task"]
        if task == IMAGE_TASK:
            # Image folders are picked from the server's folders, as for a new image run.
            dataset_path = str(data.get("dataset_path") or "")
        else:
            dataset_path = resolve_data_file(data.get("dataset_path"))
        plan = plan_retrain(run_id, dataset_path, name=data.get("name"), models=data.get("models") or None,
                            original_dataset_name=clean_display_filename(data.get("original_filename")),
                            recipe=data.get("recipe_id") or None)
    except DatasetMismatchError as exc:
        return jsonify({"status": "error", "status_code": 400, "error_message": str(exc), "message": str(exc),
                        "missing_columns": exc.missing_columns, "required_columns": exc.required_columns}), 400
    except ImportError as exc:
        return jsonify({"status": "error", "status_code": 503, "error_message": str(exc), "message": str(exc)}), 503
    except Exception as exc:
        return error_response(exc)

    path_to_save, timestamp = create_run_folder(plan.name)
    try:
        if plan.task == IMAGE_TASK:
            from easyml.cv.cv_controllers import CVController

            train_fn = CVController(kwargs={**plan.kwargs, "path_to_save": path_to_save}).execute_pipeline
        else:
            from easyml.trainer import MLTrainer

            train_fn = MLTrainer(**plan.kwargs, path_to_save=path_to_save, timestamp=timestamp).run
    except Exception as exc:
        remove_orphan_run_folder(path_to_save)
        return error_response(exc)

    job = start_training_job(train_fn, "Retrain", run_folder=path_to_save)
    return jsonify({
        "status": "started",
        "job_id": job.job_id,
        "task": plan.task,
        "parent_run_id": plan.parent_run_id,
        "name": plan.name,
        "warnings": plan.warnings,
        "message": "Retraining started",
    }), 202


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


@runs_bp.route("/runs/<int:run_id>/export", methods=["GET"])
def export_run_archive(run_id):
    include_data = request.args.get("include_data", "1") not in ("0", "false", "no")
    try:
        record = get_record_by_id(run_id)
        handle, temp_path = tempfile.mkstemp(prefix="easyml_run_", suffix=".zip")
        os.close(handle)
        export_run(run_id, output_path=temp_path, include_data=include_data)
    except Exception as exc:
        return error_response(exc)
    download_name = f"easyml-run-{run_id}-{sanitize_run_name(record.get('run_name'))}.zip"
    response = send_file(temp_path, mimetype="application/zip", as_attachment=True, download_name=download_name)
    response.call_on_close(lambda: _remove_quietly(temp_path))
    return response


@runs_bp.route("/runs/import", methods=["POST"])
def import_run_archive():
    if request.content_length and request.content_length > max_import_bytes():
        return jsonify({"status": "error", "status_code": 413,
                        "error_message": "The archive is larger than the import limit."}), 413
    if str(request.form.get("trusted", "")).lower() != "true":
        return jsonify({"status": "error", "status_code": 400,
                        "error_message": TRUST_WARNING + " Confirm that you trust this file to import it."}), 400
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"status": "error", "status_code": 400, "error_message": "Choose a run archive (.zip)."}), 400

    handle, temp_path = tempfile.mkstemp(prefix="easyml_import_", suffix=".zip")
    os.close(handle)
    try:
        upload.save(temp_path)
        run_id, notes = import_archive(temp_path, name=request.form.get("name") or None)
        record = get_record_by_id(run_id)
    except Exception as exc:
        return error_response(exc)
    finally:
        _remove_quietly(temp_path)
    return jsonify({"status": "imported", "run_id": run_id, "run_name": record.get("run_name"),
                    "task": record.get("task"), "warnings": notes}), 201


@runs_bp.route("/runs/<int:run_id>/charts", methods=["GET"])
def run_charts(run_id):
    from easyml.core.evaluation_charts import read_evaluation

    try:
        record = get_record_by_id(run_id)
        evaluation = read_evaluation(record["folder_path"])
    except Exception as exc:
        return error_response(exc)
    if not evaluation:
        return jsonify({"available": False, "run_id": run_id,
                        "reason": "Charts are available for runs trained with Easy-ML 0.6.0 or later."})
    return jsonify({"available": True, "run_id": run_id, **evaluation})
