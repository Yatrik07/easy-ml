"""
Endpoints that act on a saved run: its settings, retraining it on new data, and (later in this
module) exporting/importing it and its result charts.
"""

import os

from flask import Blueprint, jsonify, request

from easyml.retraining import IMAGE_TASK, DatasetMismatchError, plan_retrain, run_spec
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

runs_bp = Blueprint("easyml_runs", __name__)


@runs_bp.route("/runs/<int:run_id>/config", methods=["GET"])
def run_config(run_id):
    try:
        return jsonify(run_spec(run_id))
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
                            original_dataset_name=clean_display_filename(data.get("original_filename")))
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
