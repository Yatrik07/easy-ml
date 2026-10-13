import os

from flask import Blueprint, current_app, jsonify, request

from easyml.capabilities import get_capabilities
from easyml.config.paths import DATA_DIR
from easyml.route_helpers import (
    create_run_folder,
    error_response,
    optional_import_error_response,
    remove_orphan_run_folder,
    start_training_job,
)
from easyml.utils.logger import write_log

CV_INSTALL_HINT = 'Install with `pip install "easy-ml-studio[cv]"`.'

cv_bp = Blueprint("easyml_cv", __name__)


def _cv_unavailable():
    if get_capabilities()["features"]["cv"]:
        return None
    return optional_import_error_response("Computer Vision", CV_INSTALL_HINT)


@cv_bp.route("/cv/list_datasets", methods=["GET"])
def cv_list_datasets():
    try:
        data_dir = current_app.config.get("EASYML_DATA_DIR", DATA_DIR)
        exempt = {"runs", "database", "logs", "eda", "uploads"}
        folders = []

        if os.path.exists(data_dir):
            for directory in sorted(os.listdir(data_dir)):
                full_path = os.path.join(data_dir, directory)
                if os.path.isdir(full_path) and directory not in exempt:
                    folders.append({"name": directory, "path": full_path})

        uploads_dir = os.path.join(data_dir, "uploads")
        if os.path.exists(uploads_dir):
            for directory in sorted(os.listdir(uploads_dir)):
                full_path = os.path.join(uploads_dir, directory)
                if os.path.isdir(full_path):
                    folders.append({"name": f"uploads/{directory}", "path": full_path})

        return jsonify({"folders": folders})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@cv_bp.route("/cv/predict", methods=["POST"])
def cv_predict():
    unavailable = _cv_unavailable()
    if unavailable:
        return unavailable

    from easyml.cv.cv_config import AVAILABLE_CV_MODELS
    from easyml.cv.cv_controllers import CVController

    try:
        payload = request.get_json(force=True) or {}
        if not payload.get("name") or not payload.get("dataset_path"):
            return jsonify({"status_code": 400, "error_message": "Missing required fields (name, dataset_path)"}), 400
        models = payload.get("models") or []
        if not models:
            return jsonify({"status_code": 400, "error_message": "Must select at least one model to train"}), 400
        unknown = [m for m in models if m not in AVAILABLE_CV_MODELS]
        if unknown:
            return jsonify({"status_code": 400, "error_message": f"Unknown CV models: {', '.join(unknown)}"}), 400

        path_to_save, _ = create_run_folder(payload["name"])
    except Exception as exc:
        return error_response(exc)

    try:
        payload["path_to_save"] = path_to_save
        controller = CVController(kwargs=payload)
    except Exception as exc:
        remove_orphan_run_folder(path_to_save)
        return error_response(exc)

    job = start_training_job(controller.execute_pipeline, "CV", run_folder=path_to_save)
    return jsonify({"status": "started", "job_id": job.job_id, "message": "Background CV training started"}), 202


@cv_bp.route("/cv/inference", methods=["POST"])
def cv_inference():
    unavailable = _cv_unavailable()
    if unavailable:
        return unavailable

    from easyml.cv.cv_inference import CVInferenceController

    try:
        json_data = request.get_json(force=True)
        controller = CVInferenceController(
            run_id=int(json_data["run_id"]),
            image_base64=json_data["data"],
            model_name=json_data["model_name"],
        )
        return jsonify(controller.get_inference()), 200
    except Exception as exc:
        write_log(f"Error in /cv/inference: {exc}", "error", os.path.basename(__file__))
        return error_response(exc)


@cv_bp.route("/cv/validate_model", methods=["POST"])
def cv_validate_model():
    unavailable = _cv_unavailable()
    if unavailable:
        return unavailable

    try:
        from easyml.cv.custom_cnn import validate_architecture

        data = request.get_json(force=True)
        result = validate_architecture(
            data.get("layers", []),
            input_channels=3,
            input_size=tuple(data.get("input_size", [224, 224])),
            num_classes=data.get("num_classes", 10),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"valid": False, "error": str(exc)}), 400
