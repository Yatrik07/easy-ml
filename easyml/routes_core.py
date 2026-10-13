import json
import os
from io import BytesIO

import pandas as pd
from flask import Blueprint, jsonify, request, send_file

from easyml.config.GLOBAL_VARS import (
    ui2back_mappings_classification_models,
    ui2back_mappings_regression_models,
)
from easyml.capabilities import get_capabilities
from easyml.core.artifacts import validate_model_name
from easyml.core.inference import MLInferenceController
from easyml.route_helpers import (
    create_run_folder,
    error_response,
    load_class_distribution,
    optional_import_error_response,
    parse_metrics_json,
    remove_orphan_run_folder,
    resolve_data_file,
    save_dataframe_request,
    start_training_job,
)
from easyml.trainer import MLTrainer
from easyml.utils.cancellation import get_job
from easyml.utils.data_loader import get_data_info
from easyml.utils.database_operations import (
    delete_run_by_id,
    get_data_from_database,
    get_record_by_id,
)
from easyml.utils.logger import write_log

core_bp = Blueprint("easyml_core", __name__)


@core_bp.route("/load_data", methods=["POST"])
def load_data():
    try:
        data, saved_path = save_dataframe_request("data")
        data_info = get_data_info(data, dataset_path=saved_path)
        data_info["saved_path"] = saved_path
        return jsonify(data_info)
    except Exception as exc:
        write_log(f"Failed to load dataset: {exc}", "error", os.path.basename(__file__))
        return jsonify({"status": "error", "message": f"Failed to load dataset: {str(exc)}"}), 400


@core_bp.route("/class_distribution", methods=["POST"])
def class_distribution():
    try:
        payload = request.get_json() or {}
        dataset_path = payload.get("dataset_path")
        target_column = payload.get("target_column")

        if not dataset_path or not target_column:
            return jsonify({"error": "Missing dataset_path or target_column"}), 400

        return jsonify(load_class_distribution(dataset_path, target_column))
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@core_bp.route("/generate_eda", methods=["GET"])
def generate_eda():
    write_log("generating EDA report", "info", os.path.basename(__file__))
    try:
        from easyml.core.eda_generator import generate_eda_report
    except ImportError:
        return optional_import_error_response("EDA reports", 'Install with `pip install "easy-ml-studio[eda]"`.')

    try:
        dataset_path = resolve_data_file(request.args.get("path"))
        report_path = generate_eda_report(pd.read_csv(dataset_path), "current_dataset")
        return send_file(report_path, mimetype="text/html", as_attachment=False)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        write_log(f"EDA generation failed: {str(exc)}", "error", os.path.basename(__file__))
        return jsonify({"error": str(exc)}), 500


@core_bp.route("/get_models", methods=["GET"])
def get_models():
    capabilities = get_capabilities()
    return jsonify(
        {
            "MLClassification": list(ui2back_mappings_classification_models.keys()),
            "MLRegression": list(ui2back_mappings_regression_models.keys()),
            "NLPTextClassification": capabilities["models"]["NLPTextClassification"],
            "ImageClassification": capabilities["models"]["ImageClassification"],
        }
    )


@core_bp.route("/ml_predict", methods=["POST"])
def ml_predict():
    write_log("new incoming request to /ml_predict", "info", os.path.basename(__file__))
    data = request.get_json(silent=True) or {}

    task = data.get("task")
    name = data.get("name")
    target_column = data.get("target_column")

    if not task or not name or not data.get("dataset_path") or not target_column:
        return jsonify({"status_code": 400, "error_message": "Missing required fields (task, name, dataset_path, or target_column)"}), 400

    if task not in ("MLClassification", "MLRegression"):
        return jsonify({"status_code": 400, "error_message": f"Unsupported task '{task}'"}), 400

    models = data.get("models", [])
    if not models:
        return jsonify({"status_code": 400, "error_message": "Must select at least one model to train"}), 400

    available = ui2back_mappings_classification_models if task == "MLClassification" else ui2back_mappings_regression_models
    unavailable = [m for m in models if m not in available]
    if unavailable:
        return jsonify({"status_code": 400, "error_message": f"Models not available in this installation: {', '.join(unavailable)}"}), 400

    try:
        dataset_path = resolve_data_file(data.get("dataset_path"))
    except Exception as exc:
        return error_response(exc)

    scaling_raw = data.get("scaling", "No")
    reduce_features_raw = data.get("reduce_features", "No")
    path_to_save, timestamp = create_run_folder(name)

    try:
        trainer = MLTrainer(
            name=name,
            task=task,
            dataset_path=dataset_path,
            target_column=target_column,
            preprocessing=True,
            imputation=data.get("imputation_strategy", "impute") != "drop",
            scaling=scaling_raw == "Yes" if isinstance(scaling_raw, str) else bool(scaling_raw),
            models=models,
            hyper_parameter_tuning=data.get("hyper_parameter_tuning", 0),
            categorical_columns=data.get("categorical_columns", []),
            path_to_save=path_to_save,
            reduce_features=reduce_features_raw == "Yes" if isinstance(reduce_features_raw, str) else bool(reduce_features_raw),
            ignore_columns=data.get("ignore_columns", []),
            timestamp=timestamp,
            imbalance_handling=data.get("imbalance_handling", "none"),
        )
    except Exception as exc:
        remove_orphan_run_folder(path_to_save)
        return error_response(exc)

    job = start_training_job(trainer.run, "ML", run_folder=path_to_save)
    return jsonify({"status": "started", "job_id": job.job_id, "message": "Background ML training started"}), 202


@core_bp.route("/runs", methods=["GET", "DELETE"])
def manage_runs():
    if request.method == "GET":
        result = get_data_from_database()
        return jsonify(result), result.get("status_code", 200)

    data = request.get_json(silent=True) or {}
    run_id = data.get("run_id")
    if run_id is None:
        return jsonify({"error": "Missing run_id"}), 400

    result = delete_run_by_id(run_id)
    if result.get("status_code") == 200:
        return jsonify({"message": f"Run with ID {run_id} deleted successfully."}), 200
    return jsonify({"error": f"Run with ID {run_id} not found or could not be deleted."}), result.get("status_code", 500)


def _list_run_models(result, data_path):
    models = parse_metrics_json(result.get("metrics_json"))
    if not models and data_path and os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            stem, ext = os.path.splitext(filename)
            if ext in (".pkl", ".cbm", ".pth") and stem != "preprocessing_pipeline":
                models[stem.replace(".pkl", "")] = {}
    return [name for name in models.keys() if name != "preprocessing_pipeline"]


@core_bp.route("/get_run_data", methods=["GET"])
def get_run_data():
    run_id = request.args.get("run_id")
    if not run_id:
        return jsonify({"error": "Missing run_id parameter"}), 400

    try:
        result = get_record_by_id(record_id=int(run_id))
    except Exception as exc:
        return error_response(exc)

    data_path = result["folder_path"]
    column_data = []
    column_details = {}
    target = ""

    column_data_json_path = os.path.join(data_path, "column_data.json")
    config_json_path = os.path.join(data_path, "config_json.json")

    try:
        if os.path.exists(column_data_json_path):
            with open(column_data_json_path, "r", encoding="utf-8") as handle:
                col_data = json.load(handle)
            column_data = col_data.get("numerical_columns", []) + col_data.get("categorical_columns", [])
            unique_values = col_data.get("unique_values", {})
            for col in col_data.get("numerical_columns", []):
                column_details[col] = {"type": "numeric"}
            for col in col_data.get("categorical_columns", []):
                column_details[col] = {"type": "categorical", "values": unique_values.get(col, [])}

        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as handle:
                config_data = json.load(handle)
            target = config_data.get("target_column", "")
            if not column_data:
                column_data = config_data.get("columns", [])
    except Exception as exc:
        write_log(f"Error reading config files for run {run_id}: {str(exc)}", "error", os.path.basename(__file__))

    return jsonify({
        "columns": column_data,
        "column_details": column_details,
        "target_column": target,
        "task": result.get("task"),
        "models": _list_run_models(result, data_path),
    }), 200


@core_bp.route("/inference", methods=["POST"])
def inference():
    try:
        json_data = request.get_json(force=True)
        controller = MLInferenceController(
            run_id=int(json_data["run_id"]),
            data=json_data["data"],
            model_name=validate_model_name(json_data["model_name"]),
        )
        return jsonify(controller.get_inference())
    except Exception as exc:
        write_log(f"Error in /inference: {str(exc)}", "error", os.path.basename(__file__))
        return error_response(exc)


@core_bp.route("/inference_batch", methods=["POST"])
def inference_batch():
    try:
        run_id = int(request.form["run_id"])
        model_name = validate_model_name(request.form["model_name"])
        file = request.files.get("file")
        if not file:
            return jsonify({"status_code": 400, "error_message": "No file uploaded"}), 400

        df = pd.read_csv(file)
        controller = MLInferenceController(run_id=run_id, data=None, model_name=model_name)
        output_df = controller.predict_batch(df)

        output_buffer = BytesIO()
        output_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        return send_file(
            output_buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"predictions_run_{run_id}.csv",
        )
    except Exception as exc:
        write_log(f"Error in /inference_batch: {str(exc)}", "error", os.path.basename(__file__))
        return error_response(exc)


@core_bp.route("/explain", methods=["POST"])
def explain():
    if not get_capabilities()["features"]["explain"]:
        return optional_import_error_response("Explainability (SHAP)", 'Install with `pip install "easy-ml-studio[eda]"`.')

    from easyml.core.xai_explainer import XAIExplainer

    try:
        json_data = request.get_json(force=True)
        explainer = XAIExplainer(
            run_id=int(json_data["run_id"]),
            data=json_data["data"],
            model_name=validate_model_name(json_data["model_name"]),
        )
        return jsonify(explainer.explain()), 200
    except Exception as exc:
        write_log(f"Error in /explain: {str(exc)}", "error", os.path.basename(__file__))
        return error_response(exc)


@core_bp.route("/export", methods=["POST"])
def export_model():
    from easyml.utils.export_utils import create_deployment_package

    try:
        json_data = request.get_json(force=True)
        run_id = int(json_data["run_id"])
        model_name = validate_model_name(json_data["model_name"])
        zip_path = create_deployment_package(run_id, model_name)
        return send_file(zip_path, as_attachment=True, download_name=f"deploy_{model_name}_{run_id}.zip")
    except Exception as exc:
        write_log(f"Error in /export: {str(exc)}", "error", os.path.basename(__file__))
        return error_response(exc)


@core_bp.route("/stop_training", methods=["POST"])
def stop_training():
    """Cancel one training job, identified by the ``job_id`` its start endpoint returned."""
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"status_code": 400, "error_message": "Missing job_id of the training run to stop."}), 400
    job = get_job(job_id)
    if job is None:
        return jsonify({"status_code": 404, "error_message": "Unknown training job."}), 404
    if job.finished:
        return jsonify({"status": "idle", "message": "This training run has already finished."}), 200
    job.cancel()
    write_log(f"Cancellation requested for {job.label} job {job_id}", "info", os.path.basename(__file__))
    return jsonify({"status": "ok", "message": "Training stop signal sent. Training will stop at the next checkpoint."}), 200
