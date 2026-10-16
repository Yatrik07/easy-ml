import os

from flask import Blueprint, jsonify, request

from easyml.capabilities import get_capabilities
from easyml.nlp.nlp_config import AVAILABLE_NLP_MODELS
from easyml.nlp.nlp_inference import NLPInferenceController
from easyml.route_helpers import (
    recipe_summary,
    clean_display_filename,
    create_run_folder,
    error_response,
    optional_import_error_response,
    remove_orphan_run_folder,
    resolve_data_file,
    save_dataframe_request,
    start_training_job,
    uploaded_original_filename,
)
from easyml.trainer import MLTrainer
from easyml.utils.data_loader import get_data_info
from easyml.utils.logger import write_log

nlp_bp = Blueprint("easyml_nlp", __name__)


@nlp_bp.route("/nlp/load_data", methods=["POST"])
def nlp_load_data():
    try:
        data, saved_path = save_dataframe_request("data_nlp")
        data_info = get_data_info(data, dataset_path=saved_path)
        data_info["saved_path"] = saved_path
        data_info["original_filename"] = uploaded_original_filename()
        return jsonify(data_info)
    except Exception as exc:
        write_log(f"Failed to load NLP dataset: {exc}", "error", os.path.basename(__file__))
        return jsonify({"status": "error", "message": f"Failed to load dataset: {str(exc)}"}), 400


@nlp_bp.route("/nlp/get_models", methods=["GET"])
def nlp_get_models():
    return jsonify({"NLPTextClassification": list(AVAILABLE_NLP_MODELS.keys())})


@nlp_bp.route("/nlp/predict", methods=["POST"])
def nlp_predict():
    write_log("new incoming request to /nlp/predict", "info", os.path.basename(__file__))
    data = request.get_json(silent=True) or {}

    name = data.get("name")
    text_column = data.get("text_column")
    target_column = data.get("target_column")
    if not name or not data.get("dataset_path") or not text_column or not target_column:
        return jsonify({"status_code": 400, "error_message": "Missing required fields (name, dataset_path, text_column, target_column)"}), 400

    models = data.get("models", [])
    if not models:
        return jsonify({"status_code": 400, "error_message": "Must select at least one NLP model to train"}), 400
    unavailable = [m for m in models if m not in AVAILABLE_NLP_MODELS]
    if unavailable:
        return jsonify({"status_code": 400, "error_message": f"Models not available in this installation: {', '.join(unavailable)}"}), 400

    feature_mode = data.get("feature_mode", "vectorizer")
    if feature_mode not in ("vectorizer", "embedding"):
        return jsonify({"status_code": 400, "error_message": "feature_mode must be 'vectorizer' or 'embedding'"}), 400

    capabilities = get_capabilities()
    if feature_mode == "embedding" and not capabilities["features"]["embeddings"]:
        return optional_import_error_response("Sentence embeddings", f'Install with `{capabilities["install_hints"]["embeddings"]}`.')

    try:
        dataset_path = resolve_data_file(data.get("dataset_path"))
    except Exception as exc:
        return error_response(exc)

    path_to_save, timestamp = create_run_folder(name)

    try:
        trainer = MLTrainer(
            name=name,
            task="NLPTextClassification",
            dataset_path=dataset_path,
            target_column=target_column,
            text_column=text_column,
            lowercase=data.get("lowercase", True),
            remove_stopwords=data.get("remove_stopwords", True),
            lemmatize=data.get("lemmatize", False),
            regex_clean=data.get("regex_clean", True),
            regex_pattern=data.get("regex_pattern", r"[^a-zA-Z0-9\s]"),
            feature_mode=feature_mode,
            vectorizer_type=data.get("vectorizer_type", "tfidf"),
            max_features=data.get("max_features", 10000),
            ngram_range=data.get("ngram_range", [1, 2]),
            min_df=data.get("min_df", 2),
            max_df=data.get("max_df", 0.95),
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_batch_size=data.get("embedding_batch_size", 32),
            embedding_max_length=data.get("embedding_max_length", 256),
            models=models,
            hyper_parameter_tuning=data.get("hyper_parameter_tuning", 0),
            imputation=data.get("imputation_strategy", "impute") != "drop",
            path_to_save=path_to_save,
            timestamp=timestamp,
            imbalance_handling=data.get("imbalance_handling", "none"),
            original_dataset_name=clean_display_filename(data.get("original_filename")),
            recipe=recipe_summary(data.get("recipe")),
            random_seed=data.get("random_seed", 42),
            cv_folds=data.get("cv_folds", 0),
        )
    except Exception as exc:
        remove_orphan_run_folder(path_to_save)
        return error_response(exc)

    job = start_training_job(trainer.run, "NLP", run_folder=path_to_save)
    return jsonify({"status": "started", "job_id": job.job_id, "message": "Background NLP training started"}), 202


@nlp_bp.route("/nlp/inference", methods=["POST"])
def nlp_inference():
    try:
        json_data = request.get_json(force=True)
        controller = NLPInferenceController(
            run_id=int(json_data["run_id"]),
            data=json_data["data"],
            model_name=json_data["model_name"],
        )
        return jsonify(controller.get_inference())
    except Exception as exc:
        write_log(f"Error in /nlp/inference: {exc}", "error", os.path.basename(__file__))
        return error_response(exc)
