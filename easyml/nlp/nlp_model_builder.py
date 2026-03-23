"""
NLP Model Builder Module
=========================
Handles model selection, training via GridSearchCV, evaluation,
and persistence for NLP classification tasks.

Mirrors easyml/core/model_builder.py for NLP tasks.
"""

import os
import json
import traceback
import joblib
import scipy.sparse

from sklearn.model_selection import GridSearchCV

from easyml.nlp.nlp_config import (
    NLP_RequestBody, NLP_ERROR_CODES,
    get_nlp_model_instance, DEFAULT_NLP_MODELS,
    NLP_PARAM_GRID_LEVEL1, NLP_PARAM_GRID_LEVEL2,
    SPARSE_COMPATIBLE_MODELS, DENSE_ONLY_MODELS,
)
from easyml.nlp.nlp_evaluator import get_nlp_train_test_metrics
from easyml.utils.logger import write_log
from easyml.utils.utils import NumpyEncoder
from easyml.utils.socket_emitter import emit_progress


def nlp_model_builder_controller(request_body: NLP_RequestBody):
    """
    Trains and evaluates NLP models.

    For each selected model:
    1. Instantiates the model
    2. Runs GridSearchCV with the configured hyperparameter level
    3. Evaluates on train + test sets
    4. Saves best estimator to disk
    5. Stores metrics in request_body.performance

    Args:
        request_body (NLP_RequestBody): Data carrier with features and config.

    Raises:
        Exception: Re-raises with NLPModelTrainingError code.
    """
    try:
        write_log("entered nlp_model_builder_controller", "info", os.path.basename(__file__))

        # Determine which models to train
        model_names = request_body.models if request_body.models else DEFAULT_NLP_MODELS

        # Select hyperparameter grid level
        if request_body.hyper_parameter_tuning == 0:
            grid = {}
        elif request_body.hyper_parameter_tuning == 1:
            grid = NLP_PARAM_GRID_LEVEL1
        else:
            grid = NLP_PARAM_GRID_LEVEL2

        # Determine if input is sparse
        is_sparse = scipy.sparse.issparse(request_body.X_train)

        for model_name in model_names:
            write_log(f"training NLP model: {model_name}", "info", os.path.basename(__file__))
            emit_progress(f"training NLP model: {model_name}")

            try:
                model = get_nlp_model_instance(model_name)
            except ValueError as e:
                write_log(f"skipping unknown model: {model_name} - {e}", "error", os.path.basename(__file__))
                continue

            class_name = type(model).__name__

            # Handle sparse/dense compatibility
            X_train = request_body.X_train
            X_test = request_body.X_test

            if is_sparse and model_name in DENSE_ONLY_MODELS:
                write_log(f"converting sparse to dense for {model_name}", "info", os.path.basename(__file__))
                X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
                X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test

            # Get param grid for this model
            param_grid = grid.get(class_name, {})

            write_log(f"GridSearchCV for {class_name} with params: {param_grid}", "info", os.path.basename(__file__))

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=4,
                scoring='accuracy',
                n_jobs=-1,
                error_score='raise'
            )

            grid_search.fit(X_train, request_body.y_train)

            # Predict
            y_pred_train = grid_search.predict(X_train)
            y_pred_test = grid_search.predict(X_test)

            # Evaluate
            labels = list(request_body.target_encodings.keys()) if request_body.target_encodings else None
            kwargs = {'labels': labels} if labels else {}

            metrics = get_nlp_train_test_metrics(
                request_body,
                y_real_train=request_body.y_train, y_pred_train=y_pred_train,
                y_real_test=request_body.y_test, y_pred_test=y_pred_test,
                **kwargs
            )

            accuracy_train = metrics['train_metrics'].get('accuracy')
            accuracy_test = metrics['test_metrics'].get('accuracy')
            accuracy_train = round(accuracy_train, 5) if accuracy_train is not None else None
            accuracy_test = round(accuracy_test, 5) if accuracy_test is not None else None

            f1_train = metrics['train_metrics'].get('f1_weighted')
            f1_test = metrics['test_metrics'].get('f1_weighted')
            f1_train = round(f1_train, 5) if f1_train is not None else None
            f1_test = round(f1_test, 5) if f1_test is not None else None

            # Save model
            model_path = os.path.join(request_body.path_to_save, f"{class_name}.pkl")
            joblib.dump(grid_search.best_estimator_, model_path, protocol=5)

            write_log(
                f"trained {class_name}: train_acc={accuracy_train}, test_acc={accuracy_test}, "
                f"train_f1={f1_train}, test_f1={f1_test} - saved to {model_path}",
                "info", os.path.basename(__file__)
            )

            # Store performance
            request_body.performance[class_name] = {
                "model_path": model_path,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "f1_train": f1_train,
                "f1_test": f1_test,
                "train_metrics": metrics['train_metrics'],
                "test_metrics": metrics['test_metrics'],
                "best_params": grid_search.best_params_,
            }

            emit_progress(f"done with NLP model: {class_name} (test_acc={accuracy_test})")

        # Save run config and column data
        config_path = os.path.join(request_body.path_to_save, "config_json.json")
        json.dump(request_body.jsonify(), open(config_path, "w"), cls=NumpyEncoder)

        column_data = {
            "text_column": request_body.text_column,
            "target_column": request_body.target_column,
            "task_type": "nlp_classification",
        }
        column_data_path = os.path.join(request_body.path_to_save, "column_data.json")
        json.dump(column_data, open(column_data_path, "w"), cls=NumpyEncoder)

        write_log("nlp_model_builder_controller completed", "info", os.path.basename(__file__))

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "NLPModelTrainingError"
            request_body.status_code = NLP_ERROR_CODES[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()
            write_log(f"error occurred - {request_body.error_message} - {request_body.error_traceback}",
                      "error", os.path.basename(__file__))
        raise e
