import json
import os
import shutil
import tempfile
import traceback
from contextlib import contextmanager

import joblib
import numpy as np

from easyml.config.GLOBAL_VARS import (
    CATBOOST_MODEL_NAMES,
    ML_RequestBody,
    error_code_mappings,
    get_default_ml_models,
    get_ml_model_instance,
    get_model_mappings,
    get_param_grid,
)
from easyml.core.evaluation_charts import (
    build_model_charts,
    classification_parts,
    regression_points,
    tabular_importance,
    write_evaluation,
)
from easyml.core.evaluator import get_train_test_metrics
from easyml.core.cancellable_search import CancellableGridSearchCV
from easyml.core.imbalance_handler import unwrap_model, with_resampling
from easyml.utils.cancellation import is_cancelled
from easyml.utils.holdout import write_holdout
from easyml.utils.logger import write_log
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.labels import labels_in_code_order
from easyml.utils.utils import NumpyEncoder


# Models whose fit is the same (same seeding, same saved form) when it uses every core: the trees'
# seeds are drawn before the parallel part, and n_jobs is restored before predicting and saving.
_PARALLEL_FIT = {"RandomForestClassifier", "RandomForestRegressor"}
# Below this many training rows, starting the worker threads costs more than it saves.
_PARALLEL_FIT_MIN_ROWS = 1000


@contextmanager
def _all_cores(model, names, n_rows):
    """Temporarily let ``model`` use every core; its saved parameters are unchanged afterwards."""
    if type(model).__name__ not in names or not hasattr(model, "n_jobs") or n_rows < _PARALLEL_FIT_MIN_ROWS:
        yield model
        return
    previous = model.n_jobs
    model.n_jobs = -1
    try:
        yield model
    finally:
        model.n_jobs = previous


def _final_step(estimator):
    return estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator


def _model_charts(model, request_body, task, y_pred_test=None):
    """ROC/PR (classification) or scatter/residuals (regression), plus feature importance."""
    if task == "classification":
        labels = labels_in_code_order(request_body.target_encodings)
        parts = classification_parts(model, request_body.X_test, request_body.y_test, labels)
    else:
        if y_pred_test is None:
            y_pred_test = np.asarray(model.predict(request_body.X_test)).reshape(-1)
        parts = {"regression": lambda: regression_points(request_body.y_test, y_pred_test)}
    parts["importance"] = lambda: tabular_importance(model, request_body, task)
    return build_model_charts(parts)

def ml_model_builder_trainer_controller(request_body: ML_RequestBody):
    """
    Handles model training and evaluation for classification or regression tasks.

    This function:
    - Selects models based on user input or defaults
    - Applies hyperparameter tuning using GridSearchCV
    - Evaluates models using train/test metrics
    - Saves trained models and configuration JSONs

    Args:
        request_body (ML_RequestBody): Object containing all user configuration and data.

    Raises:
        Exception: If any error occurs during training, it logs and re-raises the exception.
    """

    try:

        write_log(f"entered ml_model_builder_trainer_controller()", "debug", os.path.basename(__file__))

        task = "classification" if "classification" in request_body.task.lower() else "regression" if "regression" in request_body.task.lower() else None

        if task is None:
            raise ValueError(f"Unsupported task '{request_body.task}'")

        mappings = get_model_mappings(task)
        unavailable = [model for model in request_body.models if model not in mappings]
        if unavailable:
            raise ValueError(f"These models are not available in this installation: {', '.join(unavailable)}")

        models_to_apply = (
            [get_ml_model_instance(task, model) for model in request_body.models]
            if request_body.models
            else get_default_ml_models(task)
        )

        # Resampling (SMOTE/Tomek) is part of each model's pipeline, so it only ever sees training folds.
        imbalance_method = request_body.imbalance_handling if task == "classification" else "none"
        if (imbalance_method or "none").lower() != "none":
            emit_progress(f"⚖️ Balancing training data with {imbalance_method.upper()} inside each training fold...", pct=44)

        grid = get_param_grid(task, request_body.hyper_parameter_tuning)

        scoring = 'accuracy' if 'classification' in request_body.task.lower() else 'r2'

        total_models = len(models_to_apply)
        for idx, single_model in enumerate(models_to_apply):
            model_base_pct = int(45 + (idx / max(1, total_models)) * 45)
            emit_progress(f"🤖 Training model {idx+1}/{total_models}: {type(single_model).__name__}...", pct=model_base_pct)

            # Check for cancellation before each model
            if is_cancelled():
                write_log("Training cancelled by user", "info", os.path.basename(__file__))
                raise InterruptedError("Training cancelled by user")

            model_name = type(single_model).__name__
            is_catboost = model_name in CATBOOST_MODEL_NAMES
            param_grid = grid.get(model_name, {})
            write_log(f"training - {single_model} - grid - {param_grid}", "info", os.path.basename(__file__))

            temp_dir = None
            if is_catboost:
                temp_dir = tempfile.mkdtemp()
                single_model.set_params(train_dir=temp_dir, thread_count=1, allow_writing_files=False)

            estimator, search_grid = with_resampling(single_model, param_grid, imbalance_method)
            try:
                if search_grid:
                    search = CancellableGridSearchCV(estimator=estimator, param_grid=search_grid,
                                                     cv=4, scoring=scoring, n_jobs=-1)
                    search.fit(request_body.X_train, request_body.y_train)
                    fitted, best_params = search.best_estimator_, search.best_params_
                else:
                    # No tuning requested: a single fit instead of a 4-fold search over an empty grid.
                    with _all_cores(_final_step(estimator), _PARALLEL_FIT, len(request_body.y_train)):
                        fitted, best_params = estimator.fit(request_body.X_train, request_body.y_train), {}
                # Save and evaluate the bare model: metrics below use the original (not resampled) data.
                best_estimator, best_params = unwrap_model(fitted, best_params)
            finally:
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            y_pred_train = best_estimator.predict(request_body.X_train).reshape(-1)
            y_pred_test = best_estimator.predict(request_body.X_test).reshape(-1)
            metrics = get_train_test_metrics(
                request_body,
                task=task,
                y_real_train=request_body.y_train, y_pred_train=y_pred_train,
                y_real_test=request_body.y_test, y_pred_test=y_pred_test,
                **({'labels' : list(request_body.target_encodings.keys())} if task == 'classification' else {})
            )
            if task == 'regression':
                accuracy_train = metrics['train_metrics'].get('r2_score', None)
                accuracy_train = round(accuracy_train,5) if accuracy_train is not None else accuracy_train
                accuracy_test = metrics['test_metrics'].get('r2_score', None)
                accuracy_test = round(accuracy_test,5) if accuracy_test is not None else accuracy_test
            else:
                accuracy_train = metrics['train_metrics'].get('accuracy', None)
                accuracy_train = round(accuracy_train,5) if accuracy_train is not None else accuracy_train
                accuracy_test = metrics['test_metrics'].get('accuracy', None)
                accuracy_test = round(accuracy_test,5) if accuracy_test is not None else accuracy_test

            if accuracy_train is None:
                write_log(f"Train accuracy None for {type(single_model).__name__} for task {request_body.name}.", level='error', module_name=os.path.basename(__file__))
            if accuracy_test is None:
                write_log(f"Test accuracy None for {type(single_model).__name__} for task {request_body.name}.", level='error', module_name=os.path.basename(__file__))

            extension = "cbm" if is_catboost else "pkl"
            model_path = os.path.join(request_body.path_to_save, f"{model_name}.{extension}")

            write_log(f"trained {single_model} with train accuracy: {accuracy_train} and test_accuracy:"
                      f" {accuracy_test} - saving at {model_path}", "info", os.path.basename(__file__))

            if is_catboost:
                best_estimator.save_model(model_path)
            else:
                joblib.dump(best_estimator, model_path, protocol=5)

            request_body.performance[model_name] = {
                "model_path": model_path,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "best_params": best_params,
                'train_metrics' : metrics['train_metrics'],
                "test_metrics" : metrics['test_metrics'],
            }
            emit_progress(f"📈 Preparing result charts for {model_name}...")
            request_body.charts[model_name] = _model_charts(best_estimator, request_body, task, y_pred_test)
            emit_progress(f"✅ Finished {model_name}", pct=model_base_pct + int(45 / max(1, total_models)))

        labels = labels_in_code_order(request_body.target_encodings) if task == "classification" else None
        write_evaluation(request_body.path_to_save, request_body.task, request_body.charts,
                         {"n_test": len(request_body.y_test), "classes": [str(c) for c in labels] if labels else None})

        # Write column_data.json and config_json.json ONCE after all models are done
        column_data = {}
        column_data['numerical_columns'] = request_body.numeric_columns
        column_data['categorical_columns'] = request_body.categorical_columns
        # Training-time dtypes let inference cast form input (always strings) back to the right types.
        column_data['column_dtypes'] = {col: str(dtype) for col, dtype in request_body.X.dtypes.items()}
        column_data['unique_values'] = {}
        for col in request_body.categorical_columns:
            try:
                column_data['unique_values'][col] = request_body.df[col].dropna().unique().tolist()
            except Exception:
                column_data['unique_values'][col] = []

        json_path = os.path.join(request_body.path_to_save, "config_json.json")
        column_data_path = os.path.join(request_body.path_to_save, "column_data.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(request_body.jsonify(), handle, cls=NumpyEncoder)
        with open(column_data_path, "w", encoding="utf-8") as handle:
            json.dump(column_data, handle, cls=NumpyEncoder)
        if request_body.X_test_raw is not None:
            write_holdout(request_body.path_to_save, request_body.X_test_raw.index, request_body.source_rows)

    except InterruptedError:
        raise
    except Exception as e:
        if request_body.status_code == 200:
           request_body.error_message = "ModelTrainingError"
           request_body.status_code = error_code_mappings[request_body.error_message]
           request_body.error_traceback = traceback.format_exc()

           write_log("error occurred- {} - {}".format(
               request_body.error_message, request_body.error_traceback),
               "error", os.path.basename(__file__))

        raise e
